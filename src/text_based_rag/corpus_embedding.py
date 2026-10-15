"""Download the pinned text encoder or embed corpus chunks into a local cosine index."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
import shutil
from threading import Event, Thread
import time

from .corpus_normalization import ROOT
from .corpus_image_captioning import read_lines, save

MODEL = "BAAI/bge-m3"
REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
CACHE = ROOT / "models/huggingface/hub"
SNAPSHOT = CACHE / "models--BAAI--bge-m3" / "snapshots" / REVISION
INPUT = ROOT / "artifacts/experiments/text_based_rag/corpus_chunking"
OUTPUT = ROOT / "artifacts/experiments/text_based_rag/corpus_embedding"


def file_hash(path):
    digest = sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def load_encoder():
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Cần GPU NVIDIA CUDA hỗ trợ BF16 trong .venv.")
    free, total = torch.cuda.mem_get_info()
    budget = min(8 * 1024**3, free - 2 * 1024**3)
    if budget < 3 * 1024**3:
        raise RuntimeError("Cần ít nhất 5 GiB VRAM trống; đóng Qwen hoặc ứng dụng GPU rồi chạy lại.")
    # shortcut: the allocator cap excludes other CUDA processes; lower batch size if a batch hits OOM.
    torch.cuda.set_per_process_memory_fraction(budget / total)
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(SNAPSHOT, local_files_only=True, dtype=torch.bfloat16,
                                      attn_implementation="sdpa").to("cuda").eval()
    if model.config.model_type != "xlm-roberta" or model.config.hidden_size != 1024:
        raise ValueError("Checkpoint không khớp text encoder BGE-M3 đã chọn.")
    model.config.use_cache = False
    resources = {"gpu": torch.cuda.get_device_name(0), "total_vram_bytes": total,
                 "free_vram_before_load": free, "pytorch_budget_bytes": budget}
    return tokenizer, model, resources


def embed_batch(texts, tokenizer, model):
    import numpy as np
    import torch

    inputs = tokenizer(texts, padding=True, truncation=False, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        output = model(**inputs).last_hidden_state[:, 0].float()
        norms = torch.linalg.vector_norm(output, dim=1, keepdim=True)
        if not torch.isfinite(output).all().item() or (norms <= 0).any().item():
            raise ValueError("Embedding không hợp lệ.")
        embedded = (output / norms).cpu().numpy()
    assert embedded.shape == (len(texts), 1024) and np.allclose(np.linalg.norm(embedded, axis=1), 1, atol=1e-5)
    return embedded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "index"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true", help="Replace corpus index after validating new embeddings.")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size phải >= 1")
    if args.command == "download" and args.overwrite:
        parser.error("--overwrite chỉ dùng với index")
    if args.command == "index" and OUTPUT.exists() and not args.overwrite:
        raise FileExistsError("Output corpus_embedding đã tồn tại; dùng index --overwrite để thay thế.")
    started = time.perf_counter()
    stopped = Event()
    phase, done, total = ("download", 0, 1) if args.command == "download" else ("prepare-index", 0, 0)
    received, download_bytes = 0, 0

    def report():
        elapsed = time.perf_counter() - started
        eta = f"{elapsed * (total - done) / done:.0f}s" if done else "unknown"
        print(f"[{phase}] {done}/{total} ({100 * done / max(1, total):.1f}%) "
              f"elapsed={elapsed:.0f}s ETA={eta}", flush=True)
        if download_bytes:
            eta_bytes = f"{elapsed * (download_bytes - received) / received:.0f}s" if received else "unknown"
            print(f"[download-bytes] {received}/{download_bytes} ({100 * received / download_bytes:.1f}%) "
                  f"elapsed={elapsed:.0f}s ETA={eta_bytes}", flush=True)

    def heartbeat():
        while not stopped.wait(30):
            report()

    worker = Thread(target=heartbeat, daemon=True)
    worker.start()
    report()
    manifest = None
    try:
        if args.command == "download":
            import requests
            from huggingface_hub import HfApi, hf_hub_url, snapshot_download
            info = HfApi().model_info(MODEL, revision=REVISION, files_metadata=True)
            available = {entry.rfilename for entry in info.siblings}
            weight = "model.safetensors" if "model.safetensors" in available else "pytorch_model.bin"
            if weight not in available:
                raise ValueError("Không tìm thấy weight text encoder ở revision đã chọn.")
            snapshot_download(MODEL, revision=REVISION, cache_dir=CACHE, max_workers=2,
                              allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json",
                                              "special_tokens_map.json", "sentencepiece.bpe.model"])
            metadata = next(entry for entry in info.siblings if entry.rfilename == weight)
            if not metadata.lfs or not metadata.size:
                raise ValueError("Weight thiếu kích thước/SHA256 để kiểm tra tải xuống.")
            target = SNAPSHOT / weight
            partial = target.with_suffix(target.suffix + ".incomplete")
            existing = target if target.exists() else partial
            received = existing.stat().st_size if existing.exists() else 0
            download_bytes = metadata.size
            if received > download_bytes:
                raise ValueError(f"File tải dở lớn bất thường: {existing}")
            phase = "download-weight"
            # Direct GET avoids the stalled weight HEAD redirect previously observed on this machine.
            if not target.exists() and received < download_bytes:
                with requests.get(hf_hub_url(MODEL, weight, revision=REVISION), stream=True,
                                  headers={"Range": f"bytes={received}-"} if received else {},
                                  timeout=(15, 30)) as response:
                    response.raise_for_status()
                    if received and response.status_code == 200:
                        received = 0
                    if received and not response.headers.get("Content-Range", "").startswith(f"bytes {received}-"):
                        raise ValueError("Server trả vị trí tải tiếp không khớp.")
                    with partial.open("ab" if received else "wb") as stream:
                        for part in response.iter_content(1024 * 1024):
                            stream.write(part)
                            received += len(part)
            if received != download_bytes or file_hash(existing if target.exists() else partial) != metadata.lfs.sha256:
                raise ValueError("Weight tải chưa đủ hoặc SHA256 sai; giữ file để kiểm tra, không dùng inference.")
            if not target.exists():
                partial.replace(target)
            done = 1
            report()
            print(f"[model] {SNAPSHOT}", flush=True)
            return

        source_manifest = json.loads((INPUT / "manifest.json").read_text(encoding="utf-8"))
        if source_manifest.get("status") != "COMPLETED":
            raise ValueError("Bước chia chunk chưa hoàn tất.")
        phase = "validate-chunk-input"
        input_hashes = {name: file_hash(INPUT / name) for name in
                        ("chunks.jsonl", "organized_corpus.jsonl", "manifest.json")}
        for name in ("chunks.jsonl", "organized_corpus.jsonl"):
            if source_manifest.get("output_sha256", {}).get(name) != input_hashes[name]:
                raise ValueError(f"Chunking output không khớp manifest: {name}; chạy lại chunking trước.")
        chunks = read_lines(INPUT / "chunks.jsonl")
        parents = read_lines(INPUT / "organized_corpus.jsonl")
        total = len(chunks)
        if not chunks or any(not isinstance(c.get("text"), str) or not c["text"].strip() for c in chunks):
            raise ValueError("Danh sách chunk rỗng hoặc có text không hợp lệ.")
        if len({c["chunk_id"] for c in chunks}) != total:
            raise ValueError("Trùng chunk_id.")
        parent_map = {p["article_key"]: p for p in parents}
        if len(parent_map) != len(parents):
            raise ValueError("Trùng article_key của parent.")
        if source_manifest.get("chunks") != total or source_manifest.get("articles") != len(parents):
            raise ValueError("Số chunk/parent không khớp manifest chunking.")
        for chunk in chunks:
            parent = parent_map.get(chunk.get("parent_id"))
            if (parent is None or chunk.get("article_key") != chunk["parent_id"]
                    or (chunk.get("document_id"), chunk.get("id")) != (parent.get("document_id"), parent.get("id"))):
                raise ValueError(f'Chunk không khớp parent/citation: {chunk["chunk_id"]}')
        weights = [p for p in SNAPSHOT.glob("*") if p.name in {"model.safetensors", "pytorch_model.bin"}]
        if not weights:
            raise FileNotFoundError("Chưa có weight BGE-M3. Chạy: python -m text_based_rag.corpus_embedding download")
        phase = "load-text-encoder"
        import numpy as np
        import torch
        tokenizer, model, resources = load_encoder()
        lengths = [len(tokenizer(c["text"], truncation=False)["input_ids"]) for c in chunks]
        maximum = min(8192, model.config.max_position_embeddings - 2)
        oversized = [c["chunk_id"] for c, length in zip(chunks, lengths) if length > maximum]
        if oversized:
            raise ValueError(f"Chunk vượt giới hạn {maximum} token; chia lại chunk: {oversized[:5]}")
        if any(length != chunk.get("token_count") or length > source_manifest["max_tokens"]
               for chunk, length in zip(chunks, lengths)):
            raise ValueError("Token count/ngân sách chunk không khớp tokenizer embedding.")
        OUTPUT.mkdir(parents=True, exist_ok=True)
        manifest = {
            "status": "IN_PROGRESS", "created_at": datetime.now(timezone.utc).isoformat(),
            "model": MODEL, "revision": REVISION, "model_path": str(SNAPSHOT),
            "pooling": "CLS", "normalize": "L2", "dimension": 1024, "stored_dtype": "float32",
            "inference_dtype": "bfloat16", "attention": "sdpa", "batch_size": args.batch_size,
            "index_type": "numpy_exact_inner_product", "similarity": "cosine", "input_prefix": "",
            "input_field": "text", "overwrite": args.overwrite, "parents": len(parents),
            "source_sha256": file_hash(ROOT / "src/text_based_rag/corpus_embedding.py"),
            "chunks": total, "input_max_tokens": max(lengths), "model_max_tokens": maximum,
            "input_sha256": input_hashes,
            "model_files_sha256": {p.name: file_hash(p) for p in
                                   [*weights, SNAPSHOT / "config.json", SNAPSHOT / "tokenizer.json"]},
            "libraries": {name: version(name) for name in ("torch", "transformers", "numpy")},
            "resources": resources,
        }
        save(OUTPUT / "manifest.json", manifest)
        phase = "corpus-embedding-inference"
        torch.cuda.reset_peak_memory_stats()
        vectors = np.lib.format.open_memmap(OUTPUT / "vectors.npy.tmp", mode="w+", dtype="float32",
                                           shape=(total, 1024))
        for start in range(0, total, args.batch_size):
            batch = chunks[start:start + args.batch_size]
            embedded = embed_batch([c["text"] for c in batch], tokenizer, model)
            vectors[start:start + len(batch)] = embedded
            vectors.flush()
            done = start + len(batch)
            del embedded
        del vectors
        phase = "validate-and-save-index"
        with (OUTPUT / "chunks.jsonl.tmp").open("w", encoding="utf-8") as stream:
            for row, chunk in enumerate(chunks):
                stream.write(json.dumps({**chunk, "vector_row": row}, ensure_ascii=False) + "\n")
        shutil.copyfile(INPUT / "organized_corpus.jsonl", OUTPUT / "organized_corpus.jsonl.tmp")
        # Read back the actual index files to catch row mapping or incomplete writes.
        stored = np.load(OUTPUT / "vectors.npy.tmp", mmap_mode="r", allow_pickle=False)
        mapping = read_lines(OUTPUT / "chunks.jsonl.tmp")
        assert stored.shape == (total, 1024) and np.isfinite(stored).all(), "Invalid saved vectors"
        assert np.allclose(np.linalg.norm(stored, axis=1), 1, atol=1e-5), "Vectors must be normalized"
        assert len(mapping) == total and all(m["vector_row"] == i and m["chunk_id"] == chunks[i]["chunk_id"]
                                            for i, m in enumerate(mapping)), "Vector row mapping mismatch"
        del stored
        if any(file_hash(INPUT / name) != digest for name, digest in input_hashes.items()):
            raise ValueError("Input chunking đã thay đổi trong lúc embedding; chưa thay index cũ.")
        for name in ("vectors.npy", "chunks.jsonl", "organized_corpus.jsonl"):
            (OUTPUT / (name + ".tmp")).replace(OUTPUT / name)
        manifest.update(status="COMPLETED", elapsed_seconds=time.perf_counter() - started,
                        index_bytes=(OUTPUT / "vectors.npy").stat().st_size,
                        output_sha256={name: file_hash(OUTPUT / name) for name in
                                       ("vectors.npy", "chunks.jsonl", "organized_corpus.jsonl")})
        manifest["resources"].update(peak_vram_bytes=torch.cuda.max_memory_allocated(),
                                     peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                                     free_vram_after_inference=torch.cuda.mem_get_info()[0])
        save(OUTPUT / "manifest.json", manifest)
        report()
        print(f"[output] {OUTPUT}; vectors={total}; dimensions=1024; cosine", flush=True)
    except BaseException as exc:
        if manifest is not None:
            manifest.update(status="FAILED", error=str(exc), embedded_chunks=done)
            save(OUTPUT / "manifest.json", manifest)
        raise
    finally:
        stopped.set()
        worker.join()


if __name__ == "__main__":
    main()
