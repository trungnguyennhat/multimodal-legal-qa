"""Embed full corpus images with frozen DINOv2; share encoder and artifact validation with query steps."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import re
import sys
from threading import Event, Thread
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from text_based_rag.corpus_embedding import file_hash
from text_based_rag.corpus_image_captioning import read_lines, save

BASE = ROOT / "artifacts/experiments/hybrid_rag/full_image_method"
MODEL = "facebook/dinov2-base"
MODEL_DIR = ROOT / "models/huggingface/dinov2-base"
CORPUS = ROOT / "artifacts/experiments/text_based_rag/corpus/data_preprocessing"
LAW_IMAGES = ROOT / "data/processed/images/law"
MODEL_FILES = ("config.json", "preprocessor_config.json", "model.safetensors")
CONFIG = {"model": MODEL, "pooling": "last_hidden_state_CLS", "dimension": 768,
          "normalize": "L2", "stored_dtype": "float32", "inference_dtype": "float32",
          "image_size": 518, "resize": "aspect_ratio_preserving_bicubic",
          "padding": "center", "padding_rgb": [124, 116, 104], "exif_transpose": True,
          "color": "RGB", "center_crop": False, "seed": 2025}


class Progress:
    def __init__(self):
        self.started = self.stage_started = time.monotonic()
        self.stage, self.done, self.total = "initialize", 0, 0
        self.stop = Event()
        self.worker = Thread(target=self.heartbeat, daemon=True)

    def set(self, stage, total=0):
        self.stage, self.done, self.total = stage, 0, total
        self.stage_started = time.monotonic()
        self.log()

    def log(self):
        elapsed = time.monotonic() - self.started
        percent = f"{100 * self.done / self.total:.1f}%" if self.total else "N/A"
        eta = (time.monotonic() - self.stage_started) * max(0, self.total-self.done) / self.done if self.done else None
        print(f"[{self.stage}] {self.done}/{self.total or '?'} ({percent}); elapsed={elapsed:.0f}s; "
              f"ETA={f'{eta:.0f}s' if eta is not None else 'N/A'}", flush=True)

    def heartbeat(self):
        while not self.stop.wait(30):
            self.log()


def check_hashes(hashes):
    for relative, expected in hashes.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Input thiếu hoặc thay đổi: {relative}")


def hashes_of(paths):
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


def checked_image(path, directory, expected=None):
    from PIL import Image

    path = path.resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError(f"Đường dẫn ảnh không hợp lệ: {path}")
    digest = file_hash(path)
    if expected is not None and digest != expected:
        raise ValueError(f"Hash ảnh không khớp: {path}")
    with Image.open(path) as image:
        image.load()
    return digest


def corpus_inputs(progress):
    paths = [CORPUS / name for name in ("manifest.json", "structured_corpus.jsonl", "image_manifest.jsonl")]
    hashes = hashes_of(paths)
    meta = json.loads(paths[0].read_text(encoding="utf-8"))
    if meta.get("status") != "COMPLETED" or meta.get("schema_version") != 4:
        raise ValueError("Corpus preprocessing phải COMPLETED, schema 4.")
    articles = read_lines(paths[1])
    registry = {a["article_key"]: a for a in articles}
    if len(registry) != len(articles):
        raise ValueError("article_key bị trùng trong corpus.")
    images = sorted(read_lines(paths[2]), key=lambda r: r["image_id"])
    if not images or len({r['image_id'] for r in images}) != len(images):
        raise ValueError("Danh sách ảnh corpus rỗng hoặc trùng image_id.")
    progress.set("validate corpus images", len(images))
    mapping = []
    for row in images:
        path = LAW_IMAGES / row["filename"]
        if path.resolve() != Path(row["path"]).resolve() or row["image_id"] != "sha256:" + row["sha256"]:
            raise ValueError(f"Image manifest không hợp lệ: {row['filename']}")
        checked_image(path, LAW_IMAGES, row["sha256"])
        citations = set()
        for occurrence in row["occurrences"]:
            article = registry[occurrence["article_key"]]
            if occurrence["block_id"] not in article["all_block_ids"]:
                raise ValueError(f"Block ảnh không thuộc article: {occurrence}")
            citations.add((str(article["document_id"]), str(article["id"])))
        if not citations:
            raise ValueError(f"Ảnh không có citation: {row['filename']}")
        mapping.append({"image_id": row["image_id"], "path": str(path.relative_to(ROOT)),
                        "sha256": row["sha256"], "vector_row": len(mapping),
                        "citations": [{"law_id": law, "article_id": article} for law, article in sorted(citations)],
                        "occurrences": row["occurrences"]})
        hashes[str(path.relative_to(ROOT))] = row["sha256"]
        progress.done += 1
    return mapping, {(str(a['document_id']), str(a['id'])) for a in articles}, hashes


def verified_model():
    meta = json.loads((MODEL_DIR / "manifest.json").read_text(encoding="utf-8"))
    if meta.get("status") != "COMPLETED" or meta.get("model") != MODEL or not re.fullmatch(r"[0-9a-f]{40}", meta.get("revision", "")):
        raise ValueError("Model chưa COMPLETED hoặc revision sai; chạy image_model_download.py trước.")
    expected = {str((MODEL_DIR / name).relative_to(ROOT)) for name in MODEL_FILES}
    if set(meta["output_sha256"]) != expected:
        raise ValueError("Model manifest thiếu file bắt buộc.")
    check_hashes(meta["output_sha256"])
    return meta


def validate_vectors(vectors, count):
    if (vectors.shape != (count, 768) or vectors.dtype != np.float32 or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)):
        raise ValueError("Vector phải là FP32, hữu hạn, shape (N,768), norm L2=1.")


def write_lines(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def embed_images(args, folder, meta, progress, images, queries, hashes):
    import torch
    from PIL import Image, ImageOps
    from transformers import AutoModel

    progress.set("validate local model")
    model_meta = verified_model()
    hashes.update(model_meta["output_sha256"])
    hashes.update(hashes_of([MODEL_DIR / "manifest.json"]))
    processor = json.loads((MODEL_DIR / "preprocessor_config.json").read_text(encoding="utf-8"))
    mean, std = processor["image_mean"], processor["image_std"]
    if len(mean) != 3 or len(std) != 3 or not np.isfinite(mean + std).all() or min(std) <= 0:
        raise ValueError("Normalization model không hợp lệ.")
    meta.update(config={**CONFIG, "revision": model_meta["revision"], "image_mean": mean, "image_std": std},
                batch_size=args.batch_size, input_sha256=hashes)
    save(folder / "manifest.json", meta)
    if not torch.cuda.is_available():
        raise RuntimeError("Cần CUDA trong .venv; không tự chuyển sang CPU.")
    torch.manual_seed(2025)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    meta["resources"] = {"gpu": torch.cuda.get_device_name(0), "free_vram_before_load": torch.cuda.mem_get_info()[0]}
    progress.set("load frozen DINOv2")
    model = AutoModel.from_pretrained(MODEL_DIR, local_files_only=True, use_safetensors=True,
                                      dtype=torch.float32).to("cuda").eval()
    model.requires_grad_(False)
    if model.config.model_type != "dinov2" or model.config.hidden_size != 768 or model.config.patch_size != 14:
        raise ValueError("Checkpoint không khớp DINOv2 ViT-B/14.")
    mean_tensor = torch.tensor(mean, dtype=torch.float32)[:, None, None]
    std_tensor = torch.tensor(std, dtype=torch.float32)[:, None, None]
    progress.set("embed full images", len(images))
    vectors = np.lib.format.open_memmap(folder / "vectors.npy.tmp", mode="w+", dtype="float32", shape=(len(images), 768))
    try:
        for start in range(0, len(images), args.batch_size):
            batch = []
            for row in images[start:start + args.batch_size]:
                with Image.open(ROOT / row["path"]) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                    image = ImageOps.pad(image, (518, 518), method=Image.Resampling.BICUBIC, color=(124, 116, 104))
                    pixels = torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1).copy()) / 255
                    batch.append((pixels - mean_tensor) / std_tensor)
            with torch.inference_mode():
                features = model(pixel_values=torch.stack(batch).to("cuda")).last_hidden_state[:, 0].float()
                norms = torch.linalg.vector_norm(features, dim=1, keepdim=True)
                if not torch.isfinite(features).all().item() or (norms <= 0).any().item():
                    raise ValueError("Model sinh vector không hợp lệ.")
                embedded = (features / norms).cpu().numpy()
            validate_vectors(embedded, len(batch))
            vectors[start:start + len(batch)] = embedded
            vectors.flush()
            progress.done += len(batch)
    finally:
        del vectors
        meta["resources"].update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                                 peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    progress.set("validate saved index")
    stored = np.load(folder / "vectors.npy.tmp", mmap_mode="r", allow_pickle=False)
    validate_vectors(stored, len(images))
    del stored
    check_hashes(hashes)
    (folder / "vectors.npy.tmp").replace(folder / "vectors.npy")
    write_lines(folder / "images.jsonl", images)
    names = ["vectors.npy", "images.jsonl"]
    if queries is not None:
        write_lines(folder / "queries.jsonl", queries)
        names.append("queries.jsonl")
    meta["output_sha256"] = hashes_of([folder / name for name in names])


def run_step(args, folder, action, entrypoint):
    if folder.exists() and args.command != "download" and not args.overwrite:
        raise FileExistsError(f"{folder} đã tồn tại; dùng --overwrite cho bước này.")
    output_root = ROOT / "models/huggingface" if args.command == "download" else BASE
    if (folder.is_symlink() or not folder.resolve().is_relative_to(output_root.resolve())
            or not folder.resolve().is_relative_to(ROOT)):
        raise ValueError(f"Output phải nằm trong {output_root}, không dùng symlink.")
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "manifest.json"
    meta = json.loads(manifest.read_text(encoding="utf-8")) if args.command == "download" and manifest.exists() else {}
    meta.update(command=args.command, created_at=datetime.now(timezone.utc).isoformat(),
                code_sha256=hashes_of(sorted({Path(__file__).resolve(), entrypoint.resolve()})), arguments=vars(args),
                packages={name: version(name) for name in ("torch", "transformers", "huggingface-hub", "numpy", "pillow")})
    progress = Progress()
    progress.worker.start()
    try:
        if args.command == "download":
            if meta.get("status") == "COMPLETED":
                verified_model()
                print("Model local đã được xác minh; không đổi manifest hoặc tải lại.", flush=True)
                return
            action(args, folder, meta, progress)
        else:
            meta['status'] = "RUNNING"
            save(manifest, meta)
            action(args, folder, meta, progress)
        if meta.get("input_sha256"):
            check_hashes(meta["input_sha256"])
        check_hashes(meta["output_sha256"])
        meta.update(status="COMPLETED", elapsed_seconds=time.monotonic()-progress.started)
        save(manifest, meta)
        print(f"[output] {folder}", flush=True)
    except BaseException as error:
        meta.update(status="FAILED", error=f"{type(error).__name__}: {error}",
                    failed_stage=progress.stage, processed=progress.done, total=progress.total,
                    elapsed_seconds=time.monotonic()-progress.started)
        save(manifest, meta)
        raise
    finally:
        progress.stop.set()
        progress.worker.join()
        progress.log()


def embed_corpus(args, folder, meta, progress):
    images, registry, hashes = corpus_inputs(progress)
    meta["inventory"] = {"images": len(images), "registry_citations": len(registry),
                         "citations_with_images": len({(c['law_id'], c['article_id']) for r in images for c in r['citations']})}
    embed_images(args, folder, meta, progress, images, None, hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true", help="Replace corpus index; no batch resume.")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size phải >= 1")
    args.command = "embed-corpus"
    run_step(args, BASE / "corpus_embedding", embed_corpus, Path(__file__))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()

