"""Embed pure legal-text chunks using the existing local BGE-M3 checkpoint."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from hybrid_rag.text_only.corpus_preprocessing import (
    check_hashes, hashes_of, load_step, read_lines, run_step, save, write_lines,
)
from text_based_rag.corpus_embedding import MODEL, REVISION, SNAPSHOT, load_encoder, embed_batch

BGE_SOURCE = ROOT / "src/text_based_rag/corpus_embedding.py"
CONFIG = {"model": MODEL, "revision": REVISION, "pooling": "CLS", "normalize": "L2", "dimension": 1024,
          "inference_dtype": "bfloat16", "stored_dtype": "float32", "input_prefix": "", "captions": False}


def validate_vectors(vectors, count):
    if (vectors.shape != (count, 1024) or vectors.dtype != np.float32 or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)):
        raise ValueError("Vector phải là FP32, (N,1024), hữu hạn và norm L2=1.")


def embed_records(args, folder, meta, progress, rows, filename, hashes):
    import torch

    weights = [p for p in SNAPSHOT.iterdir() if p.name in {"model.safetensors", "pytorch_model.bin"}]
    if not weights:
        raise FileNotFoundError("Thiếu model BGE-M3 local; không tự tải model.")
    hashes.update(hashes_of([*weights, *[SNAPSHOT / n for n in (
        "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model")]]))
    if not rows or any(not isinstance(r.get('text'), str) or not r['text'].strip() for r in rows):
        raise ValueError("Danh sách embedding rỗng hoặc có text không hợp lệ.")
    meta.update(input_sha256=hashes, config=CONFIG, rows=len(rows), batch_size=args.batch_size)
    save(folder / "manifest.json", meta)
    progress.set("load local BGE-M3")
    tokenizer, model, resources = load_encoder()
    model.requires_grad_(False)
    meta['resources'] = resources
    lengths = [len(tokenizer(r['text'], truncation=False)['input_ids']) for r in rows]
    maximum = min(8192, model.config.max_position_embeddings - 2)
    if any(n > maximum for n in lengths):
        raise ValueError(f"Text vượt {maximum} token; không tự truncate.")
    if any(r.get('token_count', n) != n for r, n in zip(rows, lengths)):
        raise ValueError("Token count khác chunking.")
    meta.update(max_input_tokens=max(lengths), model_max_tokens=maximum)
    torch.cuda.reset_peak_memory_stats()
    progress.set("embed original text", len(rows))
    vectors = np.lib.format.open_memmap(folder / "vectors.npy.tmp", mode="w+", dtype="float32", shape=(len(rows), 1024))
    try:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start+args.batch_size]
            embedded = embed_batch([r['text'] for r in batch], tokenizer, model)
            validate_vectors(embedded, len(batch))
            vectors[start:start+len(batch)] = embedded
            vectors.flush()
            progress.done += len(batch)
    finally:
        del vectors
        resources.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                         peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    progress.set("validate and save text index")
    stored = np.load(folder / "vectors.npy.tmp", mmap_mode="r", allow_pickle=False)
    validate_vectors(stored, len(rows))
    del stored
    check_hashes(hashes)
    (folder / "vectors.npy.tmp").replace(folder / "vectors.npy")
    mapping = [{**r, 'vector_row': i} for i, r in enumerate(rows)]
    write_lines(folder / filename, mapping)
    meta['output_sha256'] = hashes_of([folder / "vectors.npy", folder / filename])


def embed_corpus(args, folder, meta, progress):
    _, source, hashes = load_step("corpus_chunking", ["chunks.jsonl"])
    rows = read_lines(source / "chunks.jsonl")
    if len({r['chunk_id'] for r in rows}) != len(rows):
        raise ValueError("Trùng chunk_id.")
    meta['representation'] = "original_legal_text_without_images_or_vlm_captions"
    embed_records(args, folder, meta, progress, rows, "chunks.jsonl", hashes)


def load_index(name, filename):
    meta, folder, hashes = load_step(name, ["vectors.npy", filename])
    rows = read_lines(folder / filename)
    if any(r.get('vector_row') != i for i, r in enumerate(rows)):
        raise ValueError(f"Mapping vector không hợp lệ: {name}")
    vectors = np.load(folder / "vectors.npy", allow_pickle=False)
    validate_vectors(vectors, len(rows))
    if meta.get('config') != CONFIG:
        raise ValueError(f"Model/config index không đúng nhánh text_only: {name}")
    return meta, rows, vectors, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size phải >= 1")
    run_step(args, "corpus_embedding", embed_corpus, Path(__file__), [BGE_SOURCE])


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
