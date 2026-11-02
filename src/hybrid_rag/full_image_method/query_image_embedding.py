"""Embed unique train/dev images; keep query and example-bank membership explicit."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from hybrid_rag.full_image_method.corpus_image_embedding import (
    BASE, checked_image, embed_images, hashes_of, run_step,
)
from hybrid_rag.text_only.query_embedding import read_splits

QUERY_IMAGES = ROOT / "data/processed/images/train"
SPLIT_SOURCE = ROOT / "src/hybrid_rag/text_only/query_embedding.py"
SPLIT_HELPERS = ROOT / "src/hybrid_rag/text_only/corpus_preprocessing.py"


def dev_inputs(progress):
    queries, hashes = read_splits()
    progress.set("validate train/dev images", len(queries))
    unique, query_hashes = {}, {}
    for query in queries:
        if not query.get("relevant_articles") or not isinstance(query.get("image_id"), str):
            raise ValueError(f"Dev thiếu gold hoặc image_id: {query['id']}")
        path = QUERY_IMAGES / (query["image_id"] + ".jpg")
        relative = str(path.relative_to(ROOT))
        digest = hashes.get(relative)
        if digest is None:
            digest = checked_image(path, QUERY_IMAGES)
            hashes[relative] = digest
        unique.setdefault(digest, {"image_id": "sha256:" + digest, "sha256": digest, "path": relative})
        query_hashes[query['id']] = digest
        progress.done += 1
    images = [{**unique[digest], "vector_row": i} for i, digest in enumerate(sorted(unique))]
    rows = {row['sha256']: row['vector_row'] for row in images}
    mapping = [{"query_id": q["id"], "source_image_id": q["image_id"],
                "source_split": q['source_split'],
                "image_id": "sha256:" + query_hashes[q['id']], "vector_row": rows[query_hashes[q['id']]]}
               for q in queries]
    return images, mapping, queries, hashes


def embed_query(args, folder, meta, progress):
    images, queries, _, hashes = dev_inputs(progress)
    meta.update(split=args.split, query_target='dev', example_bank_split='train')
    meta["inventory"] = {"unique_images": len(images), "queries": len(queries)}
    meta['split_counts'] = {split: sum(q['source_split'] == split for q in queries) for split in ('train', 'dev')}
    hashes_by_split = {split: {q['image_id'] for q in queries if q['source_split'] == split} for split in ('train', 'dev')}
    meta['inventory']['shared_image_hashes_train_dev'] = len(hashes_by_split['train'] & hashes_by_split['dev'])
    meta['code_sha256'].update(hashes_of([SPLIT_SOURCE, SPLIT_HELPERS]))
    embed_images(args, folder, meta, progress, images, queries, hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev",), default="dev")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true", help="Replace query index; no batch resume.")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size phải >= 1")
    args.command = "embed-query"
    run_step(args, BASE / "query_embedding_dev", embed_query, Path(__file__))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
