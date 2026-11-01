"""Embed unique dev query images and map each question to its DINOv2 vector."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from hybrid_rag.full_image_method.corpus_image_embedding import (
    BASE, checked_image, embed_images, hashes_of, run_step,
)

QUERY_IMAGES = ROOT / "data/processed/images/train"
DEV = ROOT / "data/processed/splits/dev.json"


def dev_inputs(progress):
    hashes = hashes_of([DEV])
    queries = sorted(json.loads(DEV.read_text(encoding="utf-8")), key=lambda r: r["id"])
    if not queries or len({q['id'] for q in queries}) != len(queries):
        raise ValueError("Dev rỗng hoặc trùng query ID.")
    progress.set("validate dev images", len(queries))
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
                "image_id": "sha256:" + query_hashes[q['id']], "vector_row": rows[query_hashes[q['id']]]}
               for q in queries]
    return images, mapping, queries, hashes


def embed_query(args, folder, meta, progress):
    images, queries, _, hashes = dev_inputs(progress)
    meta["inventory"] = {"unique_images": len(images), "queries": len(queries)}
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
    run_step(args, BASE / "query_dev", embed_query, Path(__file__))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
