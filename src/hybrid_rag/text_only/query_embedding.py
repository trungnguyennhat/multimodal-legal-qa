"""Embed original dev questions only: no image, caption, choices or answers."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from hybrid_rag.text_only.corpus_preprocessing import DEV, hashes_of, run_step
from hybrid_rag.text_only.corpus_embedding import BGE_SOURCE, embed_records


def query_inputs():
    hashes = hashes_of([DEV])
    original = sorted(json.loads(DEV.read_text(encoding="utf-8")), key=lambda r: r['id'])
    if not original or len({r['id'] for r in original}) != len(original):
        raise ValueError("Dev rỗng hoặc trùng query ID.")
    queries = []
    for row in original:
        if not isinstance(row.get('question'), str) or not row['question'].strip():
            raise ValueError(f"Câu hỏi rỗng/không hợp lệ: {row['id']}")
        queries.append({'query_id': row['id'], 'source_image_id': row['image_id'], 'text': row['question']})
    return queries, original, hashes


def embed_query(args, folder, meta, progress):
    queries, _, hashes = query_inputs()
    meta.update(split=args.split, representation="original_question_only", image_id_usage="metadata_only")
    embed_records(args, folder, meta, progress, queries, "queries.jsonl", hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev",), default="dev")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size phải >= 1")
    run_step(args, "query_embedding_dev", embed_query, Path(__file__),
             [Path(__file__).with_name("corpus_embedding.py"), BGE_SOURCE])


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
