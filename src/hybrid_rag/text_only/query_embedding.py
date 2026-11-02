"""Embed original train/dev questions; reserve train labels for example retrieval only."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from hybrid_rag.text_only.corpus_preprocessing import DEV, hashes_of, run_step
from hybrid_rag.text_only.corpus_embedding import BGE_SOURCE, embed_records
from multimodal_legal_rag.data import DEFAULT_SOURCE, DATA_FILES


def read_splits():
    source = DEFAULT_SOURCE / DATA_FILES['train']
    paths = [DEV.with_name('train.json'), DEV, source]
    hashes = hashes_of(paths)
    raw = json.loads(source.read_text(encoding='utf-8'))
    original = {row['id']: row for row in raw}
    if not original or len(original) != len(raw):
        raise ValueError("Train gốc rỗng hoặc trùng ID.")
    records, ids = [], set()
    for split, path in zip(('train', 'dev'), paths[:2]):
        rows = json.loads(path.read_text(encoding='utf-8'))
        if not rows:
            raise ValueError(f"Split {split} rỗng.")
        for row in rows:
            if row['id'] in ids or row['id'] not in original:
                raise ValueError(f"Train/dev giao nhau, trùng hoặc sai ID: {row['id']}")
            if (any(row.get(k) != original[row['id']].get(k) for k in ('question', 'image_id', 'relevant_articles'))
                    or not isinstance(row.get('question'), str) or not row['question'].strip()
                    or not isinstance(row.get('image_id'), str) or not row['image_id']
                    or not isinstance(row.get('relevant_articles'), list) or not row['relevant_articles']):
                raise ValueError(f"Split khác train gốc hoặc thiếu question/image/citation: {row['id']}")
            ids.add(row['id'])
            records.append({**row, 'source_split': split})
    if ids != set(original):
        raise ValueError("Train + dev không phủ đúng toàn bộ train gốc.")
    return sorted(records, key=lambda r: (r['source_split'], r['id'])), hashes


def query_inputs():
    original, hashes = read_splits()
    queries = []
    for row in original:
        queries.append({'query_id': row['id'], 'source_image_id': row['image_id'],
                        'source_split': row['source_split'], 'text': row['question']})
    return queries, original, hashes


def embed_query(args, folder, meta, progress):
    queries, _, hashes = query_inputs()
    meta.update(split=args.split, representation="original_question_only", image_id_usage="metadata_only")
    meta['split_counts'] = {split: sum(q['source_split'] == split for q in queries) for split in ('train', 'dev')}
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
