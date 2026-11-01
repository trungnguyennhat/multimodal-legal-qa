"""Chunk original legal text, including table text, with BGE-M3 token budgets."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from hybrid_rag.text_only.corpus_preprocessing import load_step, read_lines, write_lines, hashes_of, run_step
from text_based_rag.corpus_chunking import Organizer, chunk_record, heading
from text_based_rag.corpus_embedding import SNAPSHOT


def chunk(args, folder, meta, progress):
    from transformers import AutoTokenizer

    _, source, hashes = load_step("data_preprocessing", ["structured_corpus.jsonl"])
    tokenizer_files = [SNAPSHOT / n for n in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")]
    hashes.update(hashes_of(tokenizer_files))
    meta.update(input_sha256=hashes, max_tokens=args.max_tokens, overlap_tokens=args.overlap_tokens,
                method="ordered_original_text", tables="text_can_be_split", figures=False, vlm_captions=False)
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("Cần fast tokenizer BGE-M3 local để chia text.")
    tokenizer.model_max_length = 10**9
    records = read_lines(source / "structured_corpus.jsonl")
    organizer = Organizer(records, {})
    result = []
    progress.set("chunk legal text", len(records))
    for record in records:
        if any(b['type'] != 'paragraph' for b in record['blocks']):
            raise ValueError("Text-only corpus còn block ảnh hoặc bảng chưa chuyển thành text.")
        units = organizer.units(record)
        article_chunks = chunk_record(record, units, tokenizer, args.max_tokens, args.overlap_tokens, organizer)
        if not article_chunks:
            text = heading(record)
            if not text.strip():
                raise ValueError(f"Article không có cả text lẫn tiêu đề: {record['article_key']}")
            article_chunks = [{"chunk_id": record['article_key'] + '/c0001', "article_key": record['article_key'],
                               "parent_id": record['article_key'], "document_id": record['document_id'], "id": record['id'],
                               "title": record['title'], "text": text, "chunk_type": "heading_only", "source_block_ids": [],
                               "token_count": len(tokenizer(text, truncation=False)['input_ids']), "part_index": 0,
                               "previous_chunk_id": None, "next_chunk_id": None}]
        for row in article_chunks:
            if row.pop('image_ids', []) or row['token_count'] > args.max_tokens or not row['text'].strip():
                raise ValueError(f"Chunk sai ngân sách hoặc có ảnh: {row['chunk_id']}")
            row['vector_row'] = len(result)
            result.append(row)
        progress.done += 1
    if len({r['chunk_id'] for r in result}) != len(result):
        raise ValueError("Trùng chunk_id.")
    write_lines(folder / "chunks.jsonl", result)
    temporary = folder / "embedding_text_preview.txt.tmp"
    temporary.write_text("\n\n".join(f"[{r['chunk_id']}]\n{r['text']}" for r in result) + "\n", encoding="utf-8")
    temporary.replace(folder / "embedding_text_preview.txt")
    meta.update(articles=len(records), chunks=len(result), max_chunk_tokens=max(r['token_count'] for r in result),
                output_sha256=hashes_of([folder / n for n in ("chunks.jsonl", "embedding_text_preview.txt")]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--overlap-tokens", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 128 <= args.max_tokens <= 8192 or not 0 <= args.overlap_tokens < args.max_tokens // 2:
        parser.error("Cần 128 <= max-tokens <= 8192; 0 <= overlap-tokens < max-tokens/2")
    run_step(args, "corpus_chunking", chunk, Path(__file__), [ROOT / "src/text_based_rag/corpus_chunking.py"])


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
