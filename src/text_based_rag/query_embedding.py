"""Embed train/dev/public-test questions and image descriptions with the corpus encoder."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from threading import Event, Thread
import time

from multimodal_legal_rag.data import DATA_FILES, DEFAULT_SOURCE, read_json_list
from .corpus_embedding import MODEL, REVISION, SNAPSHOT, file_hash, load_encoder, embed_batch
from .corpus_image_captioning import ROOT, IMAGE_CATEGORIES, ITEM_FIELDS, read_lines, save
from .corpus_chunking import caption_parts

INPUT = ROOT / "data/processed/splits/train.json"
CAPTIONS = ROOT / "artifacts/experiments/text_based_rag/query_image_captioning_train"
CORPUS = ROOT / "artifacts/experiments/text_based_rag/corpus_embedding"
OUTPUT = ROOT / "artifacts/experiments/text_based_rag/query_embedding_train"


def query_representation(question, caption, query_id):
    try:
        payload = json.loads(caption['description'])
    except json.JSONDecodeError as exc:
        if not caption.get('accepted_by_user') or caption.get('original_status') != 'INCOMPLETE':
            raise ValueError(f'Query {query_id}: JSON caption bị cắt chưa được chấp nhận.') from exc
        category, recovered, _ = caption_parts(caption)
        items = recovered
        parse_mode = 'recovered_complete_items'
    else:
        if (not isinstance(payload, dict) or set(payload) != {'image_category', 'summary', 'items'}
                or payload['image_category'] not in IMAGE_CATEGORIES
                or not isinstance(payload['summary'], str) or not payload['summary'].strip()
                or not isinstance(payload['items'], list)):
            raise ValueError(f'Query {query_id}: caption không khớp schema query mới.')
        category, items, parse_mode = payload['image_category'], list(enumerate(payload['items'])), 'json'
    selected, false_indices, invalid_indices, duplicate_indices = [], [], [], []
    seen = set()
    for index, item in items:
        if (not isinstance(item, dict) or type(item.get('is_relevant')) is not bool
                or any(not isinstance(item.get(k), str) or not item[k].strip() for k in ITEM_FIELDS)):
            if parse_mode == 'json':
                raise ValueError(f'Query {query_id}: item {index} thiếu trường hoặc is_relevant không phải boolean.')
            invalid_indices.append(index)
            continue
        if item['is_relevant'] is False:
            false_indices.append(index)
            continue
        values = tuple(item[k].strip() for k in ITEM_FIELDS)
        if values in seen:
            duplicate_indices.append(index)
            continue
        seen.add(values)
        selected.append((index, values))
    text = 'Câu hỏi:\n' + question
    if selected:
        if category:
            text += '\n\nLoại hình: ' + category
        for _, (identifier, features, meaning) in selected:
            text += f'\n\nĐối tượng: {identifier}\nĐặc điểm: {features}\nÝ nghĩa hoặc thông số: {meaning}'
    reason = None if selected else ('truncated_no_recoverable_relevant_item'
                                   if parse_mode != 'json' else 'no_relevant_item')
    return {'text': text, 'image_category': category, 'parse_mode': parse_mode,
            'representation_mode': 'question_relevant_items' if selected else 'question_only',
            'fallback_reason': reason, 'selected_item_indices': [index for index, _ in selected],
            'parsed_item_count': len(items), 'selected_item_count': len(selected),
            'false_item_indices': false_indices, 'invalid_item_indices': invalid_indices,
            'duplicate_item_indices': duplicate_indices}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument('--split', choices=('train', 'dev', 'public_test'), default='train')
    parser.add_argument('--overwrite', action='store_true', help='Replace query embedding output after validating new vectors.')
    parser.add_argument('--allow-incomplete-captions', action='store_true',
                        help='Use nonempty truncated captions, retaining their INCOMPLETE status.')
    args = parser.parse_args()
    INPUT = ROOT / f'data/processed/splits/{args.split}.json'
    suffix = f'_{args.split}'
    CAPTIONS = ROOT / f'artifacts/experiments/text_based_rag/query_image_captioning{suffix}'
    OUTPUT = ROOT / f'artifacts/experiments/text_based_rag/query_embedding{suffix}'
    IMAGES = ROOT / 'data/processed/images/train'
    if args.split == 'public_test':
        INPUT = DEFAULT_SOURCE / DATA_FILES['public_test']
        IMAGES = ROOT / 'data/processed/images/public_test'
        CAPTIONS = ROOT / 'artifacts/public-test/text_based_rag/query_image_captioning'
        OUTPUT = ROOT / 'artifacts/public-test/text_based_rag/query_embedding'
    if args.batch_size < 1:
        parser.error("batch-size phải >= 1")
    if OUTPUT.exists() and not args.overwrite:
        raise FileExistsError(f"Output {OUTPUT} đã tồn tại; dùng --overwrite để thay thế.")
    started, stopped = time.perf_counter(), Event()
    phase, done, total = "prepare-query-text", 0, 0
    manifest = None

    def report():
        elapsed = time.perf_counter() - started
        eta = f"{elapsed * (total - done) / done:.0f}s" if done else "unknown"
        print(f"[{phase}] {done}/{total} ({100 * done / max(1, total):.1f}%) "
              f"elapsed={elapsed:.0f}s ETA={eta}", flush=True)

    def heartbeat():
        while not stopped.wait(30):
            report()

    worker = Thread(target=heartbeat, daemon=True)
    worker.start()
    report()
    try:
        caption_manifest = json.loads((CAPTIONS / "manifest.json").read_text(encoding="utf-8-sig"))
        corpus_manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
        allowed_statuses = {'COMPLETED', 'INCOMPLETE'} if args.allow_incomplete_captions else {'COMPLETED'}
        if (caption_manifest.get("status") not in allowed_statuses or caption_manifest.get("split") != args.split
                or caption_manifest.get('input_sha256') != file_hash(INPUT)):
            raise ValueError("Caption chưa hoàn tất hoặc không khớp split/nguồn query.")
        if caption_manifest.get('settings', {}).get('caption_unit') != 'query_id':
            raise ValueError('Cần caption schema query mới theo ảnh + câu hỏi và is_relevant; không dùng caption cũ theo ảnh.')
        expected = {"status": "COMPLETED", "model": MODEL, "revision": REVISION,
                    "pooling": "CLS", "normalize": "L2", "dimension": 1024,
                    "stored_dtype": "float32", "inference_dtype": "bfloat16", "input_prefix": ""}
        if any(corpus_manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("Encoder của index corpus không khớp BGE-M3 hiện tại; kiểm tra corpus_embedding/manifest.json.")
        if corpus_manifest.get('input_sha256', {}).get('chunks.jsonl') != file_hash(ROOT / 'artifacts/experiments/text_based_rag/corpus_chunking/chunks.jsonl'):
            raise ValueError('Corpus index chưa khớp chunking hiện tại; chạy lại corpus_embedding index trước.')
        model_hashes = corpus_manifest["model_files_sha256"]
        for name, digest in model_hashes.items():
            path = (SNAPSHOT / name).resolve()
            if path.parent != SNAPSHOT.resolve() or file_hash(path) != digest:
                raise ValueError(f"Model local khác model đã embedding corpus: {name}")
        rows = read_json_list(INPUT)
        mapping = read_lines(CAPTIONS / "query_image_mapping.jsonl")
        by_id = {item["query_id"]: item for item in mapping}
        if (not rows or len(by_id) != len(mapping) or len({row["id"] for row in rows}) != len(rows)
                or set(by_id) != {row["id"] for row in rows}):
            raise ValueError("Danh sách query không khớp mapping caption hoặc trùng ID.")
        total = len(rows)
        queries, caption_hashes, image_hashes = [], {}, {}
        for row in rows:
            item = by_id[row["id"]]
            path = (CAPTIONS / item["caption_file"]).resolve()
            image_path = (IMAGES / item["filename"]).resolve()
            if (path.parent != (CAPTIONS / "captions").resolve()
                    or image_path.parent != IMAGES.resolve()
                    or item["source_image_id"] != row["image_id"]
                    or item["filename"] != row["image_id"] + ".jpg"):
                raise ValueError(f"Mapping ảnh không hợp lệ: {row['id']}")
            if item["filename"] not in image_hashes:
                image_hashes[item["filename"]] = "sha256:" + file_hash(image_path)
            caption = json.loads(path.read_text(encoding="utf-8-sig"))
            description, question = caption.get("description"), row.get("question")
            if (caption.get("status") not in allowed_statuses or caption.get("image_id") != item["image_id"]
                    or image_hashes[item["filename"]] != item["image_id"]
                    or (caption_manifest.get('settings', {}).get('caption_unit') == 'query_id'
                        and (caption.get('query_id') != row['id'] or caption.get('question_text') != question))
                    or not isinstance(description, str) or not description.strip()
                    or not isinstance(question, str) or not question.strip()):
                raise ValueError(f"Caption/ảnh/câu hỏi không hợp lệ: {row['id']}")
            caption_hashes[item["caption_file"]] = file_hash(path)
            queries.append({"query_id": row["id"], "source_image_id": row["image_id"],
                            "image_id": item["image_id"], "caption_file": item["caption_file"],
                            "accepted_caption": caption.get("accepted_by_user", False),
                            "caption_status": caption['status'],
                            "original_caption_status": caption.get('original_status', caption['status']),
                            **query_representation(question, caption, row['id']),
                            "vector_row": len(queries)})
        phase = "load-text-encoder"
        question_only = sum(q['representation_mode'] == 'question_only' for q in queries)
        print(f'[representation] queries={total}; question_only={question_only}; '
              f'relevant_items={sum(q["selected_item_count"] for q in queries)}', flush=True)
        incomplete_queries = sum(q['caption_status'] == 'INCOMPLETE' for q in queries)
        if incomplete_queries:
            print(f'[warning] Dùng caption INCOMPLETE cho {incomplete_queries}/{total} query; '
                  'giữ nguyên trạng thái, chưa được người dùng duyệt.', flush=True)
        import numpy as np
        import torch
        tokenizer, model, resources = load_encoder()
        maximum = min(8192, model.config.max_position_embeddings - 2)
        for query in queries:
            query["token_count"] = len(tokenizer(query["text"], truncation=False)["input_ids"])
            if query["token_count"] > maximum:
                raise ValueError(f"Query {query['query_id']} vượt {maximum} token; không tự cắt nội dung.")
        OUTPUT.mkdir(parents=True, exist_ok=True)
        manifest = {"status": "IN_PROGRESS", "created_at": datetime.now(timezone.utc).isoformat(),
                    "split": args.split, "queries": total, "model": MODEL, "revision": REVISION,
                    "pooling": "CLS", "normalize": "L2", "dimension": 1024,
                    "stored_dtype": "float32", "inference_dtype": "bfloat16", "attention": "sdpa",
                    "input_fields": ["question", "relevant_caption_items"], "input_prefix": "",
                    "representation": "question_relevant_items_v1", "relevance_filter": "is_relevant is true",
                    "include_summary": False, "deduplication": "exact_three_stripped_fields",
                    "truncated_policy": "accepted_complete_items_only", "empty_selection": "question_only",
                    "question_only_queries": question_only, "overwrite": args.overwrite,
                    "selected_items": sum(q['selected_item_count'] for q in queries),
                    "recovered_caption_queries": sum(q['parse_mode'] != 'json' for q in queries),
                    "source_sha256": {name: file_hash(ROOT / 'src/text_based_rag' / name)
                                      for name in ('query_embedding.py', 'corpus_chunking.py')},
                    "batch_size": args.batch_size, "model_max_tokens": maximum,
                    "allow_incomplete_captions": args.allow_incomplete_captions,
                    "incomplete_caption_queries": incomplete_queries,
                    "input_max_tokens": max(q["token_count"] for q in queries),
                    "input_sha256": {f"{args.split}.json": file_hash(INPUT),
                                     "query_image_mapping.jsonl": file_hash(CAPTIONS / "query_image_mapping.jsonl"),
                                     "caption_manifest.json": file_hash(CAPTIONS / "manifest.json"),
                                     "corpus_manifest.json": file_hash(CORPUS / "manifest.json")},
                    "caption_files_sha256": caption_hashes, "model_files_sha256": model_hashes,
                    "libraries": {name: version(name) for name in ("torch", "transformers", "numpy")},
                    "resources": resources}
        save(OUTPUT / "manifest.json", manifest)
        with (OUTPUT / "queries.jsonl.tmp").open("w", encoding="utf-8") as stream:
            for query in queries:
                stream.write(json.dumps(query, ensure_ascii=False) + "\n")
        (OUTPUT / "query_text_preview.txt.tmp").write_text(
            ("\n\n" + "=" * 80 + "\n\n").join(q["text"] for q in queries), encoding="utf-8")
        phase = "query-embedding-inference"
        torch.cuda.reset_peak_memory_stats()
        vectors = np.lib.format.open_memmap(OUTPUT / "vectors.npy.tmp", mode="w+", dtype="float32",
                                           shape=(total, 1024))
        for start in range(0, total, args.batch_size):
            batch = queries[start:start + args.batch_size]
            vectors[start:start + len(batch)] = embed_batch([q["text"] for q in batch], tokenizer, model)
            vectors.flush()
            done = start + len(batch)
        del vectors
        phase = 'validate-and-save-query-vectors'
        stored = np.load(OUTPUT / "vectors.npy.tmp", mmap_mode="r", allow_pickle=False)
        saved = read_lines(OUTPUT / "queries.jsonl.tmp")
        assert stored.shape == (total, 1024) and np.isfinite(stored).all(), "Invalid saved vectors"
        assert np.allclose(np.linalg.norm(stored, axis=1), 1, atol=1e-5), "Vectors must be normalized"
        assert len(saved) == total and all(q["vector_row"] == i and q["query_id"] == queries[i]["query_id"]
                                            for i, q in enumerate(saved)), "Vector row mapping mismatch"
        del stored
        for name, path in ((f'{args.split}.json', INPUT), ('query_image_mapping.jsonl', CAPTIONS / 'query_image_mapping.jsonl'),
                           ('caption_manifest.json', CAPTIONS / 'manifest.json'), ('corpus_manifest.json', CORPUS / 'manifest.json')):
            if file_hash(path) != manifest['input_sha256'][name]:
                raise ValueError(f'Input thay đổi trong lúc embedding: {name}; chưa thay vector cũ.')
        if any(file_hash(CAPTIONS / name) != digest for name, digest in caption_hashes.items()):
            raise ValueError('Caption thay đổi trong lúc embedding; chưa thay vector cũ.')
        for name in ('vectors.npy', 'queries.jsonl', 'query_text_preview.txt'):
            (OUTPUT / (name + '.tmp')).replace(OUTPUT / name)
        manifest.update(status="COMPLETED", elapsed_seconds=time.perf_counter() - started,
                        output_sha256={name: file_hash(OUTPUT / name) for name in
                                       ("vectors.npy", "queries.jsonl", "query_text_preview.txt")})
        resources.update(peak_vram_bytes=torch.cuda.max_memory_allocated(),
                         peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                         free_vram_after_inference=torch.cuda.mem_get_info()[0])
        save(OUTPUT / "manifest.json", manifest)
        print(f"[output] {OUTPUT}; queries={total}; dimensions=1024", flush=True)
    except BaseException as exc:
        if manifest is not None:
            manifest.update(status="FAILED", error=str(exc), embedded_queries=done)
            save(OUTPUT / "manifest.json", manifest)
        raise
    finally:
        stopped.set()
        worker.join()


if __name__ == "__main__":
    main()
