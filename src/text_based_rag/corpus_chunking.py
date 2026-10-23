"""Pack legal text in source order, preserving whole images/tables and overlapping chunks."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from threading import Event, Thread
import time

from .corpus_normalization import OUTPUT as PREPROCESSING, ROOT
from .corpus_image_captioning import IMAGE_CATEGORIES, ITEM_FIELDS, blocks, read_lines, save

CAPTIONS = ROOT / "artifacts/experiments/text_based_rag/corpus/corpus_image_captioning"
OUTPUT = ROOT / "artifacts/experiments/text_based_rag/corpus/corpus_chunking"
TOKENIZER = ROOT / "models/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"


def heading(record):
    return "\n".join(dict.fromkeys(str(record[key]).strip()
                                 for key in ("document_title", "title") if record[key]))


class Organizer:
    def __init__(self, records, captions):
        self.lookup = {b["block_id"]: b for r in records for b in blocks(r["blocks"])}
        self.captions = captions

    def render(self, block):
        if block["type"] == "paragraph":
            return block["text"]
        if block["type"] == "figure":
            return self.captions[block["image_id"]]["description"]
        text = block["blocks"][0]["text"]
        for child in block["blocks"][1:]:
            text = text.replace(f'<<BLOCK:{child["block_id"]}>>', self.render(child))
        return text

    def units(self, record):
        return [{"body": self.render(block), "source": [b["block_id"] for b in blocks([block])],
                 "chunk_type": {"paragraph": "legal_text", "figure": "image", "table": "table"}[block["type"]]}
                for block in record["blocks"]]
def caption_parts(caption):
    """Recover only complete JSON items; never repair an accepted truncated description."""
    raw = caption["description"]
    category, items, remainder = None, [], ""
    try:
        payload = json.loads(raw)
        if (not isinstance(payload, dict) or payload.get("image_category") not in IMAGE_CATEGORIES
                or not isinstance(payload.get("items"), list)):
            raise ValueError("Caption schema không hợp lệ")
        category, items = payload["image_category"], payload["items"]
    except (json.JSONDecodeError, ValueError):
        if not caption.get("accepted_by_user") or caption.get("original_status") != "INCOMPLETE":
            raise ValueError(f'Caption không phải JSON hợp lệ và chưa được chấp nhận: {caption["filename"]}')
        decoder = json.JSONDecoder()
        position = raw.find("{") + 1
        last_item_end = None
        try:
            while position > 0 and position < len(raw):
                while position < len(raw) and raw[position] in " \t\r\n,":
                    position += 1
                key, position = decoder.raw_decode(raw, position)
                while position < len(raw) and raw[position].isspace():
                    position += 1
                if raw[position:position + 1] != ":":
                    break
                position += 1
                while position < len(raw) and raw[position].isspace():
                    position += 1
                if key == "items" and raw[position:position + 1] == "[":
                    position += 1
                    while position < len(raw):
                        while position < len(raw) and raw[position] in " \t\r\n,":
                            position += 1
                        item, end = decoder.raw_decode(raw, position)
                        if not isinstance(item, dict):
                            break
                        items.append(item)
                        position = last_item_end = end
                    break
                value, position = decoder.raw_decode(raw, position)
                if key == "image_category" and value in IMAGE_CATEGORIES:
                    category = value
        except (json.JSONDecodeError, TypeError):
            pass
        remainder = raw[last_item_end:] if last_item_end else raw
    valid_items = []
    for index, item in enumerate(items):
        if isinstance(item, dict) and all(isinstance(item.get(k), str) and item[k].strip() for k in ITEM_FIELDS):
            valid_items.append((index, item))
        else:
            remainder += "\n" + json.dumps(item, ensure_ascii=False)
    return category, valid_items, remainder


def chunk_record(record, units, tokenizer, limit, overlap, organizer):
    prefix = heading(record) + "\n\n"

    def count(text):
        return len(tokenizer(text, truncation=False)["input_ids"])

    if count(prefix) + overlap + 32 >= limit:
        raise ValueError(f'Tiêu đề quá dài: {record["article_key"]}; tăng --max-tokens.')
    chunks, segments = [], []
    body, kinds, has_new = "", set(), False

    def append(text, unit):
        nonlocal body, has_new
        body += "\n\n" if body else ""
        start = len(body)
        body += text
        segments.append((start, len(body), unit))
        kinds.add(unit["chunk_type"])
        has_new = True

    def emit():
        nonlocal body, segments, kinds, has_new
        assert has_new, "Never emit an overlap-only chunk"
        source = list(dict.fromkeys(bid for _, _, unit in segments for bid in unit["source"]))
        kind = next(iter(kinds)) if len(kinds) == 1 else "mixed"
        text = prefix + body
        token_count = count(text)
        assert token_count <= limit or kind in {"image", "table"}, "Only whole resources may exceed budget"
        chunks.append({"chunk_id": f'{record["article_key"]}/c{len(chunks) + 1:04d}',
                       "article_key": record["article_key"], "parent_id": record["article_key"],
                       "document_id": record["document_id"], "id": record["id"], "title": record["title"],
                       "chunk_type": kind, "text": text, "token_count": token_count,
                       "source_block_ids": source,
                       "image_ids": list(dict.fromkeys(organizer.lookup[bid]["image_id"] for bid in source
                                                       if organizer.lookup[bid]["type"] == "figure")),
                       "part_index": len(chunks)})
        if overlap:
            offsets = tokenizer(body, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
            start = offsets[max(0, len(offsets) - overlap)][0] if offsets else len(body)
            # Re-tokenization at a new beginning can change the token count.
            while start < len(body) and len(tokenizer(body[start:], add_special_tokens=False)["input_ids"]) > overlap:
                start += 1
        else:
            start = len(body)
        segments = [(max(a, start) - start, b - start, unit) for a, b, unit in segments if b > start]
        body = body[start:]
        kinds, has_new = set(), False

    for unit in units:
        remaining = unit["body"]
        if unit["chunk_type"] in {"image", "table"}:
            candidate = body + ("\n\n" if body else "") + remaining
            if count(prefix + candidate) > limit and has_new:
                emit()
            append(remaining, unit)
            if count(prefix + body) > limit:
                emit()
            continue
        while remaining:
            lead = prefix + body + ("\n\n" if body else "")
            if count(lead + remaining) <= limit:
                append(remaining, unit)
                break
            offsets = tokenizer(remaining, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
            lo, hi, end = 1, min(len(offsets), limit + 8), 0
            while lo <= hi:
                mid = (lo + hi) // 2
                candidate_end = offsets[mid - 1][1]
                if count(lead + remaining[:candidate_end]) <= limit:
                    end = candidate_end
                    lo = mid + 1
                else:
                    hi = mid - 1
            if not end:
                if has_new:
                    emit()
                    continue
                raise ValueError(f'Không đủ ngân sách cho text và overlap: {record["article_key"]}')
            boundaries = [m.end() for m in re.finditer(r"\n\s*\n|(?<=[.!?;])\s+", remaining[:end])]
            if boundaries and boundaries[-1] >= end // 2:
                end = boundaries[-1]
            append(remaining[:end], unit)
            remaining = remaining[end:]
            emit()
    if has_new:
        emit()
    for index, chunk in enumerate(chunks):
        chunk["previous_chunk_id"] = chunks[index - 1]["chunk_id"] if index else None
        chunk["next_chunk_id"] = chunks[index + 1]["chunk_id"] if index + 1 < len(chunks) else None
    for unit in units:
        if unit["chunk_type"] in {"image", "table"}:
            assert any(unit["body"] in c["text"] and set(unit["source"]) <= set(c["source_block_ids"])
                       for c in chunks), "Resource was split or lost"
    return chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--overlap-tokens", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true", help="Replace chunking output after validating new chunks.")
    args = parser.parse_args()
    if not 0 <= args.overlap_tokens < args.max_tokens // 2 or args.max_tokens < 128:
        parser.error("Cần max-tokens >= 128 và 0 <= overlap-tokens < max-tokens/2")
    if not args.tokenizer.is_dir():
        raise FileNotFoundError(f"Thiếu tokenizer local: {args.tokenizer}")
    if OUTPUT.exists() and not args.overwrite:
        raise FileExistsError("Output corpus_chunking đã tồn tại; dùng --overwrite để thay thế.")
    source_manifest = json.loads((PREPROCESSING / "manifest.json").read_text(encoding="utf-8"))
    caption_manifest = json.loads((CAPTIONS / "manifest.json").read_text(encoding="utf-8"))
    if source_manifest["status"] != "COMPLETED" or source_manifest["schema_version"] != 4 or caption_manifest["status"] != "COMPLETED":
        raise ValueError("Cần preprocessing schema 4 và caption đã hoàn tất/được chấp nhận.")
    records = read_lines(PREPROCESSING / "structured_corpus.jsonl")
    captions, issues, caption_hashes = {}, [], {}
    for path in sorted((CAPTIONS / "captions").glob("*_description.json")):
        caption_hashes[path.name] = sha256(path.read_bytes()).hexdigest()
        value = json.loads(path.read_text(encoding="utf-8"))
        if value["status"] != "COMPLETED" or not value["description"].strip():
            raise ValueError(f"Caption chưa hoàn tất hoặc rỗng: {path.name}")
        if value.get("original_status") == "INCOMPLETE" and not value.get("accepted_by_user"):
            raise ValueError(f"Caption bị cắt chưa được chấp nhận: {path.name}")
        if value["image_id"] in captions:
            raise ValueError(f"Trùng caption: {value['image_id']}")
        captions[value["image_id"]] = value
        value["caption_file"] = "captions/" + path.name
        if value.get("original_status") == "INCOMPLETE":
            issues.append({"code": "accepted_truncated_caption", "filename": value["filename"],
                           "detail": "Người dùng chấp nhận mô tả vốn có trạng thái INCOMPLETE"})
    images = read_lines(PREPROCESSING / "image_manifest.jsonl")
    image_ids = set()
    for image in images:
        digest = sha256(Path(image["path"]).read_bytes()).hexdigest()
        if digest != image["sha256"] or image["image_id"] != "sha256:" + digest:
            raise ValueError(f'Ảnh đã thay đổi: {image["filename"]}')
        image_ids.add(image["image_id"])
    if image_ids != set(captions):
        raise ValueError("Caption và ảnh normalization không khớp ID/hash; không tự dùng caption ảnh khác.")
    for record in records:
        for block in blocks(record["blocks"]):
            if block["type"] == "figure" and block.get("image_id") not in captions:
                raise ValueError(f'Thiếu caption cho {block["block_id"]}')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "IN_PROGRESS", "created_at": datetime.now(timezone.utc).isoformat(),
                "tokenizer": str(args.tokenizer.resolve()), "max_tokens": args.max_tokens,
                "overlap_tokens": args.overlap_tokens, "preprocessing": source_manifest,
                "method": "ordered_text_atomic_resources",
                "caption_reuse": "verified_image_sha256", "overwrite": args.overwrite,
                "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                "caption_manifest": caption_manifest, "caption_file_sha256": caption_hashes,
                "input_sha256": {name: sha256((PREPROCESSING / name).read_bytes()).hexdigest() for name in
                                 ("structured_corpus.jsonl", "image_manifest.jsonl", "manifest.json")},
                "tokenizer_sha256": {
                    p.name: sha256(p.read_bytes()).hexdigest() for p in args.tokenizer.glob("*tokenizer*.json")}}
    save(OUTPUT / "manifest.json", manifest)
    started, done = time.perf_counter(), 0
    stopped = Event()

    def report():
        elapsed = time.perf_counter() - started
        eta = f"{elapsed * (len(records) - done) / done:.0f}s" if done else "unknown"
        print(f"[organize-and-chunk] {done}/{len(records)} ({100 * done / max(1, len(records)):.1f}%) "
              f"elapsed={elapsed:.0f}s ETA={eta}", flush=True)

    def heartbeat():
        while not stopped.wait(30):
            report()

    worker = Thread(target=heartbeat, daemon=True)
    worker.start()
    report()
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
        if not tokenizer.is_fast:
            raise ValueError("Cần fast tokenizer để giữ nguyên ký tự khi chia đoạn.")
        tokenizer.model_max_length = 10**9  # Measure full sections without truncating them.
        encoder_config = json.loads((args.tokenizer / "config.json").read_text(encoding="utf-8"))
        encoder_limit = min(8192, encoder_config["max_position_embeddings"] - 2)
        manifest["encoder_max_tokens"] = encoder_limit
        organizer = Organizer(records, captions)
        organized, chunks = [], []
        for record in records:
            units = organizer.units(record)
            figures = [b for b in blocks(record["blocks"]) if b["type"] == "figure"]
            text = heading(record) + "\n\n" + "\n\n".join(organizer.render(b) for b in record["blocks"])
            parent_captions = []
            for figure in figures:
                caption = captions[figure["image_id"]]
                parent_captions.append({"image_id": figure["image_id"], "source_block_id": figure["block_id"],
                                        "caption_file": caption["caption_file"], "description": caption["description"]})
            organized.append({key: record[key] for key in ("article_key", "document_id", "document_title", "id", "title")})
            organized[-1]["text"] = text
            organized[-1].update(parent_id=record["article_key"], captions=parent_captions)
            # Table placeholders are expanded in place; the entire resource must survive.
            assert all(unit["body"] in text for unit in units), "Lost source content"
            assert all(c["description"] in text for c in parent_captions), "Lost original caption"
            article_chunks = chunk_record(record, units, tokenizer, args.max_tokens, args.overlap_tokens, organizer)
            covered = {bid for c in article_chunks for bid in c["source_block_ids"]}
            assert all(b["block_id"] in covered for b in blocks(record["blocks"])), "Lost block provenance"
            oversized = [c["chunk_id"] for c in article_chunks if c["token_count"] > encoder_limit]
            if oversized:
                raise ValueError(f"Chunk vượt giới hạn encoder {encoder_limit} token; không tự cắt ảnh/bảng: {oversized}")
            chunks.extend(article_chunks)
            done += 1
        assert len({c["chunk_id"] for c in chunks}) == len(chunks), "Duplicate chunk ID"
        parent_ids = {p["parent_id"] for p in organized}
        assert all(c["parent_id"] in parent_ids for c in chunks), "Unknown chunk parent"
        assert {image_id for c in chunks for image_id in c["image_ids"]} == set(captions), "Lost image coverage"
        for name, values in (("organized_corpus.jsonl", organized), ("chunks.jsonl", chunks), ("issues.jsonl", issues)):
            temporary = OUTPUT / (name + ".tmp")
            with temporary.open("w", encoding="utf-8") as stream:
                for value in values:
                    stream.write(json.dumps(value, ensure_ascii=False) + "\n")
            temporary.replace(OUTPUT / name)
        for name, text in (("organized_corpus.txt", "\n\n".join(r["text"] for r in organized)),
                           ("embedding_text_preview.txt", ("\n\n" + "=" * 72 + "\n\n").join(c["text"] for c in chunks))):
            temporary = OUTPUT / (name + ".tmp")
            temporary.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n", encoding="utf-8")
            temporary.replace(OUTPUT / name)
        manifest.update(status="COMPLETED", articles=len(organized), chunks=len(chunks), issues=len(issues),
                        oversized_resource_chunks=sum(c["token_count"] > args.max_tokens for c in chunks),
                        images=len(captions), chunk_types=dict(Counter(c["chunk_type"] for c in chunks)),
                        max_chunk_tokens=max(c["token_count"] for c in chunks),
                        output_sha256={name: sha256((OUTPUT / name).read_bytes()).hexdigest() for name in
                                       ("organized_corpus.jsonl", "chunks.jsonl", "issues.jsonl", "organized_corpus.txt", "embedding_text_preview.txt")},
                        elapsed_seconds=time.perf_counter() - started)
        save(OUTPUT / "manifest.json", manifest)
        report()
        print(f"[output] {OUTPUT}; chunks={len(chunks)}; issues={len(issues)}", flush=True)
    except BaseException as exc:
        manifest.update(status="FAILED", error=str(exc))
        save(OUTPUT / "manifest.json", manifest)
        raise
    finally:
        stopped.set()
        worker.join()


if __name__ == "__main__":
    main()
