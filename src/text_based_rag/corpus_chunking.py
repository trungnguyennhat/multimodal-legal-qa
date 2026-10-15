"""Organize legal text and create bounded chunks without embedding or GPU inference."""

import argparse
from collections import Counter, defaultdict
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
CLAUSE = re.compile(r"^(?:\d+(?:\.\d+)*\.?|[a-zđ]\))\s+", re.I)
EXCEPTION = re.compile(r"^(?:trừ|ngoại lệ|tuy nhiên|không áp dụng)\b", re.I)


def heading(record):
    return "\n".join(dict.fromkeys(str(record[key]).strip()
                                 for key in ("document_title", "title") if record[key]))


class Organizer:
    def __init__(self, records, issues):
        self.lookup = {b["block_id"]: b for r in records for b in blocks(r["blocks"])}
        self.issues = issues

    def issue(self, code, source_ids, detail):
        self.issues.append({"code": code, "source_block_ids": source_ids, "detail": detail})

    def render(self, block):
        if block["type"] == "paragraph":
            return block["text"]
        if block["type"] == "figure":
            return block.get("caption_original") or block.get("label") or "Hình minh họa"
        return "\n\n".join(row["body"] for row in self.table_rows(block))

    def table_rows(self, table):
        headers = defaultdict(list)
        for cell in table["cells"]:
            if cell["is_header"]:
                for column in range(cell["column"], cell["column"] + cell["colspan"]):
                    headers[column].append("\n".join(self.render(b) for b in cell["blocks"]))
        cells = {c["cell_id"]: c for c in table["cells"]}
        title = "\n".join(filter(None, (table.get("caption_original"), table.get("unit_original"))))
        result = []
        for index, grid_row in enumerate(table["grid"]):
            selected = list(dict.fromkeys(cid for cid in grid_row if cid is not None))
            parts, source = [], [table["block_id"]]
            if table.get("caption_block_id"):
                source.append(table["caption_block_id"])
            source.extend(b["block_id"] for cell in table["cells"] if cell["is_header"]
                          for b in blocks(cell["blocks"]))
            for cid in selected:
                cell = cells[cid]
                label = " / ".join(dict.fromkeys(headers[cell["column"]])) or f'Cột {cell["column"] + 1}'
                value_parts = []
                for b in cell["blocks"]:
                    value_parts.append(self.render(b))
                    for link in b.get("references", []):
                        target = self.lookup.get(link.get("target_block_id"))
                        if target and target["type"] == "figure" and link["status"] == "resolved":
                            value_parts.append(self.render(target))
                            source.append(target["block_id"])
                        else:
                            self.issue("table_cell_reference", [b["block_id"]],
                                       "Tham chiếu cần xem lại trong mục đầy đủ: " + link["mention"])
                value = "\n".join(value_parts)
                parts.append(f"{label}: {value}")
                source.extend(b["block_id"] for b in blocks(cell["blocks"]))
            if any(cid is None for cid in grid_row):
                self.issue("table_missing_cell", source, f"Hàng {index + 1}; ô thiếu không được hiểu là số 0")
            header_text = "; ".join(f"Cột {col + 1}: " + " / ".join(dict.fromkeys(values))
                                    for col, values in sorted(headers.items()))
            result.append({"body": "\n".join(filter(None, (title, f"Hàng {index + 1}", *parts))),
                           "source": list(dict.fromkeys(source)),
                           "context": "\n".join(filter(None, (title, header_text))), "chunk_type": "table_row"})
        for block in table.get("other_blocks", []):
            result.append({"body": self.render(block),
                           "source": [table["block_id"], *(b["block_id"] for b in blocks([block]))],
                           "context": title, "chunk_type": "table_row"})
        if not result:
            self.issue("empty_table", [table["block_id"]], "Không có hàng bảng để chuyển thành văn bản")
            result.append({"body": title or "Bảng không có ô dữ liệu.", "source": [table["block_id"]],
                           "context": "", "chunk_type": "table_row"})
        return result

    def references(self, block):
        result, ids = [], []
        for link in block.get("references", []):
            target = self.lookup.get(link.get("target_block_id"))
            if link["status"] != "resolved" or target is None:
                self.issue("reference_" + link["status"], [block["block_id"]], link["mention"])
            elif target["type"] in {"figure", "table"}:
                result.append(self.render(target))
                ids.extend(b["block_id"] for b in blocks([target]))
        return "\n\n".join(dict.fromkeys(result)), list(dict.fromkeys(ids))

    def units(self, record):
        units = []
        caption_ids = {b["caption_block_id"] for b in blocks(record["blocks"]) if b.get("caption_block_id")}
        lead, lead_id = "", None
        for block in record["blocks"]:
            bid = block["block_id"]
            if bid in caption_ids:
                continue
            if block["type"] == "figure":
                continue
            if block["type"] == "table":
                units.extend(self.table_rows(block))
                lead, lead_id = "", None
                continue
            body = self.render(block)
            context, ref_ids = self.references(block)
            source = [bid, *ref_ids]
            if block.get("caption_block_id"):
                source.append(block["caption_block_id"])
            if block["type"] == "paragraph":
                if CLAUSE.match(body) and lead:
                    context = "\n\n".join(filter(None, (lead, context)))
                    source.append(lead_id)
                elif body.rstrip().endswith(":"):
                    lead, lead_id = body, bid
                elif not CLAUSE.match(body):
                    lead, lead_id = "", None
            merge = EXCEPTION.match(body) and units and units[-1]["chunk_type"] == "legal_text"
            # shortcut: adjacency only groups likely explanations; review issues and retain the full parent section.
            if merge and units:
                units[-1]["body"] += "\n\n" + body
                units[-1]["source"].extend(source)
                units[-1]["context"] = "\n\n".join(filter(None, (units[-1]["context"], context)))
            else:
                units.append({"body": body, "context": context, "source": source,
                              "chunk_type": "legal_text"})
        return units


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


def caption_units(figure, caption, issues):
    category, items, remainder = caption_parts(caption)
    source = [figure["block_id"]]
    if figure.get("caption_block_id"):
        source.append(figure["caption_block_id"])
    context = "\n".join(filter(None, (figure.get("caption_original") or figure.get("label"),
                                      "Loại hình: " + category if category else "")))
    metadata = {"image_id": figure["image_id"], "image_category": category,
                "caption_file": caption["caption_file"], "original_caption_status": caption.get("original_status", caption["status"]),
                "accepted_by_user": caption.get("accepted_by_user", False)}
    units = []
    for index, item in items:
        identity = "Đối tượng: " + item["identifier"]
        body = "\n".join(("Đặc điểm: " + item["visual_or_geometric_features"],
                           "Ý nghĩa hoặc thông số: " + item["meaning_or_specification"]))
        units.append({"body": body, "context": "\n".join(filter(None, (context, identity))),
                      "source": source, "chunk_type": "caption_item", "item_index": index, **metadata})
    if remainder.strip() or not units:
        units.append({"body": remainder if remainder.strip() else caption["description"],
                      "context": context, "source": source, "chunk_type": "caption_fragment", "item_index": None, **metadata})
        issues.append({"code": "caption_fragment", "source_block_ids": source,
                       "filename": caption["filename"], "recovered_complete_items": len(items),
                       "detail": "Giữ phần caption không đọc được thành item; không sửa hoặc bỏ nội dung nguồn."})
    return units


def split_text(text, prefix, tokenizer, limit, overlap):
    """Prefer paragraph/sentence boundaries; slice source characters, never decoded tokens."""
    start = 0
    while start < len(text):
        remaining = text[start:]
        if len(tokenizer(prefix + remaining)["input_ids"]) <= limit:
            yield remaining
            break
        offsets = tokenizer(remaining, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        available = max(1, limit - len(tokenizer(prefix)["input_ids"]) - 8)
        end = start + offsets[min(available, len(offsets)) - 1][1]
        while end > start and len(tokenizer(prefix + text[start:end])["input_ids"]) > limit:
            end -= 1
        if end <= start:
            raise ValueError("Tiêu đề/ngữ cảnh quá dài; tăng --max-tokens.")
        boundaries = [m.end() for m in re.finditer(r"\n\s*\n|(?<=[.!?;])\s+", text[start:end])]
        if boundaries and boundaries[-1] >= (end - start) // 2:
            end = start + boundaries[-1]
        yield text[start:end]
        next_start = end
        if overlap:
            emitted = tokenizer(text[start:end], add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
            if len(emitted) > overlap:
                candidate = start + emitted[-overlap][0]
                if start < candidate < end:
                    next_start = candidate
        assert next_start > start, "Chunk splitting must advance"
        start = next_start


def chunk_record(record, units, tokenizer, limit, overlap, organizer):
    title = heading(record)
    if len(tokenizer(title)["input_ids"]) + 32 >= limit:
        raise ValueError(f'Tiêu đề quá dài: {record["article_key"]}; tăng --max-tokens.')
    chunks = []

    def emit(text, unit, part_index):
        ids = unit["source"]
        chunks.append({"chunk_id": f'{record["article_key"]}/c{len(chunks) + 1:04d}',
                       "article_key": record["article_key"], "parent_id": record["article_key"],
                       "document_id": record["document_id"], "chunk_type": unit["chunk_type"],
                       "id": record["id"], "title": record["title"], "text": text,
                       "token_count": len(tokenizer(text)["input_ids"]),
                       "source_block_ids": list(dict.fromkeys(ids)),
                       "image_ids": list(dict.fromkeys(organizer.lookup[bid]["image_id"] for bid in ids
                                                       if organizer.lookup[bid]["type"] == "figure")),
                       "part_index": part_index,
                       **{k: unit[k] for k in ("image_id", "image_category", "item_index", "caption_file",
                                               "original_caption_status", "accepted_by_user") if k in unit}})

    for unit in units:
        context, body = unit["context"], unit["body"]
        prefix = "\n\n".join(filter(None, (title, context))) + "\n\n"
        text = prefix + body
        if len(tokenizer(text)["input_ids"]) <= limit:
            emit(text, unit, 0)
            continue
        if len(tokenizer(prefix)["input_ids"]) + 32 >= limit:
            if unit["chunk_type"] == "caption_item":
                raise ValueError(f'Tên/ngữ cảnh item vượt ngân sách: {record["article_key"]}, {unit["item_index"]}')
            organizer.issue("context_exceeds_budget", unit["source"],
                            "Ngữ cảnh dài được đưa vào nội dung để chia; cần lấy mục đầy đủ khi retrieval")
            body = text[len(title) + 2:]
            prefix = title + "\n\n"
        organizer.issue("group_split", unit["source"], "Nhóm quá dài; chia đoạn/câu, kèm tiêu đề/ngữ cảnh và overlap")
        for part_index, part in enumerate(split_text(body, prefix, tokenizer, limit, overlap)):
            emit(prefix + part, unit, part_index)
    for index, chunk in enumerate(chunks):
        chunk["previous_chunk_id"] = chunks[index - 1]["chunk_id"] if index else None
        chunk["next_chunk_id"] = chunks[index + 1]["chunk_id"] if index + 1 < len(chunks) else None
    assert all(0 < c["token_count"] <= limit for c in chunks), "Chunk exceeds token budget"
    return chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--overlap-tokens", type=int, default=64)
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
    if source_manifest["status"] != "COMPLETED" or source_manifest["schema_version"] != 2 or caption_manifest["status"] != "COMPLETED":
        raise ValueError("Cần preprocessing schema 2 và caption đã hoàn tất/được chấp nhận.")
    records = read_lines(PREPROCESSING / "structured_corpus.jsonl")
    for name, key in (("structured_corpus.jsonl", "corpus_sha256"), ("image_manifest.jsonl", "images_sha256")):
        if sha256((PREPROCESSING / name).read_bytes()).hexdigest() != caption_manifest["config"][key]:
            raise ValueError("Corpus đã thay đổi sau bước caption.")
    captions, issues, caption_hashes = {}, [], {}
    for path in sorted((CAPTIONS / "captions").glob("*_description.json")):
        caption_hashes[path.name] = sha256(path.read_bytes()).hexdigest()
        value = json.loads(path.read_text(encoding="utf-8"))
        if value["status"] != "COMPLETED" or not value["description"].strip():
            raise ValueError(f"Caption chưa hoàn tất hoặc rỗng: {path.name}")
        if value["image_id"] in captions:
            raise ValueError(f"Trùng caption: {value['image_id']}")
        captions[value["image_id"]] = value
        value["caption_file"] = "captions/" + path.name
        if value.get("original_status") == "INCOMPLETE":
            issues.append({"code": "accepted_truncated_caption", "filename": value["filename"],
                           "detail": "Người dùng chấp nhận mô tả vốn có trạng thái INCOMPLETE"})
    for record in records:
        for block in blocks(record["blocks"]):
            if block["type"] == "figure":
                if block.get("image_id") not in captions:
                    raise ValueError(f'Thiếu caption cho {block["block_id"]}')
                block["description_vlm"] = captions[block["image_id"]]["description"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "IN_PROGRESS", "created_at": datetime.now(timezone.utc).isoformat(),
                "tokenizer": str(args.tokenizer.resolve()), "max_tokens": args.max_tokens,
                "overlap_tokens": args.overlap_tokens, "preprocessing": source_manifest,
                "method": "article_parent_item_children", "overwrite": args.overwrite,
                "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                "caption_manifest": caption_manifest, "caption_file_sha256": caption_hashes,
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
        organizer = Organizer(records, issues)
        organized, chunks = [], []
        caption_item_count, caption_fragment_count = 0, 0
        for record in records:
            units = organizer.units(record)
            figures = [b for b in blocks(record["blocks"]) if b["type"] == "figure"]
            text = heading(record) + "\n\n" + "\n\n".join(organizer.render(b) for b in record["blocks"])
            parent_captions = []
            for figure in figures:
                caption = captions[figure["image_id"]]
                children = caption_units(figure, caption, issues)
                units.extend(children)
                caption_item_count += sum(u["chunk_type"] == "caption_item" for u in children)
                caption_fragment_count += sum(u["chunk_type"] == "caption_fragment" for u in children)
                parent_captions.append({"image_id": figure["image_id"], "source_block_id": figure["block_id"],
                                        "caption_file": caption["caption_file"], "description": caption["description"]})
                text += "\n\n" + organizer.render(figure) + "\nMô tả hình:\n" + caption["description"]
            organized.append({key: record[key] for key in ("article_key", "document_id", "document_title", "id", "title")})
            organized[-1]["text"] = text
            organized[-1].update(parent_id=record["article_key"], captions=parent_captions)
            # Every normalized paragraph survives, including paragraphs inside nested tables.
            assert all(b["text"] in text for b in blocks(record["blocks"]) if b["type"] == "paragraph"), "Lost source paragraph"
            assert all(c["description"] in text for c in parent_captions), "Lost original caption"
            article_chunks = chunk_record(record, units, tokenizer, args.max_tokens, args.overlap_tokens, organizer)
            covered = {bid for c in article_chunks for bid in c["source_block_ids"]}
            assert all(b["block_id"] in covered for b in blocks(record["blocks"]) if b["type"] != "table"), "Lost block provenance"
            chunks.extend(article_chunks)
            done += 1
        assert len({c["chunk_id"] for c in chunks}) == len(chunks), "Duplicate chunk ID"
        parent_ids = {p["parent_id"] for p in organized}
        assert all(c["parent_id"] in parent_ids for c in chunks), "Unknown chunk parent"
        assert {c["image_id"] for c in chunks if c["chunk_type"].startswith("caption_")} == set(captions), "Lost image coverage"
        for name, values in (("organized_corpus.jsonl", organized), ("chunks.jsonl", chunks), ("issues.jsonl", issues)):
            temporary = OUTPUT / (name + ".tmp")
            with temporary.open("w", encoding="utf-8") as stream:
                for value in values:
                    stream.write(json.dumps(value, ensure_ascii=False) + "\n")
            temporary.replace(OUTPUT / name)
        for name, text in (("organized_corpus.txt", "\n\n".join(r["text"] for r in organized)),
                           ("embedding_text_preview.txt", ("\n\n" + "=" * 72 + "\n\n").join(c["text"] for c in chunks))):
            temporary = OUTPUT / (name + ".tmp")
            temporary.write_text(text + "\n", encoding="utf-8")
            temporary.replace(OUTPUT / name)
        manifest.update(status="COMPLETED", articles=len(organized), chunks=len(chunks), issues=len(issues),
                        caption_items=caption_item_count, caption_fragments=caption_fragment_count,
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
