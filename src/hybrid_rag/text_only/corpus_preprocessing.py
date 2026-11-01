"""Keep original legal text and table text from normalized corpus; remove every figure."""

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

from text_based_rag.corpus_embedding import file_hash
from text_based_rag.corpus_image_captioning import blocks, read_lines, save

BASE = ROOT / "artifacts/experiments/hybrid_rag/text_only"
SOURCE = ROOT / "artifacts/experiments/text_based_rag/corpus/data_preprocessing"
DEV = ROOT / "data/processed/splits/dev.json"
MARKER = re.compile(r"<<BLOCK:([^>]+)>>")


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
        percent = f"{100*self.done/self.total:.1f}%" if self.total else "N/A"
        eta = (time.monotonic()-self.stage_started)*max(0, self.total-self.done)/self.done if self.done else None
        print(f"[{self.stage}] {self.done}/{self.total or '?'} ({percent}); "
              f"elapsed={time.monotonic()-self.started:.0f}s; ETA={f'{eta:.0f}s' if eta is not None else 'N/A'}", flush=True)

    def heartbeat(self):
        while not self.stop.wait(30):
            self.log()


def hashes_of(paths):
    return {str(p.absolute().relative_to(ROOT)): file_hash(p) for p in paths}


def check_hashes(hashes):
    for relative, expected in hashes.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Input/output thiếu hoặc thay đổi: {relative}")


def write_lines(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def load_step(name, filenames):
    folder = BASE / name
    manifest = folder / "manifest.json"
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    required = {str((folder / n).relative_to(ROOT)) for n in filenames}
    if (meta.get("status") != "COMPLETED" or not meta.get("code_sha256")
            or not required <= set(meta.get("output_sha256", {}))):
        raise ValueError(f"{name} chưa COMPLETED hoặc thiếu output; chạy lại bước đó.")
    for key in ("code_sha256", "input_sha256", "output_sha256"):
        check_hashes(meta[key])
    hashes = {**meta["input_sha256"], **meta["output_sha256"], **hashes_of([manifest])}
    return meta, folder, hashes


def run_step(args, name, action, entrypoint, dependencies=()):
    folder = BASE / name
    if folder.exists() and not args.overwrite:
        raise FileExistsError(f"{folder} đã tồn tại; dùng --overwrite cho bước này.")
    if (folder.is_symlink() or not folder.resolve().is_relative_to(BASE.resolve())
            or not folder.resolve().is_relative_to(ROOT)):
        raise ValueError("Output phải nằm trong text_only, không dùng symlink.")
    folder.mkdir(parents=True, exist_ok=True)
    code = {Path(__file__).resolve(), entrypoint.resolve(), *[p.resolve() for p in dependencies]}
    meta = {"status": "RUNNING", "step": name, "created_at": datetime.now(timezone.utc).isoformat(),
            "arguments": vars(args), "code_sha256": hashes_of(sorted(code)), "input_sha256": {},
            "packages": {n: version(n) for n in ("torch", "transformers", "numpy")}}
    save(folder / "manifest.json", meta)
    progress = Progress()
    progress.worker.start()
    try:
        action(args, folder, meta, progress)
        for key in ("code_sha256", "input_sha256", "output_sha256"):
            check_hashes(meta[key])
        meta.update(status="COMPLETED", elapsed_seconds=time.monotonic()-progress.started)
        save(folder / "manifest.json", meta)
        print(f"[output] {folder}", flush=True)
    except BaseException as error:
        meta.update(status="FAILED", error=f"{type(error).__name__}: {error}", failed_stage=progress.stage,
                    processed=progress.done, total=progress.total, elapsed_seconds=time.monotonic()-progress.started)
        save(folder / "manifest.json", meta)
        raise
    finally:
        progress.stop.set()
        progress.worker.join()
        progress.log()


def preprocess(args, folder, meta, progress):
    paths = [SOURCE / n for n in ("manifest.json", "structured_corpus.jsonl")]
    meta["input_sha256"] = hashes_of(paths)
    original = json.loads(paths[0].read_text(encoding="utf-8"))
    if original.get("status") != "COMPLETED" or original.get("schema_version") != 4:
        raise ValueError("Cần data_preprocessing schema 4, COMPLETED của text_based_rag.")
    records = read_lines(paths[1])
    if not records or len({r['article_key'] for r in records}) != len(records):
        raise ValueError("Corpus rỗng hoặc trùng article_key.")
    all_blocks = [b for r in records for b in blocks(r["blocks"])]
    lookup = {b["block_id"]: b for b in all_blocks}
    if len(lookup) != len(all_blocks):
        raise ValueError("Trùng block ID trong corpus.")

    def render(block, active):
        bid = block["block_id"]
        if bid in active:
            raise ValueError(f"Marker tạo vòng lặp: {bid}")
        kind = block["type"]
        if kind == "figure":
            return ""
        if kind == "table":
            children = block["blocks"]
            if not children or children[0]["type"] != "paragraph":
                raise ValueError(f"Bảng không khớp schema 4: {bid}")
            for child in children[1:]:
                if f'<<BLOCK:{child["block_id"]}>>' not in children[0]["text"]:
                    raise ValueError(f"Bảng thiếu marker nguồn: {child['block_id']}")
            return render(children[0], active | {bid})
        if kind != "paragraph" or not isinstance(block.get("text"), str):
            raise ValueError(f"Block text không hợp lệ: {bid}")
        return MARKER.sub(lambda m: render(lookup[m.group(1)], active | {bid}), block["text"])

    output, preview = [], []
    progress.set("remove figures and retain legal text", len(records))
    for record in records:
        kept = []
        for block in record["blocks"]:
            if block["type"] == "figure":
                continue
            text = render(block, set())
            if MARKER.search(text) or "<<IMAGE:" in text:
                raise ValueError(f"Còn marker ảnh/block: {block['block_id']}")
            kept.append({"block_id": block["block_id"], "type": "paragraph", "text": text,
                         "source_type": block["type"],
                         "source_block_ids": [b['block_id'] for b in blocks([block]) if b['type'] != 'figure']})
        row = {k: record[k] for k in ("article_key", "document_id", "document_title", "id", "title")}
        row["blocks"] = kept
        output.append(row)
        preview.append(f"[{row['document_id']} / {row['id']}] {row['title']}\n" + "\n\n".join(b['text'] for b in kept))
        progress.done += 1
    meta.update(schema_version=1, representation="original_legal_text_without_figures_or_vlm_captions",
                articles=len(output), removed_figures=sum(b['type']=='figure' for b in all_blocks),
                retained_tables_as_text=sum(b['type']=='table' for b in all_blocks))
    check_hashes(meta["input_sha256"])
    write_lines(folder / "structured_corpus.jsonl", output)
    temporary = folder / "corpus_preview.txt.tmp"
    temporary.write_text("\n\n".join(preview) + "\n", encoding="utf-8")
    temporary.replace(folder / "corpus_preview.txt")
    meta["output_sha256"] = hashes_of([folder / n for n in ("structured_corpus.jsonl", "corpus_preview.txt")])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_step(args, "data_preprocessing", preprocess, Path(__file__),
             [ROOT / "src/text_based_rag/corpus_image_captioning.py"])


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
