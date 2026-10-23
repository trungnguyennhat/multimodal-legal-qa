"""Normalize the legal corpus without modifying the source or running a model."""

from dataclasses import dataclass, field
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from html.parser import HTMLParser
from html import escape
from pathlib import Path
from threading import Event, Thread
from urllib.parse import unquote, urlsplit
import argparse
import json
import re
import time

from multimodal_legal_rag.data import read_json_list


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/raw/VLSP2025-MLQA-TSR/dataset/law_db/vlsp2025_law.json"
IMAGES = ROOT / "data/processed/images/law"
OUTPUT = ROOT / "artifacts/experiments/text_based_rag/corpus/data_preprocessing"
MARKER = re.compile(r"<<(?P<kind>IMAGE|TABLE):(?P<body>.*?)/(?P=kind)>>", re.S)
VOID = {"img", "br", "hr", "input", "meta", "link", "source", "wbr", "col", "area", "base", "embed", "param"}
BREAKS = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "br", "hr"}


def clean(value):
    return "\n".join(re.sub(r"[^\S\n]+", " ", line).strip() for line in value.replace("\r", "").split("\n")).strip()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@dataclass
class Element:
    tag: str
    attrs: dict = field(default_factory=dict)
    children: list = field(default_factory=list)


class FragmentParser(HTMLParser):
    def __init__(self, html_whitespace=False):
        super().__init__(convert_charrefs=True)
        self.root = Element("root")
        self.stack = [self.root]
        self.repairs = []
        self.html_whitespace = html_whitespace

    def handle_starttag(self, tag, attrs):
        # Word exports sometimes omit optional closing tags; retain content and report repairs.
        if tag in {"td", "th", "tr", "p"}:
            close = {"td", "th"} if tag in {"td", "th"} else {tag}
            for i in range(len(self.stack) - 1, 0, -1):
                if self.stack[i].tag in close:
                    self.repairs.append(f"implicit close: {self.stack[i].tag}")
                    del self.stack[i:]
                    break
                if self.stack[i].tag in {"table", "td", "th"}:
                    break
        node = Element(tag, dict(attrs))
        self.stack[-1].children.append(node)
        if tag not in VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                if i != len(self.stack) - 1:
                    self.repairs.append(f"close ancestors before: {tag}")
                del self.stack[i:]
                return
        if tag not in VOID:
            self.repairs.append(f"unmatched end tag: {tag}")

    def handle_data(self, data):
        self.stack[-1].children.append(re.sub(r"\s+", " ", data) if self.html_whitespace else data)


class Normalizer:
    def __init__(self, image_dir):
        self.image_dir = image_dir.resolve()
        self.issues = []
        self.images = {}
        self.blocks = []
        self.article = None
        self.span = None

    def issue(self, code, detail, block_id=None):
        self.issues.append({"code": code, "article_key": self.article["article_key"],
                            "block_id": block_id, "detail": detail})

    def block(self, kind, **fields):
        block = {"block_id": f'{self.article["article_key"]}/b{len(self.blocks) + 1:04d}',
                 "type": kind, "source_span": list(self.span), **fields}
        self.blocks.append(block)
        return block

    def figure(self, source):
        figure = self.block("figure", source_reference=source,
                             image_id=None, description_vlm=None)
        parsed = urlsplit(source.strip())
        name = Path(unquote(parsed.path).replace("\\", "/")).name
        if parsed.scheme or parsed.netloc or not name or name in {".", ".."}:
            self.issue("invalid_image_reference", source, figure["block_id"])
            return figure
        path = (self.image_dir / name).resolve()
        if path.parent != self.image_dir or not path.is_file():
            self.issue("missing_image", source, figure["block_id"])
            return figure
        if name not in self.images:
            digest = sha256(path.read_bytes()).hexdigest()
            self.images[name] = {"image_id": f"sha256:{digest}", "filename": name,
                                 "path": str(path), "sha256": digest, "occurrences": []}
        image = self.images[name]
        image["occurrences"].append({"article_key": self.article["article_key"],
                                     "block_id": figure["block_id"], "source_reference": source})
        figure["image_id"] = image["image_id"]
        figure["filename"] = name
        return figure

    def html(self, fragment, result=None):
        fragment = re.sub(r"<<IMAGE:\s*(.*?)\s*/IMAGE>>",
                          lambda m: f'<img src="{escape(m.group(1), quote=True)}">', fragment, flags=re.S)
        parser = FragmentParser(html_whitespace=bool(re.search(r"<[a-zA-Z][^>]*>", fragment)))
        parser.feed(fragment)
        parser.close()
        if len(parser.stack) > 1:
            parser.repairs.append("unclosed tags at end of fragment")
        for detail in sorted(set(parser.repairs)):
            self.issue("html_repair", detail)
        return self.content(parser.root, result)

    def content(self, root, result=None):
        result = [] if result is None else result
        buffer = []

        def flush():
            text = clean("".join(buffer))
            if text:
                if result and result[-1]["type"] == "paragraph":
                    result[-1]["text"] += "\n" + text
                    result[-1]["source_span"][1] = self.span[1]
                else:
                    result.append(self.block("paragraph", text=text))
            buffer.clear()

        def walk(node):
            if isinstance(node, str):
                buffer.append(node)
                return
            if node.tag in {"script", "style", "head"}:
                return
            if node.tag == "table":
                flush()
                result.append(self.table(node))
                return
            if node.tag == "img":
                flush()
                result.append(self.figure(node.attrs.get("src", "")))
                return
            if node.tag in BREAKS:
                buffer.append("\n")
            if node.tag in {"sub", "sup"}:
                buffer.append("_(" if node.tag == "sub" else "^(")
            for child in node.children:
                walk(child)
            if node.tag in {"sub", "sup"}:
                buffer.append(")")
            if node.tag in BREAKS:
                buffer.append("\n")

        walk(root)
        flush()
        return result

    def table(self, root):
        table = self.block("table", cells=[], grid=[])

        def rows(node):
            for child in node.children:
                if isinstance(child, Element):
                    if child.tag == "tr":
                        yield child
                    elif child.tag != "table":
                        yield from rows(child)

        row_nodes = list(rows(root))
        occupied = {}
        for r, row in enumerate(row_nodes):
            c = 0
            for node in row.children:
                if not isinstance(node, Element) or node.tag not in {"td", "th"}:
                    continue
                while (r, c) in occupied:
                    c += 1
                try:
                    rowspan = int(node.attrs.get("rowspan", "1"))
                    rowspan = len(row_nodes) - r if rowspan == 0 else rowspan
                    colspan = int(node.attrs.get("colspan", "1"))
                    if not (1 <= rowspan <= 1000 and 1 <= colspan <= 256):
                        raise ValueError
                except (TypeError, ValueError):
                    self.issue("invalid_table_span", str(node.attrs), table["block_id"])
                    rowspan = colspan = 1
                cell_id = f'{table["block_id"]}/r{r + 1}c{c + 1}'
                cell = {"cell_id": cell_id, "row": r, "column": c, "rowspan": rowspan,
                        "colspan": colspan, "is_header": node.tag == "th",
                        "blocks": self.content(node)}
                table["cells"].append(cell)
                for rr in range(r, r + rowspan):
                    for cc in range(c, c + colspan):
                        if (rr, cc) in occupied:
                            self.issue("overlapping_table_cells", cell_id, table["block_id"])
                        else:
                            occupied[rr, cc] = cell_id
                c += colspan
        if occupied:
            height = max(r for r, _ in occupied) + 1
            width = max(c for _, c in occupied) + 1
            table["grid"] = [[occupied.get((r, c)) for c in range(width)] for r in range(height)]
        if not table["cells"]:
            self.issue("empty_or_unsupported_table", "No direct td/th cells found", table["block_id"])
        if any(cell["row"] + cell["rowspan"] > len(row_nodes) for cell in table["cells"]):
            self.issue("table_span_beyond_rows", "Rowspan extends beyond explicit rows", table["block_id"])
        # Keep content outside rows as separate blocks instead of silently dropping it.
        for child in root.children:
            if isinstance(child, str):
                if child.strip():
                    self.content(Element("root", children=[child]), table.setdefault("other_blocks", []))
            elif child.tag not in {"tr", "thead", "tbody", "tfoot", "col", "colgroup"}:
                self.content(child, table.setdefault("other_blocks", []))

        resources, row_text = [], {}

        def render(sequence):
            parts = []
            for block in sequence:
                if block["type"] == "paragraph":
                    parts.append(block["text"])
                else:
                    resources.append(block)
                    parts.append(f'<<BLOCK:{block["block_id"]}>>')
            return "\n".join(parts)

        for cell in table["cells"]:
            span = f'; span={cell["rowspan"]}x{cell["colspan"]}' if cell["rowspan"] > 1 or cell["colspan"] > 1 else ""
            header = "; header" if cell["is_header"] else ""
            value = render(cell.pop("blocks"))
            row_text.setdefault(cell["row"], []).append(f'Cột {cell["column"] + 1}{span}{header}: {value}'.rstrip())
        lines = [f'Hàng {row + 1}: ' + " | ".join(values) for row, values in row_text.items()]
        lines.append(render(table.pop("other_blocks", [])))
        paragraph = self.block("paragraph", text="\n".join(line for line in lines if line) or "Bảng không có ô dữ liệu.")
        table["blocks"] = [paragraph, *resources]
        return table

    def normalize(self, doc, article, key):
        self.article = {"article_key": key, "document_id": doc["id"], "document_title": doc["title"],
                        "id": article["id"], "title": article["title"]}
        self.blocks = []
        top = []
        cursor = 0
        for marker in MARKER.finditer(article["text"]):
            self.span = (cursor, marker.start())
            self.html(article["text"][cursor:marker.start()], top)
            self.span = marker.span()
            if marker.group("kind") == "IMAGE":
                top.append(self.figure(marker.group("body").strip()))
            else:
                self.html(marker.group("body"), top)
            cursor = marker.end()
        self.span = (cursor, len(article["text"]))
        self.html(article["text"][cursor:], top)
        if re.search(r"<<(?:IMAGE|TABLE):|/(?:IMAGE|TABLE)>>", MARKER.sub("", article["text"])):
            self.issue("unparsed_marker", "Malformed marker retained in source text")

        self.article["blocks"] = top
        def reachable(sequence):
            for block in sequence:
                yield block["block_id"]
                yield from reachable(block.get("blocks", []))

        kept = set(reachable(top))
        self.blocks = [block for block in self.blocks if block["block_id"] in kept]
        self.article["all_block_ids"] = [b["block_id"] for b in self.blocks]
        return self.article, list(self.blocks)


def preview(blocks, stream, depth=0):
    prefix = "  " * depth
    for block in blocks:
        stream.write(f'\n{prefix}[{block["block_id"]}] {block["type"]}\n\n')
        if block["type"] == "paragraph":
            stream.write(prefix + block["text"] + "\n")
        elif block["type"] == "figure":
            stream.write(f'{prefix}Ảnh: {block["source_reference"]}\n')
            if block.get("description_vlm"):
                stream.write(f'{prefix}Mô tả VLM: {block["description_vlm"]}\n')
        else:
            preview(block["blocks"], stream, depth + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--image-dir", type=Path, default=IMAGES)
    args = parser.parse_args()
    docs = read_json_list(args.source)
    if not args.image_dir.is_dir():
        raise ValueError(f"Missing image directory: {args.image_dir}")
    for doc in docs:
        if not all(isinstance(doc.get(k), str) for k in ("id", "title")) or not isinstance(doc.get("articles"), list):
            raise ValueError("Document requires string id/title and an articles array")
        for article in doc["articles"]:
            if not isinstance(article, dict) or not all(isinstance(article.get(k), str) for k in ("id", "title", "text")):
                raise ValueError(f'Invalid article in {doc["id"]}: requires string id/title/text')
    total = sum(len(d["articles"]) for d in docs)
    run = OUTPUT
    run.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "IN_PROGRESS", "schema_version": 4, "source": str(args.source.resolve()),
                "source_sha256": sha256(args.source.read_bytes()).hexdigest(),
                "image_directory": str(args.image_dir.resolve()), "documents": len(docs), "articles": total,
                "created_at": datetime.now(timezone.utc).isoformat(), "vlm_run": False}
    dump(run / "manifest.json", manifest)
    normalizer = Normalizer(args.image_dir)
    records, flat = [], []
    started = time.perf_counter()
    stopped = Event()

    def report():
        elapsed = time.perf_counter() - started
        done = len(records)
        eta = f"{elapsed * (total - done) / done:.0f}s" if done else "unknown"
        print(f"[normalize] {done}/{total} ({100 * done / max(total, 1):.1f}%) elapsed={elapsed:.1f}s ETA={eta}", flush=True)

    def heartbeat():
        while not stopped.wait(30):
            report()

    worker = Thread(target=heartbeat, daemon=True)
    worker.start()
    report()
    try:
        for di, doc in enumerate(docs):
            for ai, article in enumerate(doc["articles"]):
                record, blocks = normalizer.normalize(doc, article, f"d{di + 1:02d}/a{ai + 1:04d}")
                records.append(record)
                flat.append(blocks)
        # Runtime integrity checks protect provenance before publishing normalized records.
        ids = [b["block_id"] for blocks in flat for b in blocks]
        assert len(ids) == len(set(ids)), "Duplicate block identifiers"
        assert all("text" not in record for record in records), "Original article text must not be exported"
        assert all("attributes" not in b and all("attributes" not in c for c in b.get("cells", []))
                   for blocks in flat for b in blocks), "Raw HTML attributes must not be exported"
        assert all("text" not in b and sum(child["type"] == "paragraph" for child in b["blocks"]) == 1
                   and all("blocks" not in cell for cell in b["cells"])
                   for blocks in flat for b in blocks if b["type"] == "table"), "Table must have one shared paragraph"
        for filename, values in (("structured_corpus.jsonl", records),
                                 ("image_manifest.jsonl", normalizer.images.values()),
                                 ("issues.jsonl", normalizer.issues)):
            with (run / filename).open("w", encoding="utf-8") as stream:
                for value in values:
                    stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        dump(run / "corpus_inventory.json", {
            "documents": len(docs), "articles": total,
            "block_types": dict(Counter(b["type"] for blocks in flat for b in blocks)),
            "image_occurrences": sum(len(i["occurrences"]) for i in normalizer.images.values()),
            "issue_codes": dict(Counter(i["code"] for i in normalizer.issues)),
            "per_document": [{"id": d["id"], "title": d["title"], "articles": len(d["articles"])} for d in docs],
        })
        with (run / "corpus_preview.md").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(f'\n# {record["document_id"]} / {record["id"]} — {record["title"]}\n')
                preview(record["blocks"], stream)
        manifest.update(status="COMPLETED", blocks=len(ids), image_files=len(normalizer.images),
                        unique_image_hashes=len({i["sha256"] for i in normalizer.images.values()}),
                        issues=len(normalizer.issues), elapsed_seconds=time.perf_counter() - started)
        dump(run / "manifest.json", manifest)
        report()
        print(f'[output] {run}\nIssues requiring review: {len(normalizer.issues)}', flush=True)
    except Exception as exc:
        manifest.update(status="FAILED", error=str(exc))
        dump(run / "manifest.json", manifest)
        raise
    finally:
        stopped.set()
        worker.join()


if __name__ == "__main__":
    main()
