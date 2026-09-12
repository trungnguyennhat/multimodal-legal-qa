"""BM25 retrieval baseline and task evaluators."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import sys
import time
import tracemalloc
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from .data import DATA_FILES, DEFAULT_SOURCE, read_json_list


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS = PROJECT_ROOT / "data" / "processed" / "splits"
DEFAULT_ARTIFACTS = PROJECT_ROOT / "artifacts" / "experiments"
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
MARKUP_RE = re.compile(r"<<(?:IMAGE|TABLE):.*?/(?:IMAGE|TABLE)>>", re.DOTALL)


def _tokens(value: Any) -> list[str]:
    text = MARKUP_RE.sub(" ", unicodedata.normalize("NFC", str(value)))
    return TOKEN_RE.findall(text.casefold())


def _refs(row: dict[str, Any]) -> set[tuple[str, str]]:
    refs = row.get("relevant_articles", [])
    if not isinstance(refs, list):
        raise ValueError(f"{row.get('id', '<unknown>')}: relevant_articles phải là array")
    result: set[tuple[str, str]] = set()
    for ref in refs:
        if not isinstance(ref, dict) or "law_id" not in ref or "article_id" not in ref:
            raise ValueError(f"{row.get('id', '<unknown>')}: citation sai schema")
        result.add((str(ref["law_id"]), str(ref["article_id"])))
    return result


def _by_id(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("id", ""))
        if not sample_id or sample_id in result:
            raise ValueError(f"{label} có id thiếu hoặc trùng: {sample_id!r}")
        result[sample_id] = row
    return result


def retrieval_metrics(predictions: list[dict[str, Any]], gold: list[dict[str, Any]]) -> dict[str, Any]:
    """Match the official macro per-sample F2, with missing predictions scored as zero."""
    pred = _by_id(predictions, "prediction")
    truth = _by_id(gold, "gold")
    if not truth:
        raise ValueError("gold không được rỗng")
    scores: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    for sample_id, row in truth.items():
        expected = _refs(row)
        if not expected:
            raise ValueError(f"gold {sample_id} không có citation")
        actual = _refs(pred[sample_id]) if sample_id in pred else set()
        overlap = len(expected & actual)
        precision = overlap / len(actual) if actual else 0.0
        recall = overlap / len(expected)
        denominator = 4 * precision + recall
        scores.append(5 * precision * recall / denominator if denominator else 0.0)
        precisions.append(precision)
        recalls.append(recall)
    return {
        "f2": sum(scores) / len(scores),
        "precision": sum(precisions) / len(precisions),
        "recall": sum(recalls) / len(recalls),
        "samples": len(truth),
        "missing_predictions": len(set(truth) - set(pred)),
        "extra_predictions": len(set(pred) - set(truth)),
    }


def _answer(value: Any) -> str:
    return unicodedata.normalize("NFC", str(value).strip())


def accuracy_metrics(predictions: list[dict[str, Any]], gold: list[dict[str, Any]]) -> dict[str, Any]:
    pred = _by_id(predictions, "prediction")
    truth = _by_id(gold, "gold")
    if not truth:
        raise ValueError("gold không được rỗng")
    correct = sum(
        sample_id in pred and _answer(pred[sample_id].get("answer", "")) == _answer(row.get("answer", ""))
        for sample_id, row in truth.items()
    )
    return {
        "accuracy": correct / len(truth),
        "correct": correct,
        "samples": len(truth),
        "missing_predictions": len(set(truth) - set(pred)),
        "extra_predictions": len(set(pred) - set(truth)),
    }

# corpus law sẽ có key là law_id và article_id, nội dung sẽ được tổng hợp từ 2 key trên và tokenize nội dung đó
def _corpus(source: Path) -> list[dict[str, Any]]:
    laws = read_json_list(source / DATA_FILES["law"])
    merged: dict[tuple[str, str], list[str]] = {}
    for law in laws:
        for article in law["articles"]:
            key = (str(law["id"]), str(article["id"]))
            merged.setdefault(key, []).append(f"{article.get('title', '')} {article.get('text', '')}")
    return [
        {"law_id": key[0], "article_id": key[1], "tokens": _tokens(" ".join(parts))}
        for key, parts in merged.items()
    ]


def bm25_retrieve(
    queries: list[dict[str, Any]], corpus: list[dict[str, Any]], top_k: int = 5, k1: float = 1.5, b: float = 0.75
) -> list[dict[str, Any]]:
    if not corpus:
        raise ValueError("corpus không được rỗng")
    if top_k < 1 or top_k > len(corpus):
        raise ValueError(f"top-k phải trong khoảng 1..{len(corpus)}")
    if k1 <= 0 or not 0 <= b <= 1:
        raise ValueError("k1 phải dương và b phải trong khoảng 0..1")
    term_counts = [Counter(doc["tokens"]) for doc in corpus] # tính mỗi token xuất hiện bao nhiêu lần trong mỗi document trong corpus law
    lengths = [sum(counts.values()) for counts in term_counts] # tổng số token trong từng document
    average_length = sum(lengths) / len(lengths) 
    document_frequency = Counter(term for counts in term_counts for term in counts) # xét một token xuất hiện trong bao nhiêu document
    count = len(corpus)
    results: list[dict[str, Any]] = []
    
    # xét từng query và tính điểm cho token có trong query
    for row in queries:
        # tokenize query gồm question và choices
        query_text = str(row.get("question", "")) + " " + " ".join(
            str(value) for value in row.get("choices", {}).values()
        )
        query_terms = set(_tokens(query_text))
        
        
        scored: list[tuple[float, str, str]] = []
        # tổng điểm bm25 của 1 doc là tổng điểm của mọi token có trong query
        for doc, counts, length in zip(corpus, term_counts, lengths):
            score = 0.0
            for term in query_terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                inverse_frequency = math.log(1 + (count - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
                score += inverse_frequency * frequency * (k1 + 1) / (
                    frequency + k1 * (1 - b + b * length / average_length)
                )
            scored.append((score, doc["law_id"], doc["article_id"]))
        best = sorted(scored, key=lambda item: (-item[0], item[1], item[2]))[:top_k]
        results.append({
            "id": row["id"],
            "image_id": row["image_id"],
            "question": row["question"],
            "relevant_articles": [{"law_id": law_id, "article_id": article_id} for _, law_id, article_id in best],
        })
    return results


def run_baseline(input_path: Path, output: Path, source: Path, top_k: int, k1: float, b: float) -> dict[str, Any]:
    rows = read_json_list(input_path)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    tracemalloc.start()
    corpus = _corpus(source)
    predictions = bm25_retrieve(rows, corpus, top_k, k1, b)
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed = time.perf_counter() - started
    metrics = retrieval_metrics(predictions, rows) if all("relevant_articles" in row for row in rows) else None
    config = {
        "method": "bm25-text-only",
        "input": str(input_path),
        "source": str(source),
        "top_k": top_k,
        "k1": k1,
        "b": b,
    }
    resources = {
        "elapsed_seconds": elapsed,
        "python_peak_memory_bytes": peak_memory,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "corpus_articles": len(corpus),
        "queries": len(rows),
    }
    for filename, value in (
        ("config.json", config), ("predictions.json", predictions), ("metrics.json", metrics), ("resources.json", resources)
    ):
        (output / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"output": str(output), "metrics": metrics, "resources": resources}


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate", help="Tính F2 retrieval hoặc Accuracy QA")
    evaluate.add_argument("--task", required=True, choices=("retrieval", "qa"))
    evaluate.add_argument("--gold", required=True, type=Path)
    evaluate.add_argument("--predictions", required=True, type=Path)
    baseline = subparsers.add_parser("baseline", help="Chạy BM25 text-only và lưu experiment")
    baseline.add_argument("--input", type=Path, default=DEFAULT_SPLITS / "dev.json")
    baseline.add_argument("--output", type=Path, default=DEFAULT_ARTIFACTS / "bm25-retrieval-dev")
    baseline.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    baseline.add_argument("--top-k", type=int, default=5)
    baseline.add_argument("--k1", type=float, default=1.5)
    baseline.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate":
            predictions = read_json_list(args.predictions)
            gold = read_json_list(args.gold)
            _print(retrieval_metrics(predictions, gold) if args.task == "retrieval" else accuracy_metrics(predictions, gold))
        else:
            _print(run_baseline(args.input, args.output, args.source, args.top_k, args.k1, args.b))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
