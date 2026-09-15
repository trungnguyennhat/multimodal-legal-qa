"""Hybrid retrieval with reciprocal-rank fusion of BM25 and fine-tuned Visualized-BGE."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .bm25_evaluation import DEFAULT_ARTIFACTS, DEFAULT_SPLITS, run_baseline, retrieval_metrics
from .data import DEFAULT_SOURCE, read_json_list
from .visual_retrieval import DEFAULT_IMAGES, DEFAULT_MODEL, DEFAULT_WEIGHT, retrieve as visual_retrieve


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ADAPTER = PROJECT_ROOT / "artifacts" / "experiments" / "improvements" / "citation-level-loss" / "visual-bge-citation-loss" / "adapter.pt"
DEFAULT_OUTPUT = DEFAULT_ARTIFACTS / "hybrid-retrieval-dev"
DEFAULT_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_TOP_KS = (3, 5, 7)


def _indexed(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed = {str(row.get("id", "")): row for row in rows}
    if "" in indexed or len(indexed) != len(rows):
        raise ValueError(f"{label} có id thiếu hoặc trùng")
    return indexed


def reciprocal_rank_fusion(
    rows: list[dict[str, Any]],
    bm25_predictions: list[dict[str, Any]],
    visual_predictions: list[dict[str, Any]],
    visual_weight: float,
    top_k: int,
    rank_constant: int = 60,
) -> list[dict[str, Any]]:
    if not 0 <= visual_weight <= 1:
        raise ValueError("visual-weight phải trong khoảng 0..1")
    if top_k < 1 or rank_constant < 1:
        raise ValueError("top-k và rank-constant phải là số nguyên dương")
    bm25 = _indexed(bm25_predictions, "BM25 prediction")
    visual = _indexed(visual_predictions, "visual prediction")
    expected_ids = {str(row["id"]) for row in rows}
    if set(bm25) != expected_ids or set(visual) != expected_ids:
        raise ValueError("Hai nhánh retrieval phải dự đoán đúng toàn bộ id của input")

    fused = []
    for row in rows:
        sample_id = str(row["id"])
        scores: dict[tuple[str, str], float] = {}
        for branch_weight, prediction in (
            (1 - visual_weight, bm25[sample_id]),
            (visual_weight, visual[sample_id]),
        ):
            for rank, citation in enumerate(prediction.get("relevant_articles", []), 1):
                key = (str(citation["law_id"]), str(citation["article_id"]))
                scores[key] = scores.get(key, 0.0) + branch_weight / (rank_constant + rank)
        ranked = sorted(scores, key=lambda key: (-scores[key], key[0], key[1]))[:top_k]
        fused.append({
            "id": row["id"],
            "image_id": row["image_id"],
            "question": row["question"],
            "relevant_articles": [
                {"law_id": law_id, "article_id": article_id} for law_id, article_id in ranked
            ],
        })
    return fused


def tune(
    input_path: Path,
    output: Path,
    source: Path,
    image_root: Path,
    query_images: Path,
    adapter: Path,
    model: str,
    weight: Path,
    branch_depth: int,
    weights: tuple[float, ...],
    top_ks: tuple[int, ...],
    rank_constant: int,
    k1: float,
    b: float,
) -> dict[str, Any]:
    rows = read_json_list(input_path)
    if not rows or not all("relevant_articles" in row for row in rows):
        raise ValueError("tune yêu cầu input có gold relevant_articles; chỉ chọn cấu hình trên dev")
    if not weights or not top_ks:
        raise ValueError("weights và top-ks không được rỗng")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    branches = output / "branches"
    bm25_result = run_baseline(input_path, branches / "bm25", source, branch_depth, k1, b)
    visual_result = visual_retrieve(
        input_path, branches / "visual", adapter, source, image_root, query_images,
        model, weight, branch_depth, "image-question-choices", 0,
    )
    bm25_predictions = read_json_list(branches / "bm25" / "predictions.json")
    visual_predictions = read_json_list(branches / "visual" / "predictions.json")

    search = []
    best: tuple[float, float, int, list[dict[str, Any]], dict[str, Any]] | None = None
    for visual_weight in weights:
        for top_k in top_ks:
            predictions = reciprocal_rank_fusion(
                rows, bm25_predictions, visual_predictions, visual_weight, top_k, rank_constant
            )
            metrics = retrieval_metrics(predictions, rows)
            search.append({"visual_weight": visual_weight, "top_k": top_k, **metrics})
            candidate = (metrics["f2"], visual_weight, top_k, predictions, metrics)
            if best is None or candidate[:3] > best[:3]:
                best = candidate
    assert best is not None
    _, visual_weight, top_k, predictions, metrics = best
    config = {
        "method": "weighted-reciprocal-rank-fusion",
        "input": str(input_path),
        "source": str(source),
        "adapter": str(adapter),
        "visual_query_mode": "image-question-choices",
        "visual_image_context_tokens_per_side": 0,
        "branch_depth": branch_depth,
        "rank_constant": rank_constant,
        "searched_visual_weights": list(weights),
        "searched_top_ks": list(top_ks),
        "selected_visual_weight": visual_weight,
        "selected_top_k": top_k,
        "selection": "highest dev F2; ties prefer higher visual weight, then higher top-k",
        "bm25_k1": k1,
        "bm25_b": b,
    }
    resources = {
        "elapsed_seconds": time.perf_counter() - started,
        "queries": len(rows),
        "bm25": bm25_result["resources"],
        "visual": visual_result["resources"],
    }
    for filename, value in (
        ("config.json", config),
        ("predictions.json", predictions),
        ("metrics.json", metrics),
        ("resources.json", resources),
        ("search.json", search),
    ):
        (output / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"output": str(output), "config": config, "metrics": metrics, "resources": resources}


def _float_list(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("phải là danh sách số phân tách bằng dấu phẩy") from exc
    if not result or any(item < 0 or item > 1 for item in result):
        raise argparse.ArgumentTypeError("mọi weight phải trong khoảng 0..1")
    return result


def _int_list(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("phải là danh sách số nguyên phân tách bằng dấu phẩy") from exc
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError("mọi top-k phải là số nguyên dương")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_SPLITS / "dev.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--query-images", type=Path, default=DEFAULT_IMAGES / "train")
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--weight", type=Path, default=DEFAULT_WEIGHT)
    parser.add_argument("--branch-depth", type=int, default=20)
    parser.add_argument("--visual-weights", type=_float_list, default=DEFAULT_WEIGHTS)
    parser.add_argument("--top-ks", type=_int_list, default=DEFAULT_TOP_KS)
    parser.add_argument("--rank-constant", type=int, default=60)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args(argv)
    try:
        result = tune(
            args.input, args.output, args.source, args.image_root, args.query_images,
            args.adapter, args.model, args.weight, args.branch_depth,
            args.visual_weights, args.top_ks, args.rank_constant, args.k1, args.b,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
