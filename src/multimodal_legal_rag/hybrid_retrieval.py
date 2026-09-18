"""Fuse example-based and corpus-based retrieval using the Stage 3 visual adapter."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .bm25_evaluation import DEFAULT_ARTIFACTS, DEFAULT_SPLITS, _refs, retrieval_metrics
from .data import DEFAULT_SOURCE, read_json_list
from .visual_retrieval import (
    DEFAULT_IMAGES, DEFAULT_MODEL, DEFAULT_WEIGHT, _adapter, _article_candidates,
    _citation_scores, _embed_items, _load_model, _resources,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ADAPTER = PROJECT_ROOT / "artifacts" / "experiments" / "improvements" / "citation-level-loss" / "visual-bge-citation-loss" / "adapter.pt"
DEFAULT_OUTPUT = DEFAULT_ARTIFACTS / "hybrid-example-retrieval-dev"
DEFAULT_EXAMPLE_KS = (1, 3, 5)
DEFAULT_WEIGHTS = (0.0, 0.25, 0.5, 0.75)
DEFAULT_TOP_KS = (3, 5, 7)


def _normalize(scores: dict[tuple[str, str], float]) -> dict[tuple[str, str], float]:
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if low == high:
        return {key: 1.0 for key in scores}
    return {key: (score - low) / (high - low) for key, score in scores.items()}


def _predictions(
    rows: list[dict[str, Any]], score_rows: list[dict[tuple[str, str], float]], top_k: int,
) -> list[dict[str, Any]]:
    if top_k < 1:
        raise ValueError("top-k phải là số nguyên dương")
    predictions = []
    for row, scores in zip(rows, score_rows):
        ranked = sorted(scores, key=lambda key: (-scores[key], key[0], key[1]))[:top_k]
        predictions.append({
            "id": row["id"], "image_id": row["image_id"], "question": row["question"],
            "relevant_articles": [
                {"law_id": law_id, "article_id": article_id} for law_id, article_id in ranked
            ],
        })
    return predictions


def _top_scores(
    keys: list[tuple[str, str]], values: list[float], depth: int,
) -> dict[tuple[str, str], float]:
    ranked = sorted(zip(keys, values), key=lambda item: (-item[1], item[0][0], item[0][1]))[:depth]
    return dict(ranked)


def _example_scores(
    similarities: Any,
    train_rows: list[dict[str, Any]],
    corpus_keys: set[tuple[str, str]],
    example_k: int,
) -> list[dict[tuple[str, str], float]]:
    result = []
    for query_scores in similarities:
        neighbors = sorted(range(len(train_rows)), key=lambda index: (-float(query_scores[index]), index))[:example_k]
        citations: dict[tuple[str, str], float] = {}
        for index in neighbors:
            score = float(query_scores[index])
            for citation in _refs(train_rows[index]) & corpus_keys:
                citations[citation] = citations.get(citation, 0.0) + score
        result.append(citations)
    return result


def _fuse(
    corpus_scores: list[dict[tuple[str, str], float]],
    example_scores: list[dict[tuple[str, str], float]],
    example_weight: float,
) -> list[dict[tuple[str, str], float]]:
    if not 0 <= example_weight <= 1:
        raise ValueError("example-weight phải trong khoảng 0..1")
    fused = []
    for corpus, examples in zip(corpus_scores, example_scores):
        scores: dict[tuple[str, str], float] = {}
        for branch_weight, branch in (
            (1 - example_weight, _normalize(corpus)),
            (example_weight, _normalize(examples)),
        ):
            if branch_weight:
                for citation, score in branch.items():
                    scores[citation] = scores.get(citation, 0.0) + branch_weight * score
        fused.append(scores)
    return fused


def tune(
    train_path: Path,
    dev_path: Path,
    output: Path,
    source: Path,
    image_root: Path,
    adapter_path: Path,
    model_name: str,
    weight_path: Path,
    branch_depth: int,
    example_ks: tuple[int, ...],
    example_weights: tuple[float, ...],
    top_ks: tuple[int, ...],
    query_images: Path | None = None,
    predict_only: bool = False,
) -> dict[str, Any]:
    if not adapter_path.is_file():
        raise ValueError(f"Thiếu fine-tuned adapter Stage 3: {adapter_path}")
    if branch_depth < 1 or not example_ks or not example_weights or not top_ks:
        raise ValueError("branch-depth và các danh sách tìm kiếm phải khác rỗng, chứa số dương")
    train_rows = read_json_list(train_path)
    dev_rows = read_json_list(dev_path)
    if not train_rows or not dev_rows or not all("relevant_articles" in row for row in train_rows):
        raise ValueError("train phải có gold relevant_articles và tập query không được rỗng")
    if predict_only:
        if len(example_ks) != 1 or len(example_weights) != 1 or len(top_ks) != 1:
            raise ValueError("predict-only yêu cầu đúng một example-k, example-weight và top-k đã khóa")
    elif not all("relevant_articles" in row for row in dev_rows):
        raise ValueError("dev phải có gold relevant_articles khi tuning")

    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch, model = _load_model(model_name, weight_path)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    candidates = _article_candidates(source, image_root / "law", model.tokenizer, 0)
    candidate_embeddings = _embed_items(model, torch, candidates, None, "corpus")
    train_embeddings = _embed_items(model, torch, train_rows, image_root / "train", "train-examples")
    dev_embeddings = _embed_items(model, torch, dev_rows, query_images or image_root / "train", "dev-queries")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    checkpoint = torch.load(adapter_path, map_location="cpu", weights_only=True)
    adapter = _adapter(torch, int(checkpoint["dimension"]))
    adapter.load_state_dict(checkpoint["state_dict"])
    adapter.eval()
    with torch.inference_mode():
        corpus_vectors = torch.nn.functional.normalize(adapter(candidate_embeddings), dim=1)
        train_vectors = torch.nn.functional.normalize(adapter(train_embeddings), dim=1)
        dev_vectors = torch.nn.functional.normalize(adapter(dev_embeddings), dim=1)
        candidate_similarities = dev_vectors @ corpus_vectors.T
        example_similarities = dev_vectors @ train_vectors.T

    candidate_keys = [(row["law_id"], row["article_id"]) for row in candidates]
    citation_similarities, citation_keys = _citation_scores(torch, candidate_similarities, candidate_keys)
    corpus_scores = [
        _top_scores(citation_keys, scores.tolist(), branch_depth) for scores in citation_similarities
    ]
    corpus_key_set = set(citation_keys)
    examples_by_k = {
        example_k: _example_scores(example_similarities, train_rows, corpus_key_set, example_k)
        for example_k in example_ks
    }

    if predict_only:
        selected = {
            "example_k": example_ks[0],
            "example_weight": example_weights[0],
            "top_k": top_ks[0],
        }
        fused_scores = _fuse(
            corpus_scores, examples_by_k[selected["example_k"]], selected["example_weight"],
        )
        predictions = _predictions(dev_rows, fused_scores, selected["top_k"])
        metrics = {
            "f2": None, "precision": None, "recall": None, "samples": len(dev_rows),
            "missing_predictions": 0, "extra_predictions": 0,
        }
        search = branch_metrics = None
    else:
        branch_metrics = {
            "corpus": {
                str(top_k): retrieval_metrics(_predictions(dev_rows, corpus_scores, top_k), dev_rows)
                for top_k in top_ks
            },
            "example": {
                str(example_k): {
                    str(top_k): retrieval_metrics(_predictions(dev_rows, scores, top_k), dev_rows)
                    for top_k in top_ks
                }
                for example_k, scores in examples_by_k.items()
            },
        }
        search = []
        best: tuple[dict[str, Any], list[dict[str, Any]]] | None = None
        for example_k in example_ks:
            for example_weight in example_weights:
                fused_scores = _fuse(corpus_scores, examples_by_k[example_k], example_weight)
                for top_k in top_ks:
                    predictions = _predictions(dev_rows, fused_scores, top_k)
                    result_metrics = retrieval_metrics(predictions, dev_rows)
                    record = {
                        "example_k": example_k, "example_weight": example_weight,
                        "top_k": top_k, **result_metrics,
                    }
                    search.append(record)
                    if best is None or result_metrics["f2"] > best[0]["f2"]:
                        best = (record, predictions)
        assert best is not None
        selected, predictions = best
        metric_names = ("f2", "precision", "recall", "samples", "missing_predictions", "extra_predictions")
        metrics = {key: selected[key] for key in metric_names}
    config = {
        "method": "example-and-corpus-score-fusion",
        "train": str(train_path), "dev": str(dev_path), "source": str(source),
        "adapter": str(adapter_path), "model": model_name, "weight": str(weight_path),
        "query_mode": "image-question-choices", "image_context_tokens_per_side": 0,
        "branch_depth": branch_depth,
        "normalization": "per-query min-max per branch",
        "example_score_aggregation": "sum similarity among selected train neighbors",
        "searched_example_ks": list(example_ks),
        "searched_example_weights": list(example_weights),
        "searched_top_ks": list(top_ks),
        "selected_example_k": selected["example_k"],
        "selected_example_weight": selected["example_weight"],
        "selected_top_k": selected["top_k"],
        "prediction_only": predict_only,
        "selection": (
            "fixed from dev; no private labels or tuning"
            if predict_only else "highest dev F2; ties keep the first simpler configuration"
        ),
    }
    resources = _resources(torch, started, candidates, len(train_rows) + len(dev_rows))
    resources.update({"train_examples": len(train_rows), "dev_queries": len(dev_rows)})
    artifacts = [
        ("config.json", config), ("predictions.json", predictions), ("metrics.json", metrics),
        ("resources.json", resources),
    ]
    if not predict_only:
        artifacts.extend((("search.json", search), ("branch_metrics.json", branch_metrics)))
    for filename, value in artifacts:
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
        raise argparse.ArgumentTypeError("mọi giá trị phải là số nguyên dương")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_SPLITS / "train.json")
    parser.add_argument("--dev", type=Path, default=DEFAULT_SPLITS / "dev.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--weight", type=Path, default=DEFAULT_WEIGHT)
    parser.add_argument("--branch-depth", type=int, default=20)
    parser.add_argument("--example-ks", type=_int_list, default=DEFAULT_EXAMPLE_KS)
    parser.add_argument("--example-weights", type=_float_list, default=DEFAULT_WEIGHTS)
    parser.add_argument("--top-ks", type=_int_list, default=DEFAULT_TOP_KS)
    parser.add_argument("--query-images", type=Path)
    parser.add_argument("--predict-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = tune(
            args.train, args.dev, args.output, args.source, args.image_root, args.adapter,
            args.model, args.weight, args.branch_depth, args.example_ks, args.example_weights, args.top_ks,
            args.query_images, args.predict_only,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
