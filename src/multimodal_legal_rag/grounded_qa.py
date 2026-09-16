"""Grounded multimodal legal QA over citations produced by Stage 4."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

from .bm25_evaluation import DEFAULT_ARTIFACTS, DEFAULT_SPLITS, _by_id, accuracy_metrics
from .data import DEFAULT_SOURCE, read_json_list
from .hybrid_retrieval import DEFAULT_ADAPTER
from .visual_retrieval import (
    DEFAULT_IMAGES,
    DEFAULT_MODEL,
    DEFAULT_WEIGHT,
    PROJECT_ROOT,
    _adapter,
    _article_candidates,
    _embed_items,
    _load_model,
)


DEFAULT_RETRIEVAL = DEFAULT_ARTIFACTS / "hybrid-example-retrieval-dev" / "predictions.json"
DEFAULT_OUTPUT = DEFAULT_ARTIFACTS / "grounded-qa-dev"
DEFAULT_QA_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
PROMPTS = ("direct-answer", "evidence-first")
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 512 * 28 * 28
MAX_NEW_TOKENS = 8
CITATION_ALIASES = {
    ("QCVN 41:2024/BGTVT", "G1.1"): (("QCVN 41:2024/BGTVT", "G.1"),),
    ("QCVN 41:2024/BGTVT", "I.414"): (("QCVN 41:2024/BGTVT", "E.14"),),
}


def _ordered_refs(row: dict[str, Any], label: str) -> list[tuple[str, str]]:
    refs = row.get("relevant_articles")
    if not isinstance(refs, list) or not refs:
        raise ValueError(f"{label} {row.get('id', '<unknown>')} không có relevant_articles hợp lệ")
    result: list[tuple[str, str]] = []
    for ref in refs:
        if not isinstance(ref, dict) or "law_id" not in ref or "article_id" not in ref:
            raise ValueError(f"{label} {row.get('id', '<unknown>')} có citation sai schema")
        key = (str(ref["law_id"]), str(ref["article_id"]))
        if key in result:
            raise ValueError(f"{label} {row.get('id', '<unknown>')} có citation trùng: {key}")
        result.append(key)
    return result


def _validate_inputs(
    rows: list[dict[str, Any]], retrieval: list[dict[str, Any]], candidates: list[dict[str, Any]], image_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, list[dict[str, Any]]]]]:
    gold = _by_id(rows, "gold")
    predicted = _by_id(retrieval, "retrieval prediction")
    missing = sorted(set(gold) - set(predicted))
    extra = sorted(set(predicted) - set(gold))
    if missing or extra:
        raise ValueError(f"retrieval prediction lệch dev: missing={len(missing)}, extra={len(extra)}")

    candidates_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for candidate in candidates:
        key = (str(candidate["law_id"]), str(candidate["article_id"]))
        candidates_by_key.setdefault(key, []).append(candidate)
    resolved: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        sample_id = str(row["id"])
        resolved[sample_id] = {}
        if row.get("question_type") not in {"Multiple choice", "Yes/No"}:
            raise ValueError(f"gold {sample_id} có question_type không hợp lệ")
        if row["question_type"] == "Multiple choice" and (
            not isinstance(row.get("choices"), dict) or set(row["choices"]) != {"A", "B", "C", "D"}
        ):
            raise ValueError(f"gold {sample_id} phải có choices A/B/C/D")
        image = image_root / "train" / f"{row['image_id']}.jpg"
        if not image.is_file():
            raise ValueError(f"Thiếu ảnh query: {image}")
        for label, source in (("gold", row), ("retrieval", predicted[sample_id])):
            refs = _ordered_refs(source, label)
            if label == "retrieval" and len(refs) != 5:
                raise ValueError(f"retrieval {sample_id} phải có đúng top-5 citation, nhận {len(refs)}")
            resolved[sample_id]["oracle" if label == "gold" else "retrieved"] = [
                {
                    "law_id": key[0],
                    "article_id": key[1],
                    "corpus_keys": _resolve_citation(key, candidates_by_key),
                }
                for key in refs
            ]
    return predicted, resolved


def _resolve_citation(
    key: tuple[str, str], candidates_by_key: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[tuple[str, str]]:
    """Resolve known source-label irregularities without modifying the dataset."""
    if key in candidates_by_key:
        return [key]
    law_id, article_id = key
    if key in CITATION_ALIASES:
        targets = list(CITATION_ALIASES[key])
        missing = [target for target in targets if target not in candidates_by_key]
        if missing:
            raise ValueError(f"alias citation {key} trỏ tới article không tồn tại: {missing}")
        return targets
    if article_id.endswith(".0") and article_id[:-2].isdigit():
        numeric = (law_id, article_id[:-2])
        if numeric in candidates_by_key:
            return [numeric]
    parts = article_id.split()
    composite = [(law_id, part) for part in parts]
    if len(composite) > 1 and all(part in candidates_by_key for part in composite):
        return composite
    raise ValueError(f"citation {key} không tồn tại và không ánh xạ được vào corpus")


def _select_evidence(
    rows: list[dict[str, Any]],
    retrieval_by_id: dict[str, dict[str, Any]],
    source: Path,
    image_root: Path,
    adapter_path: Path,
    retrieval_model: str,
    retrieval_weight: Path,
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[int, dict[str, Any]], dict[str, Any]]:
    if not adapter_path.is_file():
        raise ValueError(f"Thiếu fine-tuned adapter Stage 3: {adapter_path}")
    started = time.perf_counter()
    torch, model = _load_model(retrieval_model, retrieval_weight)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    all_candidates = _article_candidates(source, image_root / "law", model.tokenizer, 0)
    retrieval_by_id, resolved = _validate_inputs(rows, list(retrieval_by_id.values()), all_candidates, image_root)
    required = {
        tuple(key)
        for row in rows
        for mode in ("retrieved", "oracle")
        for citation in resolved[str(row["id"])][mode]
        for key in citation["corpus_keys"]
    }
    candidates = [
        {**candidate, "candidate_index": index}
        for index, candidate in enumerate(all_candidates)
        if (str(candidate["law_id"]), str(candidate["article_id"])) in required
    ]
    print(f"[evidence-indexing] {len(candidates)}/{len(all_candidates)} candidates thuộc citation cần dùng", flush=True)
    candidate_embeddings = _embed_items(model, torch, candidates, None, "evidence-indexing")
    query_embeddings = _embed_items(model, torch, rows, image_root / "train", "evidence-selection")
    del model
    if torch.cuda.is_available():
        evidence_peak = torch.cuda.max_memory_allocated()
        torch.cuda.empty_cache()
    else:
        evidence_peak = None

    checkpoint = torch.load(adapter_path, map_location="cpu", weights_only=True)
    metric_adapter = _adapter(torch, int(checkpoint["dimension"]))
    metric_adapter.load_state_dict(checkpoint["state_dict"])
    metric_adapter.eval()
    with torch.inference_mode():
        corpus_vectors = torch.nn.functional.normalize(metric_adapter(candidate_embeddings), dim=1)
        query_vectors = torch.nn.functional.normalize(metric_adapter(query_embeddings), dim=1)
        similarities = query_vectors @ corpus_vectors.T

    positions: dict[tuple[str, str], list[int]] = {}
    for position, candidate in enumerate(candidates):
        positions.setdefault((str(candidate["law_id"]), str(candidate["article_id"])), []).append(position)

    selected: dict[str, dict[str, list[dict[str, Any]]]] = {}
    runtime_candidates = {int(candidate["candidate_index"]): candidate for candidate in candidates}
    for row_index, row in enumerate(rows):
        sample_id = str(row["id"])
        selected[sample_id] = {}
        for mode in ("retrieved", "oracle"):
            evidence = []
            for citation in resolved[sample_id][mode]:
                choices = [position for key in citation["corpus_keys"] for position in positions[tuple(key)]]
                best = max(choices, key=lambda index: (float(similarities[row_index, index]), -index))
                candidate = candidates[best]
                evidence.append({
                    "law_id": citation["law_id"],
                    "article_id": citation["article_id"],
                    "corpus_law_id": str(candidate["law_id"]),
                    "corpus_article_id": str(candidate["article_id"]),
                    "candidate_index": int(candidate["candidate_index"]),
                    "modality": "image" if candidate["image"] is not None else "text",
                    "similarity": float(similarities[row_index, best]),
                })
            selected[sample_id][mode] = evidence

    resources = {
        "elapsed_seconds": time.perf_counter() - started,
        "gpu_peak_memory_bytes": evidence_peak,
        "indexed_candidates": len(candidates),
        "corpus_candidates": len(all_candidates),
        "required_citations": len(required),
    }
    del candidate_embeddings, query_embeddings, corpus_vectors, query_vectors, similarities, metric_adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return selected, runtime_candidates, resources


def _prompt_text(row: dict[str, Any], prompt_name: str, evidence: list[dict[str, Any]]) -> str:
    choices = row.get("choices", {})
    choice_text = "\n".join(f"{key}: {value}" for key, value in choices.items())
    question = f"Câu hỏi:\n{row['question']}"
    if choice_text:
        question += f"\n\nCác lựa chọn:\n{choice_text}"
    labels = "A, B, C hoặc D" if row["question_type"] == "Multiple choice" else "Đúng hoặc Sai"
    citations = ", ".join(f"{item['law_id']} — {item['article_id']}" for item in evidence)
    if prompt_name == "direct-answer":
        instruction = "Dựa duy nhất vào ảnh, câu hỏi và các căn cứ luật được cung cấp để chọn đáp án."
    elif prompt_name == "evidence-first":
        instruction = (
            "Trước khi chọn đáp án, hãy âm thầm đối chiếu nội dung ảnh và từng lựa chọn với các căn cứ luật; "
            "không trình bày quá trình suy luận."
        )
    else:
        raise ValueError(f"Prompt không hợp lệ: {prompt_name}")
    return (
        f"{instruction}\n\n{question}\n\nCác citation được phép sử dụng: {citations}.\n"
        f"Chỉ xuất đúng một nhãn trong tập: {labels}. Không giải thích và không thêm ký tự khác."
    )


def _messages(
    row: dict[str, Any], prompt_name: str, evidence: list[dict[str, Any]], candidates: dict[int, dict[str, Any]],
    image_root: Path,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "image",
            "image": str((image_root / "train" / f"{row['image_id']}.jpg").resolve()),
            "min_pixels": MIN_PIXELS,
            "max_pixels": MAX_PIXELS,
        },
        {"type": "text", "text": "Ảnh đầu tiên là ảnh của câu hỏi."},
    ]
    for number, item in enumerate(evidence, 1):
        candidate = candidates[int(item["candidate_index"])]
        citation = f"{item['corpus_law_id']} — Điều/Mục {item['corpus_article_id']}"
        if (item["law_id"], item["article_id"]) != (item["corpus_law_id"], item["corpus_article_id"]):
            citation += f" (nhãn nguồn: {item['article_id']})"
        if candidate["image"] is None:
            content.append({"type": "text", "text": f"Căn cứ {number} ({citation}):\n{candidate['text']}"})
        else:
            content.extend((
                {
                    "type": "image",
                    "image": str(Path(candidate["image"]).resolve()),
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                },
                {"type": "text", "text": f"Ảnh căn cứ {number} thuộc {citation}. {candidate['text']}"},
            ))
    content.append({"type": "text", "text": _prompt_text(row, prompt_name, evidence)})
    return [{"role": "user", "content": content}]


def _answer(raw: str, question_type: str) -> str:
    value = unicodedata.normalize("NFC", raw).strip()
    if question_type == "Multiple choice":
        match = re.fullmatch(r"(?:đáp\s*án\s*[:：]?\s*)?([ABCD])\s*[.)]?", value, re.IGNORECASE)
        return match.group(1).upper() if match else ""
    match = re.fullmatch(r"(?:đáp\s*án\s*[:：]?\s*)?(đúng|sai)\s*[.!]?", value, re.IGNORECASE)
    return "Đúng" if match and match.group(1).casefold() == "đúng" else "Sai" if match else ""


def _progress(label: str, done: int, total: int, started: float) -> None:
    elapsed = time.perf_counter() - started
    eta = elapsed / done * (total - done)
    print(f"[{label}] {done}/{total} ({done / total:.0%}) | elapsed {elapsed:.0f}s | ETA {eta:.0f}s", flush=True)


def _infer(
    model: Any,
    processor: Any,
    process_vision_info: Any,
    torch: Any,
    rows: list[dict[str, Any]],
    evidence_by_id: dict[str, dict[str, list[dict[str, Any]]]],
    candidates: dict[int, dict[str, Any]],
    image_root: Path,
    prompt_name: str,
    mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    predictions = []
    invalid = []
    label = "oracle-inference" if mode == "oracle" else f"prompt-evaluation:{prompt_name}"
    started = last_log = time.perf_counter()
    print(f"[{label}] generating {len(rows)} answers", flush=True)
    with torch.inference_mode():
        for done, row in enumerate(rows, 1):
            sample_id = str(row["id"])
            evidence = evidence_by_id[sample_id][mode]
            messages = _messages(row, prompt_name, evidence, candidates, image_root)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
            ).to(model.device)
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS)
            trimmed = generated[:, inputs.input_ids.shape[1]:]
            raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            answer = _answer(raw, str(row["question_type"]))
            refs = [
                {"law_id": item["corpus_law_id"], "article_id": item["corpus_article_id"]}
                for item in evidence
            ]
            prediction = {
                "id": row["id"], "image_id": row["image_id"], "question": row["question"],
                "question_type": row["question_type"], "relevant_articles": refs, "answer": answer,
            }
            if "choices" in row:
                prediction["choices"] = row["choices"]
            predictions.append(prediction)
            if not answer:
                invalid.append({"id": sample_id, "raw_output": raw})
            now = time.perf_counter()
            if done == len(rows) or now - last_log >= 30:
                _progress(label, done, len(rows), started)
                last_log = now
    return predictions, invalid


def run(
    input_path: Path,
    retrieval_path: Path,
    output: Path,
    source: Path,
    image_root: Path,
    adapter_path: Path,
    retrieval_model: str,
    retrieval_weight: Path,
    qa_model: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    rows = read_json_list(input_path)
    if not rows:
        raise ValueError("input QA không được rỗng")
    retrieval = read_json_list(retrieval_path)
    retrieval_by_id = _by_id(retrieval, "retrieval prediction")
    evidence, candidates, evidence_resources = _select_evidence(
        rows, retrieval_by_id, source, image_root, adapter_path, retrieval_model, retrieval_weight,
    )

    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / "huggingface"))
    try:
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise ValueError("Thiếu dependency Stage 5; chạy lệnh cài đặt trong README") from exc
    if not torch.cuda.is_available():
        raise ValueError("Stage 5 yêu cầu CUDA để chạy Qwen2.5-VL-3B ở BF16")
    torch.cuda.reset_peak_memory_stats()
    print(f"[qa-model] loading {qa_model} on cuda", flush=True)
    model_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        qa_model,
        cache_dir=PROJECT_ROOT / "models" / "huggingface",
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        use_fast=False,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        qa_model,
        cache_dir=PROJECT_ROOT / "models" / "huggingface",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    ).eval()
    qa_model_revision = getattr(model.config, "_commit_hash", None)
    model_load_seconds = time.perf_counter() - model_started
    print(f"[qa-model] ready in {model_load_seconds:.0f}s", flush=True)

    prompt_search = []
    best: tuple[str, list[dict[str, Any]], dict[str, Any]] | None = None
    for prompt_name in PROMPTS:
        predictions, invalid = _infer(
            model, processor, process_vision_info, torch, rows, evidence, candidates,
            image_root, prompt_name, "retrieved",
        )
        metrics = accuracy_metrics(predictions, rows)
        prompt_search.append({
            "prompt": prompt_name, "metrics": metrics, "invalid_outputs": invalid, "predictions": predictions,
        })
        if best is None or metrics["accuracy"] > best[2]["accuracy"]:
            best = (prompt_name, predictions, metrics)
    assert best is not None
    selected_prompt, predictions, metrics = best
    oracle_predictions, oracle_invalid = _infer(
        model, processor, process_vision_info, torch, rows, evidence, candidates,
        image_root, selected_prompt, "oracle",
    )
    oracle_metrics = accuracy_metrics(oracle_predictions, rows)
    qa_peak = torch.cuda.max_memory_allocated()

    config = {
        "method": "grounded-multimodal-qa",
        "input": str(input_path),
        "retrieval_predictions": str(retrieval_path),
        "source": str(source),
        "image_root": str(image_root),
        "retrieval_model": retrieval_model,
        "retrieval_weight": str(retrieval_weight),
        "adapter": str(adapter_path),
        "qa_model": qa_model,
        "qa_model_revision": qa_model_revision,
        "qa_dtype": "bfloat16",
        "attention": "sdpa",
        "prompts": list(PROMPTS),
        "selected_prompt": selected_prompt,
        "prompt_selection": "highest dev Accuracy; ties keep direct-answer",
        "evidence_per_citation": 1,
        "evidence_selection": "highest adapter cosine similarity within each supplied citation",
        "chunk_tokens": 1024,
        "chunk_overlap": 128,
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "do_sample": False,
        "max_new_tokens": MAX_NEW_TOKENS,
        "oracle_role": "upper bound only; not the primary pipeline score",
    }
    resources = {
        "elapsed_seconds": time.perf_counter() - started,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(0),
        "gpu_peak_memory_bytes": max(evidence_resources["gpu_peak_memory_bytes"] or 0, qa_peak),
        "qa_gpu_peak_memory_bytes": qa_peak,
        "qa_model_load_seconds": model_load_seconds,
        "queries": len(rows),
        "evidence": evidence_resources,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchvision", "transformers", "accelerate", "qwen-vl-utils")
        },
    }
    evidence_artifact = [
        {"id": row["id"], "retrieved": evidence[str(row["id"])]["retrieved"], "oracle": evidence[str(row["id"])]["oracle"]}
        for row in rows
    ]
    output.mkdir(parents=True, exist_ok=True)
    for filename, value in (
        ("config.json", config),
        ("evidence.json", evidence_artifact),
        ("prompt_search.json", prompt_search),
        ("predictions.json", predictions),
        ("metrics.json", metrics),
        ("oracle_predictions.json", oracle_predictions),
        ("oracle_metrics.json", {**oracle_metrics, "invalid_outputs": oracle_invalid}),
        ("resources.json", resources),
    ):
        (output / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"output": str(output), "selected_prompt": selected_prompt, "metrics": metrics, "oracle_metrics": oracle_metrics}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_SPLITS / "dev.json")
    parser.add_argument("--retrieval-predictions", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--retrieval-model", default=DEFAULT_MODEL)
    parser.add_argument("--retrieval-weight", type=Path, default=DEFAULT_WEIGHT)
    parser.add_argument("--qa-model", default=DEFAULT_QA_MODEL)
    args = parser.parse_args(argv)
    try:
        result = run(
            args.input, args.retrieval_predictions, args.output, args.source, args.image_root,
            args.adapter, args.retrieval_model, args.retrieval_weight, args.qa_model,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
