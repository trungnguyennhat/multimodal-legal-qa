"""Visualized-BGE retrieval for image-and-text legal queries."""

from __future__ import annotations

import argparse
import html
import importlib.metadata
import json
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any

from .bm25_evaluation import _refs, retrieval_metrics
from .data import DATA_FILES, DEFAULT_SOURCE, IMAGE_PATTERN, read_json_list


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS = PROJECT_ROOT / "data" / "processed" / "splits"
DEFAULT_IMAGES = PROJECT_ROOT / "data" / "processed" / "images"
DEFAULT_ARTIFACTS = PROJECT_ROOT / "artifacts" / "experiments"
DEFAULT_WEIGHT = PROJECT_ROOT / "models" / "Visualized_m3.pth"
DEFAULT_FINETUNE_OUTPUT = DEFAULT_ARTIFACTS / "visual-bge-finetune"
DEFAULT_ADAPTER = DEFAULT_FINETUNE_OUTPUT / "adapter.pt"
DEFAULT_VISUAL_BGE_SOURCE = PROJECT_ROOT / "models" / "FlagEmbedding" / "research" / "visual_bge"
DEFAULT_MODEL = "BAAI/bge-m3"
TABLE_RE = re.compile(r"<<TABLE:\s*(.*?)\s*/TABLE>>", re.DOTALL)
HTML_TAG_RE = re.compile(r"<[^>]+>")
TEXT_MAX_TOKENS = 8192
MULTIMODAL_TEXT_MAX_TOKENS = 7936
CHUNK_TOKENS = 1024
CHUNK_OVERLAP = 128


def _text_chunks(tokenizer: Any, text: str) -> list[str]:
    token_ids = tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"]
    if not token_ids:
        return [""]
    chunks = []
    start = 0
    while start < len(token_ids):
        end = min(start + CHUNK_TOKENS, len(token_ids))
        chunks.append(tokenizer.decode(token_ids[start:end], skip_special_tokens=True))
        if end == len(token_ids):
            break
        start = end - CHUNK_OVERLAP
    return chunks


def _article_candidates(source: Path, image_dir: Path, tokenizer: Any) -> list[dict[str, Any]]:
    """Make overlapping text chunks and separate image candidates per citation."""
    candidates: list[dict[str, Any]] = []
    for law in read_json_list(source / DATA_FILES["law"]):
        for article in law["articles"]:
            text = TABLE_RE.sub(
                lambda match: html.unescape(HTML_TAG_RE.sub(" ", match.group(1))),
                str(article.get("text", "")),
            )
            images = IMAGE_PATTERN.findall(text)
            text = IMAGE_PATTERN.sub(" ", text)
            base = {
                "law_id": str(law["id"]),
                "article_id": str(article["id"]),
            }
            title = str(article.get("title", ""))
            candidates.extend(
                {**base, "text": chunk, "image": None}
                for chunk in _text_chunks(tokenizer, f"{title} {text}".strip())
            )
            for filename in images:
                path = image_dir / filename
                if not path.is_file():
                    raise ValueError(f"Thiếu ảnh luật: {path}")
                candidates.append({**base, "text": title, "image": path})
    if not candidates:
        raise ValueError("corpus không có candidate")
    return candidates


def _query_text(row: dict[str, Any]) -> str:
    choices = row.get("choices", {})
    return str(row.get("question", "")) + " " + " ".join(str(value) for value in choices.values())


def _rank_citations(scores: list[float], candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, str]]:
    """Keep the best candidate score for each unique corpus citation."""
    best: dict[tuple[str, str], float] = {}
    for score, candidate in zip(scores, candidates):
        key = (candidate["law_id"], candidate["article_id"])
        best[key] = max(score, best.get(key, float("-inf")))
    if top_k < 1 or top_k > len(best):
        raise ValueError(f"top-k phải trong khoảng 1..{len(best)}")
    ranked = sorted(best, key=lambda key: (-best[key], key[0], key[1]))[:top_k]
    return [{"law_id": law_id, "article_id": article_id} for law_id, article_id in ranked]


def _encode(model: Any, text: str, image: Path | None = None) -> Any:
    """Encode within BGE-M3's 8192 positions, reserving 256 for image tokens."""
    tokens = model.tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MULTIMODAL_TEXT_MAX_TOKENS if image is not None else TEXT_MAX_TOKENS,
    ).to(model.device)
    if image is None:
        return model.encode_text(tokens)
    from PIL import Image

    with Image.open(image) as source:
        pixels = model.preprocess_val(source).unsqueeze(0).to(model.device)
    return model.encode_mm(pixels, tokens)


def _progress(label: str, done: int, total: int, started: float) -> None:
    elapsed = time.perf_counter() - started
    eta = elapsed / done * (total - done)
    print(f"[{label}] {done}/{total} ({done / total:.0%}) | elapsed {elapsed:.0f}s | ETA {eta:.0f}s", flush=True)


def _load_model(model_name: str, weight: Path) -> tuple[Any, Any]:
    if not (DEFAULT_VISUAL_BGE_SOURCE / "visual_bge" / "modeling.py").is_file():
        raise ValueError(f"Thiếu source Visualized-BGE: {DEFAULT_VISUAL_BGE_SOURCE}")
    sys.path.insert(0, str(DEFAULT_VISUAL_BGE_SOURCE))
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / "huggingface"))
    try:
        import torch
        from visual_bge.modeling import Visualized_BGE
    except ImportError as exc:
        raise ValueError("Thiếu dependency Stage 3; chạy các lệnh cài đặt trong README") from exc
    if not weight.is_file():
        raise ValueError(f"Thiếu model weight: {weight}")

    print(f"[model] loading {model_name} on {'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)
    model_started = time.perf_counter()
    model = Visualized_BGE(model_name_bge=model_name, model_weight=str(weight))
    print(f"[model] ready in {time.perf_counter() - model_started:.0f}s", flush=True)
    return torch, model


def _embed_items(model: Any, torch: Any, items: list[dict[str, Any]], image_dir: Path | None, label: str) -> Any:
    encoded = []
    started = last_log = time.perf_counter()
    print(f"[{label}] encoding {len(items)} items", flush=True)
    model.eval()
    with torch.inference_mode():
        for done, item in enumerate(items, 1):
            image = item["image"] if image_dir is None else image_dir / f"{item['image_id']}.jpg"
            if image is not None and not image.is_file():
                raise ValueError(f"Thiếu ảnh: {image}")
            text = item["text"] if image_dir is None else _query_text(item)
            encoded.append(_encode(model, text, image).detach().cpu())
            now = time.perf_counter()
            if done == len(items) or now - last_log >= 30:
                _progress(label, done, len(items), started)
                last_log = now
    return torch.cat(encoded)


def _adapter(torch: Any, dimension: int) -> Any:
    layer = torch.nn.Linear(dimension, dimension, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.eye(dimension))
    return layer


def _predictions(
    torch: Any,
    rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    query_embeddings: Any,
    candidate_embeddings: Any,
    adapter: Any,
    top_k: int,
) -> list[dict[str, Any]]:
    adapter.eval()
    with torch.inference_mode():
        queries = torch.nn.functional.normalize(adapter(query_embeddings), dim=1)
        corpus = torch.nn.functional.normalize(adapter(candidate_embeddings), dim=1)
        similarities = queries @ corpus.T
    return [
        {
            "id": row["id"],
            "image_id": row["image_id"],
            "question": row["question"],
            "relevant_articles": _rank_citations(similarities[index].tolist(), candidates, top_k),
        }
        for index, row in enumerate(rows)
    ]


def _resources(torch: Any, started: float, candidates: list[dict[str, Any]], queries: int) -> dict[str, Any]:
    return {
        "elapsed_seconds": time.perf_counter() - started,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "candidates": len(candidates),
        "corpus_citations": len({(row["law_id"], row["article_id"]) for row in candidates}),
        "queries": queries,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchvision", "transformers", "timm")
        },
    }


def train(
    train_path: Path,
    dev_path: Path,
    output: Path,
    source: Path,
    image_root: Path,
    model_name: str,
    weight: Path,
    epochs: int,
    learning_rate: float,
    temperature: float,
    top_k: int,
    seed: int,
) -> dict[str, Any]:
    if epochs < 1 or learning_rate <= 0 or temperature <= 0:
        raise ValueError("epochs, learning-rate và temperature phải dương")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch, model = _load_model(model_name, weight)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    candidates = _article_candidates(source, image_root / "law", model.tokenizer)
    train_rows = read_json_list(train_path)
    dev_rows = read_json_list(dev_path)
    candidate_embeddings = _embed_items(model, torch, candidates, None, "corpus")
    train_embeddings = _embed_items(model, torch, train_rows, image_root / "train", "train-queries")
    dev_embeddings = _embed_items(model, torch, dev_rows, image_root / "train", "dev-queries")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    keys = [(row["law_id"], row["article_id"]) for row in candidates]
    masks = [[key in _refs(row) for key in keys] for row in train_rows]
    keep = [index for index, mask in enumerate(masks) if any(mask)]
    if not keep:
        raise ValueError("train.json không có citation nào tồn tại trong corpus")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_embeddings = train_embeddings[keep].to(device)
    candidate_embeddings_device = candidate_embeddings.to(device)
    dev_embeddings_device = dev_embeddings.to(device)
    positive_mask = torch.tensor([masks[index] for index in keep], dtype=torch.bool, device=device)
    adapter = _adapter(torch, train_embeddings.shape[1]).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=learning_rate)
    history = []
    best_epoch = 0
    best_f2 = -1.0
    best_state = None
    train_started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        adapter.train()
        queries = torch.nn.functional.normalize(adapter(train_embeddings), dim=1)
        corpus = torch.nn.functional.normalize(adapter(candidate_embeddings_device), dim=1)
        scores = queries @ corpus.T / temperature
        loss = (torch.logsumexp(scores, dim=1) - torch.logsumexp(scores.masked_fill(~positive_mask, float("-inf")), dim=1)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_predictions = _predictions(
            torch, dev_rows, candidates, dev_embeddings_device, candidate_embeddings_device, adapter, top_k
        )
        dev_f2 = retrieval_metrics(epoch_predictions, dev_rows)["f2"]
        history.append({"epoch": epoch, "loss": loss.item(), "dev_f2": dev_f2})
        if dev_f2 > best_f2:
            best_epoch = epoch
            best_f2 = dev_f2
            best_state = {name: value.detach().cpu().clone() for name, value in adapter.state_dict().items()}
        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            _progress("training", epoch, epochs, train_started)
            print(f"[validation] epoch {epoch} | loss {loss.item():.4f} | dev F2 {dev_f2:.4f}", flush=True)

    if best_state is None:
        raise ValueError("fine-tune không tạo được checkpoint hợp lệ")
    adapter.load_state_dict(best_state)
    adapter = adapter.cpu()
    predictions = _predictions(torch, dev_rows, candidates, dev_embeddings, candidate_embeddings, adapter, top_k)
    metrics = retrieval_metrics(predictions, dev_rows)
    metrics.update({
        "best_epoch": best_epoch,
        "train_initial_loss": history[0]["loss"],
        "train_final_loss": history[-1]["loss"],
    })
    torch.save({"dimension": adapter.in_features, "state_dict": adapter.state_dict()}, output / "adapter.pt")
    config = {
        "method": "visualized-bge-frozen-linear-adapter",
        "train": str(train_path),
        "dev": str(dev_path),
        "source": str(source),
        "model": model_name,
        "visual_bge_source": "FlagEmbedding v1.4.2",
        "weight": str(weight),
        "base_model_frozen": True,
        "trainable_parameters": sum(parameter.numel() for parameter in adapter.parameters()),
        "epochs": epochs,
        "checkpoint_selection": "highest dev F2",
        "best_epoch": best_epoch,
        "learning_rate": learning_rate,
        "temperature": temperature,
        "top_k": top_k,
        "seed": seed,
        "train_samples": len(keep),
        "skipped_train_samples": len(train_rows) - len(keep),
        "text_max_tokens": TEXT_MAX_TOKENS,
        "multimodal_text_max_tokens": MULTIMODAL_TEXT_MAX_TOKENS,
        "chunk_tokens": CHUNK_TOKENS,
        "chunk_overlap": CHUNK_OVERLAP,
    }
    resources = _resources(torch, started, candidates, len(train_rows) + len(dev_rows))
    for filename, value in (
        ("config.json", config),
        ("predictions.json", predictions),
        ("metrics.json", metrics),
        ("resources.json", resources),
        ("history.json", history),
    ):
        (output / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"output": str(output), "metrics": metrics, "resources": resources}


def retrieve(
    input_path: Path,
    output: Path,
    adapter_path: Path,
    source: Path,
    image_root: Path,
    query_images: Path,
    model_name: str,
    weight: Path,
    top_k: int,
) -> dict[str, Any]:
    if not adapter_path.is_file():
        raise ValueError(f"Thiếu fine-tuned adapter: {adapter_path}; chạy command train trước")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch, model = _load_model(model_name, weight)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    rows = read_json_list(input_path)
    candidates = _article_candidates(source, image_root / "law", model.tokenizer)
    candidate_embeddings = _embed_items(model, torch, candidates, None, "corpus")
    query_embeddings = _embed_items(model, torch, rows, query_images, "queries")
    checkpoint = torch.load(adapter_path, map_location="cpu", weights_only=True)
    adapter = _adapter(torch, int(checkpoint["dimension"]))
    adapter.load_state_dict(checkpoint["state_dict"])
    predictions = _predictions(torch, rows, candidates, query_embeddings, candidate_embeddings, adapter, top_k)
    metrics = retrieval_metrics(predictions, rows) if all("relevant_articles" in row for row in rows) else None
    config = {
        "method": "visualized-bge-finetuned-adapter",
        "input": str(input_path),
        "source": str(source),
        "query_images": str(query_images),
        "model": model_name,
        "weight": str(weight),
        "adapter": str(adapter_path),
        "top_k": top_k,
        "chunk_tokens": CHUNK_TOKENS,
        "chunk_overlap": CHUNK_OVERLAP,
    }
    resources = _resources(torch, started, candidates, len(rows))
    for filename, value in (
        ("config.json", config),
        ("predictions.json", predictions),
        ("metrics.json", metrics),
        ("resources.json", resources),
    ):
        (output / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"output": str(output), "metrics": metrics, "resources": resources}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train", help="Fine-tune metric adapter trên train và đánh giá dev")
    train_parser.add_argument("--train", type=Path, default=DEFAULT_SPLITS / "train.json")
    train_parser.add_argument("--dev", type=Path, default=DEFAULT_SPLITS / "dev.json")
    train_parser.add_argument("--output", type=Path, default=DEFAULT_FINETUNE_OUTPUT)
    train_parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    train_parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGES)
    train_parser.add_argument("--model", default=DEFAULT_MODEL)
    train_parser.add_argument("--weight", type=Path, default=DEFAULT_WEIGHT)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--temperature", type=float, default=0.05)
    train_parser.add_argument("--top-k", type=int, default=5)
    train_parser.add_argument("--seed", type=int, default=2025)
    retrieve_parser = subparsers.add_parser("retrieve", help="Truy hồi bằng fine-tuned adapter")
    retrieve_parser.add_argument("--input", type=Path, default=DEFAULT_SPLITS / "dev.json")
    retrieve_parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACTS / "visual-bge-finetuned-retrieval-dev")
    retrieve_parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    retrieve_parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    retrieve_parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGES)
    retrieve_parser.add_argument("--query-images", type=Path, default=DEFAULT_IMAGES / "train")
    retrieve_parser.add_argument("--model", default=DEFAULT_MODEL)
    retrieve_parser.add_argument("--weight", type=Path, default=DEFAULT_WEIGHT)
    retrieve_parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        result = train(
            args.train, args.dev, args.output, args.source, args.image_root, args.model, args.weight,
            args.epochs, args.learning_rate, args.temperature, args.top_k, args.seed,
        ) if args.command == "train" else retrieve(
            args.input, args.output, args.adapter, args.source, args.image_root, args.query_images,
            args.model, args.weight, args.top_k,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
