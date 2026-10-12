"""Run the original retrieval baseline on the fixed train split, without changing its ranking."""
from pathlib import Path
import sys
import json
import time
from threading import Event, Thread
import importlib.util
from io import StringIO

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/VLSP2025-MLQA-TSR"
OUT = RAW / "results" / ("baseline_train_" + time.strftime("%Y%m%d_%H%M%S"))
MAX_TEXT_TOKENS = 2048
sys.path.insert(0, str(RAW))
sys.path.insert(0, str(ROOT / "models/FlagEmbedding/research/visual_bge"))
import numpy as np
import torch
from visual_bge.modeling import Visualized_BGE

spec = importlib.util.spec_from_file_location("official_retrieval", RAW / "retrieval.py")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
read_html = baseline.pd.read_html
baseline.pd.read_html = lambda html, **kwargs: read_html(StringIO(html), **kwargs)
OUT.mkdir(parents=True, exist_ok=False)
started = time.monotonic()
state = {"phase": "load-model", "done": 0, "total": 1}
stop = Event()

def report():
    elapsed = time.monotonic() - started
    done, total = state["done"], state["total"]
    eta = f"{elapsed * (total-done)/done:.0f}s" if done else "unknown"
    print(f"[{state['phase']}] {done}/{total} {100*done/max(1,total):.1f}% elapsed={elapsed:.0f}s ETA={eta}", flush=True)

def heartbeat():
    while not stop.wait(30):
        report()

Thread(target=heartbeat, daemon=True).start()
manifest = {"status": "IN_PROGRESS", "split": "train", "top_k_vectors": 5,
            "ranking": "official top-5 vectors then deduplicate citations",
            "model": "Visualized_m3.pth", "dtype": "float32", "truncation": True,
            "max_text_tokens": MAX_TEXT_TOKENS, "truncated_corpus_vectors": 0,
            "truncated_query_vectors": 0, "truncated_examples": []}

def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

try:
    save("manifest.json", manifest)
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(min(12*1024**3, free-2*1024**3)/total)
    model = Visualized_BGE(model_name_bge="BAAI/bge-m3", model_weight=str(ROOT / "models/Visualized_m3.pth"),
                          from_pretrained=str(ROOT / "models/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"))
    model.eval()
    class CappedTokenizer:
        def __init__(self, original):
            self.original = original

        def __call__(self, text, **kwargs):
            length = len(self.original(text, truncation=False)["input_ids"])
            if length > MAX_TEXT_TOKENS:
                key = "truncated_query_vectors" if state["phase"] == "query-inference" else "truncated_corpus_vectors"
                manifest[key] += 1
                if len(manifest["truncated_examples"]) < 20:
                    manifest["truncated_examples"].append({"phase": state["phase"],
                                                            "tokens": length, "text_start": text[:80]})
            return self.original(text, **kwargs, truncation=True, max_length=MAX_TEXT_TOKENS)

    model.tokenizer = CappedTokenizer(model.tokenizer)
    original_encode = model.encode

    def encode(*args, **kwargs):
        with torch.inference_mode():
            result = original_encode(*args, **kwargs)
        state["done"] += 1
        return result

    model.encode = encode
    laws = json.loads((RAW / "dataset/law_db/vlsp2025_law.json").read_text(encoding="utf-8"))
    train = json.loads((ROOT / "data/processed/splits/train.json").read_text(encoding="utf-8"))
    import re
    state.update(phase="corpus-inference", done=0, total=sum(max(1, len(re.findall(r"<<IMAGE: \s*(.*?)\s* /IMAGE>>", a["text"]))) for law in laws for a in law["articles"]))
    ids, vectors = baseline.embedding_articles(laws, str(ROOT / "data/processed/images/law"), model)
    np.save(OUT / "law_id.npy", ids)
    np.save(OUT / "law_embedding.npy", vectors)
    state.update(phase="query-inference", done=0, total=len(train))
    predictions = []
    for row in train:
        vector = model.encode(image=str(ROOT / "data/processed/images/train" / (row["image_id"] + ".jpg")), text=row["question"])
        similarity = (vector.cpu().detach() @ vectors.T).T.numpy()
        selected = baseline.get_top_k(similarity, 5)
        citations = []
        for i in range(len(ids)):
            if selected[i] > 0:
                citation = {"law_id": ids[i].split("_")[1], "article_id": ids[i].split("_")[0].split("-")[0]}
                if citation not in citations:
                    citations.append(citation)
        predictions.append({"id": row["id"], "image_id": row["image_id"], "question": row["question"], "relevant_articles": citations})
        save("predictions.json", predictions)
    metrics = baseline.eval_retrieval(predictions, train)
    save("metrics.json", metrics)
    manifest.update(status="COMPLETED", queries=len(train), corpus_vectors=len(ids), elapsed_seconds=time.monotonic()-started,
                    gpu=torch.cuda.get_device_name(), peak_vram_bytes=torch.cuda.max_memory_allocated())
    print(f"[metrics] {metrics}\n[output] {OUT}", flush=True)
except BaseException as exc:
    manifest.update(status="FAILED", error=str(exc), progress=state.copy())
    raise
finally:
    stop.set()
    save("manifest.json", manifest)
