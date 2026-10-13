"""Retrieve five unique legal citations by cosine and evaluate on train labels."""

from datetime import datetime, timezone
from importlib.metadata import version
import json
from threading import Event, Thread
import time

import numpy as np

from multimodal_legal_rag.data import read_json_list
from .corpus_embedding import file_hash
from .image_captioning import ROOT, read_lines, save

BASE = ROOT / "artifacts/experiments/text_based_rag"
OUTPUT = BASE / "query_retrieval"
TRAIN = ROOT / "data/processed/splits/train.json"
TOP_K = 5


def scores(tp, predicted, relevant):
    precision = tp / predicted if predicted else 0.0
    recall = tp / relevant if relevant else 0.0
    return {"precision": precision, "recall": recall,
            "f1": 2 * tp / (predicted + relevant) if predicted + relevant else 0.0,
            "f2": 5 * tp / (predicted + 4 * relevant) if predicted + relevant else 0.0}


def main():
    if OUTPUT.exists():
        raise FileExistsError("Output query_retrieval đã tồn tại. Tự lưu thư mục cũ sang nơi khác trước khi chạy lại.")
    started, stopped = time.perf_counter(), Event()
    phase, done, total = "validate-input", 0, 0
    manifest = None

    def report():
        elapsed = time.perf_counter() - started
        eta = f"{elapsed * (total - done) / done:.0f}s" if done else "unknown"
        print(f"[{phase}] {done}/{total} ({100 * done / max(1, total):.1f}%) "
              f"elapsed={elapsed:.0f}s ETA={eta}", flush=True)

    def heartbeat():
        while not stopped.wait(30):
            report()

    worker = Thread(target=heartbeat, daemon=True)
    worker.start()
    report()
    try:
        inputs, hashes, manifests = {}, {}, {}
        for folder, mapping_name, id_field in (("query_embedding_train", "queries.jsonl", "query_id"),
                                                ("corpus_embedding", "chunks.jsonl", "chunk_id")):
            source = BASE / folder
            meta = json.loads((source / "manifest.json").read_text(encoding="utf-8-sig"))
            if meta.get("status") != "COMPLETED":
                raise ValueError(f"{folder} chưa COMPLETED.")
            for name in ("vectors.npy", mapping_name):
                digest = file_hash(source / name)
                if meta.get("output_sha256", {}).get(name) != digest:
                    raise ValueError(f"File đã thay đổi hoặc không khớp manifest: {folder}/{name}")
                hashes[f"{folder}/{name}"] = digest
            hashes[f"{folder}/manifest.json"] = file_hash(source / "manifest.json")
            mapping = read_lines(source / mapping_name)
            vectors = np.load(source / "vectors.npy", mmap_mode="r", allow_pickle=False)
            if (not mapping or vectors.dtype != np.float32 or vectors.shape != (len(mapping), 1024)
                    or not np.isfinite(vectors).all()
                    or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
                    or len({m[id_field] for m in mapping}) != len(mapping)
                    or any(m["vector_row"] != i for i, m in enumerate(mapping))):
                raise ValueError(f"Vector hoặc ánh xạ hàng không hợp lệ: {folder}")
            inputs[folder], manifests[folder] = (mapping, vectors), meta
        qm, cm = manifests["query_embedding_train"], manifests["corpus_embedding"]
        for key in ("model", "revision", "model_files_sha256", "pooling", "normalize", "dimension",
                    "stored_dtype", "inference_dtype", "input_prefix"):
            if key not in qm or qm[key] != cm.get(key):
                raise ValueError(f"Encoder query/corpus không khớp: {key}")
        if qm.get("split") != "train" or qm["normalize"] != "L2":
            raise ValueError("Cần vector train được chuẩn hóa L2.")
        train_hash = file_hash(TRAIN)
        if qm.get("input_sha256", {}).get("train.json") != train_hash:
            raise ValueError("Train đã thay đổi sau embedding query; kiểm tra nguồn trước khi đánh giá.")
        hashes["train.json"] = train_hash
        queries, query_vectors = inputs["query_embedding_train"]
        chunks, chunk_vectors = inputs["corpus_embedding"]
        total = len(queries)
        groups = {}
        for i, chunk in enumerate(chunks):
            citation = (str(chunk["document_id"]), str(chunk["id"]))
            groups.setdefault(citation, []).append(i)
        if len(groups) < TOP_K:
            raise ValueError("Corpus có ít hơn 5 điều khoản duy nhất.")
        phase = "cosine-and-article-ranking"
        similarity = query_vectors @ chunk_vectors.T
        if not np.isfinite(similarity).all():
            raise ValueError("Điểm cosine không hợp lệ.")
        results = []
        for row, query in enumerate(queries):
            best = []
            for citation, indices in groups.items():
                index = indices[int(np.argmax(similarity[row, indices]))]
                best.append((float(similarity[row, index]), citation, index))
            best.sort(key=lambda item: (-item[0], item[1]))
            hits = []
            for rank, (score, citation, index) in enumerate(best[:TOP_K], 1):
                chunk = chunks[index]
                hits.append({"rank": rank, "law_id": citation[0], "article_id": citation[1],
                             "score": score, "article_key": chunk["article_key"],
                             "title": chunk["title"], "best_chunk_id": chunk["chunk_id"],
                             "best_chunk_text": chunk["text"]})
            assert len(hits) == TOP_K and len({(h["law_id"], h["article_id"]) for h in hits}) == TOP_K
            results.append({"query_id": query["query_id"], "query_text": query["text"], "results": hits})
            done += 1
        report()
        # Labels are read only after ranking; they never influence the scores or selected citations.
        phase = "evaluate-train"
        gold_rows = read_json_list(TRAIN)
        gold = {r["id"]: r["relevant_articles"] for r in gold_rows}
        if len(gold) != len(gold_rows) or set(gold) != {q["query_id"] for q in queries}:
            raise ValueError("ID nhãn train không khớp query.")
        per_query, missing_gold = [], set()
        for result in results:
            relevant = {(str(r["law_id"]), str(r["article_id"])) for r in gold[result["query_id"]]}
            if not relevant:
                raise ValueError(f"Query thiếu citation đúng: {result['query_id']}")
            missing_gold.update(relevant - groups.keys())
            predicted = {(r["law_id"], r["article_id"]) for r in result["results"]}
            tp = len(predicted & relevant)
            per_query.append({"query_id": result["query_id"], "true_positives": tp,
                              "predicted_count": len(predicted), "relevant_count": len(relevant),
                              **scores(tp, len(predicted), len(relevant)), "hit": int(tp > 0),
                              "gold_citations": [{"law_id": law, "article_id": article}
                                                 for law, article in sorted(relevant)]})
        metrics = {"split": "train", "top_k": TOP_K, "queries": total,
                   "macro": {key: sum(r[key] for r in per_query) / total
                             for key in ("precision", "recall", "f1", "f2")},
                   "micro": scores(sum(r["true_positives"] for r in per_query),
                                   sum(r["predicted_count"] for r in per_query),
                                   sum(r["relevant_count"] for r in per_query)),
                   "hit_rate_at_5": sum(r["hit"] for r in per_query) / total,
                   "gold_citations_missing_from_index": [{"law_id": law, "article_id": article}
                                                         for law, article in sorted(missing_gold)]}
        OUTPUT.mkdir(parents=True, exist_ok=False)
        manifest = {"status": "IN_PROGRESS", "created_at": datetime.now(timezone.utc).isoformat(),
                    "split": "train", "queries": total, "chunks": len(chunks), "articles": len(groups),
                    "top_k": TOP_K, "similarity": "cosine", "article_aggregation": "max_chunk_score",
                    "citation_key": ["law_id", "article_id"], "tie_break": "citation_lexicographic_then_chunk_row",
                    "input_sha256": hashes, "numpy_version": version("numpy"),
                    "resources": {"device": "cpu", "similarity_matrix_bytes": similarity.nbytes}}
        save(OUTPUT / "manifest.json", manifest)
        for name, records in (("retrieval_results.jsonl", results), ("per_query_metrics.jsonl", per_query)):
            with (OUTPUT / name).open("x", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        save(OUTPUT / "metrics.json", metrics)
        with (OUTPUT / "retrieval_preview.txt").open("x", encoding="utf-8") as stream:
            for result, evaluation in zip(results, per_query):
                stream.write(f"===== {result['query_id']} =====\n{result['query_text']}\n\n")
                stream.write(f"Precision@5={evaluation['precision']:.4f}; Recall@5={evaluation['recall']:.4f}; "
                             f"F2@5={evaluation['f2']:.4f}\nGold: {evaluation['gold_citations']}\n\n")
                for hit in result["results"]:
                    stream.write(f"#{hit['rank']} | {hit['law_id']} | {hit['article_id']} | "
                                 f"cosine={hit['score']:.6f} | {hit['title']}\n"
                                 f"{hit['best_chunk_id']}\n{hit['best_chunk_text']}\n\n")
        manifest.update(status="COMPLETED", elapsed_seconds=time.perf_counter() - started,
                        output_sha256={name: file_hash(OUTPUT / name) for name in
                                       ("retrieval_results.jsonl", "per_query_metrics.jsonl", "metrics.json",
                                        "retrieval_preview.txt")})
        save(OUTPUT / "manifest.json", manifest)
        report()
        print(f"[metrics@5] {json.dumps(metrics['macro'])}", flush=True)
        if missing_gold:
            print(f"[warning] {len(missing_gold)} gold citation không có trong index; vẫn tính vào recall.", flush=True)
        print(f"[output] {OUTPUT}", flush=True)
    except BaseException as exc:
        if manifest is not None:
            manifest.update(status="FAILED", error=str(exc))
            save(OUTPUT / "manifest.json", manifest)
        raise
    finally:
        stopped.set()
        worker.join()


if __name__ == "__main__":
    main()
