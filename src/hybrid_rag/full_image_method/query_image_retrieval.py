"""Retrieve citations from full-image embeddings and evaluate unchanged dev gold labels."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from hybrid_rag.full_image_method.corpus_image_embedding import (
    BASE, check_hashes, corpus_inputs, hashes_of, read_lines, run_step, save, validate_vectors, write_lines,
)
from hybrid_rag.full_image_method.query_image_embedding import dev_inputs
from text_based_rag.query_retrieval import scores

KS = (1, 3, 5, 10)


def load_index(folder):
    meta = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    names = ["images.jsonl", "vectors.npy"] + (["queries.jsonl"] if folder.name == "query_dev" else [])
    expected = {str((folder / name).relative_to(ROOT)) for name in names}
    code_paths = [Path(__file__).with_name("corpus_image_embedding.py")]
    if folder.name == "query_dev":
        code_paths.append(Path(__file__).with_name("query_image_embedding.py"))
    if (meta.get("status") != "COMPLETED" or meta.get("code_sha256") != hashes_of(code_paths)
            or set(meta.get("output_sha256", {})) != expected):
        raise ValueError(f"Index chưa COMPLETED hoặc fingerprint sai: {folder}; chạy lại với --overwrite.")
    check_hashes(meta["input_sha256"])
    check_hashes(meta["output_sha256"])
    images = read_lines(folder / "images.jsonl")
    if not images or len({r['image_id'] for r in images}) != len(images) or any(r['vector_row'] != i for i, r in enumerate(images)):
        raise ValueError(f"Mapping vector không hợp lệ: {folder}")
    vectors = np.load(folder / "vectors.npy", allow_pickle=False)
    validate_vectors(vectors, len(images))
    return meta, images, vectors


def retrieve(args, folder, meta, progress):
    progress.set("validate indexes and inputs")
    cm, corpus, cv = load_index(BASE / "corpus")
    qm, images, qv = load_index(BASE / "query_dev")
    if cm["config"] != qm["config"]:
        raise ValueError("Corpus/query phải cùng model, revision và preprocessing.")
    current_corpus, registry, hashes = corpus_inputs(progress)
    current_images, query_map, queries, query_hashes = dev_inputs(progress)
    if current_corpus != corpus or current_images != images or read_lines(BASE / "query_dev/queries.jsonl") != query_map:
        raise ValueError("Mapping hiện tại khác index; chạy lại bước embedding bị thay đổi.")
    hashes.update(query_hashes)
    for index_meta in (cm, qm):
        hashes.update(index_meta["input_sha256"])
        hashes.update(index_meta["output_sha256"])
    hashes.update(hashes_of([BASE / "corpus/manifest.json", BASE / "query_dev/manifest.json"]))
    meta.update(config=cm["config"], input_sha256=hashes,
                index_type="numpy_exact_cosine", aggregation="max_image_score_per_citation",
                tie_break="law_id_article_id_lexicographic; image_id_lexicographic", split="dev")
    save(folder / "manifest.json", meta)
    groups = {}
    for i, image in enumerate(corpus):
        for c in image['citations']:
            groups.setdefault((c['law_id'], c['article_id']), []).append(i)
    if len(groups) < max(KS):
        raise ValueError("Corpus không đủ 10 citation có ảnh để chấm các cutoff đã chọn.")
    progress.set("cosine search", len(images))
    similarity = qv @ cv.T
    rankings = []
    for row in similarity:
        hits = []
        for (law, article), indices in sorted(groups.items()):
            winner = indices[int(np.argmax(row[indices]))]
            hits.append({"law_id": law, "article_id": article, "score": float(row[winner]),
                         "best_image_id": corpus[winner]['image_id'], "best_image_path": corpus[winner]['path']})
        hits.sort(key=lambda hit: (-hit['score'], hit['law_id'], hit['article_id']))
        rankings.append([{**hit, "rank": i} for i, hit in enumerate(hits[:max(KS)], 1)])
        progress.done += 1
    predictions, evaluations = [], []
    totals = {k: {"tp": 0, "predicted": 0, "relevant": 0, "hits": 0, "rows": []} for k in KS}
    covered = set(groups)
    corpus_hashes = {r['sha256'] for r in corpus}
    gold_total = gold_covered = 0
    missing_registry, without_image = set(), set()
    uncovered_queries, overlapping_queries = [], []
    progress.set("evaluate dev queries", len(queries))
    for query, mapping in zip(queries, query_map):
        gold = {(str(c['law_id']), str(c['article_id'])) for c in query['relevant_articles']}
        hits = rankings[mapping['vector_row']]
        predictions.append({**mapping, "results": hits})
        per_query = {"query_id": query['id'], "gold_citations": query['relevant_articles'], "at_k": {}}
        for k in KS:
            predicted = {(h['law_id'], h['article_id']) for h in hits[:k]}
            tp = len(predicted & gold)
            values = scores(tp, len(predicted), len(gold))
            per_query['at_k'][str(k)] = {**values, "hit": bool(tp), "true_positives": tp}
            total = totals[k]
            total['tp'] += tp
            total['predicted'] += len(predicted)
            total['relevant'] += len(gold)
            total['hits'] += bool(tp)
            total['rows'].append(values)
        absent = gold - covered
        per_query['gold_without_index_image'] = [{"law_id": law, "article_id": article} for law, article in sorted(absent)]
        per_query['query_image_in_corpus'] = images[mapping['vector_row']]['sha256'] in corpus_hashes
        if absent:
            uncovered_queries.append(query['id'])
        if per_query['query_image_in_corpus']:
            overlapping_queries.append(query['id'])
        missing_registry.update(gold - registry)
        without_image.update((gold & registry) - covered)
        gold_total += len(gold)
        gold_covered += len(gold & covered)
        evaluations.append(per_query)
        progress.done += 1
    at_k = {str(k): {"macro": {key: float(np.mean([r[key] for r in t['rows']])) for key in ('precision', 'recall', 'f1', 'f2')},
                    "micro": scores(t['tp'], t['predicted'], t['relevant']), "hit_rate": t['hits']/len(queries)}
            for k, t in totals.items()}
    metrics = {"method": "full_image_dinov2", "split": "dev", "queries": len(queries),
               "unique_query_images": len(images), "corpus_images": len(corpus), "citations_with_images": len(covered),
               "primary_metric": "macro_f2_at_5", "macro_f2_at_5": at_k['5']['macro']['f2'], "at_k": at_k,
               "coverage": {"gold_citation_occurrences": gold_total, "covered_gold_occurrences": gold_covered,
                            "gold_image_coverage": gold_covered/gold_total,
                            "queries_with_uncovered_gold": len(uncovered_queries), "query_ids_with_uncovered_gold": uncovered_queries,
                            "gold_citations_outside_registry": [list(c) for c in sorted(missing_registry)],
                            "gold_registry_citations_without_images": [list(c) for c in sorted(without_image)],
                            "query_images_in_corpus": len({r['sha256'] for r in images} & corpus_hashes),
                            "queries_with_image_in_corpus": overlapping_queries},
               "evaluation_scope": "dev_development_not_independent_test; gold_unchanged"}
    check_hashes(hashes)
    write_lines(folder / "predictions.jsonl", predictions)
    write_lines(folder / "per_query_metrics.jsonl", evaluations)
    save(folder / "metrics.json", metrics)
    meta["output_sha256"] = hashes_of([folder / n for n in ("predictions.jsonl", "per_query_metrics.jsonl", "metrics.json")])
    meta["macro_f2_at_5"] = metrics["macro_f2_at_5"]
    print(f"[dev] macro F2@5={metrics['macro_f2_at_5']:.6f}; gold image coverage={gold_covered/gold_total:.2%}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev",), default="dev")
    parser.add_argument("--overwrite", action="store_true", help="Replace dev retrieval predictions and metrics.")
    args = parser.parse_args()
    args.command = "retrieve"
    run_step(args, BASE / "retrieval_dev", retrieve, Path(__file__))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
