"""Fuse saved pure-text and full-image dev rankings with weighted reciprocal rank fusion."""

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from hybrid_rag.full_image_method.corpus_image_embedding import (
    BASE, CORPUS, check_hashes, hashes_of, read_lines, run_step, save, write_lines,
)
from text_based_rag.query_retrieval import scores

TEXT = ROOT / "artifacts/experiments/hybrid_rag/text_only/query_retrieval_dev"
IMAGE = BASE / "retrieval_dev"
DEV = ROOT / "data/processed/splits/dev.json"
KS = (1, 3, 5, 10)


def load_branch(folder, original, registry):
    manifest = folder / "manifest.json"
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    required = {str((folder / name).relative_to(ROOT)) for name in ("predictions.jsonl", "metrics.json")}
    if (meta.get("status") != "COMPLETED" or meta.get("split") != "dev"
            or not meta.get("code_sha256") or not required <= set(meta.get("output_sha256", {}))):
        raise ValueError(f"Nhánh chưa COMPLETED/dev hoặc thiếu file/hash: {folder}")
    for field in ("code_sha256", "input_sha256", "output_sha256"):
        check_hashes(meta[field])
    rows = read_lines(folder / "predictions.jsonl")
    lookup = {row['query_id']: row for row in rows}
    if len(lookup) != len(rows) or set(lookup) != set(original):
        raise ValueError(f"Query ID của nhánh không khớp dev: {folder}")
    for qid, row in lookup.items():
        query = original[qid]
        if (row.get('source_image_id') != query['image_id']
                or ('query_text' in row and row['query_text'] != query['question'])):
            raise ValueError(f"Query metadata khác dev: {folder.name}/{qid}")
        hits = row['results']
        citations = [(hit['law_id'], hit['article_id']) for hit in hits]
        if (len(hits) != 10 or len(set(citations)) != len(hits) or not set(citations) <= registry
                or any(hit.get('rank') != i or not math.isfinite(hit['score']) for i, hit in enumerate(hits, 1))
                or hits != sorted(hits, key=lambda h: (-h['score'], h['law_id'], h['article_id']))):
            raise ValueError(f"Citation/rank/score không hợp lệ: {folder.name}/{qid}")
    hashes = {**meta['input_sha256'], **meta['output_sha256'], **hashes_of([manifest])}
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    return lookup, hashes, metrics


def fuse(args, folder, meta, progress):
    progress.set("validate dev and citation registry")
    paths = [DEV, CORPUS / "manifest.json", CORPUS / "structured_corpus.jsonl"]
    hashes = hashes_of(paths)
    registry_meta = json.loads(paths[1].read_text(encoding="utf-8"))
    if registry_meta.get('status') != 'COMPLETED' or registry_meta.get('schema_version') != 4:
        raise ValueError("Citation registry cần preprocessing schema 4, COMPLETED.")
    parents = read_lines(paths[2])
    registry = {(str(row['document_id']), str(row['id'])) for row in parents}
    dev = json.loads(DEV.read_text(encoding="utf-8"))
    original = {row['id']: row for row in dev}
    if not original or len(original) != len(dev) or any(not r.get('relevant_articles') for r in dev):
        raise ValueError("Dev rỗng, trùng ID hoặc thiếu gold citation.")
    progress.set("validate text retrieval")
    text, text_hashes, text_metrics = load_branch(TEXT, original, registry)
    hashes.update(text_hashes)
    progress.set("validate image retrieval")
    image, image_hashes, image_metrics = load_branch(IMAGE, original, registry)
    hashes.update(image_hashes)
    meta.update(input_sha256=hashes, split="dev", queries=len(dev), method="weighted_rrf",
                text_weight=args.text_weight, image_weight=1-args.text_weight, rrf_k=args.rrf_k,
                final_top_n=args.final_top_n, branch_candidate_depth=10,
                candidate_policy="union_saved_top10_citations", tie_break="citation_lexicographic",
                weight_selection="fixed_by_cli_no_gold_search", score_interpretation="rank_fusion_not_probability")
    meta['code_sha256'].update(hashes_of([ROOT / "src/text_based_rag/query_retrieval.py"]))
    save(folder / "manifest.json", meta)
    predictions, final, evaluations = [], [], []
    cutoffs = sorted(set(KS + (args.final_top_n,)))
    totals = {k: {'tp': 0, 'predicted': 0, 'relevant': 0, 'hits': 0, 'rows': []} for k in cutoffs}
    branch_f2 = {'text': [], 'image': []}
    improvements = {'improved': 0, 'worsened': 0, 'unchanged': 0}
    union_gold = gold_total = extra_image_gold = 0
    outside_registry = set()
    progress.set("fuse rankings and evaluate", len(dev))
    for query in dev:
        qid = query['id']
        candidates = {}
        for branch, weight, rows in (('text', args.text_weight, text[qid]['results']),
                                     ('image', 1-args.text_weight, image[qid]['results'])):
            for hit in rows:
                citation = (hit['law_id'], hit['article_id'])
                item = candidates.setdefault(citation, {'law_id': citation[0], 'article_id': citation[1], 'score': 0.0})
                contribution = weight/(args.rrf_k+hit['rank'])
                item['score'] += contribution
                item[branch] = {**hit, 'rrf_contribution': contribution}
        ranked = sorted((r for r in candidates.values() if r['score'] > 0),
                        key=lambda r: (-r['score'], r['law_id'], r['article_id']))
        hits = [{**r, 'rank': rank} for rank, r in enumerate(ranked[:10], 1)]
        if len(hits) < 10:
            raise ValueError(f"Fusion thiếu ứng viên: {qid}")
        selected = hits[:args.final_top_n]
        predictions.append({'query_id': qid, 'source_image_id': query['image_id'], 'results': hits})
        final.append({'id': qid, 'image_id': query['image_id'], 'question': query['question'],
                      'relevant_articles': [{'law_id': r['law_id'], 'article_id': r['article_id']} for r in selected]})
        gold = {(str(r['law_id']), str(r['article_id'])) for r in query['relevant_articles']}
        text_set = {(r['law_id'], r['article_id']) for r in text[qid]['results']}
        image_set = {(r['law_id'], r['article_id']) for r in image[qid]['results']}
        union_gold += len(gold & (text_set | image_set))
        extra_image_gold += len((gold & image_set) - text_set)
        gold_total += len(gold)
        outside_registry.update(gold-registry)
        per_query = {'query_id': qid, 'gold_citations': query['relevant_articles'], 'at_k': {},
                     'union_candidate_recall': len(gold & (text_set | image_set))/len(gold),
                     'image_additional_gold': len((gold & image_set)-text_set), 'branch_f2_at_5': {}}
        for branch, rows in (('text', text[qid]['results']), ('image', image[qid]['results'])):
            predicted = {(r['law_id'], r['article_id']) for r in rows[:5]}
            value = scores(len(predicted & gold), len(predicted), len(gold))['f2']
            branch_f2[branch].append(value)
            per_query['branch_f2_at_5'][branch] = value
        for k in cutoffs:
            predicted = {(r['law_id'], r['article_id']) for r in hits[:k]}
            tp = len(predicted & gold)
            values = scores(tp, len(predicted), len(gold))
            per_query['at_k'][str(k)] = {**values, 'hit': bool(tp), 'true_positives': tp}
            total = totals[k]
            total['tp'] += tp
            total['predicted'] += len(predicted)
            total['relevant'] += len(gold)
            total['hits'] += bool(tp)
            total['rows'].append(values)
        delta = per_query['at_k']['5']['f2'] - per_query['branch_f2_at_5']['text']
        per_query['delta_f2_at_5_vs_text'] = delta
        improvements['improved' if delta > 1e-12 else 'worsened' if delta < -1e-12 else 'unchanged'] += 1
        evaluations.append(per_query)
        progress.done += 1
    at_k = {str(k): {'macro': {key: float(np.mean([r[key] for r in t['rows']])) for key in ('precision','recall','f1','f2')},
                    'micro': scores(t['tp'],t['predicted'],t['relevant']), 'hit_rate': t['hits']/len(dev)} for k,t in totals.items()}
    comparison = {name: float(np.mean(values)) for name,values in branch_f2.items()}
    for name, branch_metrics in (('text', text_metrics), ('image', image_metrics)):
        if not math.isclose(comparison[name], branch_metrics['macro_f2_at_5'], abs_tol=1e-12):
            raise ValueError(f"Metrics lưu của nhánh {name} không khớp predictions/gold dev.")
    comparison['hybrid'] = at_k['5']['macro']['f2']
    metrics = {'method': 'weighted_rrf_pure_text_full_image', 'split': 'dev', 'queries': len(dev),
               'text_weight': args.text_weight, 'image_weight': 1-args.text_weight, 'rrf_k': args.rrf_k,
               'final_top_n': args.final_top_n, 'primary_metric': 'macro_f2_at_5',
               'macro_f2_at_5': comparison['hybrid'], 'at_k': at_k, 'final_citation_metrics': at_k[str(args.final_top_n)],
               'comparison_macro_f2_at_5': comparison, 'delta_f2_at_5_vs_text': comparison['hybrid']-comparison['text'],
               'query_changes_vs_text_at_5': improvements, 'union_candidate_recall_micro': union_gold/gold_total,
               'image_additional_gold_occurrences_in_top10': extra_image_gold,
               'gold_citations_outside_registry': [{'law_id': l, 'article_id': a} for l,a in sorted(outside_registry)],
               'evaluation_scope': 'dev_development_not_independent_test; gold_unchanged; fixed_weights'}
    check_hashes(hashes)
    write_lines(folder / "predictions.jsonl", predictions)
    write_lines(folder / "per_query_metrics.jsonl", evaluations)
    save(folder / "final_citations.json", final)
    save(folder / "metrics.json", metrics)
    meta.update(macro_f2_at_5=comparison['hybrid'], output_sha256=hashes_of([folder / n for n in (
        'predictions.jsonl','final_citations.json','per_query_metrics.jsonl','metrics.json')]))
    print(f"[F2@5] text={comparison['text']:.6f}; image={comparison['image']:.6f}; hybrid={comparison['hybrid']:.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev",), default="dev")
    parser.add_argument("--text-weight", type=float, default=0.5)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--final-top-n", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.text_weight) or not 0 <= args.text_weight <= 1:
        parser.error("text-weight phải trong [0,1]")
    if args.rrf_k < 1 or not 1 <= args.final_top_n <= 10:
        parser.error("rrf-k phải >= 1; final-top-n trong [1,10]")
    args.command = "hybrid-retrieval"
    run_step(args, BASE / "hybrid_retrieval_dev", fuse, Path(__file__))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
