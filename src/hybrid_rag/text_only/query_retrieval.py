"""Retrieve citations from pure question/legal-text embeddings and evaluate dev."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from hybrid_rag.text_only.corpus_preprocessing import (
    load_step, read_lines, run_step, save, write_lines, hashes_of,
)
from hybrid_rag.text_only.corpus_embedding import BGE_SOURCE, load_index
from hybrid_rag.text_only.query_embedding import query_inputs
from text_based_rag.query_retrieval import scores

KS = (1, 3, 5, 10)


def retrieve(args, folder, meta, progress):
    progress.set("validate pure-text indexes")
    _, corpus, cv, hashes = load_index("corpus_embedding", "chunks.jsonl")
    _, queries, qv, query_hashes = load_index("query_embedding_dev", "queries.jsonl")
    hashes.update(query_hashes)
    _, source, registry_hashes = load_step("data_preprocessing", ["structured_corpus.jsonl"])
    hashes.update(registry_hashes)
    parents = read_lines(source / "structured_corpus.jsonl")
    parent_map = {r['article_key']: (str(r['document_id']), str(r['id'])) for r in parents}
    registry = set(parent_map.values())
    current, original, dev_hashes = query_inputs()
    hashes.update(dev_hashes)
    if queries != [{**r, 'vector_row': i} for i, r in enumerate(current)]:
        raise ValueError("Dev hiện tại khác query index.")
    groups = {}
    for row, chunk in enumerate(corpus):
        citation = (str(chunk['document_id']), str(chunk['id']))
        if parent_map.get(chunk['article_key']) != citation:
            raise ValueError(f"Chunk sai citation: {chunk['chunk_id']}")
        groups.setdefault(citation, []).append(row)
    if set(groups) != registry or len(groups) < max(KS):
        raise ValueError("Index không bao phủ đủ citation corpus.")
    meta.update(input_sha256=hashes, split=args.split, representation="question_and_original_legal_text_only",
                similarity="cosine", aggregation="max_child_chunk_score", tie_break="citation_lexicographic; chunk_id_lexicographic",
                queries=len(queries), chunks=len(corpus), citations=len(groups))
    progress.set("retrieve and evaluate pure text", len(queries))
    similarity = qv @ cv.T
    predictions, evaluations, missing = [], [], set()
    totals = {k: {'tp': 0, 'predicted': 0, 'relevant': 0, 'hits': 0, 'rows': []} for k in KS}
    for i, query in enumerate(queries):
        hits = []
        for (law, article), indices in sorted(groups.items()):
            winner = min(indices, key=lambda j: (-float(similarity[i,j]), corpus[j]['chunk_id']))
            hits.append({'law_id': law, 'article_id': article, 'score': float(similarity[i,winner]),
                         'best_chunk_id': corpus[winner]['chunk_id']})
        hits.sort(key=lambda r: (-r['score'], r['law_id'], r['article_id']))
        hits = [{**r, 'rank': rank} for rank, r in enumerate(hits[:max(KS)], 1)]
        predictions.append({'query_id': query['query_id'], 'source_image_id': query['source_image_id'],
                            'query_text': query['text'], 'results': hits})
        gold = {(str(r['law_id']), str(r['article_id'])) for r in original[i]['relevant_articles']}
        if not gold:
            raise ValueError(f"Dev thiếu gold: {query['query_id']}")
        missing.update(gold - registry)
        per_query = {'query_id': query['query_id'], 'gold_citations': original[i]['relevant_articles'], 'at_k': {}}
        for k in KS:
            predicted = {(r['law_id'], r['article_id']) for r in hits[:k]}
            tp = len(gold & predicted)
            values = scores(tp, len(predicted), len(gold))
            per_query['at_k'][str(k)] = {**values, 'hit': bool(tp), 'true_positives': tp}
            total = totals[k]
            total['tp'] += tp
            total['predicted'] += len(predicted)
            total['relevant'] += len(gold)
            total['hits'] += bool(tp)
            total['rows'].append(values)
        evaluations.append(per_query)
        progress.done += 1
    at_k = {str(k): {'macro': {key: float(np.mean([r[key] for r in t['rows']])) for key in ('precision','recall','f1','f2')},
                    'micro': scores(t['tp'],t['predicted'],t['relevant']), 'hit_rate': t['hits']/len(queries)} for k,t in totals.items()}
    metrics = {'method': 'pure_text_bge_m3', 'split': args.split, 'queries': len(queries),
               'primary_metric': 'macro_f2_at_5', 'macro_f2_at_5': at_k['5']['macro']['f2'], 'at_k': at_k,
               'gold_citations_outside_registry': [{'law_id': l, 'article_id': a} for l,a in sorted(missing)],
               'evaluation_scope': 'dev_development_not_independent_test; gold_unchanged'}
    write_lines(folder / "predictions.jsonl", predictions)
    write_lines(folder / "per_query_metrics.jsonl", evaluations)
    save(folder / "metrics.json", metrics)
    meta.update(macro_f2_at_5=metrics['macro_f2_at_5'],
                output_sha256=hashes_of([folder / n for n in ('predictions.jsonl','per_query_metrics.jsonl','metrics.json')]))
    print(f"[dev text_only] macro F2@5={metrics['macro_f2_at_5']:.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev",), default="dev")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_step(args, "query_retrieval_dev", retrieve, Path(__file__),
             [Path(__file__).with_name(n) for n in ('corpus_embedding.py','query_embedding.py')]
             + [BGE_SOURCE, ROOT / "src/text_based_rag/query_retrieval.py"])


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
