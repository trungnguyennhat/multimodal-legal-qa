"""Retrieve corpus citations or train-example citations using pure question embeddings."""

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


def example_retrieval(values, bank, labels, registry):
    if (len(values) != len(bank) or len(bank) < 5 or not np.isfinite(values).all()
            or any(r.get('source_split') != 'train' for r in bank)
            or {r['query_id'] for r in bank} != set(labels)):
        raise ValueError("Example bank phải chỉ gồm train, đủ top-5, có labels/vector khớp nhau.")
    neighbors = sorted(range(len(bank)), key=lambda i: (-float(values[i]), bank[i]['query_id']))[:5]
    candidates, examples = {}, []
    for rank, index in enumerate(neighbors, 1):
        query = bank[index]
        gold = labels[query['query_id']]['relevant_articles']
        score = float(values[index])
        examples.append({'rank': rank, 'query_id': query['query_id'], 'source_split': 'train',
                         'source_image_id': query['source_image_id'], 'score': score, 'gold_citations': gold})
        for citation in sorted({(str(r['law_id']), str(r['article_id'])) for r in gold} & registry):
            item = candidates.setdefault(citation, {'law_id': citation[0], 'article_id': citation[1],
                                                    'score': score, 'source_query_ids': []})
            item['score'] = max(item['score'], score)
            item['source_query_ids'].append(query['query_id'])
    hits = sorted(candidates.values(), key=lambda r: (-r['score'], r['law_id'], r['article_id']))[:10]
    return [{**r, 'rank': i} for i, r in enumerate(hits, 1)], examples


def evaluate_predictions(predictions, original, registry, method, cutoffs=KS):
    gold_by_id = {r['id']: r for r in original}
    if (not original or len(gold_by_id) != len(original) or len(predictions) != len(original)
            or {r['query_id'] for r in predictions} != set(gold_by_id)):
        raise ValueError("Predictions phải khớp đúng tập query cần chấm.")
    totals = {k: {'tp': 0, 'predicted': 0, 'relevant': 0, 'hits': 0, 'rows': []} for k in cutoffs}
    evaluations, missing = [], set()
    for prediction in predictions:
        row = gold_by_id[prediction['query_id']]
        gold = {(str(r['law_id']), str(r['article_id'])) for r in row['relevant_articles']}
        hits = prediction['results']
        citations = [(r['law_id'], r['article_id']) for r in hits]
        if (not gold or prediction['source_image_id'] != row['image_id']
                or len(set(citations)) != len(citations) or not set(citations) <= registry
                or any(r['rank'] != i or not np.isfinite(r['score']) for i,r in enumerate(hits,1))):
            raise ValueError(f"Prediction/gold không hợp lệ: {row['id']}")
        missing.update(gold-registry)
        per_query = {'query_id': row['id'], 'gold_citations': row['relevant_articles'], 'at_k': {}}
        for k in cutoffs:
            predicted = set(citations[:k])
            tp = len(predicted & gold)
            values = scores(tp, len(predicted), len(gold))
            per_query['at_k'][str(k)] = {**values, 'hit': bool(tp), 'true_positives': tp}
            total = totals[k]
            total['tp'] += tp
            total['predicted'] += len(predicted)
            total['relevant'] += len(gold)
            total['hits'] += bool(tp)
            total['rows'].append(values)
        evaluations.append(per_query)
    at_k = {str(k): {'macro': {key: float(np.mean([r[key] for r in t['rows']])) for key in ('precision','recall','f1','f2')},
                    'micro': scores(t['tp'],t['predicted'],t['relevant']), 'hit_rate': t['hits']/len(original)} for k,t in totals.items()}
    metrics = {'method': method, 'split': 'dev', 'queries': len(original), 'primary_metric': 'macro_f2_at_5',
               'macro_f2_at_5': at_k['5']['macro']['f2'], 'at_k': at_k,
               'gold_citations_outside_registry': [{'law_id': l, 'article_id': a} for l,a in sorted(missing)],
               'evaluation_scope': 'dev_development_not_independent_test; gold_unchanged'}
    return metrics, evaluations


def retrieve(args, folder, meta, progress):
    progress.set("validate pure-text indexes")
    _, corpus, cv, hashes = load_index("corpus_embedding", "chunks.jsonl")
    _, all_queries, all_qv, query_hashes = load_index("query_embedding_dev", "queries.jsonl")
    hashes.update(query_hashes)
    _, source, registry_hashes = load_step("data_preprocessing", ["structured_corpus.jsonl"])
    hashes.update(registry_hashes)
    parents = read_lines(source / "structured_corpus.jsonl")
    parent_map = {r['article_key']: (str(r['document_id']), str(r['id'])) for r in parents}
    registry = set(parent_map.values())
    current, original, dev_hashes = query_inputs()
    hashes.update(dev_hashes)
    if all_queries != [{**r, 'vector_row': i} for i,r in enumerate(current)]:
        raise ValueError("Train/dev khác query index; chạy lại query_embedding.py --overwrite.")
    dev_indices = [i for i,r in enumerate(all_queries) if r['source_split']=='dev']
    bank_indices = [i for i,r in enumerate(all_queries) if r['source_split']=='train']
    queries = [all_queries[i] for i in dev_indices]
    bank = [all_queries[i] for i in bank_indices]
    dev = [r for r in original if r['source_split']=='dev']
    labels = {r['id']:r for r in original if r['source_split']=='train'}
    groups = {}
    for i, chunk in enumerate(corpus):
        citation = (str(chunk['document_id']), str(chunk['id']))
        if parent_map.get(chunk['article_key']) != citation:
            raise ValueError(f"Chunk sai citation: {chunk['chunk_id']}")
        groups.setdefault(citation, []).append(i)
    if set(groups) != registry or len(groups) < 10:
        raise ValueError("Index không bao phủ đủ citation corpus.")
    excluded = {(str(c['law_id']),str(c['article_id'])) for r in labels.values() for c in r['relevant_articles']} - registry
    meta.update(input_sha256=hashes, split=args.split, method=args.method,
                representation="original_question_only", queries=len(queries), example_bank_queries=len(bank),
                example_bank_split="train", top_k_examples=5, same_image_policy="allow_shared_images",
                aggregation="max_child_chunk_score" if args.method=='corpus' else "max_train_example_cosine",
                tie_break="citation_lexicographic; source_id_lexicographic")
    progress.set("retrieve pure-text " + args.method, len(queries))
    similarity = all_qv[dev_indices] @ (cv.T if args.method=='corpus' else all_qv[bank_indices].T)
    if not np.isfinite(similarity).all():
        raise ValueError("Cosine không hữu hạn.")
    predictions = []
    for i, query in enumerate(queries):
        if args.method == 'examples':
            hits, examples = example_retrieval(similarity[i], bank, labels, registry)
        else:
            examples, hits = [], []
            for (law, article), indices in sorted(groups.items()):
                winner = min(indices, key=lambda j: (-float(similarity[i,j]), corpus[j]['chunk_id']))
                hits.append({'law_id': law, 'article_id': article, 'score': float(similarity[i,winner]),
                             'best_chunk_id': corpus[winner]['chunk_id']})
            hits.sort(key=lambda r: (-r['score'], r['law_id'], r['article_id']))
            hits = [{**r, 'rank': rank} for rank,r in enumerate(hits[:10],1)]
        prediction = {'query_id': query['query_id'], 'source_image_id': query['source_image_id'],
                      'source_split': 'dev', 'query_text': query['text'], 'results': hits}
        if args.method == 'examples':
            prediction['examples'] = examples
        predictions.append(prediction)
        progress.done += 1
    metrics, evaluations = evaluate_predictions(predictions, dev, registry, 'pure_text_' + args.method)
    metrics.update(example_bank_queries=len(bank), top_k_examples=5,
                   excluded_train_citations=[{'law_id':l,'article_id':a} for l,a in sorted(excluded)])
    write_lines(folder / "predictions.jsonl", predictions)
    write_lines(folder / "per_query_metrics.jsonl", evaluations)
    save(folder / "metrics.json", metrics)
    meta.update(macro_f2_at_5=metrics['macro_f2_at_5'],
                output_sha256=hashes_of([folder/n for n in ('predictions.jsonl','per_query_metrics.jsonl','metrics.json')]))
    print(f"[dev text {args.method}] F2@5={metrics['macro_f2_at_5']:.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev",), default="dev")
    parser.add_argument("--method", choices=("corpus","examples"), default="corpus")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    name = "query_retrieval_example" if args.method=='examples' else "query_retrieval_corpus"
    run_step(args, name, retrieve, Path(__file__),
             [Path(__file__).with_name(n) for n in ('corpus_embedding.py','query_embedding.py')]
             + [BGE_SOURCE, ROOT / "src/text_based_rag/query_retrieval.py"])


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
