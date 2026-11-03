"""Retrieve each sign against corpus crops, or match crop sets to train examples; score dev."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np

from hybrid_rag.crop_image_method.corpus_image_cropping import (
    BASE, CORPUS, check_hashes, file_hash, hashes_of, load_step, read_lines, run_step, save, write_lines,
)
from hybrid_rag.crop_image_method.corpus_image_embedding import load_index
from hybrid_rag.text_only.query_embedding import read_splits
from hybrid_rag.text_only.query_retrieval import evaluate_predictions, example_retrieval

FULL = ROOT / 'artifacts/experiments/hybrid_rag/full_image_method'
EVALUATION_CODE = [ROOT / ('src/' + p) for p in (
    'hybrid_rag/text_only/query_embedding.py', 'hybrid_rag/text_only/corpus_preprocessing.py',
    'hybrid_rag/text_only/query_retrieval.py', 'text_based_rag/query_retrieval.py',
)]
DEPENDENCIES = [Path(__file__).with_name('corpus_image_embedding.py'), *EVALUATION_CODE]
RRF_K = 60


def evaluation_inputs():
    paths = [CORPUS / n for n in ('manifest.json', 'structured_corpus.jsonl')]
    meta = json.loads(paths[0].read_text(encoding='utf-8'))
    records = read_lines(paths[1])
    registry = {(str(r['document_id']), str(r['id'])) for r in records}
    if (meta.get('status') != 'COMPLETED' or meta.get('schema_version') != 4 or not registry
            or len({r['article_key'] for r in records}) != len(records)):
        raise ValueError('Citation registry phải COMPLETED, schema 4, không trùng article_key.')
    original, hashes = read_splits()
    hashes.update(hashes_of(paths))
    dev = [q for q in original if q['source_split'] == 'dev']
    labels = {q['id']: q for q in original if q['source_split'] == 'train'}
    if len(dev) != 112 or len(labels) != 418:
        raise ValueError('Split khác 112 dev / 418 train đã duyệt.')
    return dev, labels, registry, hashes


def validate_ranking(hits, registry):
    keys = [(r['law_id'], r['article_id']) for r in hits]
    if (len(hits) > 10 or len(set(keys)) != len(keys) or not set(keys) <= registry
            or any(r['rank'] != i or not math.isfinite(r['score']) for i, r in enumerate(hits, 1))
            or hits != sorted(hits, key=lambda r: (-r['score'], r['law_id'], r['article_id']))):
        raise ValueError('Ranking phải có 0–10 citation duy nhất, score hữu hạn, thứ tự xác định.')


def load_predictions(name, dev, labels, registry):
    meta, folder, hashes = load_step(name, ['predictions.jsonl', 'per_query_metrics.jsonl', 'metrics.json'])
    expected_method = {'query_retrieval_corpus': 'corpus', 'query_retrieval_example': 'examples',
                       'query_retrieval_hybrid': 'corpus_examples_rrf'}[name]
    if (meta.get('method') != expected_method or meta.get('split') != 'dev'
            or meta.get('example_bank_split') != 'train' or meta.get('example_bank_queries') != len(labels)):
        raise ValueError(f'Ranking không đúng method/dev/train bank: {name}')
    rows = read_lines(folder / 'predictions.jsonl')
    lookup = {r['query_id']: r for r in rows}
    if len(lookup) != len(rows) or set(lookup) != {q['id'] for q in dev}:
        raise ValueError('Query ID trong ranking khác tập dev.')
    for q in dev:
        row = lookup[q['id']]
        if row.get('source_split') != 'dev' or row.get('source_image_id') != q['image_id']:
            raise ValueError('Metadata query không đúng dev.')
        validate_ranking(row['results'], registry)
        if expected_method == 'examples':
            neighbors = row.get('examples', [])
            if len(neighbors) != (5 if row['crop_ids'] else 0) or len({e['query_id'] for e in neighbors}) != len(neighbors):
                raise ValueError('Example retrieval thiếu top-5 hoặc có hàng xóm trùng.')
            for e in neighbors:
                source = labels.get(e['query_id'])
                if (source is None or e.get('source_split') != 'train' or e.get('source_image_id') != source['image_id']
                        or e.get('gold_citations') != source['relevant_articles']):
                    raise ValueError('Hàng xóm không đúng nhãn train gốc.')
    computed, _ = evaluate_predictions(rows, dev, registry, 'loaded_crop_' + expected_method)
    metrics = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
    if not math.isclose(computed['macro_f2_at_5'], metrics['macro_f2_at_5'], rel_tol=0, abs_tol=1e-12):
        raise ValueError('Metrics không khớp prediction và gold dev.')
    return lookup, metrics, meta, hashes


def save_results(folder, meta, predictions, dev, registry, method, baseline, extra=None, details=None):
    metrics, evaluations = evaluate_predictions(predictions, dev, registry, method)
    if extra:
        metrics.update(extra)
    metrics['returned_citation_count_histogram'] = dict(sorted(Counter(len(p['results']) for p in predictions).items()))
    metrics['queries_with_fewer_than_5_citations'] = [p['query_id'] for p in predictions if len(p['results']) < 5]
    manifest = baseline / 'manifest.json'
    baseline_meta = json.loads(manifest.read_text(encoding='utf-8'))
    required = {str((baseline / n).relative_to(ROOT)) for n in ('predictions.jsonl', 'metrics.json')}
    if (baseline_meta.get('status') != 'COMPLETED' or not baseline_meta.get('code_sha256')
            or not required <= set(baseline_meta.get('output_sha256', {}))):
        raise ValueError('Baseline chưa COMPLETED hoặc thiếu fingerprint/predictions/metrics.')
    # Frozen predictions remain a comparison baseline even when their original source code has since changed.
    for field in ('input_sha256', 'output_sha256'):
        check_hashes(baseline_meta[field])
    hashes = {**baseline_meta['input_sha256'], **baseline_meta['output_sha256'], **hashes_of([manifest])}
    meta['baseline_reference'] = {'folder': str(baseline.relative_to(ROOT)), 'usage': 'saved_predictions_for_comparison_only',
                                  'code_sha256_at_run': baseline_meta['code_sha256']}
    metrics['baseline_current_code_matches_recorded_run'] = all(
        (ROOT / p).resolve().is_relative_to(ROOT) and (ROOT / p).is_file() and file_hash(ROOT / p) == digest
        for p, digest in baseline_meta['code_sha256'].items())
    baseline_rows = read_lines(baseline / 'predictions.jsonl')
    base_metrics, base_eval = evaluate_predictions(baseline_rows, dev, registry, 'baseline')
    stored = json.loads((baseline / 'metrics.json').read_text(encoding='utf-8'))
    if not math.isclose(base_metrics['macro_f2_at_5'], stored['macro_f2_at_5'], rel_tol=0, abs_tol=1e-12):
        raise ValueError('Baseline metrics không khớp prediction/gold hiện tại.')
    by_id = {r['query_id']: r for r in base_eval}
    changes = Counter()
    for row in evaluations:
        if details is not None:
            row.update(details[row['query_id']])
        delta = row['at_k']['5']['f2'] - by_id[row['query_id']]['at_k']['5']['f2']
        row['delta_f2_at_5_vs_full_image'] = delta
        changes['improved' if delta > 1e-12 else 'worsened' if delta < -1e-12 else 'unchanged'] += 1
    metrics.update(baseline_macro_f2_at_5=base_metrics['macro_f2_at_5'],
                   delta_macro_f2_at_5_vs_full_image=metrics['macro_f2_at_5'] - base_metrics['macro_f2_at_5'],
                   query_changes_vs_full_image=dict(changes))
    meta['input_sha256'].update(hashes)
    check_hashes(meta['input_sha256'])
    for row in predictions:
        validate_ranking(row['results'], registry)
    write_lines(folder / 'predictions.jsonl', predictions)
    write_lines(folder / 'per_query_metrics.jsonl', evaluations)
    save(folder / 'metrics.json', metrics)
    meta.update(macro_f2_at_5=metrics['macro_f2_at_5'], output_sha256=hashes_of([
        folder / n for n in ('predictions.jsonl', 'per_query_metrics.jsonl', 'metrics.json')]))
    print(f'[{method}] F2@5={metrics["macro_f2_at_5"]:.6f}; full-image={base_metrics["macro_f2_at_5"]:.6f}', flush=True)
    return metrics


def merge_crop_rankings(rankings):
    candidates = {}
    count = len(rankings)
    for crop in rankings:
        for hit in crop['results']:
            key = (hit['law_id'], hit['article_id'])
            item = candidates.setdefault(key, {'law_id': key[0], 'article_id': key[1], 'score': 0.0,
                                               'crop_evidence': []})
            contribution = 1 / (count * (RRF_K + hit['rank']))
            item['score'] += contribution
            item['crop_evidence'].append({'query_crop_id': crop['crop_id'], **hit,
                                          'rrf_contribution': contribution})
    ranked = sorted(candidates.values(), key=lambda r: (-r['score'], r['law_id'], r['article_id']))
    return [{**r, 'rank': i} for i, r in enumerate(ranked, 1)]


def corpus_ranking(query_crops, qv, corpus, cv, groups):
    rankings = []
    similarity = qv[[r['vector_row'] for r in query_crops]] @ cv.T
    if not np.isfinite(similarity).all():
        raise ValueError('Cosine không hữu hạn.')
    for i, crop in enumerate(query_crops):
        hits = []
        for (law, article), indices in groups.items():
            winner = min(indices, key=lambda j: (-float(similarity[i, corpus[j]['vector_row']]), corpus[j]['crop_id']))
            evidence = corpus[winner]
            hits.append({'law_id': law, 'article_id': article,
                         'score': float(similarity[i, evidence['vector_row']]),
                         'best_corpus_crop_id': evidence['crop_id'], 'best_corpus_crop_path': evidence['path'],
                         'corpus_parent_image_id': evidence['parent_image_id'], 'corpus_xyxy': evidence['xyxy']})
        hits.sort(key=lambda r: (-r['score'], r['law_id'], r['article_id']))
        rankings.append({k: crop[k] for k in ('crop_id', 'parent_image_id', 'path', 'xyxy', 'confidence', 'prompt')}
                        | {'results': [{**r, 'rank': j} for j, r in enumerate(hits[:10], 1)]})
    union = merge_crop_rankings(rankings)
    return {'results': union[:10], 'crop_rankings': rankings, 'candidate_union': union}


def set_similarity(query_crops, example_crops, vectors, trace=False):
    if not query_crops or not example_crops:
        raise ValueError('Đối sánh tập crop cần hai tập không rỗng.')
    values = vectors[[r['vector_row'] for r in query_crops]] @ vectors[[r['vector_row'] for r in example_crops]].T
    if not np.isfinite(values).all():
        raise ValueError('Cosine tập crop không hữu hạn.')
    score = float((values.max(axis=1).mean() + values.max(axis=0).mean()) / 2)
    if not trace:
        return score
    forward, reverse = [], []
    for i, crop in enumerate(query_crops):
        j = int(np.argmax(values[i]))
        forward.append({'query_crop_id': crop['crop_id'], 'example_crop_id': example_crops[j]['crop_id'],
                        'cosine': float(values[i, j])})
    for j, crop in enumerate(example_crops):
        i = int(np.argmax(values[:, j]))
        reverse.append({'query_crop_id': query_crops[i]['crop_id'], 'example_crop_id': crop['crop_id'],
                        'cosine': float(values[i, j])})
    return {'query_to_example': forward, 'example_to_query': reverse, 'symmetric_score': score}


def retrieve(args, folder, meta, progress):
    progress.set('validate crop corpus/query indexes')
    cm, corpus_parents, corpus, _, cv, hashes = load_index('corpus_embedding')
    qm, query_parents, images, queries, qv, query_hashes = load_index('query_embedding_dev')
    if cm['config'] != qm['config'] or cm['detector_config'] != qm['detector_config']:
        raise ValueError('Corpus/query phải dùng cùng detector, encoder và preprocessing.')
    hashes.update(query_hashes)
    dev, labels, registry, source_hashes = evaluation_inputs()
    hashes.update(source_hashes)
    originals = {**labels, **{q['id']: q for q in dev}}
    if len(queries) != len(originals) or {q['query_id'] for q in queries} != set(originals):
        raise ValueError('Query index không phủ đúng train/dev.')
    for q in queries:
        source = originals[q['query_id']]
        if q['source_image_id'] != source['image_id'] or q['source_split'] != source['source_split']:
            raise ValueError('Membership query index khác train/dev gốc.')
    by_parent = {p['image_id']: [] for p in query_parents}
    for crop in images:
        by_parent[crop['parent_image_id']].append(crop)
    for crops in by_parent.values():
        crops.sort(key=lambda r: r['crop_id'])
    groups = {}
    for i, row in enumerate(corpus):
        for c in row['citations']:
            groups.setdefault((c['law_id'], c['article_id']), []).append(i)
    if not groups or not set(groups) <= registry:
        raise ValueError('Crop corpus không có citation hợp lệ.')
    before = {(c['law_id'], c['article_id']) for p in corpus_parents for c in p['citations']}
    bank = [q for q in queries if q['source_split'] == 'train' and q['crop_ids']]
    eligible_labels = {q['query_id']: labels[q['query_id']] for q in bank}
    if args.method == 'examples' and len(bank) < 5:
        raise ValueError('Có ít hơn 5 train example có crop; kiểm tra detection.')
    meta.update(input_sha256=hashes, split='dev', method=args.method, queries=len(dev), config=cm['config'],
                detector_config=cm['detector_config'], example_bank_split='train', example_bank_queries=len(labels),
                eligible_example_bank_queries=len(bank), top_k_examples=5, rrf_k=RRF_K,
                same_image_policy='allow_shared_images', candidate_depth=10,
                aggregation='per_crop_max_citation_then_uniform_rrf' if args.method == 'corpus' else 'symmetric_mean_max_crop_cosine',
                tie_break='citation_lexicographic; crop_id_lexicographic; example_query_id_lexicographic')
    query_by_id = {q['query_id']: q for q in queries}
    cache, predictions = {}, []
    progress.set('retrieve crop ' + args.method, len(dev))
    for original in dev:
        query = query_by_id[original['id']]
        parent = query['image_id']
        crops = by_parent[parent]
        if parent not in cache:
            if not crops:
                result = {'results': [], 'reason': 'no_query_crop'}
                result.update({'crop_rankings': [], 'candidate_union': []} if args.method == 'corpus' else {'examples': []})
            elif args.method == 'corpus':
                result = corpus_ranking(crops, qv, corpus, cv, groups)
            else:
                similarities = {p: set_similarity(crops, by_parent[p], qv) for p in sorted({q['image_id'] for q in bank})}
                hits, neighbors = example_retrieval(np.array([similarities[q['image_id']] for q in bank]),
                                                     bank, eligible_labels, registry)
                bank_by_id = {q['query_id']: q for q in bank}
                for e in neighbors:
                    e['crop_matching'] = set_similarity(crops, by_parent[bank_by_id[e['query_id']]['image_id']], qv, trace=True)
                result = {'results': hits, 'examples': neighbors}
            cache[parent] = result
        predictions.append({**query, **cache[parent], 'source_split': 'dev'})
        progress.done += 1
    golds = [{(str(c['law_id']), str(c['article_id'])) for c in q['relevant_articles']} for q in dev]
    total_gold = sum(map(len, golds))
    coverage = {'gold_citation_occurrences': total_gold,
                'covered_gold_occurrences_before_detection': sum(len(g & before) for g in golds),
                'covered_gold_occurrences_after_detection': sum(len(g & set(groups)) for g in golds),
                'citations_before_detection': len(before), 'citations_after_detection': len(groups)}
    coverage.update(gold_image_coverage_before_detection=coverage['covered_gold_occurrences_before_detection'] / total_gold,
                    gold_image_coverage_after_detection=coverage['covered_gold_occurrences_after_detection'] / total_gold)
    extra = {'coverage': coverage, 'query_crop_count_histogram': dict(sorted(Counter(len(p['crop_ids']) for p in predictions).items())),
             'no_crop_query_ids': [p['query_id'] for p in predictions if not p['crop_ids']],
             'eligible_example_bank_queries': len(bank), 'excluded_train_no_crop_query_ids': sorted(set(labels) - set(eligible_labels)),
             'excluded_train_citations': [{'law_id': l, 'article_id': a} for l, a in sorted(
                 {(str(c['law_id']), str(c['article_id'])) for q in labels.values() for c in q['relevant_articles']} - registry)],
             'dev_queries_with_bank_image': [q['query_id'] for q in predictions if q['image_id'] in {b['image_id'] for b in bank}],
             'query_images_byte_equal_corpus': [q['query_id'] for q in predictions if q['image_id'] in {p['image_id'] for p in corpus_parents}]}
    if args.method == 'corpus':
        extra['candidate_union_recall_macro'] = float(np.mean([
            len(g & {(r['law_id'], r['article_id']) for r in p['candidate_union']}) / len(g)
            for p, g in zip(predictions, golds)]))
        extra['crops_represented_in_query_top5'] = {
            p['query_id']: sum(bool({(r['law_id'], r['article_id']) for r in c['results']}
                                   & {(r['law_id'], r['article_id']) for r in p['results'][:5]})
                              for c in p['crop_rankings']) for p in predictions}
    subgroup = {}
    for group in ('0', '1', 'many'):
        selected = [p for p in predictions if ('many' if len(p['crop_ids']) > 1 else str(len(p['crop_ids']))) == group]
        targets = [q for q in dev if q['id'] in {p['query_id'] for p in selected}]
        subgroup[group] = evaluate_predictions(selected, targets, registry, group)[0] if selected else {'queries': 0}
    extra['by_query_crop_count'] = subgroup
    details = {}
    corpus_hashes = {p['image_id'] for p in corpus_parents}
    for p, gold in zip(predictions, golds):
        row = {'query_crop_ids': p['crop_ids'], 'query_image_in_corpus': p['image_id'] in corpus_hashes,
               'gold_without_image_before_detection': [{'law_id': l, 'article_id': a} for l, a in sorted(gold - before)],
               'gold_without_crop_after_detection': [{'law_id': l, 'article_id': a} for l, a in sorted(gold - set(groups))]}
        if args.method == 'corpus':
            row['candidate_union_recall'] = len(gold & {(r['law_id'], r['article_id']) for r in p['candidate_union']}) / len(gold)
            row['crops_represented_in_query_top5'] = extra['crops_represented_in_query_top5'][p['query_id']]
        details[p['query_id']] = row
    name = 'query_retrieval_corpus' if args.method == 'corpus' else 'query_retrieval_example'
    save_results(folder, meta, predictions, dev, registry, 'crop_' + args.method, FULL / name, extra, details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('dev',), default='dev')
    parser.add_argument('--method', choices=('corpus', 'examples'), default='corpus')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    name = 'query_retrieval_corpus' if args.method == 'corpus' else 'query_retrieval_example'
    run_step(args, name, retrieve, Path(__file__), DEPENDENCIES)


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
