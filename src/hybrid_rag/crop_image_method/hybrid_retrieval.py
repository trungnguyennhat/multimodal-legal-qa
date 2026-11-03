"""Fuse saved crop-image and pure-text rankings with fixed weights; export final dev citations."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))

from hybrid_rag.crop_image_method.corpus_image_cropping import hashes_of, run_step, save
from hybrid_rag.crop_image_method.query_image_retrieval import FULL, evaluation_inputs, load_predictions, save_results
from hybrid_rag.crop_image_method.query_image_hybrid import HYBRID_DEPENDENCIES
from hybrid_rag.full_image_method.query_image_hybrid import load_branch as load_text_branch, rrf

TEXT = ROOT / 'artifacts/experiments/hybrid_rag/text_only'


def fuse(args, folder, meta, progress):
    progress.set('validate saved text and crop-image rankings')
    dev, labels, registry, hashes = evaluation_inputs()
    by_id = {q['id']: q for q in dev}
    dev = [by_id[q['id']] for q in json.loads((ROOT / 'data/processed/splits/dev.json').read_text(encoding='utf-8'))]
    images, im, image_meta, ih = load_predictions('query_retrieval_hybrid', dev, labels, registry)
    text, th, tm = load_text_branch(TEXT / 'query_retrieval_hybrid', {q['id']: q for q in dev}, registry, labels)
    if any(m.get('rrf_k') != args.rrf_k or m.get('within_modality_weights') != {'corpus': 0.5, 'examples': 0.5}
           for m in (image_meta, tm)):
        raise ValueError('Ranking text/ảnh phải cùng k=60 và trọng số corpus/example 0.5/0.5.')
    hashes.update(ih)
    hashes.update(th)
    meta.update(input_sha256=hashes, split='dev', method='hierarchical_weighted_rrf', queries=len(dev),
                example_bank_split='train', example_bank_queries=len(labels), image_method='crop_image_method',
                detector_config=image_meta['detector_config'], config=image_meta['config'],
                text_weight=args.text_weight, image_weight=1 - args.text_weight, rrf_k=args.rrf_k,
                final_top_n=5, weight_selection='fixed_by_cli_no_dev_search')
    predictions, availability = [], Counter()
    progress.set('fuse text and crop-image', len(dev))
    for q in dev:
        t, i = text[q['id']], images[q['id']]
        state = 'both' if t['results'] and i['results'] else 'text_only_fallback' if t['results'] else 'image_only_fallback' if i['results'] else 'none'
        hits = (rrf({'text': t['results'], 'image': i['results']},
                    {'text': args.text_weight, 'image': 1 - args.text_weight}, args.rrf_k)
                if state == 'both' else t['results'] or i['results'])
        availability[state] += 1
        predictions.append({'query_id': q['id'], 'source_image_id': q['image_id'], 'source_split': 'dev',
                            'results': hits, 'availability': state,
                            'modality_rankings': {'text': t['results'], 'image': i['results']}, 'image_evidence': i})
        progress.done += 1
    metrics = save_results(folder, meta, predictions, dev, registry, 'crop_image_text_rrf', FULL / 'image_text_retrieval',
                           {'text_weight': args.text_weight, 'image_weight': 1 - args.text_weight,
                            'rrf_k': args.rrf_k, 'final_top_n': 5, 'availability_queries': dict(availability),
                            'weight_selection': 'fixed_by_cli_no_dev_search',
                            'branch_metrics': {'text_corpus_examples': tm, 'image_corpus_examples': im}})
    metrics['final_citation_metrics'] = metrics['at_k']['5']
    save(folder / 'metrics.json', metrics)
    final = [{'id': q['id'], 'image_id': q['image_id'], 'question': q['question'],
              'relevant_articles': [{'law_id': r['law_id'], 'article_id': r['article_id']} for r in p['results'][:5]]}
             for q, p in zip(dev, predictions)]
    save(folder / 'final_citations.json', final)
    meta['output_sha256'].update(hashes_of([folder / 'metrics.json', folder / 'final_citations.json']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('dev',), default='dev')
    parser.add_argument('--text-weight', type=float, default=0.75)
    parser.add_argument('--rrf-k', type=int, choices=(60,), default=60)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.text_weight) or not 0 <= args.text_weight <= 1:
        parser.error('text-weight phải nằm trong [0,1]')
    run_step(args, 'image_text_retrieval', fuse, Path(__file__),
             [*HYBRID_DEPENDENCIES, Path(__file__).with_name('query_image_hybrid.py')])


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
