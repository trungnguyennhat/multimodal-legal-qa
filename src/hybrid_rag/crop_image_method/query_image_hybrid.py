"""Fuse saved crop-image corpus/example rankings, retaining empty-evidence states."""

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))

from hybrid_rag.crop_image_method.corpus_image_cropping import run_step
from hybrid_rag.crop_image_method.query_image_retrieval import (
    DEPENDENCIES, FULL, evaluation_inputs, load_predictions, save_results,
)
from hybrid_rag.full_image_method.query_image_hybrid import rrf

HYBRID_DEPENDENCIES = [*DEPENDENCIES, Path(__file__).with_name('query_image_retrieval.py'),
                       ROOT / 'src/hybrid_rag/full_image_method/query_image_hybrid.py']


def fuse_images(args, folder, meta, progress):
    progress.set('validate saved crop-image corpus/example rankings')
    dev, labels, registry, hashes = evaluation_inputs()
    corpus, cm, cmeta, ch = load_predictions('query_retrieval_corpus', dev, labels, registry)
    examples, em, emeta, eh = load_predictions('query_retrieval_example', dev, labels, registry)
    if cmeta['config'] != emeta['config'] or cmeta['detector_config'] != emeta['detector_config']:
        raise ValueError('Hai ranking crop không cùng detector/encoder.')
    hashes.update(ch)
    hashes.update(eh)
    meta.update(input_sha256=hashes, split='dev', method='corpus_examples_rrf', queries=len(dev),
                example_bank_split='train', example_bank_queries=len(labels), config=cmeta['config'],
                detector_config=cmeta['detector_config'], rrf_k=args.rrf_k,
                within_modality_weights={'corpus': 0.5, 'examples': 0.5}, candidate_depth=10)
    predictions, availability = [], Counter()
    progress.set('fuse crop-image corpus/examples', len(dev))
    for q in dev:
        c, e = corpus[q['id']], examples[q['id']]
        if c['crop_ids'] != e['crop_ids'] or c['image_id'] != e['image_id']:
            raise ValueError('Hai ranking không tham chiếu cùng ảnh/crop.')
        state = 'both' if c['results'] and e['results'] else 'corpus_only' if c['results'] else 'examples_only' if e['results'] else 'none'
        hits = (rrf({'corpus': c['results'], 'examples': e['results']}, {'corpus': 0.5, 'examples': 0.5}, args.rrf_k)
                if state == 'both' else c['results'] or e['results'])
        availability[state] += 1
        predictions.append({'query_id': q['id'], 'source_image_id': q['image_id'], 'source_split': 'dev',
                            'image_id': c['image_id'], 'crop_ids': c['crop_ids'], 'results': hits,
                            'availability': state, 'example_fallback': state == 'corpus_only',
                            'source_rankings': {'corpus': c, 'examples': e}})
        progress.done += 1
    save_results(folder, meta, predictions, dev, registry, 'crop_corpus_examples', FULL / 'query_retrieval_hybrid',
                 {'rrf_k': args.rrf_k, 'within_modality_weights': {'corpus': 0.5, 'examples': 0.5},
                  'availability_queries': dict(availability), 'branch_metrics': {'image_corpus': cm, 'image_examples': em}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('dev',), default='dev')
    parser.add_argument('--rrf-k', type=int, choices=(60,), default=60)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    run_step(args, 'query_retrieval_hybrid', fuse_images, Path(__file__), HYBRID_DEPENDENCIES)


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
