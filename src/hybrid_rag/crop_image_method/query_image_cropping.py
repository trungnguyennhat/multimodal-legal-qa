"""Detect each unique train/dev image once; map all questions to their sign crops."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))

from hybrid_rag.crop_image_method.corpus_image_cropping import (
    crop_arguments, detect_images, run_step, validate_arguments,
)
from hybrid_rag.full_image_method.query_image_embedding import dev_inputs, SPLIT_SOURCE, SPLIT_HELPERS

DEPENDENCIES = [ROOT / 'src/hybrid_rag/full_image_method/query_image_embedding.py', SPLIT_SOURCE, SPLIT_HELPERS]


def query_inventory(progress):
    images, queries, original, hashes = dev_inputs(progress)
    counts = {s: sum(q['source_split'] == s for q in queries) for s in ('train', 'dev')}
    if len(images) != 298 or counts != {'train': 418, 'dev': 112}:
        raise ValueError('Train/dev khác 418/112 hoặc khác 298 ảnh duy nhất đã duyệt; kiểm tra dữ liệu.')
    return ([{k: v for k, v in r.items() if k != 'vector_row'} for r in images],
            [{k: v for k, v in r.items() if k != 'vector_row'} for r in queries], original, hashes)


def crop_query(args, folder, meta, progress):
    images, queries, _, hashes = query_inventory(progress)
    sets = {s: {q['image_id'] for q in queries if q['source_split'] == s} for s in ('train', 'dev')}
    meta.update(split='dev', query_target='dev', example_bank_split='train',
                split_counts={'train': 418, 'dev': 112}, shared_image_hashes_train_dev=len(sets['train'] & sets['dev']))
    detect_images(args, folder, meta, progress, images, queries, hashes)


def main():
    parser = crop_arguments()
    parser.description = __doc__
    parser.add_argument('--split', choices=('dev',), default='dev', help='Prepare train bank and dev targets together.')
    args = parser.parse_args()
    validate_arguments(parser, args)
    run_step(args, 'query_cropping_dev', crop_query, Path(__file__), DEPENDENCIES)


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
