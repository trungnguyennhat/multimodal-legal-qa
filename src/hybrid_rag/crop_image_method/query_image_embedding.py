"""Embed unique sign crops from train/dev images while preserving every query mapping."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))

from hybrid_rag.crop_image_method.corpus_image_cropping import run_step
from hybrid_rag.crop_image_method.corpus_image_embedding import embed_crops, embedding_arguments


def embed_query(args, folder, meta, progress):
    embed_crops(args, folder, meta, progress, 'query_cropping_dev')


def main():
    parser = embedding_arguments()
    parser.description = __doc__
    parser.add_argument('--split', choices=('dev',), default='dev')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size phải >= 1')
    run_step(args, 'query_embedding_dev', embed_query, Path(__file__),
             [Path(__file__).with_name('corpus_image_embedding.py')])


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
