"""Embed corpus sign crops with the existing frozen DINOv2; deduplicate equal RGB pixels."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np

from hybrid_rag.crop_image_method.corpus_image_cropping import (
    BASE, MODEL_DIR, load_cropping, load_step, pixel_hash, read_lines, rgb_image, run_step, save,
    hashes_of, verified_model, write_lines,
)
from hybrid_rag.full_image_method.corpus_image_embedding import CONFIG, embed_images, validate_vectors


def vector_mapping(crops, queries):
    unique = {}
    for crop in crops:
        unique.setdefault(crop['pixel_sha256'], crop)
    representatives = sorted(unique.values(), key=lambda r: r['crop_id'])
    rows = {r['pixel_sha256']: i for i, r in enumerate(representatives)}
    mapping = [{**r, 'vector_row': rows[r['pixel_sha256']]} for r in crops]
    by_id = {r['crop_id']: r['vector_row'] for r in mapping}
    query_mapping = None if queries is None else [
        {**q, 'vector_rows': [by_id[cid] for cid in q['crop_ids']]} for q in queries]
    return representatives, mapping, query_mapping


def embed_crops(args, folder, meta, progress, source):
    progress.set('validate crop artifacts and content')
    cropping, parents, crops, queries, hashes = load_cropping(source)
    if source == 'corpus_cropping' and not crops:
        raise ValueError('Corpus không có crop; kiểm tra YOLOE trước khi embedding.')
    progress.set('verify crop RGB hashes', len(crops))
    for row in crops:
        image = rgb_image(ROOT / row['path'])
        if image.size != (row['width'], row['height']) or pixel_hash(image) != row['pixel_sha256']:
            raise ValueError(f'RGB crop khác metadata: {row["crop_id"]}')
        progress.done += 1
    representatives, mapping, query_mapping = vector_mapping(crops, queries)
    meta.update(cropping_step=source, detector_config=cropping['detector_config'],
                detector_config_sha256=cropping['detector_config_sha256'],
                inventory={'parent_images': len(parents), 'crop_instances': len(crops),
                           'unique_vectors': len(representatives)}, index_schema='crop_instances_with_shared_vector_rows')
    if queries is not None:
        meta.update(split='dev', example_bank_split='train', split_counts=cropping['split_counts'])
    if representatives:
        embed_images(args, folder, meta, progress, representatives, None, hashes)
    else:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError('Cần CUDA trong .venv; không tự chuyển CPU.')
        model = verified_model()
        processor = json.loads((MODEL_DIR / 'preprocessor_config.json').read_text(encoding='utf-8'))
        hashes.update(model['output_sha256'])
        hashes.update(hashes_of([MODEL_DIR / 'manifest.json']))
        meta.update(input_sha256=hashes, config={**CONFIG, 'revision': model['revision'],
                    'image_mean': processor['image_mean'], 'image_std': processor['image_std']},
                    resources={'gpu': torch.cuda.get_device_name(0), 'peak_cuda_allocated_bytes': 0,
                               'note': 'No query crops; model inference skipped.'})
        with (folder / 'vectors.npy.tmp').open('wb') as stream:
            np.save(stream, np.empty((0, 768), dtype=np.float32), allow_pickle=False)
        (folder / 'vectors.npy.tmp').replace(folder / 'vectors.npy')
    write_lines(folder / 'images.jsonl', mapping)
    names = ['images.jsonl', 'vectors.npy']
    if query_mapping is not None:
        write_lines(folder / 'queries.jsonl', query_mapping)
        names.append('queries.jsonl')
    vectors = np.load(folder / 'vectors.npy', allow_pickle=False, mmap_mode='r')
    validate_vectors(vectors, len(representatives))
    meta['output_sha256'] = hashes_of([folder / n for n in names])
    print(f'[crop embedding] {len(crops)} crop instances; {len(representatives)} vectors', flush=True)


def load_index(name):
    query_index = name == 'query_embedding_dev'
    names = ['images.jsonl', 'vectors.npy'] + (['queries.jsonl'] if query_index else [])
    meta, folder, hashes = load_step(name, names)
    source = 'query_cropping_dev' if query_index else 'corpus_cropping'
    cropping, parents, crops, queries, crop_hashes = load_cropping(source)
    representatives, expected, expected_queries = vector_mapping(crops, queries)
    mapping = read_lines(folder / 'images.jsonl')
    saved_queries = read_lines(folder / 'queries.jsonl') if query_index else None
    if (mapping != expected or saved_queries != expected_queries or meta.get('cropping_step') != source
            or meta.get('detector_config') != cropping['detector_config']
            or meta.get('detector_config_sha256') != cropping['detector_config_sha256']
            or meta.get('index_schema') != 'crop_instances_with_shared_vector_rows'):
        raise ValueError(f'Index/crop/query mapping không khớp: {name}')
    model = verified_model()
    processor = json.loads((MODEL_DIR / 'preprocessor_config.json').read_text(encoding='utf-8'))
    expected_config = {**CONFIG, 'revision': model['revision'], 'image_mean': processor['image_mean'],
                       'image_std': processor['image_std']}
    if meta.get('config') != expected_config:
        raise ValueError('Index không cùng cấu hình DINOv2 đã duyệt.')
    vectors = np.load(folder / 'vectors.npy', allow_pickle=False, mmap_mode='r')
    validate_vectors(vectors, len(representatives))
    if not query_index and not mapping:
        raise ValueError('Corpus index không có crop.')
    hashes.update(crop_hashes)
    return meta, parents, mapping, saved_queries, vectors, hashes


def embed_corpus(args, folder, meta, progress):
    embed_crops(args, folder, meta, progress, 'corpus_cropping')


def embedding_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--overwrite', action='store_true')
    return parser


def main():
    parser = embedding_arguments()
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size phải >= 1')
    run_step(args, 'corpus_embedding', embed_corpus, Path(__file__))


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
