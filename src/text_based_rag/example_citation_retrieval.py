"""Transfer train citations from similar questions, excluding the query's image group."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from threading import Event, Thread
import time

import numpy as np
from PIL import Image, ImageOps

from multimodal_legal_rag.data import read_json_list
from .corpus_embedding import file_hash
from .image_captioning import ROOT, read_lines, save
from .query_retrieval import BASE, OUTPUT, TRAIN, TOP_K, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--neighbors', type=int, default=5)
    parser.add_argument('--overwrite', action='store_true', help='Replace retrieval output files.')
    args = parser.parse_args()
    if args.neighbors < 1:
        parser.error('neighbors phải >= 1')
    if OUTPUT.exists() and not args.overwrite:
        raise FileExistsError('Output đã tồn tại; dùng --overwrite để thay thế kết quả retrieval.')
    started, stopped = time.perf_counter(), Event()
    phase, done, total = 'validate-input', 0, 0
    manifest = None

    def report():
        elapsed = time.perf_counter() - started
        eta = f'{elapsed * (total - done) / done:.0f}s' if done else 'unknown'
        print(f'[{phase}] {done}/{total} ({100 * done / max(1, total):.1f}%) '
              f'elapsed={elapsed:.0f}s ETA={eta}', flush=True)

    def heartbeat():
        while not stopped.wait(30):
            report()

    worker = Thread(target=heartbeat, daemon=True)
    worker.start()
    try:
        hashes, metadata = {}, {}
        for folder, names in (('query_embedding', ('queries.jsonl', 'vectors.npy')),
                              ('corpus_embedding', ('chunks.jsonl',))):
            source = BASE / folder
            meta = json.loads((source / 'manifest.json').read_text(encoding='utf-8-sig'))
            if meta.get('status') != 'COMPLETED':
                raise ValueError(f'{folder} chưa COMPLETED')
            for name in names:
                digest = file_hash(source / name)
                if digest != meta.get('output_sha256', {}).get(name):
                    raise ValueError(f'Hash không khớp: {folder}/{name}')
                hashes[f'{folder}/{name}'] = digest
            hashes[f'{folder}/manifest.json'] = file_hash(source / 'manifest.json')
            metadata[folder] = meta
        qm = metadata['query_embedding']
        hashes['train.json'] = file_hash(TRAIN)
        if (qm.get('split') != 'train' or qm.get('normalize') != 'L2'
                or qm.get('input_sha256', {}).get('train.json') != hashes['train.json']):
            raise ValueError('Cần vector L2 train và train.json đúng nguồn embedding.')
        queries = read_lines(BASE / 'query_embedding/queries.jsonl')
        vectors = np.load(BASE / 'query_embedding/vectors.npy', allow_pickle=False)
        rows = read_json_list(TRAIN)
        by_id = {r['id']: r for r in rows}
        total = len(queries)
        if (not total or len(by_id) != len(rows) or len(rows) != total
                or {q['query_id'] for q in queries} != set(by_id)
                or any(q['vector_row'] != i for i, q in enumerate(queries))
                or vectors.dtype != np.float32 or vectors.shape != (total, 1024)
                or not np.isfinite(vectors).all()
                or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)):
            raise ValueError('Vector, ID hoặc mapping train không hợp lệ.')
        corpus = {(str(c['document_id']), str(c['id']))
                  for c in read_lines(BASE / 'corpus_embedding/chunks.jsonl')}
        labels = [{(str(c['law_id']), str(c['article_id']))
                   for c in by_id[q['query_id']]['relevant_articles']} for q in queries]
        if any(not label for label in labels):
            raise ValueError('Nhãn train rỗng.')
        missing = set().union(*labels) - corpus
        if missing:
            print(f'[warning] {len(missing)} citation nhãn không có trong index: {sorted(missing)}; '
                  'không chuyển sang prediction, vẫn tính trong mẫu số recall.', flush=True)
        transferable = [label & corpus for label in labels]
        phase = 'hash-decoded-images'
        pixel_hashes, image_hashes = {}, {}
        for q in queries:
            image_id = by_id[q['query_id']]['image_id']
            path = (ROOT / 'data/processed/images/train' / f'{image_id}.jpg').resolve()
            if path.parent != (ROOT / 'data/processed/images/train').resolve():
                raise ValueError('Đường dẫn ảnh không hợp lệ.')
            if image_id not in pixel_hashes:
                image_hashes[image_id] = file_hash(path)
                with Image.open(path) as image:
                    rgb = ImageOps.exif_transpose(image).convert('RGB')
                    # shortcut: exact decoded pixels only; add near-duplicate detection for resized/cropped images.
                    pixel_hashes[image_id] = sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            if q['source_image_id'] != image_id or q['image_id'] != 'sha256:' + image_hashes[image_id]:
                raise ValueError(f'Ảnh khác nguồn embedding: {q["query_id"]}')
            done += 1
        report()
        phase, done = 'example-retrieval-inference-cpu', 0
        similarity = vectors @ vectors.T
        results, per_query = [], []
        for i, query in enumerate(queries):
            image_id = query['source_image_id']
            eligible = [j for j, other in enumerate(queries)
                        if j != i and other['source_image_id'] != image_id
                        and pixel_hashes[other['source_image_id']] != pixel_hashes[image_id]]
            if not eligible:
                raise ValueError(f'Không còn mẫu train hợp lệ: {query["query_id"]}')
            neighbors = sorted(eligible, key=lambda j: (-float(similarity[i, j]), queries[j]['query_id']))[:args.neighbors]
            support = {}
            for j in neighbors:
                for citation in sorted(transferable[j]):
                    support.setdefault(citation, j)
            ranked = sorted(support, key=lambda c: (-float(similarity[i, support[c]]), c))[:TOP_K]
            hits = [{'rank': rank, 'law_id': c[0], 'article_id': c[1],
                     'score': float(similarity[i, support[c]]),
                     'source_query_id': queries[support[c]]['query_id']} for rank, c in enumerate(ranked, 1)]
            assert all(j != i and queries[j]['source_image_id'] != image_id
                       and pixel_hashes[queries[j]['source_image_id']] != pixel_hashes[image_id] for j in neighbors)
            assert len(ranked) == len(set(ranked)) and set(ranked) <= corpus
            results.append({'query_id': query['query_id'], 'query_text': query['text'],
                            'excluded_count': total - len(eligible), 'eligible_count': len(eligible),
                            'neighbors': [{'query_id': queries[j]['query_id'],
                                           'source_image_id': queries[j]['source_image_id'],
                                           'pixel_sha256': pixel_hashes[queries[j]['source_image_id']],
                                           'score': float(similarity[i, j]),
                                           'citations': [{'law_id': c[0], 'article_id': c[1]} for c in sorted(transferable[j])]}
                                          for j in neighbors], 'results': hits})
            relevant, predicted = labels[i], set(ranked)
            tp = len(relevant & predicted)
            per_query.append({'query_id': query['query_id'], 'true_positives': tp,
                              'predicted_count': len(predicted), 'relevant_count': len(relevant),
                              **scores(tp, len(predicted), len(relevant)), 'hit': int(tp > 0),
                              'candidate_recall': len(relevant & support.keys()) / len(relevant),
                              'eligible_label_coverage': len(relevant & set().union(*(transferable[j] for j in eligible))) / len(relevant),
                              'gold_citations': [{'law_id': c[0], 'article_id': c[1]} for c in sorted(relevant)]})
            done += 1
        report()
        metrics = {'split': 'train', 'method': 'example_citation_transfer', 'top_k': TOP_K,
                   'neighbors': args.neighbors, 'queries': total,
                   'macro': {key: sum(r[key] for r in per_query) / total for key in ('precision', 'recall', 'f1', 'f2')},
                   'micro': scores(sum(r['true_positives'] for r in per_query),
                                   sum(r['predicted_count'] for r in per_query), sum(r['relevant_count'] for r in per_query)),
                   'hit_rate_at_5': sum(r['hit'] for r in per_query) / total,
                   'candidate_recall': sum(r['candidate_recall'] for r in per_query) / total,
                   'eligible_label_coverage': sum(r['eligible_label_coverage'] for r in per_query) / total,
                   'mean_predicted_count': sum(r['predicted_count'] for r in per_query) / total,
                   'gold_citations_missing_from_index': [{'law_id': c[0], 'article_id': c[1]} for c in sorted(missing)]}
        # Validate everything before replacing the authorized retrieval output files.
        OUTPUT.mkdir(parents=True, exist_ok=True)
        manifest = {'status': 'IN_PROGRESS', 'created_at': datetime.now(timezone.utc).isoformat(),
                    'method': 'example_citation_transfer', 'split': 'train', 'queries': total,
                    'neighbors': args.neighbors, 'top_k': TOP_K, 'aggregation': 'max_neighbor_cosine',
                    'exclusion': ['self', 'same_source_image_id', 'same_exif_transposed_rgb_pixels'],
                    'tie_break': 'query_id_then_citation', 'input_sha256': hashes,
                    'source_sha256': file_hash(ROOT / 'src/text_based_rag/example_citation_retrieval.py'),
                    'image_file_sha256': image_hashes, 'image_pixel_sha256': pixel_hashes,
                    'encoder': qm, 'libraries': {name: version(name) for name in ('numpy', 'Pillow')},
                    'resources': {'device': 'cpu', 'similarity_matrix_bytes': similarity.nbytes,
                                  'vector_bytes': vectors.nbytes}, 'overwrite': args.overwrite}
        save(OUTPUT / 'manifest.json', manifest)
        for name, records in (('retrieval_results.jsonl', results), ('per_query_metrics.jsonl', per_query)):
            with (OUTPUT / name).open('w', encoding='utf-8') as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        save(OUTPUT / 'metrics.json', metrics)
        with (OUTPUT / 'retrieval_preview.txt').open('w', encoding='utf-8') as stream:
            for result, metric in zip(results, per_query):
                stream.write(f'===== {result["query_id"]} =====\n{result["query_text"]}\n')
                stream.write(f'F2={metric["f2"]:.6f}; Gold: {metric["gold_citations"]}\n')
                for hit in result['results']:
                    stream.write(f'#{hit["rank"]} | {hit["law_id"]} | {hit["article_id"]} | '
                                 f'cosine={hit["score"]:.6f} | source={hit["source_query_id"]}\n')
                stream.write('\n')
        manifest.update(status='COMPLETED', elapsed_seconds=time.perf_counter() - started,
                        output_sha256={name: file_hash(OUTPUT / name) for name in
                                       ('retrieval_results.jsonl', 'per_query_metrics.jsonl', 'metrics.json', 'retrieval_preview.txt')})
        save(OUTPUT / 'manifest.json', manifest)
        print(f'[metrics] {json.dumps(metrics, ensure_ascii=False)}', flush=True)
    except BaseException as exc:
        if manifest is not None:
            manifest.update(status='FAILED', error=str(exc))
            save(OUTPUT / 'manifest.json', manifest)
        raise
    finally:
        stopped.set()
        worker.join()


if __name__ == '__main__':
    main()
