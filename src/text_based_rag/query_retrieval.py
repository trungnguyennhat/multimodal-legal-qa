"""Retrieve citations, fuse results, and export private-test Subtask 1 predictions."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from threading import Event, Thread
import time

import numpy as np

from multimodal_legal_rag.data import DATA_FILES, DEFAULT_SOURCE, PRIVATE_TASK1_FILE, read_json_list
from .corpus_embedding import file_hash
from .corpus_image_captioning import ROOT, read_lines, save

BASE = ROOT / "artifacts/experiments/text_based_rag/query"
CORPUS_BASE = ROOT / "artifacts/experiments/text_based_rag/corpus"
OUTPUT = BASE / "query_retrieval_corpus"
DEV = ROOT / "data/processed/splits/dev.json"
TOP_K = 5
PRIVATE = ROOT / 'artifacts/private_test/text_based_rag/retrieval'


def scores(tp, predicted, relevant):
    precision = tp / predicted if predicted else 0.0
    recall = tp / relevant if relevant else 0.0
    return {"precision": precision, "recall": recall,
            "f1": 2 * tp / (predicted + relevant) if predicted + relevant else 0.0,
            "f2": 5 * tp / (predicted + 4 * relevant) if predicted + relevant else 0.0}


def held_out_retrieval_inputs(hashes, query_split):
    """Build held-out branches with the full original train set as the example bank."""
    test_root = PRIVATE if query_split == 'private_test' else ROOT / 'artifacts/public-test/text_based_rag'

    def load(folder, mapping_name, id_field, prefix):
        meta = json.loads((folder / 'manifest.json').read_text(encoding='utf-8-sig'))
        if meta.get('status') != 'COMPLETED':
            raise ValueError(f'{folder} chưa COMPLETED.')
        hashes[f'{prefix}/manifest.json'] = file_hash(folder / 'manifest.json')
        for name in (mapping_name, 'vectors.npy'):
            digest = file_hash(folder / name)
            if meta.get('output_sha256', {}).get(name) != digest:
                raise ValueError(f'File không khớp manifest: {folder}/{name}')
            hashes[f'{prefix}/{name}'] = digest
        mapping = read_lines(folder / mapping_name)
        vectors = np.load(folder / 'vectors.npy', mmap_mode='r', allow_pickle=False)
        if (not mapping or vectors.dtype != np.float32 or vectors.shape != (len(mapping), 1024)
                or meta.get('queries' if id_field == 'query_id' else 'chunks') != len(mapping)
                or not np.isfinite(vectors).all()
                or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
                or len({r[id_field] for r in mapping}) != len(mapping)
                or any(r['vector_row'] != i for i, r in enumerate(mapping))):
            raise ValueError(f'Vector/mapping không hợp lệ: {folder}')
        return mapping, vectors, meta

    queries, qv, qm = load(test_root / 'query_embedding', 'queries.jsonl', 'query_id', f'{query_split}_query_embedding')
    chunks, cv, cm = load(CORPUS_BASE / 'corpus_embedding', 'chunks.jsonl', 'chunk_id', 'corpus_embedding')
    encoder_keys = ('model', 'revision', 'model_files_sha256', 'pooling', 'normalize', 'dimension',
                    'stored_dtype', 'inference_dtype', 'input_prefix')
    if any(key not in qm or qm[key] != cm.get(key) for key in encoder_keys):
        raise ValueError(f'Encoder {query_split} query/corpus không khớp.')
    test_source = PRIVATE_TASK1_FILE if query_split == 'private_test' else DEFAULT_SOURCE / DATA_FILES['public_test']
    hashes[f'{query_split}.json'] = file_hash(test_source)
    if (qm.get('split') != query_split or qm.get('representation') != 'question_relevant_items_v1'
            or qm.get('normalize') != 'L2'
            or qm.get('input_sha256', {}).get(f'{query_split}.json') != hashes[f'{query_split}.json']):
        raise ValueError(f'{query_split} embedding không đúng split/representation/nguồn.')
    for name, path in (('caption_manifest.json', test_root / 'query_image_captioning/manifest.json'),
                       ('query_image_mapping.jsonl', test_root / 'query_image_captioning/query_image_mapping.jsonl'),
                       ('corpus_manifest.json', CORPUS_BASE / 'corpus_embedding/manifest.json')):
        if qm.get('input_sha256', {}).get(name) != file_hash(path):
            raise ValueError(f'Nguồn {query_split} embedding đã thay đổi: {name}')
    if cm.get('input_sha256', {}).get('chunks.jsonl') != file_hash(CORPUS_BASE / 'corpus_chunking/chunks.jsonl'):
        raise ValueError('Corpus index không khớp chunking hiện tại.')
    bank, bank_vectors, bank_metas, split_counts = [], [], {}, {}
    for split in ('train', 'dev'):
        folder_name = f'query_embedding_{split}'
        rows, vectors, meta = load(BASE / folder_name, 'queries.jsonl', 'query_id', folder_name)
        keys = (*encoder_keys, 'representation', 'relevance_filter', 'include_summary',
                'deduplication', 'truncated_policy', 'empty_selection')
        if meta.get('split') != split or any(key not in qm or meta.get(key) != qm[key] for key in keys):
            raise ValueError(f'Encoder/representation {split} không khớp {query_split}.')
        hashes[f'{split}.json'] = file_hash(DEV.with_name(f'{split}.json'))
        if meta.get('input_sha256', {}).get(f'{split}.json') != hashes[f'{split}.json']:
            raise ValueError(f'Split {split} đã thay đổi sau embedding.')
        bank.extend(rows)
        bank_vectors.append(vectors)
        bank_metas[split], split_counts[split] = meta, len(rows)
    if (len({q['query_id'] for q in bank}) != len(bank) or len(bank) < TOP_K
            or {q['query_id'] for q in bank} & {q['query_id'] for q in queries}):
        raise ValueError('Kho example bị trùng ID/không đủ examples hoặc lẫn query test.')
    ev = np.concatenate(bank_vectors, axis=0)
    example_similarity, corpus_similarity = qv @ ev.T, qv @ cv.T
    if not np.isfinite(example_similarity).all() or not np.isfinite(corpus_similarity).all():
        raise ValueError('Cosine không hợp lệ.')
    neighbors = [sorted(range(len(bank)), key=lambda i: (-float(values[i]), bank[i]['query_id']))[:TOP_K]
                 for values in example_similarity]
    # Only train/dev labels are attached after ranking. Public gold is not read here.
    bank_labels = {}
    for split in ('train', 'dev'):
        rows = read_json_list(DEV.with_name(f'{split}.json'))
        mapping = {r['id']: r for r in rows}
        expected = {q['query_id']: q for q in bank if q['query_id'] in mapping}
        if (len(mapping) != len(rows) or len(mapping) != split_counts[split]
                or set(expected) != set(mapping)
                or any(mapping[qid]['image_id'] != q['source_image_id'] for qid, q in expected.items())):
            raise ValueError(f'Nhãn {split} không khớp example embedding.')
        bank_labels.update(mapping)
    original_path = DEFAULT_SOURCE / DATA_FILES['train']
    hashes['original_train.json'] = file_hash(original_path)
    original_rows = read_json_list(original_path)
    original = {r['id']: r for r in original_rows}
    if len(original) != len(original_rows) or set(original) != set(bank_labels):
        raise ValueError('Train + dev không phủ đúng toàn bộ train gốc.')
    for qid, row in bank_labels.items():
        if (row['image_id'] != original[qid]['image_id'] or row['question'] != original[qid]['question']
                or row['relevant_articles'] != original[qid]['relevant_articles'] or not row['relevant_articles']):
            raise ValueError(f'Example khác train gốc hoặc thiếu gold citation: {qid}')
    groups = {}
    for i, chunk in enumerate(chunks):
        groups.setdefault((str(chunk['document_id']), str(chunk['id'])), []).append(i)
    if len(groups) < TOP_K:
        raise ValueError('Corpus không đủ 5 citation.')
    valid = set(groups)
    excluded = {(str(r['law_id']), str(r['article_id'])) for row in bank_labels.values()
                for r in row['relevant_articles']} - valid
    corpus, examples, article_scores, best_chunks = {}, {}, {}, {}
    for row, query in enumerate(queries):
        qid = query['query_id']
        values, winners = {}, {}
        for citation, indices in groups.items():
            index = indices[int(np.argmax(corpus_similarity[row, indices]))]
            values[citation], winners[citation] = float(corpus_similarity[row, index]), chunks[index]['chunk_id']
        article_scores[qid], best_chunks[qid] = values, winners
        top_corpus = sorted(values, key=lambda c: (-values[c], c))[:TOP_K]
        corpus[qid] = {'query_id': qid, 'query_text': query['text'], 'results': [
            {'law_id': c[0], 'article_id': c[1], 'score': values[c], 'best_chunk_id': winners[c], 'rank': rank}
            for rank, c in enumerate(top_corpus, 1)]}
        selected, citations = [], {}
        for rank, index in enumerate(neighbors[row], 1):
            example = bank[index]
            score = float(example_similarity[row, index])
            labels = bank_labels[example['query_id']]['relevant_articles']
            selected.append({'rank': rank, 'query_id': example['query_id'], 'query_text': example['text'],
                             'score': score, 'gold_citations': labels})
            for citation in {(str(r['law_id']), str(r['article_id'])) for r in labels} & valid:
                hit = citations.setdefault(citation, {'law_id': citation[0], 'article_id': citation[1],
                                                      'score': score, 'source_query_ids': []})
                hit['score'] = max(hit['score'], score)
                hit['source_query_ids'].append(example['query_id'])
        hits = sorted(citations.values(), key=lambda h: (-h['score'], h['law_id'], h['article_id']))
        for rank, hit in enumerate(hits, 1):
            hit['rank'] = rank
        examples[qid] = {'query_id': qid, 'query_text': query['text'], 'examples': selected, 'results': hits}
    metadata = {'example_bank_splits': ['train', 'dev'], 'example_bank_queries': len(bank),
                'example_bank_split_counts': split_counts, 'query_encoder': qm, 'corpus_encoder': cm,
                'example_encoders': bank_metas, 'same_image_policy': 'allow_shared_images_as_requested',
                'excluded_example_citations': [{'law_id': law, 'article_id': article} for law, article in sorted(excluded)]}
    print(f'[{query_split}-retrieval-input] queries={len(queries)}; examples={len(bank)}; '
          f'splits={split_counts}; excluded_citation_ids={len(excluded)}', flush=True)
    return corpus, examples, article_scores, best_chunks, valid, qv.nbytes + cv.nbytes + ev.nbytes, metadata


def private_branch_retrieval(args):
    branch = 'example' if args.method == 'examples' else 'corpus'
    output = PRIVATE / f'query_retrieval_{branch}'
    names = ('manifest.json', 'retrieval_results.jsonl', 'retrieval_preview.txt')
    if any((output / name).exists() for name in names) and not args.overwrite:
        raise FileExistsError(f'{output} đã tồn tại; dùng --overwrite.')
    started, stopped = time.perf_counter(), Event()
    done, total, manifest = 0, 0, None

    def report():
        elapsed = time.perf_counter() - started
        eta = f'{elapsed * (total - done) / done:.0f}s' if done else 'unknown'
        print(f'[private-{branch}-retrieval-inference-cpu] {done}/{total} '
              f'({100 * done / max(1, total):.1f}%) elapsed={elapsed:.0f}s ETA={eta}', flush=True)

    def heartbeat():
        while not stopped.wait(30):
            report()

    worker = Thread(target=heartbeat, daemon=True)
    worker.start()
    report()
    try:
        hashes = {}
        corpus, examples, _, _, _, vector_bytes, metadata = held_out_retrieval_inputs(hashes, 'private_test')
        records = list((examples if branch == 'example' else corpus).values())
        total = len(records)
        manifest = {'status': 'IN_PROGRESS', 'created_at': datetime.now(timezone.utc).isoformat(),
                    'method': 'example_retrieval' if branch == 'example' else 'dense_corpus_retrieval',
                    'split': 'private_test', 'queries': total, 'input_sha256': hashes,
                    'source_sha256': file_hash(ROOT / 'src/text_based_rag/query_retrieval.py'),
                    'similarity': 'cosine', ('top_k_examples' if branch == 'example' else 'top_k'): TOP_K,
                    'evaluation_available': False, 'overwrite': args.overwrite, **metadata,
                    'resources': {'device': 'cpu', 'vector_bytes_loaded': vector_bytes}}
        output.mkdir(parents=True, exist_ok=True)
        save(output / 'manifest.json', manifest)
        with (output / 'retrieval_results.jsonl').open('w' if args.overwrite else 'x', encoding='utf-8') as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        with (output / 'retrieval_preview.txt').open('w' if args.overwrite else 'x', encoding='utf-8') as stream:
            for record in records:
                stream.write(f"===== {record['query_id']} =====\n{record['query_text']}\n")
                for example in record.get('examples', []):
                    stream.write(f"Example #{example['rank']} | {example['query_id']} | cosine={example['score']:.6f}\n")
                for hit in record['results']:
                    stream.write(f"#{hit['rank']} | {hit['law_id']} | {hit['article_id']} | cosine={hit['score']:.6f}\n")
                stream.write('\n')
                done += 1
        manifest.update(status='COMPLETED', elapsed_seconds=time.perf_counter() - started,
                        output_sha256={name: file_hash(output / name) for name in names if name != 'manifest.json'})
        save(output / 'manifest.json', manifest)
        report()
        print(f'[output] {output}; no private gold evaluation', flush=True)
    except BaseException as exc:
        if manifest is not None:
            manifest.update(status='FAILED', error=str(exc))
            save(output / 'manifest.json', manifest)
        raise
    finally:
        stopped.set()
        worker.join()


def hybrid_retrieval(args):
    public = args.split == 'public_test'
    private = args.split == 'private_test'
    output = (PRIVATE if private else ROOT / 'artifacts/public-test/text_based_rag' if public else BASE) / 'query_retrieval_hybrid'
    labels_path = DEFAULT_SOURCE / DATA_FILES['public_test'] if public else DEV
    names = ('manifest.json', 'retrieval_results.jsonl', 'per_query_metrics.jsonl',
             'metrics.json', 'retrieval_preview.txt')
    if private:
        names = ('manifest.json', 'retrieval_results.jsonl', 'retrieval_preview.txt')
        if (PRIVATE / 'submission_task1.json').exists() and not args.overwrite:
            raise FileExistsError('Private submission_task1.json đã tồn tại; dùng --overwrite.')
    if any((output / name).exists() for name in names) and not args.overwrite:
        raise FileExistsError(f'{output} đã tồn tại; dùng --overwrite để thay thế.')
    started, stopped = time.perf_counter(), Event()
    phase, done, total, manifest = 'validate-hybrid-input', 0, 0, None

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
    report()
    try:
        hashes = {}
        gated = args.hybrid_strategy == 'gated'
        if public or private:
            phase = f'{args.split}-corpus-and-example-retrieval-inference-cpu'
            corpus, examples, article_scores, best_chunks, valid, vector_bytes, source_metadata = held_out_retrieval_inputs(hashes, args.split)
            if private:
                required_hashes = dict(hashes)
                expected_texts = {qid: r['query_text'] for qid, r in corpus.items()}
                for branch, method in (('corpus', 'dense_corpus_retrieval'), ('example', 'example_retrieval')):
                    folder = PRIVATE / f'query_retrieval_{branch}'
                    meta = json.loads((folder / 'manifest.json').read_text(encoding='utf-8-sig'))
                    digest = file_hash(folder / 'retrieval_results.jsonl')
                    if (meta.get('status') != 'COMPLETED' or meta.get('split') != 'private_test'
                            or meta.get('method') != method or meta.get('example_bank_queries') != source_metadata['example_bank_queries']
                            or meta.get('output_sha256', {}).get('retrieval_results.jsonl') != digest
                            or any(meta.get('input_sha256', {}).get(name) != value for name, value in required_hashes.items())):
                        raise ValueError(f'Nhánh private {branch} không khớp nguồn hiện tại; chạy lại nhánh đó.')
                    saved = read_lines(folder / 'retrieval_results.jsonl')
                    mapping = {r['query_id']: r for r in saved}
                    if (len(mapping) != len(saved) or set(mapping) != set(expected_texts) or meta.get('queries') != len(saved)
                            or any(mapping[qid]['query_text'] != text for qid, text in expected_texts.items())):
                        raise ValueError(f'Mapping private {branch} không khớp query.')
                    hashes[f'query_retrieval_{branch}/manifest.json'] = file_hash(folder / 'manifest.json')
                    hashes[f'query_retrieval_{branch}/retrieval_results.jsonl'] = digest
                    if branch == 'corpus':
                        corpus = mapping
                    else:
                        examples = mapping
        else:
            branches, metas, hashes = {}, {}, {}
            for branch, method in (('corpus', 'dense_corpus_retrieval'), ('example', 'example_retrieval')):
                folder = BASE / f'query_retrieval_{branch}'
                meta = json.loads((folder / 'manifest.json').read_text(encoding='utf-8-sig'))
                digest = file_hash(folder / 'retrieval_results.jsonl')
                if (meta.get('status') != 'COMPLETED' or meta.get('split') != 'dev'
                        or meta.get('method') != method
                        or meta.get('output_sha256', {}).get('retrieval_results.jsonl') != digest):
                    raise ValueError(f'Kết quả {branch} chưa hoàn tất hoặc không khớp manifest.')
                rows = read_lines(folder / 'retrieval_results.jsonl')
                mapping = {r['query_id']: r for r in rows}
                if not rows or len(mapping) != len(rows) or meta.get('queries') != len(rows):
                    raise ValueError(f'Mapping query không hợp lệ: {branch}')
                branches[branch], metas[branch] = mapping, meta
                hashes[f'query_retrieval_{branch}/manifest.json'] = file_hash(folder / 'manifest.json')
                hashes[f'query_retrieval_{branch}/retrieval_results.jsonl'] = digest
            corpus, examples = branches['corpus'], branches['example']
            if set(corpus) != set(examples) or metas['corpus']['query_encoder'] != metas['example']['query_encoder']:
                raise ValueError('Hai nhánh không cùng query/encoder; chạy lại nguồn tương ứng.')
            for name in ('dev.json', 'query_embedding_dev/queries.jsonl', 'query_embedding_dev/vectors.npy',
                         'corpus_embedding/chunks.jsonl'):
                # Preserve manifest artifact keys while resolving the grouped filesystem layout.
                path = DEV if name == 'dev.json' else (CORPUS_BASE if name.startswith('corpus_') else BASE) / name
                digest = file_hash(path)
                if any(meta.get('input_sha256', {}).get(name) != digest for meta in metas.values()):
                    raise ValueError(f'Hai nhánh không cùng nguồn hiện tại: {name}')
                hashes[name] = digest
            registry = CORPUS_BASE / 'corpus_embedding'
            registry_meta = json.loads((registry / 'manifest.json').read_text(encoding='utf-8-sig'))
            if (registry_meta.get('status') != 'COMPLETED' or
                    registry_meta.get('output_sha256', {}).get('chunks.jsonl') != hashes['corpus_embedding/chunks.jsonl']):
                raise ValueError('Danh mục corpus không khớp index đã hoàn tất.')
            hashes['corpus_embedding/manifest.json'] = file_hash(registry / 'manifest.json')
            chunks = read_lines(registry / 'chunks.jsonl')
            valid = {(str(c['document_id']), str(c['id'])) for c in chunks}
            gated = args.hybrid_strategy == 'gated'
            vector_bytes, article_scores, best_chunks = 0, {}, {}
            if gated:
                phase = 'hybrid-candidate-corpus-scoring-cpu'
                hashes['corpus_embedding/vectors.npy'] = file_hash(registry / 'vectors.npy')
                if (registry_meta.get('output_sha256', {}).get('vectors.npy') != hashes['corpus_embedding/vectors.npy']
                        or metas['corpus'].get('input_sha256', {}).get('corpus_embedding/vectors.npy') != hashes['corpus_embedding/vectors.npy']):
                    raise ValueError('Vector corpus không khớp index/kết quả nguồn.')
                queries = read_lines(BASE / 'query_embedding_dev/queries.jsonl')
                qv = np.load(BASE / 'query_embedding_dev/vectors.npy', mmap_mode='r', allow_pickle=False)
                cv = np.load(registry / 'vectors.npy', mmap_mode='r', allow_pickle=False)
                for mapping, vectors in ((queries, qv), (chunks, cv)):
                    if (vectors.dtype != np.float32 or vectors.shape != (len(mapping), 1024)
                            or not np.isfinite(vectors).all()
                            or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
                            or any(row['vector_row'] != i for i, row in enumerate(mapping))):
                        raise ValueError('Vector/mapping scoring corpus không hợp lệ.')
                if len({q['query_id'] for q in queries}) != len(queries) or {q['query_id'] for q in queries} != set(corpus):
                    raise ValueError('Mapping embedding không khớp query retrieval.')
                vector_bytes = qv.nbytes + cv.nbytes
                similarity = qv @ cv.T
                if not np.isfinite(similarity).all():
                    raise ValueError('Cosine corpus không hợp lệ.')
                groups = {}
                for i, chunk in enumerate(chunks):
                    groups.setdefault((str(chunk['document_id']), str(chunk['id'])), []).append(i)
                for row, query in enumerate(queries):
                    if query['text'] != corpus[query['query_id']]['query_text']:
                        raise ValueError(f"Text embedding không khớp: {query['query_id']}")
                    values, winners = {}, {}
                    for citation, indices in groups.items():
                        index = indices[int(np.argmax(similarity[row, indices]))]
                        values[citation], winners[citation] = float(similarity[row, index]), chunks[index]['chunk_id']
                    article_scores[query['query_id']], best_chunks[query['query_id']] = values, winners
            source_metadata = {'same_image_policy': metas['example'].get('same_image_policy')}
        results, total = [], len(corpus)
        gate_counts = {'high': 0, 'middle': 0, 'low': 0}
        phase = 'hybrid-fusion-inference-cpu'
        for query_id, query in corpus.items():
            if query['query_text'] != examples[query_id]['query_text']:
                raise ValueError(f'Query text khác giữa hai nhánh: {query_id}')
            candidates, gate, smax = {}, 'fixed', None
            weight = args.example_weight
            corpus_rows, example_rows = query['results'], examples[query_id]['results']
            if gated:
                neighbors = sorted(examples[query_id]['examples'], key=lambda e: e['rank'])
                if (len(neighbors) != TOP_K or [e['rank'] for e in neighbors] != list(range(1, TOP_K + 1))
                        or not np.isfinite([e['score'] for e in neighbors]).all()
                        or any(neighbors[i]['score'] < neighbors[i + 1]['score'] for i in range(TOP_K - 1))):
                    raise ValueError(f'Top examples không hợp lệ: {query_id}')
                smax = max(float(e['score']) for e in neighbors)
                gate = 'high' if smax >= 0.82 else 'low' if smax < 0.70 else 'middle'
                gate_counts[gate] += 1
                weight = {'high': 0.9, 'middle': args.example_weight, 'low': 0.3}[gate]
                if gate == 'high':
                    allowed = {(str(r['law_id']), str(r['article_id']))
                               for e in neighbors[:3] for r in e['gold_citations']} & valid
                    example_rows = [r for r in example_rows if (str(r['law_id']), str(r['article_id'])) in allowed]
                    if {(str(r['law_id']), str(r['article_id'])) for r in example_rows} != allowed:
                        raise ValueError(f'Top-3 citations không khớp example results: {query_id}')
                else:
                    allowed = {(str(r['law_id']), str(r['article_id'])) for r in corpus_rows + example_rows}
                corpus_rows = [{'law_id': law, 'article_id': article,
                                'score': article_scores[query_id][(law, article)],
                                'best_chunk_id': best_chunks[query_id][(law, article)]}
                               for law, article in sorted(allowed)]
            for branch, rows in (('corpus', corpus_rows), ('example', example_rows)):
                values = [float(r['score']) for r in rows]
                if not np.isfinite(values).all():
                    raise ValueError(f'Điểm không hợp lệ: {branch}/{query_id}')
                lo, hi = (min(values), max(values)) if values else (0.0, 0.0)
                seen = set()
                for row, score in zip(rows, values):
                    citation = (str(row['law_id']), str(row['article_id']))
                    if citation not in valid or citation in seen:
                        raise ValueError(f'Citation không hợp lệ hoặc trùng: {branch}/{query_id}/{citation}')
                    seen.add(citation)
                    hit = candidates.setdefault(citation, {
                        'law_id': citation[0], 'article_id': citation[1],
                        'corpus_score': None, 'example_score': None,
                        'normalized_corpus_score': 0.0, 'normalized_example_score': 0.0,
                        'sources': [], 'source_query_ids': []})
                    hit[f'{branch}_score'] = score
                    hit[f'normalized_{branch}_score'] = ((score + 1) / 2 if gated else
                                                       (score - lo) / (hi - lo) if hi > lo else 1.0)
                    hit['sources'].append(branch)
                    if branch == 'example':
                        hit['source_query_ids'] = ([e['query_id'] for e in neighbors[:3]
                                                   if citation in {(str(r['law_id']), str(r['article_id']))
                                                                   for r in e['gold_citations']}]
                                                  if gated and gate == 'high' else row['source_query_ids'])
                    else:
                        hit['best_chunk_id'] = row['best_chunk_id']
            for hit in candidates.values():
                hit['score'] = (weight * hit['normalized_example_score'] +
                                (1 - weight) * hit['normalized_corpus_score'])
            hits = sorted(candidates.values(), key=lambda h: (-h['score'], h['law_id'], h['article_id']))
            if gate != 'high':
                hits = hits[:args.final_top_n]
            for rank, hit in enumerate(hits, 1):
                hit['rank'] = rank
            assert len({(h['law_id'], h['article_id']) for h in hits}) == len(hits)
            record = {'query_id': query_id, 'query_text': query['query_text'], 'results': hits}
            if gated:
                record.update(gate=gate, s_max=smax, example_weight=weight, corpus_weight=1 - weight)
                assert gate != 'high' or {(h['law_id'], h['article_id']) for h in hits} == allowed
            results.append(record)
            done += 1
        report()
        per_query = []
        if not private:
            phase, done = f'evaluate-hybrid-{args.split}', 0
            gold_rows = read_json_list(labels_path)
            gold = {r['id']: r['relevant_articles'] for r in gold_rows}
            if len(gold) != len(gold_rows) or set(gold) != set(corpus):
                raise ValueError(f'ID gold {args.split} không khớp prediction.')
            if public:
                source_ids = {r['id']: r['image_id'] for r in gold_rows}
                mapping = read_lines(ROOT / 'artifacts/public-test/text_based_rag/query_embedding/queries.jsonl')
                if any(source_ids[q['query_id']] != q['source_image_id'] for q in mapping):
                    raise ValueError('Image ID public gold không khớp query embedding.')
            per_query = []
            for result in results:
                relevant = {(str(r['law_id']), str(r['article_id'])) for r in gold[result['query_id']]}
                if not relevant:
                    raise ValueError(f"Query thiếu gold citation: {result['query_id']}")
                predicted = {(r['law_id'], r['article_id']) for r in result['results']}
                tp = len(predicted & relevant)
                per_query.append({'query_id': result['query_id'], 'true_positives': tp,
                                  'predicted_count': len(predicted), 'relevant_count': len(relevant),
                                  **scores(tp, len(predicted), len(relevant)), 'hit': int(tp > 0),
                                  'gold_citations': [{'law_id': law, 'article_id': article}
                                                     for law, article in sorted(relevant)]})
                done += 1
        config = {'method': 'hybrid_retrieval', 'split': args.split, 'queries': total,
                  'example_weight': args.example_weight, 'corpus_weight': 1 - args.example_weight,
                  'final_top_n': args.final_top_n, 'normalization': 'per_query_per_branch_minmax',
                  'flat_score_policy': 'one', 'missing_branch_score': 0.0}
        if gated:
            config.update(strategy='confidence_gated', tau_high=0.82, tau_low=0.70,
                          weight_scope='example_weight_and_corpus_weight_apply_to_middle_gate',
                          high_example_weight=0.9, low_example_weight=0.3,
                          high_top_k_examples=3, high_cutoff='all_valid_top3_citations',
                          final_top_n_applies_to=['middle', 'low'], gate_counts=gate_counts,
                          normalization='fixed_cosine_affine_(s+1)/2',
                          confidence_interpretation='similarity_not_probability')
            config.pop('flat_score_policy')
        metrics = None
        if not private:
            metrics = {**config, 'macro': {key: sum(r[key] for r in per_query) / total
                                         for key in ('precision', 'recall', 'f1', 'f2')},
                       'micro': scores(sum(r['true_positives'] for r in per_query),
                                       sum(r['predicted_count'] for r in per_query),
                                       sum(r['relevant_count'] for r in per_query)),
                       'hit_rate': sum(r['hit'] for r in per_query) / total}
            if gated:
                metrics['evaluation_scope'] = ('public_test_frozen_dev_configuration' if public else
                                               'dev_development_not_independent_test')
                metrics['by_gate'] = {}
                for gate in gate_counts:
                    rows = [r for r, result in zip(per_query, results) if result['gate'] == gate]
                    metrics['by_gate'][gate] = {'queries': len(rows), 'macro': {
                        key: sum(r[key] for r in rows) / len(rows) if rows else None
                        for key in ('precision', 'recall', 'f1', 'f2')}}
                references = {'corpus_only': [], 'example_only': [], 'gate_only_without_corpus_reranking': []}
                for result in results:
                    query_id = result['query_id']
                    relevant = {(str(r['law_id']), str(r['article_id'])) for r in gold[query_id]}
                    baseline_predictions = {
                        'corpus_only': corpus[query_id]['results'], 'example_only': examples[query_id]['results'],
                        'gate_only_without_corpus_reranking': (result['results'] if result['gate'] == 'high'
                                                              else examples[query_id]['results'])}
                    for name, rows in baseline_predictions.items():
                        predicted = {(str(r['law_id']), str(r['article_id'])) for r in rows}
                        references[name].append(scores(len(predicted & relevant), len(predicted), len(relevant))['f2'])
                metrics['reference_macro_f2'] = {name: sum(values) / total for name, values in references.items()}
        submission = None
        if private:
            phase, done = 'validate-private-submission', 0
            source_rows = read_json_list(PRIVATE_TASK1_FILE)
            predictions = {r['query_id']: r for r in results}
            source_ids = [r['id'] for r in source_rows]
            query_mapping = read_lines(PRIVATE / 'query_embedding/queries.jsonl')
            if (len(set(source_ids)) != len(source_rows) or len(predictions) != len(results)
                    or len(results) != len(source_rows) or set(source_ids) != set(predictions)
                    or {q['query_id'] for q in query_mapping} != set(source_ids)):
                raise ValueError('Submission thiếu/trùng ID hoặc không đủ query từ input private.')
            source_by_id = {r['id']: r for r in source_rows}
            if any(source_by_id[q['query_id']]['image_id'] != q['source_image_id'] for q in query_mapping):
                raise ValueError('Image ID private không khớp embedding.')
            submission = []
            for row in source_rows:
                hits = predictions[row['id']]['results']
                citations = [(h['law_id'], h['article_id']) for h in hits]
                if (set(row) != {'id', 'image_id', 'question'} or not citations or len(set(citations)) != len(citations)
                        or any(not isinstance(law, str) or not isinstance(article, str) or (law, article) not in valid
                               for law, article in citations)):
                    raise ValueError(f"Submission không hợp lệ/không có citation: {row['id']}; chưa xuất file benchmark.")
                submission.append({**row, 'relevant_articles': [{'law_id': law, 'article_id': article}
                                                               for law, article in citations]})
                done += 1
            report()
        output.mkdir(parents=True, exist_ok=True)
        manifest = {**config, 'status': 'IN_PROGRESS', 'created_at': datetime.now(timezone.utc).isoformat(),
                    'input_sha256': hashes, 'source_sha256': file_hash(ROOT / 'src/text_based_rag/query_retrieval.py'),
                    'candidate_policy': 'union_saved_branch_results', 'tie_break': 'citation_lexicographic',
                    **source_metadata,
                    'overwrite': args.overwrite, 'resources': {'device': 'cpu', 'vector_bytes_loaded': vector_bytes}}
        if gated:
            manifest.update(candidate_policy='high_top3_examples_only_else_union_saved_results',
                            corpus_scoring='max_child_cosine_for_every_candidate',
                            high_reranking='weighted_within_locked_candidate_set_not_strict_tie_break',
                            selection_split='dev', independent_evaluation=public)
        if public:
            manifest.update(candidate_policy='high_top3_examples_only_else_top5_corpus_union_top5_examples',
                            configuration_frozen_from_dev=True, intermediate_branch_outputs=False)
        if private:
            manifest.update(candidate_policy='high_top3_examples_only_else_top5_corpus_union_top5_examples',
                            evaluation_available=False, configuration_frozen_from_dev=True,
                            intermediate_branch_outputs=True, submission_file='../submission_task1.json')
        save(output / 'manifest.json', manifest)
        exports = (('retrieval_results.jsonl', results),) if private else (
            ('retrieval_results.jsonl', results), ('per_query_metrics.jsonl', per_query))
        for name, rows in exports:
            with (output / name).open('w' if args.overwrite else 'x', encoding='utf-8') as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        if not private:
            save(output / 'metrics.json', metrics)
        with (output / 'retrieval_preview.txt').open('w' if args.overwrite else 'x', encoding='utf-8') as stream:
            for result, evaluation in zip(results, [None] * len(results) if private else per_query):
                stream.write(f"===== {result['query_id']} =====\n{result['query_text']}\n")
                if evaluation is not None:
                    stream.write(f"F2={evaluation['f2']:.4f}; gold={evaluation['gold_citations']}\n")
                if gated:
                    stream.write(f"gate={result['gate']}; s_max={result['s_max']:.6f}; "
                                 f"example_weight={result['example_weight']}; corpus_weight={result['corpus_weight']}\n")
                for hit in result['results']:
                    stream.write(f"#{hit['rank']} | {hit['law_id']} | {hit['article_id']} | "
                                 f"fusion={hit['score']:.6f} | corpus={hit['normalized_corpus_score']:.6f} | "
                                 f"example={hit['normalized_example_score']:.6f} | sources={hit['sources']}\n")
                stream.write('\n')
        if private:
            save(PRIVATE / 'submission_task1.json', submission)
        manifest.update(status='COMPLETED', elapsed_seconds=time.perf_counter() - started,
                        output_sha256={name: file_hash(output / name) for name in names if name != 'manifest.json'})
        if private:
            manifest['output_sha256']['../submission_task1.json'] = file_hash(PRIVATE / 'submission_task1.json')
        save(output / 'manifest.json', manifest)
        report()
        if private:
            print(f'[submission] {PRIVATE / "submission_task1.json"}; queries={len(submission)}; no private gold evaluation', flush=True)
        else:
            print(f"[hybrid-metrics] {json.dumps(metrics['macro'])}", flush=True)
        print(f'[output] {output}', flush=True)
    except BaseException as exc:
        if manifest is not None:
            manifest.update(status='FAILED', error=str(exc))
            save(output / 'manifest.json', manifest)
        raise
    finally:
        stopped.set()
        worker.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('dev', 'public_test', 'private_test'), default='dev')
    parser.add_argument('--method', choices=('corpus', 'examples', 'hybrid'), default='corpus')
    parser.add_argument('--hybrid-strategy', choices=('gated', 'fixed'), default='gated')
    parser.add_argument('--example-weight', type=float, default=0.6, help='Hybrid example weight; corpus weight = 1 - weight.')
    parser.add_argument('--final-top-n', type=int, default=5, help='Hybrid final citation cutoff.')
    parser.add_argument("--overwrite", action="store_true", help="Replace retrieval output files.")
    args = parser.parse_args()
    if args.method == 'hybrid':
        if not 0 < args.example_weight < 1 or args.final_top_n < 1:
            parser.error('Hybrid yêu cầu 0 < example-weight < 1, final-top-n >= 1.')
        if args.split in ('public_test', 'private_test') and (args.hybrid_strategy != 'gated'
                                            or args.example_weight != 0.6 or args.final_top_n != 5):
            parser.error('Test dùng cấu hình đã chốt trên dev: strategy=gated, example-weight=0.6, final-top-n=5.')
        hybrid_retrieval(args)
        return
    if args.split == 'private_test':
        private_branch_retrieval(args)
        return
    examples = args.method == 'examples'
    if examples and args.split != 'dev':
        parser.error('--method examples chỉ hỗ trợ --split dev ở bước này.')
    public_root = ROOT / 'artifacts/public-test/text_based_rag'
    OUTPUT = (public_root / 'query_retrieval_corpus' if args.split == 'public_test' else
              BASE / ('query_retrieval_example' if examples else 'query_retrieval_corpus'))
    LABELS = DEFAULT_SOURCE / DATA_FILES['public_test'] if args.split == 'public_test' else DEV
    query_folder = 'query_embedding' if args.split == 'public_test' else 'query_embedding_dev'
    query_source = public_root / query_folder if args.split == 'public_test' else BASE / query_folder
    bank_folder = 'query_embedding_train' if examples else 'corpus_embedding'
    output_names = ('manifest.json', 'retrieval_results.jsonl', 'per_query_metrics.jsonl',
                    'metrics.json', 'retrieval_preview.txt')
    if any((OUTPUT / name).exists() for name in output_names) and not args.overwrite:
        raise FileExistsError(f"Output retrieval tại {OUTPUT} đã tồn tại; dùng --overwrite để thay thế.")
    started, stopped = time.perf_counter(), Event()
    phase, done, total = "validate-input", 0, 0
    manifest = None

    def report():
        elapsed = time.perf_counter() - started
        eta = f"{elapsed * (total - done) / done:.0f}s" if done else "unknown"
        print(f"[{phase}] {done}/{total} ({100 * done / max(1, total):.1f}%) "
              f"elapsed={elapsed:.0f}s ETA={eta}", flush=True)

    def heartbeat():
        while not stopped.wait(30):
            report()

    worker = Thread(target=heartbeat, daemon=True)
    worker.start()
    report()
    try:
        inputs, hashes, manifests = {}, {}, {}
        for folder, mapping_name, id_field in ((query_folder, "queries.jsonl", "query_id"),
                                                (bank_folder, 'queries.jsonl' if examples else 'chunks.jsonl',
                                                 'query_id' if examples else 'chunk_id')):
            source = query_source if folder == query_folder else (BASE if examples else CORPUS_BASE) / folder
            meta = json.loads((source / "manifest.json").read_text(encoding="utf-8-sig"))
            if meta.get("status") != "COMPLETED":
                raise ValueError(f"{folder} chưa COMPLETED.")
            for name in ("vectors.npy", mapping_name):
                digest = file_hash(source / name)
                if meta.get("output_sha256", {}).get(name) != digest:
                    raise ValueError(f"File đã thay đổi hoặc không khớp manifest: {folder}/{name}")
                hashes[f"{folder}/{name}"] = digest
            hashes[f"{folder}/manifest.json"] = file_hash(source / "manifest.json")
            mapping = read_lines(source / mapping_name)
            vectors = np.load(source / "vectors.npy", mmap_mode="r", allow_pickle=False)
            if (not mapping or vectors.dtype != np.float32 or vectors.shape != (len(mapping), 1024)
                    or not np.isfinite(vectors).all()
                    or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
                    or len({m[id_field] for m in mapping}) != len(mapping)
                    or any(m["vector_row"] != i for i, m in enumerate(mapping))):
                raise ValueError(f"Vector hoặc ánh xạ hàng không hợp lệ: {folder}")
            inputs[folder], manifests[folder] = (mapping, vectors), meta
        qm, cm = manifests[query_folder], manifests[bank_folder]
        for key in ("model", "revision", "model_files_sha256", "pooling", "normalize", "dimension",
                    "stored_dtype", "inference_dtype", "input_prefix"):
            if key not in qm or qm[key] != cm.get(key):
                raise ValueError(f"Encoder query/{bank_folder} không khớp: {key}")
        if qm.get("split") != args.split or qm["normalize"] != "L2":
            raise ValueError("Cần vector đúng split được chuẩn hóa L2.")
        label_hash = file_hash(LABELS)
        if qm.get("input_sha256", {}).get(f"{args.split}.json") != label_hash:
            raise ValueError("Dữ liệu đã thay đổi sau embedding query; kiểm tra nguồn trước khi đánh giá.")
        hashes[f"{args.split}.json"] = label_hash
        queries, query_vectors = inputs[query_folder]
        chunks, chunk_vectors = inputs[bank_folder]
        total = len(queries)
        excluded = set()
        groups = {}
        if examples:
            if (cm.get('split') != 'train' or qm.get('representation') != 'question_relevant_items_v1'
                    or cm.get('representation') != qm['representation']):
                raise ValueError('Train/dev phải cùng representation strict-relevant và đúng split.')
            if {q['query_id'] for q in queries} & {q['query_id'] for q in chunks}:
                raise ValueError('Train/dev bị trùng query_id; kiểm tra split trước retrieval.')
            train_path = DEV.with_name('train.json')
            hashes['train.json'] = file_hash(train_path)
            if cm.get('input_sha256', {}).get('train.json') != hashes['train.json']:
                raise ValueError('Train split đã thay đổi sau embedding.')
            if len(chunks) < TOP_K:
                raise ValueError('Cần ít nhất 5 train examples.')
            # The corpus mapping is an ID registry only; corpus vectors never enter example scoring.
            registry = CORPUS_BASE / 'corpus_embedding'
            registry_meta = json.loads((registry / 'manifest.json').read_text(encoding='utf-8-sig'))
            hashes['corpus_embedding/manifest.json'] = file_hash(registry / 'manifest.json')
            hashes['corpus_embedding/chunks.jsonl'] = file_hash(registry / 'chunks.jsonl')
            if (registry_meta.get('status') != 'COMPLETED' or
                    registry_meta.get('output_sha256', {}).get('chunks.jsonl') != hashes['corpus_embedding/chunks.jsonl']):
                raise ValueError('Danh mục citation corpus không khớp index đã hoàn tất.')
            valid_citations = {(str(c['document_id']), str(c['id'])) for c in read_lines(registry / 'chunks.jsonl')}
        else:
            for i, chunk in enumerate(chunks):
                citation = (str(chunk["document_id"]), str(chunk["id"]))
                groups.setdefault(citation, []).append(i)
        if not examples and len(groups) < TOP_K:
            raise ValueError("Corpus có ít hơn 5 điều khoản duy nhất.")
        phase = 'example-retrieval-inference-cpu' if examples else 'dense-corpus-retrieval-inference-cpu'
        similarity = query_vectors @ chunk_vectors.T
        if not np.isfinite(similarity).all():
            raise ValueError("Điểm cosine không hợp lệ.")
        if examples:
            neighbors = [sorted(range(len(chunks)), key=lambda i: (-float(values[i]), chunks[i]['query_id']))[:TOP_K]
                         for values in similarity]
            # Attach train labels only after the nearest examples have been selected.
            train_rows = read_json_list(train_path)
            train = {r['id']: r for r in train_rows}
            if (len(train) != len(train_rows) or set(train) != {q['query_id'] for q in chunks}
                    or any(train[q['query_id']]['image_id'] != q['source_image_id'] for q in chunks)):
                raise ValueError('Nhãn train không khớp mapping embedding.')
            for q in chunks:
                labels = train[q['query_id']]['relevant_articles']
                if not labels:
                    raise ValueError(f"Train example thiếu citation: {q['query_id']}")
                for label in labels:
                    citation = (str(label['law_id']), str(label['article_id']))
                    if citation in valid_citations:
                        groups.setdefault(citation, [])
                    else:
                        excluded.add(citation)
        results = []
        for row, query in enumerate(queries):
            if examples:
                selected, citations = [], {}
                for index in neighbors[row]:
                    example = chunks[index]
                    score = float(similarity[row, index])
                    labels = train[example['query_id']]['relevant_articles']
                    selected.append({'rank': len(selected) + 1, 'query_id': example['query_id'],
                                     'query_text': example['text'], 'score': score, 'gold_citations': labels})
                    for citation in {(str(r['law_id']), str(r['article_id'])) for r in labels}:
                        if citation not in valid_citations:
                            continue
                        hit = citations.setdefault(citation, {'law_id': citation[0], 'article_id': citation[1],
                                                              'score': score, 'source_query_ids': []})
                        hit['score'] = max(hit['score'], score)
                        hit['source_query_ids'].append(example['query_id'])
                hits = sorted(citations.values(), key=lambda h: (-h['score'], h['law_id'], h['article_id']))
                for rank, hit in enumerate(hits, 1):
                    hit['rank'] = rank
                assert len(selected) == TOP_K and len(hits) == len(citations)
                results.append({'query_id': query['query_id'], 'query_text': query['text'],
                                'examples': selected, 'results': hits})
                done += 1
                continue
            best = []
            for citation, indices in groups.items():
                index = indices[int(np.argmax(similarity[row, indices]))]
                best.append((float(similarity[row, index]), citation, index))
            best.sort(key=lambda item: (-item[0], item[1]))
            hits = []
            for rank, (score, citation, index) in enumerate(best[:TOP_K], 1):
                chunk = chunks[index]
                hits.append({"rank": rank, "law_id": citation[0], "article_id": citation[1],
                             "score": score, "article_key": chunk["article_key"],
                             "title": chunk["title"], "best_chunk_id": chunk["chunk_id"],
                             "best_chunk_text": chunk["text"]})
            assert len(hits) == TOP_K and len({(h["law_id"], h["article_id"]) for h in hits}) == TOP_K
            results.append({"query_id": query["query_id"], "query_text": query["text"], "results": hits})
            done += 1
        report()
        # Evaluation labels never influence ranking or predicted citations.
        phase, done = f"evaluate-{args.split}", 0
        gold_rows = read_json_list(LABELS)
        gold = {r["id"]: r["relevant_articles"] for r in gold_rows}
        if len(gold) != len(gold_rows) or set(gold) != {q["query_id"] for q in queries}:
            raise ValueError("ID nhãn không khớp query.")
        image_ids = {r["id"]: r["image_id"] for r in gold_rows}
        if any(image_ids[q["query_id"]] != q["source_image_id"] for q in queries):
            raise ValueError("Image ID không khớp mapping query.")
        per_query, missing_gold = [], set()
        for result in results:
            relevant = {(str(r["law_id"]), str(r["article_id"])) for r in gold[result["query_id"]]}
            if not relevant:
                raise ValueError(f"Query thiếu citation đúng: {result['query_id']}")
            missing_gold.update(relevant - groups.keys())
            predicted = {(r["law_id"], r["article_id"]) for r in result["results"]}
            tp = len(predicted & relevant)
            per_query.append({"query_id": result["query_id"], "true_positives": tp,
                              "predicted_count": len(predicted), "relevant_count": len(relevant),
                              **scores(tp, len(predicted), len(relevant)), "hit": int(tp > 0),
                              "gold_citations": [{"law_id": law, "article_id": article}
                                                 for law, article in sorted(relevant)]})
            done += 1
        metrics = {"method": "dense_corpus_retrieval", "split": args.split, "top_k": TOP_K, "queries": total,
                   "macro": {key: sum(r[key] for r in per_query) / total
                             for key in ("precision", "recall", "f1", "f2")},
                   "micro": scores(sum(r["true_positives"] for r in per_query),
                                   sum(r["predicted_count"] for r in per_query),
                                   sum(r["relevant_count"] for r in per_query)),
                   "hit_rate_at_5": sum(r["hit"] for r in per_query) / total,
                   "gold_citations_missing_from_index": [{"law_id": law, "article_id": article}
                                                         for law, article in sorted(missing_gold)]}
        if examples:
            metrics['method'] = 'example_retrieval'
            metrics['top_k_examples'] = metrics.pop('top_k')
            metrics['hit_rate'] = metrics.pop('hit_rate_at_5')
            metrics['gold_citations_missing_from_examples'] = metrics.pop('gold_citations_missing_from_index')
            metrics['excluded_train_citations'] = [{'law_id': law, 'article_id': article}
                                                    for law, article in sorted(excluded)]
        # Validate all inputs and predictions before replacing the authorized output files.
        OUTPUT.mkdir(parents=True, exist_ok=True)
        manifest = {"status": "IN_PROGRESS", "created_at": datetime.now(timezone.utc).isoformat(),
                    "method": "dense_corpus_retrieval", "split": args.split, "queries": total,
                    "chunks": len(chunks), "articles": len(groups),
                    "top_k": TOP_K, "similarity": "cosine", "article_aggregation": "max_chunk_score",
                    "citation_key": ["law_id", "article_id"], "tie_break": "citation_lexicographic_then_chunk_row",
                    "input_sha256": hashes, "numpy_version": version("numpy"),
                    "source_sha256": file_hash(ROOT / "src/text_based_rag/query_retrieval.py"),
                    "query_encoder": qm, "corpus_encoder": cm, "overwrite": args.overwrite,
                    "resources": {"device": "cpu", "similarity_matrix_bytes": similarity.nbytes,
                                  "vector_bytes": query_vectors.nbytes + chunk_vectors.nbytes}}
        if examples:
            manifest.update(method='example_retrieval', examples=manifest.pop('chunks'),
                            top_k_examples=manifest.pop('top_k'), example_encoder=manifest.pop('corpus_encoder'),
                            article_aggregation='union_top_examples_max_score',
                            tie_break='example_query_id_then_citation_lexicographic',
                            same_image_policy='allow_shared_images_as_requested',
                            invalid_citation_policy='exclude_ids_absent_from_index_registry',
                            excluded_train_citations=metrics['excluded_train_citations'])
        save(OUTPUT / "manifest.json", manifest)
        for name, records in (("retrieval_results.jsonl", results), ("per_query_metrics.jsonl", per_query)):
            with (OUTPUT / name).open("w" if args.overwrite else "x", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        save(OUTPUT / "metrics.json", metrics)
        with (OUTPUT / "retrieval_preview.txt").open("w" if args.overwrite else "x", encoding="utf-8") as stream:
            for result, evaluation in zip(results, per_query):
                stream.write(f"===== {result['query_id']} =====\n{result['query_text']}\n\n")
                suffix = ' (top 5 examples)' if examples else '@5'
                stream.write(f"Precision{suffix}={evaluation['precision']:.4f}; Recall{suffix}={evaluation['recall']:.4f}; "
                             f"F2{suffix}={evaluation['f2']:.4f}\nGold: {evaluation['gold_citations']}\n\n")
                if examples:
                    for example in result['examples']:
                        stream.write(f"Example #{example['rank']} | {example['query_id']} | cosine={example['score']:.6f}\n"
                                     f"{example['query_text']}\nGold: {example['gold_citations']}\n\n")
                for hit in result["results"]:
                    if examples:
                        stream.write(f"#{hit['rank']} | {hit['law_id']} | {hit['article_id']} | "
                                     f"score={hit['score']:.6f} | sources={hit['source_query_ids']}\n\n")
                        continue
                    stream.write(f"#{hit['rank']} | {hit['law_id']} | {hit['article_id']} | "
                                 f"cosine={hit['score']:.6f} | {hit['title']}\n"
                                 f"{hit['best_chunk_id']}\n{hit['best_chunk_text']}\n\n")
        manifest.update(status="COMPLETED", elapsed_seconds=time.perf_counter() - started,
                        output_sha256={name: file_hash(OUTPUT / name) for name in output_names
                                       if name != 'manifest.json'})
        save(OUTPUT / "manifest.json", manifest)
        report()
        metric_label = 'metrics-top-5-examples' if examples else 'metrics@5'
        print(f"[{metric_label}] {json.dumps(metrics['macro'])}", flush=True)
        if missing_gold:
            print(f"[warning] {len(missing_gold)} gold citation ngoài nguồn retrieval; vẫn tính vào recall.", flush=True)
        if excluded:
            print(f'[warning] Loại {len(excluded)} train citation ID không có trong index; xem metrics.json.', flush=True)
        print(f"[output] {OUTPUT}", flush=True)
    except BaseException as exc:
        if manifest is not None:
            manifest.update(status="FAILED", error=str(exc))
            save(OUTPUT / "manifest.json", manifest)
        raise
    finally:
        stopped.set()
        worker.join()


if __name__ == "__main__":
    main()
