"""Retrieve citations from full-image corpus or train examples; evaluate dev only."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from hybrid_rag.full_image_method.corpus_image_embedding import (
    BASE, check_hashes, corpus_inputs, hashes_of, read_lines, run_step, save, validate_vectors, write_lines,
)
from hybrid_rag.full_image_method.query_image_embedding import dev_inputs, SPLIT_SOURCE, SPLIT_HELPERS
from hybrid_rag.text_only.query_retrieval import example_retrieval, evaluate_predictions

RETRIEVAL_HELPERS = ROOT / "src/hybrid_rag/text_only/query_retrieval.py"


def load_index(folder):
    meta = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    query_index = folder.name == "query_embedding_dev"
    names = ["images.jsonl", "vectors.npy"] + (["queries.jsonl"] if query_index else [])
    expected = {str((folder/name).relative_to(ROOT)) for name in names}
    code_paths = [Path(__file__).with_name("corpus_image_embedding.py")]
    if query_index:
        code_paths += [Path(__file__).with_name("query_image_embedding.py"), SPLIT_SOURCE, SPLIT_HELPERS]
    if (meta.get('status') != 'COMPLETED' or meta.get('code_sha256') != hashes_of(code_paths)
            or set(meta.get('output_sha256',{})) != expected):
        raise ValueError(f"Index chưa COMPLETED hoặc fingerprint sai: {folder}; chạy lại query embedding với --overwrite nếu là index query.")
    check_hashes(meta['input_sha256'])
    check_hashes(meta['output_sha256'])
    images = read_lines(folder / 'images.jsonl')
    if (not images or len({r['image_id'] for r in images}) != len(images)
            or any(r['vector_row'] != i for i,r in enumerate(images))):
        raise ValueError(f"Mapping ảnh không hợp lệ: {folder}")
    vectors = np.load(folder / 'vectors.npy', allow_pickle=False)
    validate_vectors(vectors, len(images))
    return meta, images, vectors


def retrieve(args, folder, meta, progress):
    progress.set("validate full-image indexes")
    cm, corpus, cv = load_index(BASE / 'corpus_embedding')
    qm, images, qv = load_index(BASE / 'query_embedding_dev')
    if cm['config'] != qm['config']:
        raise ValueError("Corpus/query phải cùng model, revision và preprocessing.")
    current_corpus, registry, hashes = corpus_inputs(progress)
    current_images, all_queries, original, query_hashes = dev_inputs(progress)
    if (current_corpus != corpus or current_images != images
            or read_lines(BASE / 'query_embedding_dev/queries.jsonl') != all_queries):
        raise ValueError("Mapping train/dev khác index; chạy lại query_image_embedding.py --overwrite.")
    hashes.update(query_hashes)
    for index_meta in (cm,qm):
        hashes.update(index_meta['input_sha256'])
        hashes.update(index_meta['output_sha256'])
    hashes.update(hashes_of([BASE/'corpus_embedding/manifest.json', BASE/'query_embedding_dev/manifest.json']))
    queries = [r for r in all_queries if r['source_split']=='dev']
    bank = [r for r in all_queries if r['source_split']=='train']
    dev = [r for r in original if r['source_split']=='dev']
    labels = {r['id']:r for r in original if r['source_split']=='train'}
    groups = {}
    for i, row in enumerate(corpus):
        for citation in row['citations']:
            groups.setdefault((citation['law_id'],citation['article_id']),[]).append(i)
    if not set(groups) <= registry or len(groups) < 10:
        raise ValueError("Corpus không đủ citation hợp lệ.")
    excluded = {(str(c['law_id']),str(c['article_id'])) for r in labels.values() for c in r['relevant_articles']} - registry
    meta.update(input_sha256=hashes, config=cm['config'], split='dev', method=args.method, queries=len(queries),
                example_bank_split='train', example_bank_queries=len(bank), top_k_examples=5,
                same_image_policy='allow_shared_images', index_type='numpy_exact_cosine',
                aggregation='max_image_score_per_citation' if args.method=='corpus' else 'max_train_example_cosine',
                tie_break='citation_lexicographic; source_id_lexicographic')
    meta['code_sha256'].update(hashes_of([SPLIT_SOURCE,SPLIT_HELPERS,RETRIEVAL_HELPERS,
                                          Path(__file__).with_name('query_image_embedding.py'),
                                          ROOT/'src/text_based_rag/query_retrieval.py']))
    progress.set('retrieve image ' + args.method,len(queries))
    dev_vectors = qv[[r['vector_row'] for r in queries]]
    similarity = dev_vectors @ (cv.T if args.method=='corpus' else qv[[r['vector_row'] for r in bank]].T)
    if not np.isfinite(similarity).all():
        raise ValueError("Cosine không hữu hạn.")
    predictions = []
    corpus_hashes = {r['sha256'] for r in corpus}
    bank_hashes = {r['image_id'] for r in bank}
    shared_bank_queries = []
    for i, query in enumerate(queries):
        if args.method=='examples':
            hits, examples = example_retrieval(similarity[i],bank,labels,registry)
        else:
            examples, hits = [], []
            for (law,article), indices in sorted(groups.items()):
                winner = min(indices,key=lambda j:(-float(similarity[i,j]),corpus[j]['image_id']))
                hits.append({'law_id':law,'article_id':article,'score':float(similarity[i,winner]),
                             'best_image_id':corpus[winner]['image_id'],'best_image_path':corpus[winner]['path']})
            hits.sort(key=lambda r:(-r['score'],r['law_id'],r['article_id']))
            hits = [{**r,'rank':rank} for rank,r in enumerate(hits[:10],1)]
        prediction = {**query,'results':hits}
        if args.method=='examples':
            prediction['examples'] = examples
        if query['image_id'] in bank_hashes:
            shared_bank_queries.append(query['query_id'])
        predictions.append(prediction)
        progress.done += 1
    metrics, evaluations = evaluate_predictions(predictions,dev,registry,'full_image_' + args.method)
    gold_total = sum(len({(str(c['law_id']),str(c['article_id'])) for c in r['relevant_articles']}) for r in dev)
    covered_total = 0
    for row, evaluation, prediction in zip(dev,evaluations,predictions):
        gold = {(str(c['law_id']),str(c['article_id'])) for c in row['relevant_articles']}
        absent = gold-set(groups)
        covered_total += len(gold & set(groups))
        evaluation['gold_without_index_image'] = [{'law_id':l,'article_id':a} for l,a in sorted(absent)]
        evaluation['query_image_in_corpus'] = images[prediction['vector_row']]['sha256'] in corpus_hashes
    metrics.update(corpus_images=len(corpus), citations_with_images=len(groups),
                   unique_query_images=len({r['image_id'] for r in queries}), example_bank_queries=len(bank),
                   top_k_examples=5, dev_queries_with_bank_image=shared_bank_queries,
                   excluded_train_citations=[{'law_id':l,'article_id':a} for l,a in sorted(excluded)],
                   coverage={'gold_citation_occurrences':gold_total,'covered_gold_occurrences':covered_total,
                             'gold_image_coverage':covered_total/gold_total})
    write_lines(folder/'predictions.jsonl',predictions)
    write_lines(folder/'per_query_metrics.jsonl',evaluations)
    save(folder/'metrics.json',metrics)
    meta.update(macro_f2_at_5=metrics['macro_f2_at_5'],
                output_sha256=hashes_of([folder/n for n in ('predictions.jsonl','per_query_metrics.jsonl','metrics.json')]))
    print(f"[dev image {args.method}] F2@5={metrics['macro_f2_at_5']:.6f}",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=('dev',),default='dev')
    parser.add_argument('--method',choices=('corpus','examples'),default='corpus')
    parser.add_argument('--overwrite',action='store_true')
    args = parser.parse_args()
    args.command = 'retrieve'
    name = 'query_retrieval_example' if args.method=='examples' else 'query_retrieval_corpus'
    run_step(args,BASE/name,retrieve,Path(__file__))


if __name__ == '__main__':
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
