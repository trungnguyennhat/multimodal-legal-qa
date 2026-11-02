"""Fuse image corpus/example rankings and save the image-only ranking before text fusion."""

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from hybrid_rag.full_image_method.corpus_image_embedding import (
    BASE, CORPUS, check_hashes, hashes_of, read_lines, run_step, save, write_lines,
)
from hybrid_rag.text_only.query_embedding import read_splits
from hybrid_rag.text_only.query_retrieval import evaluate_predictions

DEV = ROOT / "data/processed/splits/dev.json"


def load_branch(folder, original, registry, labels):
    manifest = folder / 'manifest.json'
    meta = json.loads(manifest.read_text(encoding='utf-8'))
    method = {'query_retrieval_corpus':'corpus', 'query_retrieval_example':'examples',
              'query_retrieval_hybrid':'corpus_examples_rrf'}[folder.name]
    required = {str((folder/n).relative_to(ROOT)) for n in ('predictions.jsonl','metrics.json')}
    if (meta.get('status')!='COMPLETED' or meta.get('split')!='dev' or meta.get('method')!=method
            or meta.get('example_bank_split')!='train' or meta.get('example_bank_queries')!=len(labels)
            or not required <= set(meta.get('output_sha256',{})) or not meta.get('code_sha256')):
        raise ValueError(f"Nhánh chưa COMPLETED hoặc chưa có đúng example bank train: {folder}")
    for field in ('code_sha256','input_sha256','output_sha256'):
        check_hashes(meta[field])
    rows = read_lines(folder/'predictions.jsonl')
    lookup = {r['query_id']:r for r in rows}
    if len(lookup)!=len(rows) or set(lookup)!=set(original):
        raise ValueError(f"Query ID không khớp dev: {folder}")
    for qid,row in lookup.items():
        if (row.get('source_split')!='dev' or row.get('source_image_id')!=original[qid]['image_id']
                or ('query_text' in row and row['query_text']!=original[qid]['question'])):
            raise ValueError(f"Metadata query sai: {qid}")
        hits = row['results']
        citations = [(r['law_id'],r['article_id']) for r in hits]
        if (not 0<=len(hits)<=10 or (method!='examples' and len(hits)!=10)
                or len(set(citations))!=len(hits) or not set(citations)<=registry
                or any(r.get('rank')!=i or not math.isfinite(r['score']) for i,r in enumerate(hits,1))
                or hits!=sorted(hits,key=lambda r:(-r['score'],r['law_id'],r['article_id']))):
            raise ValueError(f"Rank/citation/score sai: {folder}/{qid}")
        if method=='examples':
            examples = row.get('examples',[])
            if len(examples)!=5 or len({e['query_id'] for e in examples})!=5:
                raise ValueError(f"Thiếu top-5 train examples: {qid}")
            for e in examples:
                source = labels.get(e['query_id'])
                if (source is None or e.get('source_split')!='train'
                        or e.get('source_image_id')!=source['image_id']
                        or e.get('gold_citations')!=source['relevant_articles']):
                    raise ValueError(f"Example không thuộc train hoặc sai nhãn: {qid}")
    metrics,_ = evaluate_predictions(rows,list(original.values()),registry,'loaded_' + method)
    stored = json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
    if not math.isclose(metrics['macro_f2_at_5'],stored['macro_f2_at_5'],rel_tol=0,abs_tol=1e-12):
        raise ValueError(f"Metric không khớp predictions/gold: {folder}")
    hashes = {**meta['input_sha256'],**meta['output_sha256'],**hashes_of([manifest])}
    return lookup,hashes,stored



def rrf(branches, weights, k):
    candidates = {}
    for name,hits in branches.items():
        for hit in hits:
            key = (hit['law_id'],hit['article_id'])
            item = candidates.setdefault(key,{'law_id':key[0],'article_id':key[1],'score':0.0})
            contribution = weights[name]/(k+hit['rank'])
            item['score'] += contribution
            item[name] = {**hit,'rrf_contribution':contribution}
    ranked = sorted((r for r in candidates.values() if r['score']>0),key=lambda r:(-r['score'],r['law_id'],r['article_id']))
    return [{**r,'rank':i} for i,r in enumerate(ranked[:10],1)]



def fuse_images(args, folder, meta, progress):
    progress.set('validate image corpus/example inputs')
    paths = [DEV, CORPUS/'manifest.json', CORPUS/'structured_corpus.jsonl']
    hashes = hashes_of(paths)
    registry_meta = json.loads(paths[1].read_text(encoding='utf-8'))
    if registry_meta.get('status')!='COMPLETED' or registry_meta.get('schema_version')!=4:
        raise ValueError('Citation registry cần preprocessing schema 4, COMPLETED.')
    registry = {(str(r['document_id']),str(r['id'])) for r in read_lines(paths[2])}
    splits,split_hashes = read_splits()
    hashes.update(split_hashes)
    labels = {r['id']:r for r in splits if r['source_split']=='train'}
    dev = json.loads(DEV.read_text(encoding='utf-8'))
    original = {r['id']:r for r in dev}
    inputs,branch_metrics = {},{}
    for method in ('corpus','example'):
        inputs[method],branch_hashes,branch_metrics['image_'+('examples' if method=='example' else method)] = load_branch(
            BASE/('query_retrieval_'+method), original, registry, labels)
        hashes.update(branch_hashes)
    meta.update(input_sha256=hashes,split='dev',method='corpus_examples_rrf',queries=len(dev),
                example_bank_split='train',example_bank_queries=len(labels),rrf_k=args.rrf_k,
                within_modality_weights={'corpus':0.5,'examples':0.5},candidate_depth=10,
                tie_break='citation_lexicographic')
    meta['code_sha256'].update(hashes_of([ROOT/'src/hybrid_rag/text_only'/n for n in (
        'query_embedding.py','query_retrieval.py','corpus_preprocessing.py')]
        + [ROOT/'src/text_based_rag/query_retrieval.py']))
    predictions = []
    fallback_count = 0
    progress.set('fuse image corpus/examples',len(dev))
    for query in dev:
        qid = query['id']
        corpus = inputs['corpus'][qid]['results']
        examples = inputs['example'][qid]['results']
        fallback = not examples
        hits = corpus if fallback else rrf({'corpus':corpus,'examples':examples},{'corpus':0.5,'examples':0.5},args.rrf_k)
        fallback_count += fallback
        predictions.append({'query_id':qid,'source_image_id':query['image_id'],'source_split':'dev',
                            'results':hits,'example_fallback':fallback})
        progress.done += 1
    metrics,evaluations = evaluate_predictions(predictions,dev,registry,'image_corpus_examples')
    metrics.update(rrf_k=args.rrf_k,within_modality_weights={'corpus':0.5,'examples':0.5},
                   example_fallback_queries=fallback_count,branch_metrics=branch_metrics)
    check_hashes(hashes)
    write_lines(folder/'predictions.jsonl',predictions)
    write_lines(folder/'per_query_metrics.jsonl',evaluations)
    save(folder/'metrics.json',metrics)
    meta.update(macro_f2_at_5=metrics['macro_f2_at_5'],output_sha256=hashes_of([folder/n for n in (
        'predictions.jsonl','per_query_metrics.jsonl','metrics.json')]))
    print(f"[image corpus+example] F2@5={metrics['macro_f2_at_5']:.6f}",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=('dev',),default='dev')
    parser.add_argument('--rrf-k',type=int,default=60)
    parser.add_argument('--overwrite',action='store_true')
    args = parser.parse_args()
    if args.rrf_k<1:
        parser.error('rrf-k phải >= 1')
    args.command = 'image-hybrid-retrieval'
    run_step(args,BASE/'query_retrieval_hybrid',fuse_images,Path(__file__))


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
