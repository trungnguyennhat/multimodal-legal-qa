"""Fuse pure-text corpus/example rankings before fusion with the saved image ranking."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from hybrid_rag.full_image_method.query_image_hybrid import load_branch, rrf
from hybrid_rag.text_only.corpus_preprocessing import (
    BASE, DEV, SOURCE, check_hashes, hashes_of, read_lines, run_step, save, write_lines,
)
from hybrid_rag.text_only.query_embedding import read_splits
from hybrid_rag.text_only.query_retrieval import evaluate_predictions


def fuse_text(args, folder, meta, progress):
    progress.set('validate text corpus/example inputs')
    paths = [DEV, SOURCE/'manifest.json', SOURCE/'structured_corpus.jsonl']
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
        inputs[method],branch_hashes,branch_metrics['text_'+('examples' if method=='example' else method)] = load_branch(
            BASE/('query_retrieval_'+method), original, registry, labels)
        hashes.update(branch_hashes)
    meta.update(input_sha256=hashes,split='dev',method='corpus_examples_rrf',queries=len(dev),
                representation='original_question_only',example_bank_split='train',example_bank_queries=len(labels),
                rrf_k=args.rrf_k,within_modality_weights={'corpus':0.5,'examples':0.5},
                candidate_depth=10,tie_break='citation_lexicographic')
    predictions = []
    fallback_count = 0
    progress.set('fuse text corpus/examples',len(dev))
    for query in dev:
        qid = query['id']
        corpus = inputs['corpus'][qid]['results']
        examples = inputs['example'][qid]['results']
        fallback = not examples
        hits = corpus if fallback else rrf({'corpus':corpus,'examples':examples},{'corpus':0.5,'examples':0.5},args.rrf_k)
        fallback_count += fallback
        predictions.append({'query_id':qid,'source_image_id':query['image_id'],'source_split':'dev',
                            'query_text':query['question'],'results':hits,'example_fallback':fallback})
        progress.done += 1
    metrics,evaluations = evaluate_predictions(predictions,dev,registry,'text_corpus_examples')
    metrics.update(rrf_k=args.rrf_k,within_modality_weights={'corpus':0.5,'examples':0.5},
                   example_fallback_queries=fallback_count,branch_metrics=branch_metrics)
    check_hashes(hashes)
    write_lines(folder/'predictions.jsonl',predictions)
    write_lines(folder/'per_query_metrics.jsonl',evaluations)
    save(folder/'metrics.json',metrics)
    meta.update(macro_f2_at_5=metrics['macro_f2_at_5'],output_sha256=hashes_of([folder/n for n in (
        'predictions.jsonl','per_query_metrics.jsonl','metrics.json')]))
    print(f"[text corpus+example] F2@5={metrics['macro_f2_at_5']:.6f}",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=('dev',),default='dev')
    parser.add_argument('--rrf-k',type=int,default=60)
    parser.add_argument('--overwrite',action='store_true')
    args = parser.parse_args()
    if args.rrf_k<1:
        parser.error('rrf-k phải >= 1')
    run_step(args,'query_retrieval_hybrid',fuse_text,Path(__file__),
             [Path(__file__).with_name(n) for n in ('query_embedding.py','query_retrieval.py')]
             + [ROOT/'src/hybrid_rag/full_image_method'/n for n in ('query_image_hybrid.py','corpus_image_embedding.py')]
             + [ROOT/'src/text_based_rag/query_retrieval.py'])


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
