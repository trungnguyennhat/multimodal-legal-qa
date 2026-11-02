"""Fuse text corpus/examples with the saved image hybrid ranking; optionally tune on dev."""

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
from hybrid_rag.full_image_method.query_image_hybrid import load_branch, rrf
from hybrid_rag.text_only.query_embedding import read_splits
from hybrid_rag.text_only.query_retrieval import evaluate_predictions

TEXT = ROOT / "artifacts/experiments/hybrid_rag/text_only"
DEV = ROOT / "data/processed/splits/dev.json"
KS = (1,3,5,10)


def fuse(args, folder, meta, progress):
    progress.set('validate registry and original splits')
    paths = [DEV,CORPUS/'manifest.json',CORPUS/'structured_corpus.jsonl']
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
    inputs,input_metrics = {},{}
    for name,branch,method in (('text_corpus',TEXT,'corpus'),('text_examples',TEXT,'example'),
                               ('image_hybrid',BASE,'hybrid')):
        progress.set('validate ' + name)
        inputs[name],branch_hashes,input_metrics[name] = load_branch(branch/('query_retrieval_'+method),original,registry,labels)
        hashes.update(branch_hashes)
    image_metrics = input_metrics.pop('image_hybrid')
    if (image_metrics.get('rrf_k')!=args.rrf_k
            or image_metrics.get('within_modality_weights')!={'corpus':0.5,'examples':0.5}):
        raise ValueError('Ranking ảnh khác cấu hình RRF; chạy query_image_hybrid.py cùng --rrf-k trước.')
    input_metrics.update(image_metrics['branch_metrics'])
    meta.update(input_sha256=hashes,split='dev',queries=len(dev),method='hierarchical_weighted_rrf',
                example_bank_queries=len(labels),within_modality_weights={'corpus':0.5,'examples':0.5},
                rrf_k=args.rrf_k,final_top_n=args.final_top_n,candidate_depth=10,
                tie_break='citation_lexicographic',weight_selection='tuned_on_dev' if args.tune_on_dev else 'fixed_by_cli')
    meta['code_sha256'].update(hashes_of([ROOT/'src/hybrid_rag/text_only'/n for n in (
        'query_embedding.py','query_retrieval.py','corpus_preprocessing.py')]
        + [ROOT/'src/text_based_rag/query_retrieval.py',Path(__file__).with_name('query_image_hybrid.py')]))
    save(folder/'manifest.json',meta)
    modalities = {'text':{},'image':{}}
    fallback = {'text':0,'image':0}
    progress.set('combine corpus and examples',len(dev))
    for query in dev:
        qid = query['id']
        corpus = inputs['text_corpus'][qid]['results']
        examples = inputs['text_examples'][qid]['results']
        text_fallback = not examples
        fallback['text'] += text_fallback
        hits = rrf({'corpus':corpus,'examples':examples},{'corpus':0.5,'examples':0.5},args.rrf_k) if examples else corpus
        modalities['text'][qid] = {'query_id':qid,'source_image_id':query['image_id'],'results':hits}
        image_row = inputs['image_hybrid'][qid]
        if type(image_row.get('example_fallback')) is not bool:
            raise ValueError(f'Ranking ảnh thiếu trạng thái fallback: {qid}')
        fallback['image'] += image_row['example_fallback']
        modalities['image'][qid] = image_row
        progress.done += 1
    modality_metrics = {}
    modality_evaluations = {}
    for name,rows in modalities.items():
        modality_metrics[name],modality_evaluations[name] = evaluate_predictions(list(rows.values()),dev,registry,name+'_corpus_examples')
    weights = (0,0.25,0.5,0.75,1) if args.tune_on_dev else (args.text_weight,)
    cutoffs = sorted(set(KS+(args.final_top_n,)))
    sweep, best = [], None
    initial_equal_weight = None
    progress.set('fuse modalities and score dev',len(weights))
    for weight in weights:
        predictions = []
        for query in dev:
            qid = query['id']
            hits = rrf({name:rows[qid]['results'] for name,rows in modalities.items()},
                       {'text':weight,'image':1-weight},args.rrf_k)
            if len(hits)!=10:
                raise ValueError(f'Fusion thiếu top-10: {qid}')
            predictions.append({'query_id':qid,'source_image_id':query['image_id'],'source_split':'dev','results':hits,
                                'modality_rankings':{name:rows[qid]['results'] for name,rows in modalities.items()},
                                'example_fallback':{'text':not inputs['text_examples'][qid]['results'],
                                                    'image':inputs['image_hybrid'][qid]['example_fallback']}})
        metrics,evaluations = evaluate_predictions(predictions,dev,registry,'hierarchical_weighted_rrf',cutoffs)
        score = metrics['macro_f2_at_5']
        sweep.append({'text_weight':weight,'image_weight':1-weight,'macro_f2_at_5':score})
        if weight==0.5:
            initial_equal_weight = score
        if (best is None or score>best[0]+1e-12 or (abs(score-best[0])<=1e-12 and weight>best[1])):
            best = (score,weight,predictions,metrics,evaluations)
        progress.done += 1
    score,weight,predictions,metrics,evaluations = best
    comparison = {name:m['macro_f2_at_5'] for name,m in input_metrics.items()}
    comparison.update({name+'_corpus_examples':m['macro_f2_at_5'] for name,m in modality_metrics.items()})
    comparison['hybrid'] = score
    changes = {'improved':0,'worsened':0,'unchanged':0}
    union_tp = gold_total = extra_image_gold = 0
    text_eval = {r['query_id']:r for r in modality_evaluations['text']}
    for prediction,evaluation,query in zip(predictions,evaluations,dev):
        qid = query['id']
        delta = evaluation['at_k']['5']['f2']-text_eval[qid]['at_k']['5']['f2']
        evaluation['delta_f2_at_5_vs_text_corpus_examples'] = delta
        changes['improved' if delta>1e-12 else 'worsened' if delta < -1e-12 else 'unchanged'] += 1
        gold = {(str(c['law_id']),str(c['article_id'])) for c in query['relevant_articles']}
        text_set = {(r['law_id'],r['article_id']) for r in modalities['text'][qid]['results']}
        image_set = {(r['law_id'],r['article_id']) for r in modalities['image'][qid]['results']}
        union_tp += len(gold & (text_set|image_set))
        extra_image_gold += len((gold & image_set)-text_set)
        gold_total += len(gold)
        evaluation['example_fallback'] = prediction['example_fallback']
    metrics.update(text_weight=weight,image_weight=1-weight,rrf_k=args.rrf_k,final_top_n=args.final_top_n,
                   final_citation_metrics=metrics['at_k'][str(args.final_top_n)],
                   within_modality_weights={'corpus':0.5,'examples':0.5},example_fallback_queries=fallback,
                   comparison_macro_f2_at_5=comparison,branch_metrics={**input_metrics,**{k+'_corpus_examples':v for k,v in modality_metrics.items()}},
                   weight_search=sweep,initial_equal_weight_macro_f2_at_5=initial_equal_weight,
                   image_branch_enabled=weight<1,query_changes_vs_text_at_5=changes,
                   union_candidate_recall_micro=union_tp/gold_total,image_additional_gold_occurrences_in_top10=extra_image_gold,
                   evaluation_scope='dev_development_not_independent_test; gold_unchanged; '+('tuned_on_dev' if args.tune_on_dev else 'fixed_weights'))
    final = [{'id':q['id'],'image_id':q['image_id'],'question':q['question'],
              'relevant_articles':[{'law_id':r['law_id'],'article_id':r['article_id']} for r in p['results'][:args.final_top_n]]}
             for q,p in zip(dev,predictions)]
    check_hashes(hashes)
    write_lines(folder/'predictions.jsonl',predictions)
    write_lines(folder/'per_query_metrics.jsonl',evaluations)
    save(folder/'final_citations.json',final)
    save(folder/'metrics.json',metrics)
    meta.update(text_weight=weight,image_weight=1-weight,macro_f2_at_5=score,
                output_sha256=hashes_of([folder/n for n in ('predictions.jsonl','final_citations.json','per_query_metrics.jsonl','metrics.json')]))
    print(f'[fusion dev] text_weight={weight}; F2@5={score:.6f}; tuned={args.tune_on_dev}',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=('dev',),default='dev')
    parser.add_argument('--text-weight',type=float,default=0.5)
    parser.add_argument('--tune-on-dev',action='store_true',help='Select text weight from 0,.25,.5,.75,1 using dev F2@5; store sweep in metrics.json.')
    parser.add_argument('--rrf-k',type=int,default=60)
    parser.add_argument('--final-top-n',type=int,default=5)
    parser.add_argument('--overwrite',action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.text_weight) or not 0<=args.text_weight<=1:
        parser.error('text-weight phải trong [0,1]')
    if args.rrf_k<1 or not 1<=args.final_top_n<=10:
        parser.error('rrf-k >= 1; final-top-n trong [1,10]')
    args.command = 'hybrid-retrieval'
    run_step(args,BASE/'image_text_retrieval',fuse,Path(__file__))


if __name__ == '__main__':
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
