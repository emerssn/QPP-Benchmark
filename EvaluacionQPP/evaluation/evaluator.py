import pandas as pd
import numpy as np
from typing import Dict, Union, Iterable
import os
import ir_measures
from ir_measures import nDCG, P, AP, RR, Judged
from ..utils.file_utils import ensure_dir

# Dataset format configurations
DATASET_FORMATS = {
    "antique_test": {
        "doc_id_transform": lambda x: x.split('_')[0],  # Remove suffix after underscore
        "needs_doc_prefix": False,
        "qrels_columns": {
            'qid': 'query_id',
            'docno': 'doc_id',
            'label': 'relevance'
        },
        "run_columns": {
            'qid': 'query_id',
            'docno': 'doc_id',
            'docScore': 'score'
        },
        "run_doc_id_transform": lambda x: x.replace('doc', '').split('_')[0]
    },
    "iquique_dataset": {
        "doc_id_transform": lambda x: x if x.startswith('doc') else f"doc{x}",  # Ensure doc prefix
        "needs_doc_prefix": True,
        "qrels_columns": {
            'qid': 'query_id',
            'docno': 'doc_id',
            'label': 'relevance'
        },
        "run_columns": {
            'qid': 'query_id',
            'docno': 'doc_id',
            'docScore': 'score'
        },
        "run_doc_id_transform": lambda x: x if x.startswith('doc') else f"doc{x}"  # Ensure consistent doc prefix
    }
}

def evaluate_results(
    qrels_df: pd.DataFrame,
    results_df: pd.DataFrame,
    metrics: list = ['ndcg@10'],
    output_dir: str = None,
    dataset_name: str = "antique_test"
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Evaluate retrieval results using ir_measures.
    """
    dataset_config = DATASET_FORMATS.get(dataset_name, DATASET_FORMATS["antique_test"])
    
    qrels = qrels_df.loc[:, ~qrels_df.columns.duplicated()]
    run = results_df.loc[:, ~results_df.columns.duplicated()]
    
    qrels = qrels.rename(columns=dataset_config["qrels_columns"])
    run = run.rename(columns=dataset_config["run_columns"])
    
    qrels['doc_id'] = qrels['doc_id'].astype(str).apply(dataset_config["doc_id_transform"])
    run['doc_id'] = run['doc_id'].astype(str).apply(dataset_config["run_doc_id_transform"])
    
    qrels['query_id'] = qrels['query_id'].astype(str)
    qrels['doc_id'] = qrels['doc_id'].astype(str)
    qrels['relevance'] = qrels['relevance'].astype(int)
    qrels['relevance'] = qrels['relevance'].clip(lower=0)
    
    run['query_id'] = run['query_id'].astype(str)
    run['doc_id'] = run['doc_id'].astype(str)
    run['score'] = run['score'].astype(float)
    
    queries_with_qrels = set(qrels['query_id'].unique())
    original_run_queries = set(run['query_id'].unique())
    run = run[run['query_id'].isin(queries_with_qrels)]
    
    run = run.sort_values(['query_id', 'score'], ascending=[True, False])
    
    if run.empty:
        print("\nWarning: No valid queries to evaluate after filtering!")
        return {metric: {'per_query': {}, 'mean': 0.0} for metric in metrics}
    
    ir_metrics = []
    for metric in metrics:
        if metric.startswith('ndcg@'):
            k = int(metric.split('@')[1])
            ir_metrics.append(ir_measures.nDCG(cutoff=k))
        elif metric.startswith('p@') or metric.startswith('P@'):
            k = int(metric.split('@')[1])
            ir_metrics.append(ir_measures.P(cutoff=k))
        elif metric.lower() in ['map', 'ap']:
            ir_metrics.append(ir_measures.AP)
        elif metric.startswith('rr@') or metric.startswith('RR@'):
            ir_metrics.append(ir_measures.RR(rel=2))
        elif metric.startswith('judged@'):
            k = int(metric.split('@')[1])
            ir_metrics.append(ir_measures.Judged(cutoff=k))
    
    evaluator = ir_measures.evaluator(ir_metrics, qrels)
    query_results = list(evaluator.iter_calc(run))
    
    all_queries = set(qrels['query_id'].unique())
    evaluated_queries = set(r.query_id for r in query_results)
    missing_queries = all_queries - evaluated_queries
    
    results = {}
    for metric in metrics:
        metric_results = [r for r in query_results if str(r.measure).lower() == metric.lower()]
        
        query_scores = {str(r.query_id): r.value for r in metric_results}
        for qid in missing_queries:
            query_scores[str(qid)] = 0.0
            
        results[metric] = {
            'per_query': query_scores,
            'mean': np.mean(list(query_scores.values())) if query_scores else 0.0
        }
    
    if output_dir:
        output_dir = ensure_dir(output_dir)
        with open(os.path.join(output_dir, 'evaluation_results.txt'), 'w') as f:
            f.write("Evaluation Results\n")
            f.write("==================\n\n")
            for metric, scores in results.items():
                f.write(f"\n{metric.upper()} Results:\n")
                f.write(f"Mean: {scores['mean']:.4f}\n")
                f.write("Per-query results:\n")
                for qid, score in sorted(scores['per_query'].items()):
                    f.write(f"  Query {qid}: {score:.4f}")
                    if qid in missing_queries:
                        f.write(" (no relevant documents)")
                    f.write("\n")
    
    return results 