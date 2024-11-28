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
    # Get dataset format configuration
    dataset_config = DATASET_FORMATS.get(dataset_name, DATASET_FORMATS["antique_test"])
    
    # First clean up any duplicate columns in input DataFrames
    qrels = qrels_df.loc[:, ~qrels_df.columns.duplicated()]
    run = results_df.loc[:, ~results_df.columns.duplicated()]
    
    print("\nDEBUG: Initial column names:")
    print("Qrels columns:", qrels.columns.tolist())
    print("Run columns:", run.columns.tolist())
    
    # Prepare qrels and run data using dataset-specific column mappings
    qrels = qrels.rename(columns=dataset_config["qrels_columns"])
    
    # For run data, first ensure we have the right columns
    if 'docno' in run.columns:
        run = run.rename(columns={'docno': 'doc_id'})
    elif 'docid' in run.columns:
        run = run.rename(columns={'docid': 'doc_id'})
    
    if 'qid' in run.columns:
        run = run.rename(columns={'qid': 'query_id'})
    
    if 'docScore' in run.columns:
        run = run.rename(columns={'docScore': 'score'})
    
    print("\nDEBUG: After column renaming:")
    print("Qrels columns:", qrels.columns.tolist())
    print("Run columns:", run.columns.tolist())
    
    # Apply dataset-specific document ID transformation
    print("\nDEBUG: Before doc_id transformation:")
    print("Sample qrels doc_ids:", qrels['doc_id'].head().tolist())
    print("Sample run doc_ids:", run['doc_id'].head().tolist())
    
    # Apply document ID transformations based on dataset configuration
    qrels['doc_id'] = qrels['doc_id'].astype(str).apply(dataset_config["doc_id_transform"])
    run['doc_id'] = run['doc_id'].astype(str).apply(dataset_config["run_doc_id_transform"])
    
    print("\nDEBUG: After doc_id transformation:")
    print("Sample qrels doc_ids:", qrels['doc_id'].head().tolist())
    print("Sample run doc_ids:", run['doc_id'].head().tolist())
    
    # Ensure correct data types
    qrels['query_id'] = qrels['query_id'].astype(str)
    qrels['doc_id'] = qrels['doc_id'].astype(str)
    qrels['relevance'] = qrels['relevance'].astype(int)
    qrels['relevance'] = qrels['relevance'].clip(lower=0)
    
    run['query_id'] = run['query_id'].astype(str)
    run['doc_id'] = run['doc_id'].astype(str)
    run['score'] = run['score'].astype(float)
    
    # Print sample of final dataframes
    print("\nDEBUG: Final dataframes sample:")
    print("\nQrels:")
    print(qrels.head())
    print("\nRun:")
    print(run.head())
    
    # Check document overlap for each query
    print("\nDEBUG: Document overlap check for each query:")
    for qid in qrels['query_id'].unique():
        qrels_docs = set(qrels[qrels['query_id'] == qid]['doc_id'])
        run_docs = set(run[run['query_id'] == qid]['doc_id'])
        #print(f"\nQuery {qid}:")
        #print(f"Qrels docs: {sorted(list(qrels_docs))[:5]}")
        #print(f"Run docs: {sorted(list(run_docs))[:5]}")
        #print(f"Overlapping docs: {sorted(list(qrels_docs.intersection(run_docs)))}")
    
    # Filter run to only include queries that have qrels
    queries_with_qrels = set(qrels['query_id'].unique())
    original_run_queries = set(run['query_id'].unique())
    run = run[run['query_id'].isin(queries_with_qrels)]
    
    # Check document overlap for a sample query
    if queries_with_qrels.intersection(original_run_queries):
        sample_query = list(queries_with_qrels.intersection(original_run_queries))[0]
        qrels_docs = set(qrels[qrels['query_id'] == sample_query]['doc_id'])
        run_docs = set(run[run['query_id'] == sample_query]['doc_id'])
        print(f"\nDocument overlap check for query {sample_query}:")
        print(f"Number of relevant docs: {len(qrels_docs)}")
        print(f"Number of retrieved docs: {len(run_docs)}")
        print(f"Sample of relevant docs: {list(qrels_docs)[:5]}")
        print(f"Sample of retrieved docs: {list(run_docs)[:5]}")
        print(f"Number of overlapping docs: {len(qrels_docs.intersection(run_docs))}")
    else:
        print("\nNo overlapping queries between qrels and run.")
    
    # Print filtering statistics
    print(f"\nFiltering Statistics:")
    print(f"Queries in run but not in qrels: {len(original_run_queries - queries_with_qrels)}")
    print(f"Queries in qrels but not in run: {len(queries_with_qrels - original_run_queries)}")
    print(f"Queries with both run and qrels: {len(queries_with_qrels.intersection(original_run_queries))}")
    
    # Sort results by score (descending) for each query
    run = run.sort_values(['query_id', 'score'], ascending=[True, False])
    
    if run.empty:
        print("\nWarning: No valid queries to evaluate after filtering!")
        return {metric: {'per_query': {}, 'mean': 0.0} for metric in metrics}
    
    # Parse metrics into ir_measures format
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
    
    # Add this right before the ir_measures evaluation
    print("\nDEBUG: Final document ID formats:")
    print("Sample qrels:")
    print(qrels[['query_id', 'doc_id', 'relevance']].head())
    print("\nUnique qrels doc_id formats:", sorted(qrels['doc_id'].unique())[:5])
    print("\nSample run:")
    print(run[['query_id', 'doc_id', 'score']].head())
    print("\nUnique run doc_id formats:", sorted(run['doc_id'].unique())[:5])
    
    # Create evaluator and calculate results
    evaluator = ir_measures.evaluator(ir_metrics, qrels)
    query_results = list(evaluator.iter_calc(run))
    
    # Handle queries with no relevant documents
    all_queries = set(qrels['query_id'].unique())
    evaluated_queries = set(r.query_id for r in query_results)
    missing_queries = all_queries - evaluated_queries
    
    # Organize results by metric
    results = {}
    for metric in metrics:
        metric_results = [r for r in query_results if str(r.measure).lower() == metric.lower()]
        
        # Add zero scores for queries with no relevant documents
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