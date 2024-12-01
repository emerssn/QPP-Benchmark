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
        "run_doc_id_transform": lambda x: x.replace('doc', '').split('_')[0],
        "relevance_levels": {
            1: "Out of context/nonsensical",
            2: "Not relevant but on topic",
            3: "Marginally relevant",
            4: "Highly relevant"
        },
        "binary_threshold": 3,  # Scores >= 3 are considered relevant for binary metrics
        "gain_values": {  # Custom gain values for nDCG
            1: 0.0,
            2: 0.0,
            3: 0.5,
            4: 1.0
        }
    },
    "iquique_dataset": {
        "doc_id_transform": lambda x: x if x.startswith('doc') else f"doc{x}",
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
        "run_doc_id_transform": lambda x: x if x.startswith('doc') else f"doc{x}",
        "relevance_levels": {
            0: "Not Relevant",
            1: "Relevant",
            2: "Highly Relevant"
        },
        "binary_threshold": 1,  # Scores >= 1 are considered relevant for binary metrics
        "gain_values": {  # Updated gain values for nDCG
            0: 0,     # Not Relevant
            1: 1,     # Relevant
            2: 2      # Highly Relevant - using linear scale for better NDCG calculation
        }
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
    
    # Create copies of dataframes to avoid modifying originals
    qrels = qrels_df.copy()
    run = results_df.copy()
    
    # Rename columns to match ir_measures expectations
    qrels = qrels.rename(columns=dataset_config["qrels_columns"])
    run = run.rename(columns=dataset_config["run_columns"])
    
    # Transform document IDs
    qrels['doc_id'] = qrels['doc_id'].astype(str).apply(dataset_config["doc_id_transform"])
    run['doc_id'] = run['doc_id'].astype(str).apply(dataset_config["run_doc_id_transform"])
    
    # Convert types
    qrels['query_id'] = qrels['query_id'].astype(str)
    qrels['doc_id'] = qrels['doc_id'].astype(str)
    qrels['relevance'] = qrels['relevance'].astype(int)
    
    run['query_id'] = run['query_id'].astype(str)
    run['doc_id'] = run['doc_id'].astype(str)
    run['score'] = run['score'].astype(float)
    
    # Filter out queries not in qrels
    valid_queries = set(qrels['query_id'].unique())
    run = run[run['query_id'].isin(valid_queries)]
    
    if run.empty:
        return {metric: {'per_query': {}, 'mean': 0.0} for metric in metrics}
    
    # Sort results by score in descending order within each query
    run = run.sort_values(['query_id', 'score'], ascending=[True, False])
    
    # Initialize metrics
    results = {}
    metric_objects = []
    
    # Create metric objects with appropriate parameters
    for metric in metrics:
        if metric.startswith('ndcg@'):
            k = int(metric.split('@')[1])
            metric_objects.append((metric, ir_measures.nDCG(cutoff=k)))
        elif metric.lower() in ['map', 'ap']:
            # Use explicit relevance threshold for MAP/AP
            metric_objects.append((metric, ir_measures.AP(rel=1)))
        elif metric.startswith('p@'):
            k = int(metric.split('@')[1])
            metric_objects.append((metric, ir_measures.P(cutoff=k, rel=1)))
        elif metric.startswith('rr@'):
            metric_objects.append((metric, ir_measures.RR(rel=1)))
    
    # Debug information
    print(f"\nDebug - Number of queries in qrels: {len(valid_queries)}")
    print(f"Debug - Number of documents in run: {len(run)}")
    print(f"Debug - Relevance judgments distribution:\n{qrels['relevance'].value_counts()}")
    
    # Create evaluator for all metrics
    evaluator = ir_measures.evaluator([m for _, m in metric_objects], qrels)
    all_results = list(evaluator.iter_calc(run))
    
    # Process results for each metric
    for metric_name, metric_obj in metric_objects:
        metric_results = [r for r in all_results if str(r.measure).lower() == metric_name.lower()]
        
        # Debug information for each metric
        print(f"\nDebug - {metric_name} results count: {len(metric_results)}")
        
        query_scores = {str(r.query_id): r.value for r in metric_results}
        
        # For NDCG metrics, we let ir_measures handle the normalization
        results[metric_name] = {
            'per_query': query_scores,
            'mean': np.mean(list(query_scores.values())) if query_scores else 0.0
        }
    
    # Write results if output directory specified
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
                    f.write(f"  Query {qid}: {score:.4f}\n")
    
    return results 