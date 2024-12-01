import nltk
import argparse
import sys
import os
import pyterrier as pt
import pandas as pd 

from EvaluacionQPP.data.dataset_processor import DatasetProcessor
from EvaluacionQPP.indexing.index_builder import IndexBuilder
from EvaluacionQPP.metodos.qpp_factory import QPPMethodFactory
from EvaluacionQPP.retrieval.retrieval import get_batch_scores
from EvaluacionQPP.evaluation.evaluator import evaluate_results
from EvaluacionQPP.evaluation.correlation_analyzer import QPPCorrelationAnalyzer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

AVAILABLE_DATASETS = {
    "antique_test": "irds:antique/test",
    "iquique_small": "iquique_dataset",
    # Add more datasets as needed
    # "msmarco": "irds:msmarco/...",
    # "trec": "irds:trec/...",
}

AVAILABLE_METRICS = ['ndcg@5', 'ndcg@10', 'ndcg@20', 'ap']
AVAILABLE_CORRELATIONS = ['kendall', 'spearman', 'pearson']

def process_dataset(dataset_name: str, dataset_path: str, args) -> QPPCorrelationAnalyzer:
    """Process a single dataset and return its correlation analyzer."""
    print(f"\nProcessing dataset: {dataset_name}")
    
    if not pt.started():
        pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])
    
    try:
        dataset_processor = DatasetProcessor(dataset_path)
    except Exception as e:
        print(f"Error loading dataset {dataset_path}: {e}")
        return None
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(script_dir, "indices", dataset_name)
    
    # Build or load index
    index_builder = IndexBuilder(dataset_processor, dataset_name)
    index = index_builder.load_or_build_index(index_path)
    
    # Save term frequencies for analysis (as in old version)
    sample_file_path = os.path.join(script_dir, "sample_term_frequencies.json")
    index_builder.save_sample_frequencies_to_json(sample_file_path)
    
    # Get queries
    try:
        queries = dataset_processor.get_queries()
        if args.max_queries:
            query_ids = list(queries.keys())[:args.max_queries]
            queries = {qid: queries[qid] for qid in query_ids}
    except ValueError as e:
        print(f"Error getting queries: {e}")
        return None
    
    # Prepare queries DataFrame
    queries_df = pd.DataFrame(list(queries.items()), columns=['qid', 'query'])
    
    # Get retrieval results (matching old version's parameters)
    retrieval_results = get_batch_scores(
        queries_df=queries_df,
        index=index,
        dataset=dataset_processor.dataset,
        method='BM25',
        num_results=1000,  # Fixed value as in old version
        controls={
            'BM25': {'k1': 1.5, 'b': 0.8}
        }
    )
    
    # Get RM3 results for UEF method (matching old version's parameters)
    rm_results_df = None
    if args.use_uef:
        rm_results_df = get_batch_scores(
            queries_df=queries_df,
            index=index,
            dataset=dataset_processor.dataset,
            method='RM3',
            num_results=1000,  # Fixed value as in old version
            controls={
                'RM3': {
                    'fb_terms': 10,
                    'fb_docs': 10,
                    'original_weight': 0.5
                }
            }
        )
    
    # Get qrels
    qrels = dataset_processor.get_qrels()
    
    # Use both nDCG and AP metrics by default
    evaluation_metrics = ['ndcg@10', 'ap']
    if args.metrics:  # Add any additional metrics requested, but only if they're nDCG or AP
        additional_metrics = [m for m in args.metrics 
                            if m not in evaluation_metrics and 
                            (m.startswith('ndcg@') or m == 'ap')]
        evaluation_metrics.extend(additional_metrics)
    
    # Evaluate retrieval results
    evaluation_results = evaluate_results(
        qrels_df=qrels,
        results_df=retrieval_results,
        metrics=evaluation_metrics,
        output_dir=os.path.join(script_dir, "evaluation_results"),
        dataset_name=dataset_path
    )
    
    # Print evaluation results (as in old version)
    print("\nRetrieval Evaluation Results:")
    for metric, scores in evaluation_results.items():
        print(f"\n{metric.upper()}:")
        print(f"Mean: {scores['mean']:.4f}")
        print("Sample of per-query scores:")
        for qid, score in list(scores['per_query'].items())[:5]:
            print(f"  Query {qid}: {score:.4f}")
    
    # Create QPP factory and compute scores
    qpp_factory = QPPMethodFactory(
        index_builder=index_builder,
        retrieval_results=retrieval_results,
        rm_results=rm_results_df,
        dataset_name=dataset_path
    )
    
    qpp_scores = qpp_factory.compute_all_scores(
        queries=queries,
        list_size_param=args.list_size
    )
    
    # Create and return correlation analyzer
    return QPPCorrelationAnalyzer(
        qpp_scores=qpp_scores,
        retrieval_metrics=evaluation_results,
        output_dir=os.path.join(script_dir, "correlation_analysis", dataset_name)
    )

def main():
    parser = argparse.ArgumentParser(description='Run QPP evaluation on specified datasets')
    
    # Dataset selection
    parser.add_argument('--datasets', nargs='+', choices=AVAILABLE_DATASETS.keys(),
                       default=list(AVAILABLE_DATASETS.keys()),
                       help='Datasets to process (default: all)')
    
    # Query processing
    parser.add_argument('--max-queries', type=int, default=None,
                       help='Maximum number of queries to process per dataset')
    parser.add_argument('--list-size', type=int, default=10,
                       help='List size parameter for QPP methods (default: 10)')
    parser.add_argument('--num-results', type=int, default=1000,
                       help='Number of results to retrieve (default: 1000)')
    
    # Evaluation options
    parser.add_argument('--metrics', nargs='+', choices=AVAILABLE_METRICS,
                       default=['ndcg@10', 'ap'],
                       help='Evaluation metrics to use (default: ndcg@10 and ap)')
    parser.add_argument('--correlations', nargs='+', choices=AVAILABLE_CORRELATIONS,
                       default=['kendall'],
                       help='Correlation coefficients to compute (default: kendall)')
    
    # Analysis options
    parser.add_argument('--use-uef', action='store_true',
                       help='Enable UEF-based QPP methods')
    parser.add_argument('--skip-plots', action='store_true',
                       help='Skip generating plots')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Custom output directory for results')
    
    args = parser.parse_args()
    
    # Process datasets
    dataset_analyzers = {}
    for dataset_name in args.datasets:
        dataset_path = AVAILABLE_DATASETS[dataset_name]
        analyzer = process_dataset(dataset_name, dataset_path, args)
        if analyzer:
            dataset_analyzers[dataset_name] = analyzer
            
            if not args.skip_plots:
                # Generate individual dataset report and plots
                analyzer.generate_report(args.correlations)
                for corr_type in args.correlations:
                    analyzer.plot_correlations_boxplot(corr_type)
    
    # Generate cross-dataset analysis if multiple datasets
    if len(dataset_analyzers) > 1 and not args.skip_plots:
        output_dir = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "correlation_analysis")
        for corr_type in args.correlations:
            QPPCorrelationAnalyzer.plot_correlations_across_datasets(
                datasets=dataset_analyzers,
                correlation_type=corr_type,
                output_dir=output_dir
            )

if __name__ == "__main__":
    main()