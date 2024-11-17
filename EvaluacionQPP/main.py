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

def main():
    parser = argparse.ArgumentParser(description='Run QPP evaluation on a specified dataset')
    parser.add_argument('--dataset', type=str, default="antique_test",
                       choices=AVAILABLE_DATASETS.keys(),
                       help='Dataset identifier (default: antique_test)')
    
    args = parser.parse_args()
    dataset_name = AVAILABLE_DATASETS[args.dataset]
    safe_dataset_name = args.dataset

    if not pt.started():
        pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])
    
    try:
        dataset_processor = DatasetProcessor(dataset_name)
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        sys.exit(1)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(script_dir, "indices", safe_dataset_name)
    
    # Build or load index
    index_builder = IndexBuilder(dataset_processor, safe_dataset_name)
    index = index_builder.load_or_build_index(index_path)

    # Save term frequencies for analysis
    sample_file_path = os.path.join(script_dir, "sample_term_frequencies.json")
    index_builder.save_sample_frequencies_to_json(sample_file_path)
    
    # Get queries
    try:
        queries = dataset_processor.get_queries()
        print(f"Total queries: {len(queries)}")
        print("Sample of queries:")
        for qid, query in list(queries.items())[:5]:
            print(f"  Query ID: {qid}, Query: {query}")
    except ValueError as e:
        print(f"Error getting queries: {e}")
        return
    
    # Prepare queries DataFrame
    queries_df = pd.DataFrame(list(queries.items()), columns=['qid', 'query'])
    
    # Get retrieval results
    retrieval_results = get_batch_scores(
        queries_df=queries_df,
        index=index,
        dataset=dataset_processor.dataset,
        method='BM25',
        num_results=1000,
        controls={
            'BM25': {'k1': 1.5, 'b': 0.8}
        }
    )
    
    # Get RM3 results for UEF method
    rm_results_df = get_batch_scores(
        queries_df=queries_df,
        index=index,
        dataset=dataset_processor.dataset,
        method='RM3',
        num_results=1000,
        controls={
            'RM3': {
                'fb_terms': 10,
                'fb_docs': 10,
                'original_weight': 0.5
            }
        }
    )
    
    # Create QPP factory and compute all scores
    qpp_factory = QPPMethodFactory(
        index_builder=index_builder,
        retrieval_results=retrieval_results,
        rm_results=rm_results_df,
        dataset_name=dataset_name
    )
    
    qpp_scores = qpp_factory.compute_all_scores(
        queries,
        list_size_param=10
    )
    
    # Print results
    for query_id, scores in qpp_scores.items():
        print(f"\nQuery ID: {query_id}")
        print(f"Query: {queries[query_id]}")
        for method, score in scores.items():
            print(f"{method.upper()} Score: {score:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    main()





































