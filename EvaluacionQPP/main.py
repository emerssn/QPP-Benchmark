import nltk
import argparse
import sys
import os
import pyterrier as pt
import pandas as pd 

from EvaluacionQPP.data.dataset_processor import DatasetProcessor
from EvaluacionQPP.indexing.index_builder import IndexBuilder
from EvaluacionQPP.metodos.pre_retrieval.idf import IDF
from EvaluacionQPP.metodos.post_retrieval.nqc import NQC
from EvaluacionQPP.metodos.post_retrieval.wig import WIG
from EvaluacionQPP.metodos.post_retrieval.clarity import Clarity
from EvaluacionQPP.retrieval.retrieval import get_batch_scores
from EvaluacionQPP.utils.text_processing import preprocess_text
from EvaluacionQPP.metodos.post_retrieval.uef import UEF
from EvaluacionQPP.metodos.pre_retrieval.scq import SCQ

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
    
    index_builder = IndexBuilder(dataset_processor, safe_dataset_name)
    index = index_builder.load_or_build_index(index_path)

    sample_file_path = os.path.join(script_dir, "sample_term_frequencies.json")
    index_builder.save_sample_frequencies_to_json(sample_file_path)
    
    idf = IDF(index)
    scq = SCQ(index)
    
    try:
        queries = dataset_processor.get_queries()
        print(f"Total queries: {len(queries)}")
        print("Sample of queries:")
        for qid, query in list(queries.items())[:5]:
            print(f"  Query ID: {qid}, Query: {query}")
    except ValueError as e:
        print(f"Error getting queries: {e}")
        return
    
    queries_df = pd.DataFrame(list(queries.items()), columns=['qid', 'query'])
    
    idf_scores_avg = idf.compute_scores_batch(queries, method='avg')
    idf_scores_max = idf.compute_scores_batch(queries, method='max')
    
    scq_scores_avg = scq.compute_scores_batch(queries, method='avg')
    scq_scores_max = scq.compute_scores_batch(queries, method='max')
    
    processed_queries = {qid: preprocess_text(query_text) for qid, query_text in queries.items()}
    
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
    
    nqc = NQC(index_builder, retrieval_results)
    wig = WIG(index_builder, retrieval_results)
    clarity = Clarity(index_builder, retrieval_results)
    uef = UEF(index_builder, retrieval_results, rm_results_df)
    
    nqc_scores = nqc.compute_scores_batch(processed_queries, list_size_param=10)
    wig_scores = wig.compute_scores_batch(processed_queries, list_size_param=10)
    clarity_scores = clarity.compute_scores_batch(processed_queries)
    
    uef_wig_scores = uef.compute_scores_batch(processed_queries, wig_scores)
    uef_nqc_scores = uef.compute_scores_batch(processed_queries, nqc_scores)
    
    for query_id in queries.keys():
        print(f"\nQuery ID: {query_id}")
        print(f"Query: {queries[query_id]}")
        print(f"IDF Score (avg): {idf_scores_avg.get(query_id, 0.0):.4f}")
        print(f"IDF Score (max): {idf_scores_max.get(query_id, 0.0):.4f}")
        print(f"SCQ Score (avg): {scq_scores_avg.get(query_id, 0.0):.4f}")
        print(f"SCQ Score (max): {scq_scores_max.get(query_id, 0.0):.4f}")
        print(f"NQC Score: {nqc_scores.get(query_id, 0.0):.4f}")
        print(f"WIG Score: {wig_scores.get(query_id, 0.0):.4f}")
        print(f"Clarity Score: {clarity_scores.get(query_id, 0.0):.4f}")
        print(f"UEF-WIG Score: {uef_wig_scores.get(query_id, 0.0):.4f}")
        print(f"UEF-NQC Score: {uef_nqc_scores.get(query_id, 0.0):.4f}")
        print("-" * 50)

if __name__ == "__main__":
    main()





































