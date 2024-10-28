import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')

import pyterrier as pt

from EvaluacionQPP.data.dataset_processor import DatasetProcessor
from EvaluacionQPP.indexing.index_builder import IndexBuilder
from EvaluacionQPP.metodos.pre_retrieval.idf import IDF
from EvaluacionQPP.metodos.post_retrieval.nqc import NQC
from EvaluacionQPP.metodos.post_retrieval.wig import WIG
from EvaluacionQPP.metodos.post_retrieval.clarity import Clarity
from EvaluacionQPP.retrieval.retrieval import get_batch_scores
from EvaluacionQPP.utils.text_processing import preprocess_text
from EvaluacionQPP.metodos.post_retrieval.uef import UEF

import os
import shutil
import pandas as pd 
import json



def main():

    # Initialize PyTerrier

    if not pt.started():

        pt.init()

    

    # Initialize DatasetProcessor

    dataset_processor = DatasetProcessor("irds:antique/test")

    

    # Get the absolute path for the index directory

    script_dir = os.path.dirname(os.path.abspath(__file__))

    index_path = os.path.join(script_dir, "indices", "antique")

    

    # Initialize IndexBuilder and build/load index

    index_builder = IndexBuilder(dataset_processor)

    index = index_builder.load_or_build_index(index_path)

    # Save sample term frequencies to JSON
    sample_file_path = os.path.join(script_dir, "sample_term_frequencies.json")
    index_builder.save_sample_frequencies_to_json(sample_file_path)

    

    # Initialize IDF

    idf = IDF(index)
    

    # Get queries

    try:

        queries = dataset_processor.get_queries()

        print(f"Total queries: {len(queries)}")

        print("Sample of queries:")

        for qid, query in list(queries.items())[:5]:  # Print first 5 queries as a sample

            print(f"  Query ID: {qid}, Query: {query}")

    except ValueError as e:

        print(f"Error getting queries: {e}")

        return

    

    # Convert queries dictionary to DataFrame

    queries_df = pd.DataFrame(list(queries.items()), columns=['qid', 'query'])

    

    # Process queries

    processed_queries = {qid: preprocess_text(query_text) for qid, query_text in queries.items()}

    

    # Get original BM25 results
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

    # Get RM3 re-ranked results
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

    # Initialize predictors
    nqc = NQC(index_builder, retrieval_results)
    wig = WIG(index_builder, retrieval_results)
    clarity = Clarity(index_builder, retrieval_results)
    uef = UEF(index_builder, retrieval_results, rm_results_df)

    # Compute predictor scores
    nqc_scores = nqc.compute_scores_batch(processed_queries, list_size_param=10)
    wig_scores = wig.compute_scores_batch(processed_queries, list_size_param=10)
    clarity_scores = clarity.compute_scores_batch(processed_queries)

    # Compute UEF scores
    uef_wig_scores = uef.compute_scores_batch(processed_queries, wig_scores)
    uef_nqc_scores = uef.compute_scores_batch(processed_queries, nqc_scores)

    # Print all scores
    for query_id in queries.keys():
        print(f"\nQuery ID: {query_id}")
        print(f"Query: {queries[query_id]}")
        print(f"NQC Score: {nqc_scores.get(query_id, 0.0):.4f}")
        print(f"WIG Score: {wig_scores.get(query_id, 0.0):.4f}")
        print(f"Clarity Score: {clarity_scores.get(query_id, 0.0):.4f}")
        print(f"UEF-WIG Score: {uef_wig_scores.get(query_id, 0.0):.4f}")
        print(f"UEF-NQC Score: {uef_nqc_scores.get(query_id, 0.0):.4f}")
        print("-" * 50)

if __name__ == "__main__":

    main()








