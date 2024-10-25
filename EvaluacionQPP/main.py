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

    

    # Perform retrieval using the get_batch_scores function
    retrieval_results = get_batch_scores(
        queries_df=queries_df,
        index=index,
        dataset=dataset_processor.dataset,
        method='BM25',
        num_results=1000,
        controls={
            'BM25': {'k1': 1.5, 'b': 0.8},
            'TF_IDF': {},
            'DirichletLM': {'mu': 2000},
            'PL2': {'c': 1.2}
        }
    ) 

    

    # Debug: Print retrieval results columns

    print(f"Retrieval Results Columns: {retrieval_results.columns.tolist()}")

    

    # Optional: Inspect the first few retrieval results

    print(retrieval_results.head())

    

    # Initialize NQC with retrieval results

    nqc = NQC(index_builder, retrieval_results)


    clarity = Clarity(index_builder, retrieval_results)

    # Prepare a dictionary of processed queries for batch scoring

    processed_queries = {qid: preprocess_text(query_text) for qid, query_text in queries.items()}

    # Compute NQC scores in batch

    nqc_scores = nqc.compute_scores_batch(processed_queries, list_size_param=10)
    
    # Compute Clarity scores in batch
    clarity_scores = clarity.compute_scores_batch(processed_queries)

    # Initialize WIG with retrieval results

    wig = WIG(index_builder, retrieval_results)

    # Compute WIG scores in batch

    wig_scores = wig.compute_scores_batch(processed_queries, list_size_param=10)

    # Compute and print IDF scores, NQC scores, WIG scores, and Clarity scores for each query

    for query_id in queries.keys():

        query_text = queries[query_id]

        processed_query = processed_queries[query_id]

        try:

            if not processed_query:

                print(f"\nQuery ID: {query_id} has no valid terms after preprocessing.")

                print("-" * 50)

                continue



            max_idf_score = idf.compute_score(query_text, method='max')

            avg_idf_score = idf.compute_score(query_text, method='avg')

            nqc_score = nqc_scores.get(query_id, 0.0)

            wig_score = wig_scores.get(query_id, 0.0)

            clarity_score = clarity_scores.get(query_id, 0.0)

            print(f"\nQuery ID: {query_id}")

            print(f"Query: {query_text}")

            print(f"Processed Query Terms: {processed_query}")

            print(f"Max IDF Score: {max_idf_score:.4f}")

            print(f"Avg IDF Score: {avg_idf_score:.4f}")

            print(f"NQC Score: {nqc_score:.4f}")

            print(f"WIG Score: {wig_score:.4f}")

            print(f"Clarity Score: {clarity_score:.4f}")

            print("-" * 50)

        except Exception as e:

            print(f"Error processing query {query_id}: {e}")


if __name__ == "__main__":

    main()









