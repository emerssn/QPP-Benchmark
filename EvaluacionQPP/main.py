import nltk

import argparse

import sys



nltk.download('punkt')



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



from EvaluacionQPP.metodos.pre_retrieval.scq import SCQ







import os



import shutil



import pandas as pd 



import json















def main():







    # Add argument parsing



    parser = argparse.ArgumentParser(description='Run QPP evaluation on a specified dataset')



    parser.add_argument('--dataset', type=str, default="irds:antique/test",



                       help='Dataset identifier (default: irds:antique/test)')



    



    args = parser.parse_args()



    dataset_name = args.dataset







    # Initialize PyTerrier



    if not pt.started():



        pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])



    



    # Initialize DatasetProcessor with the specified dataset



    try:



        dataset_processor = DatasetProcessor(dataset_name)



    except Exception as e:



        print(f"Error loading dataset {dataset_name}: {e}")



        sys.exit(1)



    



    # Get the absolute path for the index directory



    script_dir = os.path.dirname(os.path.abspath(__file__))



    # Use dataset name in index path (sanitize it first)



    safe_dataset_name = dataset_name.replace(':', '_').replace('/', '_')



    index_path = os.path.join(script_dir, "indices", safe_dataset_name)



    



    # Initialize IndexBuilder and build/load index



    index_builder = IndexBuilder(dataset_processor)



    index = index_builder.load_or_build_index(index_path)







    # Save sample term frequencies to JSON



    sample_file_path = os.path.join(script_dir, "sample_term_frequencies.json")



    index_builder.save_sample_frequencies_to_json(sample_file_path)



    



    # Initialize IDF and SCQ



    idf = IDF(index)



    scq = SCQ(index)



    



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



    



    # Compute IDF scores (both max and avg)



    idf_scores_avg = idf.compute_scores_batch(queries, method='avg')



    idf_scores_max = idf.compute_scores_batch(queries, method='max')



    



    # Compute SCQ scores (all three methods)



    scq_scores_avg = scq.compute_scores_batch(queries, method='avg')



    scq_scores_max = scq.compute_scores_batch(queries, method='max')


    
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






































