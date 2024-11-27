import pyterrier as pt
import pandas as pd
from EvaluacionQPP.data.dataset_processor import DatasetProcessor
from EvaluacionQPP.indexing.index_builder import IndexBuilder
from EvaluacionQPP.metodos.pre_retrieval.idf import IDF
from EvaluacionQPP.metodos.post_retrieval.nqc import NQC  # Ensure NQC is properly imported
import os
import shutil
from typing import Union, Dict
from EvaluacionQPP.data.iquique_dataset import IquiqueDataset

def perform_retrieval(index, queries_df, dataset, method='BM25'):  # {{ edit_1 }}
    """
    Performs retrieval based on the specified method.

    Args:
        index: The PyTerrier index.
        queries_df (pd.DataFrame): DataFrame with columns ['qid', 'query'].
        method (str): Retrieval method to use.

    Returns:
        retrieval_results: DataFrame with retrieval results.
    """
    if method == 'BM25':
        bm25 = pt.BatchRetrieve(index, wmodel='BM25')
        retrieval_results = bm25.transform(queries_df)
        retrieval_results = retrieval_results.rename(columns={'score': 'docScore'})  # Rename here
        retrieval_results['doc_id'] = retrieval_results['doc_id'].astype(str)  # {{ edit_1 }}
        return retrieval_results
    elif method == 'QL':
        ql = pt.BatchRetrieve(index, wmodel='QL')
        retrieval_results = ql.transform(queries_df)
        retrieval_results = retrieval_results.rename(columns={'score': 'docScore'})  # Rename here
        retrieval_results['doc_id'] = retrieval_results['doc_id'].astype(str)  # {{ edit_1 }}
        return retrieval_results
    elif method == 'RM':
        rm = pt.BatchRetrieve(index, wmodel='RM3')
        retrieval_results = rm.transform(queries_df)
        retrieval_results = retrieval_results.rename(columns={'score': 'docScore'})  # Rename here
        retrieval_results['doc_id'] = retrieval_results['doc_id'].astype(str)  # {{ edit_1 }}
        return retrieval_results
    else:
        raise ValueError(f"Unknown retrieval method: {method}")

def perform_rm3_retrieval(
    index,
    queries_df: pd.DataFrame,
    dataset,
    fb_terms: int = 10,
    fb_docs: int = 10,
    original_weight: float = 0.5,
    num_results: int = 1000
) -> pd.DataFrame:
    """
    Performs retrieval using the RM3 (Relevance Model 3) method.
    """
    # Create simple RM3 pipeline
    bm25 = pt.BatchRetrieve(index, wmodel="BM25", num_results=num_results)
    rm3_pipe = bm25 >> pt.rewrite.RM3(index) >> bm25
    
    # Add text based on dataset type
    if isinstance(dataset, IquiqueDataset):
        def add_text(res):
            res['text'] = res['docno'].map(dataset.documents)
            return res
        rm3_pipe = rm3_pipe >> add_text
    else:
        rm3_pipe = rm3_pipe >> pt.text.get_text(dataset, "text")
    
    # Perform retrieval
    results = rm3_pipe.transform(queries_df)
    
    # Clean up column names first
    # Keep the first occurrence of each column name
    results = results.loc[:, ~results.columns.duplicated(keep='first')]
    
    # Rename columns
    column_mapping = {
        'score': 'docScore',
        'docid': 'docno'  # Map docid to docno if it exists
    }
    # Only rename columns that exist
    for old_col, new_col in column_mapping.items():
        if old_col in results.columns and new_col not in results.columns:
            results = results.rename(columns={old_col: new_col})
    
    # Ensure docno is string type
    if 'docno' in results.columns:
        results['docno'] = results['docno'].astype(str)
    
    # Reset index to ensure proper ordering
    results = results.reset_index(drop=True)
    
    print("\nDEBUG - RM3 results:")
    print("Columns:", results.columns.tolist())
    print("Sample results:")
    print(results[['qid', 'docno', 'docScore']].head())
    
    return results

def get_batch_scores(
    dataset,
    queries_df: pd.DataFrame,
    index: Union[pt.IndexFactory, str], 
    method: str = 'BM25',
    num_results: int = 1000,
    controls: Dict = None
) -> pd.DataFrame:
    """
    Get retrieval scores for multiple queries using PyTerrier
    """
    # Convert string path to index if needed
    if isinstance(index, str):
        index = pt.IndexFactory.of(index)
    
    # Set default controls if none provided
    if controls is None:
        controls = {
            'BM25': {'k1': 1.2, 'b': 0.75},
            'DirichletLM': {'mu': 2500},
            'TF_IDF': {},
            'PL2': {'c': 1.0},
            'RM3': {
                'fb_terms': 10,
                'fb_docs': 10,
                'original_weight': 0.5
            }
        }
    
    # Initialize retrieval model based on method
    if method == 'BM25':
        retriever = pt.BatchRetrieve(
            index, 
            wmodel='BM25',
            controls=controls.get('BM25', {}),
            num_results=num_results
        )
    elif method == 'TF_IDF':
        retriever = pt.BatchRetrieve(
            index, 
            wmodel='TF_IDF',
            controls=controls.get('TF_IDF', {}),
            num_results=num_results
        )
    elif method == 'DirichletLM':
        retriever = pt.BatchRetrieve(
            index, 
            wmodel='DirichletLM',
            controls=controls.get('DirichletLM', {}),
            num_results=num_results
        )
    elif method == 'RM3':
        rm3_controls = controls.get('RM3', {})
        return perform_rm3_retrieval(
            index,
            queries_df,
            dataset,
            fb_terms=rm3_controls.get('fb_terms', 10),
            fb_docs=rm3_controls.get('fb_docs', 10),
            original_weight=rm3_controls.get('original_weight', 0.5),
            num_results=num_results
        )
    else:
        raise ValueError(f"Unsupported retrieval method: {method}")

    # Get initial results
    results = retriever.transform(queries_df)
    
    # Add text based on dataset type
    if isinstance(dataset, IquiqueDataset):
        # For IquiqueDataset, directly map docno to text
        results['text'] = results['docno'].map(dataset.documents)
    else:
        # Use the getter of text from PyTerrier for other datasets
        text_getter = pt.text.get_text(dataset, "text")
        results = text_getter.transform(results)
    
    # Clean up column names first
    # Keep the first occurrence of each column name
    results = results.loc[:, ~results.columns.duplicated(keep='first')]
    
    # Rename columns
    column_mapping = {
        'score': 'docScore',
        'docid': 'docno'  # Map docid to docno if it exists
    }
    # Only rename columns that exist
    for old_col, new_col in column_mapping.items():
        if old_col in results.columns and new_col not in results.columns:
            results = results.rename(columns={old_col: new_col})
    
    # Ensure docno is string type
    if 'docno' in results.columns:
        results['docno'] = results['docno'].astype(str)
    
    # Reset index to ensure proper ordering
    results = results.reset_index(drop=True)
    
    print(f"\nDEBUG - {method} results:")
    print("Columns:", results.columns.tolist())
    print("Sample results:")
    print(results[['qid', 'docno', 'docScore']].head())
    
    return results
