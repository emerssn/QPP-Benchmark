import pyterrier as pt
import pandas as pd
from EvaluacionQPP.data.dataset_processor import DatasetProcessor
from EvaluacionQPP.indexing.index_builder import IndexBuilder
from EvaluacionQPP.metodos.pre_retrieval.idf import IDF
from EvaluacionQPP.metodos.post_retrieval.nqc import NQC  # Ensure NQC is properly imported
import os
import shutil
from typing import Union, Dict

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
        return retrieval_results
    elif method == 'QL':
        ql = pt.BatchRetrieve(index, wmodel='QL')
        retrieval_results = ql.transform(queries_df)
        retrieval_results = retrieval_results.rename(columns={'score': 'docScore'})  # Rename here
        return retrieval_results
    elif method == 'RM':
        rm = pt.BatchRetrieve(index, wmodel='RM3')
        retrieval_results = rm.transform(queries_df)
        retrieval_results = retrieval_results.rename(columns={'score': 'docScore'})  # Rename here
        return retrieval_results
    else:
        raise ValueError(f"Unknown retrieval method: {method}")

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

    Parameters:
    -----------
    queries_df: pd.DataFrame
        DataFrame containing queries with columns 'qid' and 'query'
    index: Union[pt.Index, str]
        PyTerrier index or path to index
    method: str
        Retrieval method ('BM25', 'TF_IDF', 'DirichletLM', 'PL2')
    num_results: int
        Number of results per query
    controls: Dict
        Optional parameter controls for the retrieval model

    Returns:
    --------
    pd.DataFrame: Results with columns ['qid', 'docno', 'docScore', 'rank']
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
            'PL2': {'c': 1.0}
        }
    
    # Initialize retrieval model based on method
    if method == 'BM25':
        retriever = pt.terrier.Retriever(
            index, 
            wmodel='BM25',
            controls=controls.get('BM25', {}),
            num_results=num_results
        ) >> pt.text.get_text(dataset, "text")
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
    else:
        raise ValueError(f"Unsupported retrieval method: {method}")
    
    # Perform batch retrieval
    results = retriever.transform(queries_df)
    
    # Clean and format results
    results = results.rename(columns={
        'score': 'docScore',
        'docid': 'docno' if 'docno' not in results.columns else 'docno'
    })
    
    # Add rank column if not present
    if 'rank' not in results.columns:
        results['rank'] = results.groupby('qid').cumcount() + 1
    
    # Ensure consistent column order
    columns = ['qid', 'docno', 'docScore', 'rank']
    extra_cols = [col for col in results.columns if col not in columns]
    results = results[columns + extra_cols]
    
    return results
