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
from EvaluacionQPP.evaluation.evaluator import DATASET_FORMATS
import logging

def get_dataset_config(dataset):
    """Get the appropriate dataset configuration"""
    if isinstance(dataset, IquiqueDataset):
        return DATASET_FORMATS["iquique_dataset"]
    # Add more dataset type checks as needed
    return DATASET_FORMATS["antique_test"]

def perform_retrieval(index, queries_df, dataset, method='BM25'):
    """
    Performs retrieval based on the specified method.

    Args:
        index: The PyTerrier index.
        queries_df (pd.DataFrame): DataFrame with columns ['qid', 'query'].
        dataset: Dataset instance for handling document text and formatting.
        method (str): Retrieval method to use ('BM25', 'QL', or 'RM').

    Returns:
        pd.DataFrame: DataFrame with retrieval results containing columns:
            - qid: Query ID
            - query: Original query text
            - docno: Document ID
            - docScore: Retrieval score
            - rank: Document rank for the query
            For queries with no results, returns placeholder rows with 'EMPTY' docno and 0.0 score.
    """
    logger = logging.getLogger(__name__)
    
    # Ensure qid is string type
    queries_df = queries_df.copy()
    queries_df['qid'] = queries_df['qid'].astype(str)
    
    logger.info(f"Performing retrieval for {len(queries_df)} queries using {method}")
    
    # Get dataset configuration
    dataset_config = get_dataset_config(dataset)
    
    # Configure retrieval parameters with better defaults
    retrieval_params = {
        'wmodel': method,
        'verbose': True,
        'num_results': 1000,  # Ensure we get enough results per query
        'controls': {
            'BM25': {'k1': 1.2, 'b': 0.75},  # Standard BM25 parameters
            'QL': {'c': 2500},  # Dirichlet smoothing parameter
            'RM3': {'fb.terms': 10, 'fb.docs': 10, 'fb.original': 0.5}  # RM3 parameters
        }
    }
    
    try:
        # Initialize retriever with appropriate parameters
        if method in ['BM25', 'QL', 'RM']:
            retriever = pt.BatchRetrieve(
                index, 
                wmodel=method,
                controls=retrieval_params['controls'].get(method, {}),
                num_results=retrieval_params['num_results'],
                verbose=True
            )
        else:
            raise ValueError(f"Unknown retrieval method: {method}")

        # Perform retrieval
        retrieval_results = retriever.transform(queries_df)
        
        # Log retrieval statistics
        unique_queries = len(retrieval_results['qid'].unique())
        total_results = len(retrieval_results)
        avg_results = total_results / unique_queries if unique_queries > 0 else 0
        
        logger.info(
            f"Retrieved {total_results} results for {unique_queries}/{len(queries_df)} queries "
            f"(avg {avg_results:.1f} results per query)"
        )

        # Ensure consistent column names and types
        retrieval_results = retrieval_results.rename(columns={'score': 'docScore'})
        retrieval_results['qid'] = retrieval_results['qid'].astype(str)
        retrieval_results['docno'] = retrieval_results['docno'].astype(str)
        
        # Apply document ID transformation
        retrieval_results['docno'] = retrieval_results['docno'].apply(dataset_config["doc_id_transform"])
        
        # Check result counts per query
        results_per_query = retrieval_results.groupby('qid').size()
        queries_with_few_results = results_per_query[results_per_query < 10]
        
        if not queries_with_few_results.empty:
            logger.warning(
                f"Found {len(queries_with_few_results)} queries with fewer than 10 results: \n"
                f"Query counts: \n{queries_with_few_results.head()}"
            )
        
        return retrieval_results
        
    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        raise

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
    bm25 = pt.BatchRetrieve(index, wmodel="BM25", num_results=num_results)
    rm3_pipe = bm25 >> pt.rewrite.RM3(index) >> bm25
    
    if isinstance(dataset, IquiqueDataset):
        def add_text(res):
            res['text'] = res['docno'].map(dataset.documents)
            return res
        rm3_pipe = rm3_pipe >> add_text
    else:
        rm3_pipe = rm3_pipe >> pt.text.get_text(dataset, "text")
    
    results = rm3_pipe.transform(queries_df)
    
    results = results.loc[:, ~results.columns.duplicated(keep='first')]
    
    column_mapping = {
        'score': 'docScore',
        'docid': 'docno'
    }
    for old_col, new_col in column_mapping.items():
        if old_col in results.columns and new_col not in results.columns:
            results = results.rename(columns={old_col: new_col})
    
    if 'docno' in results.columns:
        results['docno'] = results['docno'].astype(str)
    
    results = results.reset_index(drop=True)
    
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
    
    # Get dataset configuration
    dataset_config = get_dataset_config(dataset)
    
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
    
    # Apply document ID transformation based on dataset configuration
    results['docno'] = results['docno'].astype(str).apply(dataset_config["doc_id_transform"])
    
    # Reset index to ensure proper ordering
    results = results.reset_index(drop=True)
    
    return results
