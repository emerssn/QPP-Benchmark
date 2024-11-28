import numpy as np
import pandas as pd
from typing import Dict
from ..base import PostRetrievalMethod
from ...evaluation.evaluator import DATASET_FORMATS

class UEF(PostRetrievalMethod):
    def __init__(self, index_builder, retrieval_results, rm_results_df=None, dataset_name="antique_test"):
        super().__init__(index_builder, retrieval_results)
        self.rm_results_df = rm_results_df if rm_results_df is not None else retrieval_results
        self.dataset_config = DATASET_FORMATS.get(dataset_name, DATASET_FORMATS["antique_test"])
        
        # Clean up column names in both dataframes
        self.retrieval_results = self._clean_dataframe(self.retrieval_results)
        self.rm_results_df = self._clean_dataframe(self.rm_results_df)
    
    def _clean_dataframe(self, df):
        """Clean up DataFrame columns and ensure consistent naming."""
        # Remove duplicate columns
        df = df.loc[:, ~df.columns.duplicated(keep='first')]
        
        # Standardize column names
        column_mapping = {
            'docno': 'doc_id',
            'score': 'docScore'
        }
        
        # Only rename columns that exist
        df = df.rename(columns=column_mapping)
        
        # Ensure doc_id is string type and apply dataset-specific transformation
        if 'doc_id' in df.columns:
            df['doc_id'] = df['doc_id'].astype(str).apply(self.dataset_config["run_doc_id_transform"])
        
        return df

    def compute_scores_batch(self, processed_queries: Dict[str, list], predictor_scores: Dict[str, float], 
                           list_size: int = 100) -> Dict[str, float]:
        """Compute UEF scores for a batch of queries."""
        uef_scores = {}
        
        for qid in processed_queries.keys():
            original_results = self.retrieval_results[
                self.retrieval_results['qid'] == qid
            ].head(list_size).copy()
            
            rm_results = self.rm_results_df[
                self.rm_results_df['qid'] == qid
            ].head(list_size).copy()
            
            if not original_results.empty and not rm_results.empty:
                try:
                    original_scores = original_results.set_index('doc_id')['docScore']
                    rm_scores = rm_results.set_index('doc_id')['docScore']
                    
                    common_docs = original_scores.index.intersection(rm_scores.index)
                    
                    if len(common_docs) >= 2:
                        orig_common = original_scores[common_docs]
                        rm_common = rm_scores[common_docs]
                        
                        similarity = orig_common.corr(rm_common)
                        uef_scores[qid] = similarity * predictor_scores.get(qid, 0.0)
                    else:
                        uef_scores[qid] = 0.0
                except Exception as e:
                    print(f"Error computing UEF score for query {qid}: {str(e)}")
                    uef_scores[qid] = 0.0
            else:
                uef_scores[qid] = 0.0
        
        return uef_scores

    def compute_score(self, qid: str, original_results: pd.DataFrame, 
                     rm_results: pd.DataFrame, predictor_score: float) -> float:
        """Compute UEF score for a single query."""
        if original_results.empty or rm_results.empty:
            return 0.0

        try:
            # Use doc_id for correlation calculation
            original_scores = original_results.set_index('doc_id')['docScore']
            rm_scores = rm_results.set_index('doc_id')['docScore']
            
            # Calculate correlation between original and RM rankings
            common_docs = original_scores.index.intersection(rm_scores.index)
            if len(common_docs) < 2:  # Need at least 2 documents for correlation
                return 0.0
                
            correlation = original_scores[common_docs].corr(rm_scores[common_docs])
            
            # Calculate final UEF score
            uef_score = correlation * predictor_score
            
            return uef_score
        except Exception as e:
            print(f"Error computing UEF score: {str(e)}")
            return 0.0