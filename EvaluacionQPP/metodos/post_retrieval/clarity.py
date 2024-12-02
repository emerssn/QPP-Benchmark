import numpy as np
import pandas as pd
from typing import Dict
from EvaluacionQPP.indexing.index_builder import IndexBuilder
from EvaluacionQPP.utils.text_processing import preprocess_text
from ..base import PostRetrievalMethod
import logging

logger = logging.getLogger(__name__)

class Clarity(PostRetrievalMethod):
    def __init__(self, index_builder, retrieval_results, results_df=None, dataset_name=None):
        super().__init__(index_builder, retrieval_results, results_df)
        self.index = index_builder
        self.vocabulary = index_builder.get_vocabulary()
        self.dataset_name = dataset_name

    def compute_scores_batch(self, processed_queries: Dict[str, list], top_k: int = 1000) -> Dict[str, float]:
        """
        Compute Clarity scores for a batch of queries.

        Args:
            processed_queries (Dict[str, list]): Dictionary of query IDs to processed query terms.
            top_k (int): Number of top documents to consider for each query.

        Returns:
            Dict[str, float]: Dictionary of query IDs to their Clarity scores.
        """
        clarity_scores = {}
        
        # Handle empty results DataFrame
        if self.retrieval_results is None or self.retrieval_results.empty:
            return {qid: 0.0 for qid in processed_queries}
        
        for qid, query_terms in processed_queries.items():
            try:
                topk_docs = self.retrieval_results[self.retrieval_results['qid'] == qid].head(top_k)
                clarity_score = self.compute_score(query_terms, topk_docs)
                clarity_scores[qid] = clarity_score
            except Exception as e:
                logger.error(f"Error computing clarity score for query {qid}: {e}")
                clarity_scores[qid] = 0.0
            
        return clarity_scores

    def compute_score(self, query_terms: list, topk_docs: pd.DataFrame) -> float:
        """Compute Clarity score with improved robustness"""
        if topk_docs.empty or not query_terms:
            logger.warning(f"Empty docs or query terms. Docs shape: {topk_docs.shape}, Terms: {query_terms}")
            return 0.0

        # Filter out invalid documents
        valid_docs = topk_docs[topk_docs['text'].notna()]
        if valid_docs.empty:
            logger.warning("No valid documents found after filtering")
            return 0.0

        # Compute term frequencies with error handling
        try:
            topk_term_freq = self._compute_term_frequencies(valid_docs)
            if not topk_term_freq:
                return 0.0
                
            total_terms = sum(topk_term_freq.values())
            p_w_topk = {term: freq / total_terms for term, freq in topk_term_freq.items()}
            p_w_collection = self._get_collection_probabilities(p_w_topk.keys())
            
            return self._calculate_kl_divergence(p_w_topk, p_w_collection)
            
        except Exception as e:
            logger.error(f"Error computing Clarity score: {e}")
            return 0.0

    def _compute_term_frequencies(self, docs: pd.DataFrame) -> Dict[str, int]:
        """Compute term frequencies with error handling"""
        topk_term_freq = {}
        
        for text in docs['text']:
            if pd.isna(text):
                print(f"Warning: Found NaN text in document")
                continue
            
            terms = preprocess_text(text, dataset_name=self.dataset_name)
            for term in terms:
                topk_term_freq[term] = topk_term_freq.get(term, 0) + 1

        return topk_term_freq

    def _get_collection_probabilities(self, terms: list) -> Dict[str, float]:
        """Get collection probabilities with error handling"""
        p_w_collection = {}
        for term in terms:
            # Check both stemmed term and potential original form
            cf = self.index.term_cf.get(term, 0)
            if cf == 0 and term == 'play':
                # If 'play' has 0 frequency, check for 'playa'
                cf = self.index.term_cf.get('playa', 0)
            p_w_collection[term] = cf / self.index.total_terms
        return p_w_collection

    def _calculate_kl_divergence(self, p_w_topk: Dict[str, float], p_w_collection: Dict[str, float]) -> float:
        """
        Calculate KL-divergence with error handling and smoothing.
        
        Args:
            p_w_topk: Term probabilities in top-k documents
            p_w_collection: Term probabilities in collection
            
        Returns:
            float: Non-negative KL-divergence score
        """
        clarity = 0.0
        epsilon = 1e-10  # Small constant for smoothing
        
        for term in p_w_topk:
            # Add smoothing to avoid log(0)
            p_topk = p_w_topk[term]
            p_coll = p_w_collection[term]
            
            # Apply smoothing to collection probabilities
            p_coll_smoothed = max(p_coll, epsilon)
            
            if p_topk > 0:  # Only calculate for non-zero probabilities in top-k
                # Use smoothed collection probability
                clarity += p_topk * np.log(p_topk / p_coll_smoothed)
        
        # Ensure non-negative score
        return max(0.0, clarity)
