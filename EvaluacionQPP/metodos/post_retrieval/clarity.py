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
        for qid, query_terms in processed_queries.items():
            topk_docs = self.retrieval_results[self.retrieval_results['qid'] == qid].head(top_k)
            clarity_score = self.compute_score(query_terms, topk_docs)
            clarity_scores[qid] = clarity_score
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
            # Skip None or NaN values
            if pd.isna(text):
                print(f"Warning: Found NaN text in document")
                continue
            
            terms = preprocess_text(text, dataset_name=self.dataset_name)
            for term in terms:
                topk_term_freq[term] = topk_term_freq.get(term, 0) + 1

        return topk_term_freq

    def _get_collection_probabilities(self, terms: list) -> Dict[str, float]:
        """Get collection probabilities with error handling"""
        p_w_collection = {term: self.index.term_cf.get(term, 0) / self.index.total_terms for term in terms}
        return p_w_collection

    def _calculate_kl_divergence(self, p_w_topk: Dict[str, float], p_w_collection: Dict[str, float]) -> float:
        """Calculate KL-divergence with error handling"""
        clarity = 0.0
        for term in p_w_topk:
            if p_w_collection[term] > 0 and p_w_topk[term] > 0:
                clarity += p_w_topk[term] * np.log(p_w_topk[term] / p_w_collection[term])

        return clarity
