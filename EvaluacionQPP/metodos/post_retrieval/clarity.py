import numpy as np
import pandas as pd
from typing import Dict
from EvaluacionQPP.indexing.index_builder import IndexBuilder
from EvaluacionQPP.utils.text_processing import preprocess_text
from ..base import PostRetrievalMethod

class Clarity(PostRetrievalMethod):
    def __init__(self, index_builder, retrieval_results, results_df=None):
        super().__init__(index_builder, retrieval_results, results_df)
        self.index = index_builder
        self.vocabulary = index_builder.get_vocabulary()

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
        """
        Compute the Clarity score for a single query.

        Args:
            query_terms (list): List of processed query terms.
            topk_docs (pd.DataFrame): DataFrame of top-k retrieved documents for the query.

        Returns:
            float: Clarity score.
        """
        if topk_docs.empty or not query_terms:
            return 0.0

        # Compute term frequencies in top-k documents
        topk_term_freq = {}
        for text in topk_docs['text']:
            terms = preprocess_text(text)
            for term in terms:
                topk_term_freq[term] = topk_term_freq.get(term, 0) + 1

        total_topk_terms = sum(topk_term_freq.values())

        # Compute P(w|D^k_{q,M})
        p_w_topk = {term: freq / total_topk_terms for term, freq in topk_term_freq.items()}

        # Compute P(w|D) using collection frequencies
        p_w_collection = {term: self.index.term_cf.get(term, 0) / self.index.total_terms for term in p_w_topk.keys()}

        # Compute Clarity score using KL-divergence
        clarity = 0.0
        for term in p_w_topk:
            if p_w_collection[term] > 0 and p_w_topk[term] > 0:
                clarity += p_w_topk[term] * np.log(p_w_topk[term] / p_w_collection[term])

        return clarity
