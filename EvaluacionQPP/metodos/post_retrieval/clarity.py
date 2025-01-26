import numpy as np
import pandas as pd
from typing import Dict, Iterable
from collections import defaultdict
from EvaluacionQPP.indexing.index_builder import IndexBuilder
from EvaluacionQPP.utils.text_processing import preprocess_text
from ..base import PostRetrievalMethod
import logging

logger = logging.getLogger(__name__)

class Clarity(PostRetrievalMethod):
    def __init__(
        self,
        index_builder: IndexBuilder,
        retrieval_results: pd.DataFrame,
        dataset_name: str = None,
        top_k: int = 100,
        term_cutoff: int = 100,
        score_column: str = "docScore"
    ):
        super().__init__(index_builder, retrieval_results, dataset_name)
        
        # Validate critical inputs
        if score_column not in retrieval_results.columns:
            raise ValueError(f"Retrieval results must contain '{score_column}' column")
        if (self.retrieval_results[score_column] < 0).any():
            raise ValueError("Retrieval scores must be non-negative")

        self.top_k = top_k
        self.term_cutoff = term_cutoff
        self.score_column = score_column
        self.index_stats = {
            'total_terms': index_builder.total_terms,
            'term_cf': index_builder.term_cf
        }
        self.dataset_name = dataset_name

    def compute_scores_batch(self, processed_queries: Dict[str, list]) -> Dict[str, float]:
        """Batch compute clarity scores using retrieval scores"""
        if self.retrieval_results.empty:
            logger.warning("Empty retrieval results - returning zero scores")
            return {qid: 0.0 for qid in processed_queries}

        clarity_scores = {}
        for qid in processed_queries:
            try:
                docs = self.retrieval_results[self.retrieval_results['qid'] == qid]
                clarity_scores[qid] = self.compute_score(docs)
            except Exception as e:
                logger.error(f"Error processing {qid}: {e}", exc_info=True)
                clarity_scores[qid] = 0.0
        return clarity_scores

    def compute_score(self, docs: pd.DataFrame) -> float:
        """Core clarity computation for a single query"""
        # Get top documents by retrieval score
        top_docs = docs.nlargest(self.top_k, self.score_column)
        
        if len(top_docs) < 5:
            logger.debug(f"Insufficient docs ({len(top_docs)}) for reliable score")
            return 0.0

        # 1. Build score-weighted term distribution
        term_weights = self._compute_term_weights(top_docs)
        if not term_weights:
            return 0.0

        # 2. Normalize weights using total retrieval score
        total_score = top_docs[self.score_column].sum()
        if total_score <= 0:
            return 0.0
            
        p_w_rm = {term: score/total_score for term, score in term_weights.items()}

        # 3. Get collection probabilities
        p_w_coll = self._get_collection_probabilities(p_w_rm.keys())

        # 4. Calculate KL divergence with numerical stability
        kl_divergence = 0.0
        for term, p in p_w_rm.items():
            coll_p = max(p_w_coll.get(term, 1e-10), 1e-10)
            kl_divergence += p * np.log2(p / coll_p)

        return max(0.0, kl_divergence)

    def _compute_term_weights(self, docs: pd.DataFrame) -> Dict[str, float]:
        """Build term weights using document retrieval scores"""
        term_weights = defaultdict(float)
        
        for doc_score, text in zip(docs[self.score_column], docs['text']):
            if pd.isna(text):
                continue
                
            terms = preprocess_text(text, self.dataset_name)
            for term in terms:
                term_weights[term] += doc_score

        # Apply term cutoff from paper
        sorted_terms = sorted(term_weights.items(), 
                            key=lambda x: x[1], 
                            reverse=True)[:self.term_cutoff]
        
        return dict(sorted_terms)

    def _get_collection_probabilities(self, terms: Iterable[str]) -> Dict[str, float]:
        """Calculate collection probabilities without smoothing"""
        return {
            term: self.index_stats['term_cf'].get(term, 0) / self.index_stats['total_terms']
            for term in terms
        }