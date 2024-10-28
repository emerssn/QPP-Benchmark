import numpy as np

import pandas as pd

from typing import Dict

from ..base import PostRetrievalMethod



class UEF(PostRetrievalMethod):

    def __init__(self, index_builder, retrieval_results, rm_results_df=None):

        """

        Initialize UEF predictor.

        

        Args:

            index_builder: IndexBuilder instance

            retrieval_results: Original retrieval results

            rm_results_df: Relevance model re-ranked results

        """

        super().__init__(index_builder, retrieval_results)

        self.rm_results_df = rm_results_df if rm_results_df is not None else retrieval_results



    def compute_scores_batch(self, processed_queries: Dict[str, list], predictor_scores: Dict[str, float], 

                           list_size: int = 100) -> Dict[str, float]:

        """

        Compute UEF scores for a batch of queries.



        Args:

            processed_queries (Dict[str, list]): Dictionary of query IDs to processed query terms

            predictor_scores (Dict[str, float]): Base predictor scores (e.g., WIG, NQC, etc.)

            list_size (int): Number of top documents to consider



        Returns:

            Dict[str, float]: Dictionary of query IDs to their UEF scores

        """

        uef_scores = {}

        

        for qid in processed_queries.keys():

            # Get original results for this query

            original_results = self.retrieval_results[

                self.retrieval_results['qid'] == qid

            ].head(list_size)

            

            # Get RM re-ranked results for this query

            rm_results = self.rm_results_df[

                self.rm_results_df['qid'] == qid

            ].head(list_size)

            

            # Calculate UEF score using correlation method

            if not original_results.empty and not rm_results.empty:

                # Create score series indexed by docno

                original_scores = original_results.set_index('docno')['docScore']

                rm_scores = rm_results.set_index('docno')['docScore']

                

                # Calculate correlation using only documents that appear in both rankings

                common_docs = original_scores.index.intersection(rm_scores.index)

                if len(common_docs) >= 2:  # Need at least 2 documents for correlation

                    similarity = original_scores[common_docs].corr(rm_scores[common_docs])

                    uef_scores[qid] = similarity * predictor_scores.get(qid, 0.0)

                else:

                    uef_scores[qid] = 0.0

            else:

                uef_scores[qid] = 0.0

            

        return uef_scores



    def compute_score(self, qid: str, original_results: pd.DataFrame, 

                     rm_results: pd.DataFrame, predictor_score: float) -> float:

        """

        Compute UEF score for a single query.



        Args:

            qid (str): Query ID

            original_results (pd.DataFrame): Original retrieval results

            rm_results (pd.DataFrame): RM re-ranked results

            predictor_score (float): Base predictor score



        Returns:

            float: UEF score

        """

        if original_results.empty or rm_results.empty:

            return 0.0



        # Prepare score series indexed by docno for correlation calculation

        original_scores = original_results.set_index('docno')['docScore']

        rm_scores = rm_results.set_index('docno')['docScore']

        

        # Calculate correlation between original and RM rankings

        # Using only documents that appear in both rankings

        common_docs = original_scores.index.intersection(rm_scores.index)

        if len(common_docs) < 2:  # Need at least 2 documents for correlation

            return 0.0

            

        correlation = original_scores[common_docs].corr(rm_scores[common_docs])

        

        # Calculate final UEF score

        uef_score = correlation * predictor_score

        

        return uef_score


