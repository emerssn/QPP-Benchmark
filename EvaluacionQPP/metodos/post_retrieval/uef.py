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

        

        # Clean up column names in both dataframes

        self.retrieval_results = self._clean_dataframe(self.retrieval_results)

        self.rm_results_df = self._clean_dataframe(self.rm_results_df)

    

    def _clean_dataframe(self, df):

        """

        Clean up DataFrame columns and ensure consistent naming.

        

        Args:

            df: DataFrame to clean up

        

        Returns:

            Cleaned DataFrame

        """

        # Remove duplicate columns

        df = df.loc[:, ~df.columns.duplicated(keep='first')]

        

        # Standardize column names

        column_mapping = {

            'docno': 'doc_id',  # Map both docno columns to doc_id

            'score': 'docScore'

        }

        

        # Only rename columns that exist

        for old_col, new_col in column_mapping.items():

            if old_col in df.columns:

                df = df.rename(columns={old_col: new_col})

        

        # Ensure doc_id is string type and strip any whitespace

        if 'doc_id' in df.columns:

            df['doc_id'] = df['doc_id'].astype(str).str.strip()

        

        return df



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

            ].head(list_size).copy()

            

            # Get RM re-ranked results for this query

            rm_results = self.rm_results_df[

                self.rm_results_df['qid'] == qid

            ].head(list_size).copy()

            

            print(f"\nDEBUG UEF - Query {qid}:")

            print("Original results sample with types:")

            print(original_results[['qid', 'doc_id', 'docScore']].head().to_string())

            print("\nDoc_id type:", original_results['doc_id'].dtype)

            print("Sample doc_id values:", original_results['doc_id'].head().tolist())

            

            print("\nRM results sample with types:")

            print(rm_results[['qid', 'doc_id', 'docScore']].head().to_string())

            print("\nDoc_id type:", rm_results['doc_id'].dtype)

            print("Sample doc_id values:", rm_results['doc_id'].head().tolist())

            

            if not original_results.empty and not rm_results.empty:

                try:

                    # Ensure doc_id is string type and clean

                    original_results['doc_id'] = original_results['doc_id'].astype(str).str.strip()

                    rm_results['doc_id'] = rm_results['doc_id'].astype(str).str.strip()

                    

                    # Create score series indexed by doc_id

                    original_scores = original_results.set_index('doc_id')['docScore']

                    rm_scores = rm_results.set_index('doc_id')['docScore']

                    

                    # Debug print

                    print("\nOriginal scores index:", original_scores.index.tolist())

                    print("RM scores index:", rm_scores.index.tolist())

                    

                    # Calculate correlation using only documents that appear in both rankings

                    common_docs = original_scores.index.intersection(rm_scores.index)

                    print(f"\nNumber of common documents: {len(common_docs)}")

                    print("Common documents:", sorted(list(common_docs)))

                    

                    if len(common_docs) >= 2:

                        # Get scores for common documents

                        orig_common = original_scores[common_docs]

                        rm_common = rm_scores[common_docs]

                        

                        print("\nOriginal scores for common docs:", orig_common.tolist())

                        print("RM scores for common docs:", rm_common.tolist())

                        

                        similarity = orig_common.corr(rm_common)

                        uef_scores[qid] = similarity * predictor_scores.get(qid, 0.0)

                        print(f"Correlation: {similarity}, Final UEF score: {uef_scores[qid]}")

                    else:

                        print("Not enough common documents for correlation")

                        uef_scores[qid] = 0.0

                except Exception as e:

                    print(f"Error computing UEF score for query {qid}: {str(e)}")

                    uef_scores[qid] = 0.0

            else:

                print("Empty results for query")

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


