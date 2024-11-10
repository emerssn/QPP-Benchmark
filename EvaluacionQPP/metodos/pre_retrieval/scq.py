from ..base import PreRetrievalMethod
import numpy as np
from ...utils.text_processing import preprocess_text
import json

class SCQ(PreRetrievalMethod):
    def __init__(self, index):
        super().__init__(index)
        self.total_docs = self.index.getCollectionStatistics().getNumberOfDocuments()
        
        meta_index = self.index.getMetaIndex()
        last_doc_id = self.total_docs - 1
        
        try:
            term_df_str = meta_index.getItem("term_df", last_doc_id)
            term_cf_str = meta_index.getItem("term_cf", last_doc_id)
            self.term_df = json.loads(term_df_str) if term_df_str else {}
            self.term_cf = json.loads(term_cf_str) if term_cf_str else {}
        except:
            print("Warning: Could not load term statistics from metadata. SCQ calculations may be inaccurate.")
            self.term_df = {}
            self.term_cf = {}

    def compute_score(self, query, method='avg', **kwargs):
        """
        Computes SCQ score for a query using specified method.
        
        Args:
            query (str): The query text
            method (str): The SCQ calculation method ('max', 'avg', or 'sum')
            
        Returns:
            float: The SCQ score
        """
        terms = preprocess_text(query)
        raw_scq_scores = self._calc_raw_scq(terms)
        
        if method == 'max':
            return self.calc_max_scq(raw_scq_scores)
        elif method == 'avg':
            return self.calc_avg_scq(raw_scq_scores)
        elif method == 'sum':
            return self.calc_scq(raw_scq_scores)
        else:
            raise ValueError("Invalid method. Choose 'max', 'avg', or 'sum'.")

    def _calc_raw_scq(self, terms):
        """
        Calculates raw SCQ scores for terms.
        
        Zhao, Y. et al. 2008.
        Effective Pre-retrieval Query Performance Prediction Using Similarity and Variability Evidence
        """
        raw_scores = []
        
        for term in terms:
            if term in self.term_cf and term in self.term_df:
                cf = self.term_cf[term]
                df = self.term_df[term]
            else:
                # Fallback to using the index's lexicon
                lexicon = self.index.getLexicon()
                lex_entry = lexicon.getLexiconEntry(term)
                if lex_entry is not None:
                    cf = lex_entry.getFrequency()
                    df = lex_entry.getDocumentFrequency()
                else:
                    cf = 0
                    df = 0
            
            if cf > 0 and df > 0:
                score = (1 + np.log(cf)) * np.log(1 + self.total_docs / df)
                raw_scores.append(score)
            else:
                raw_scores.append(0)
                
        return np.array(raw_scores)

    def calc_scq(self, raw_scores):
        """
        Calculates sum SCQ score.
        """
        return np.sum(raw_scores)

    def calc_max_scq(self, raw_scores):
        """
        Calculates maximum SCQ score.
        """
        return np.max(raw_scores) if len(raw_scores) > 0 else 0.0

    def calc_avg_scq(self, raw_scores):
        """
        Calculates average SCQ score (NSCQ in the original paper).
        """
        return np.mean(raw_scores) if len(raw_scores) > 0 else 0.0

    def compute_scores_batch(self, queries_dict=None, method='avg'):
        """
        Computes SCQ scores for multiple queries in batch.

        Args:
            queries_dict (dict): Optional mapping from query_id to query text.
                               If None, uses the default queries from the dataset.
            method (str): The SCQ calculation method ('max', 'avg', or 'sum').

        Returns:
            dict: Mapping from query_id to its corresponding SCQ score.
        """
        scores_dict = {}
        for query_id, query_text in queries_dict.items():
            scores_dict[query_id] = self.compute_score(query_text, method=method)
        return scores_dict