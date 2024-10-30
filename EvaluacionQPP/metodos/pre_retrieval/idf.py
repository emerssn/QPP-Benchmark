from ..base import PreRetrievalMethod
import numpy as np
from ...utils.text_processing import preprocess_text
import json

class IDF(PreRetrievalMethod):
    def __init__(self, index):
        super().__init__(index)
        self.total_docs = self.index.getCollectionStatistics().getNumberOfDocuments()
        
        meta_index = self.index.getMetaIndex()
        last_doc_id = self.total_docs - 1
        
        try:
            term_df_str = meta_index.getItem("term_df", last_doc_id)
            self.term_df = json.loads(term_df_str) if term_df_str else {}
        except:
            print("Warning: Could not load term_df from metadata. IDF calculations may be inaccurate.")
            self.term_df = {}

    def compute_score(self, query, method='avg', **kwargs):
        terms = preprocess_text(query)
        idfs = [self.calc_idf(term) for term in terms]
        
        if method == 'max':
            return self.calc_max_idf(idfs)
        elif method == 'avg':
            return self.calc_avg_idf(idfs)
        else:
            raise ValueError("Invalid method. Choose 'max' or 'avg'.")

    def calc_idf(self, term):
        if term in self.term_df:
            return np.log(self.total_docs / self.term_df[term])
        else:
            # Fallback to using the index's lexicon
            lexicon = self.index.getLexicon()
            lex_entry = lexicon.getLexiconEntry(term)
            if lex_entry is not None:
                return np.log(self.total_docs / lex_entry.getDocumentFrequency())
            else:
                return 0 

    def calc_max_idf(self, idfs):
        """
        Scholer, F. et al. 2004. Query association surrogates for Web search.
        """
        return max(idfs) if idfs else 0.0

    def calc_avg_idf(self, idfs):
        """
        Cronen-Townsend, S. et al. 2002. Predicting query performance.
        """
        return np.mean(idfs) if idfs else 0.0

    def compute_scores_batch(self, queries_dict=None, method='avg'):
        """
        Computes IDF scores for multiple queries in batch.

        Args:
            queries_dict (dict): Optional mapping from query_id to query text.
                                   If None, uses the default queries from the dataset.
            method (str): The IDF calculation method ('max' or 'avg').

        Returns:
            dict: Mapping from query_id to its corresponding IDF score.
        """
        scores_dict = {}
        for query_id, query_text in queries_dict.items():
            scores_dict[query_id] = self.compute_score(query_text, method=method)
        return scores_dict
