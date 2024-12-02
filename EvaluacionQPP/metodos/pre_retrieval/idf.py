from ..base import PreRetrievalMethod
import numpy as np
import json

class IDF(PreRetrievalMethod):
    def __init__(self, index):
        super().__init__(index)
        if hasattr(index, 'term_df'):
            # If we're passed an IndexBuilder, use its statistics
            self.term_df = index.term_df
            self.total_docs = index.total_docs
            self.index_builder = index
            self.index = index.index  # Get PyTerrier index for fallback
        else:
            # Otherwise, load from PyTerrier index
            self.index_builder = None
            self.index = index
            self.total_docs = index.getCollectionStatistics().getNumberOfDocuments()
            self.term_df = {}
            # Build term_df from lexicon
            lexicon = index.getLexicon()
            for entry in lexicon:
                term = entry.getKey()
                stats = entry.getValue()
                self.term_df[term] = stats.getDocumentFrequency()

    def compute_score(self, query_terms, method='avg', **kwargs):
        """
        Compute IDF score for a query.
        
        Args:
            query_terms (list): List of preprocessed query terms
            method (str): Aggregation method ('avg' or 'max')
            
        Returns:
            float: IDF score
        """
        if not query_terms:
            return 0.0
            
        idfs = []
        for term in query_terms:
            df = self._get_term_df(term)
            if df > 0:  # Avoid log(0)
                idf = np.log(self.total_docs / df)
                idfs.append(idf)
            else:
                # If term not found, use maximum possible IDF
                idfs.append(np.log(self.total_docs))
        
        if not idfs:
            return 0.0
            
        if method == 'max':
            return max(idfs)
        elif method == 'avg':
            return np.mean(idfs)
        else:
            raise ValueError("Invalid method. Choose 'max' or 'avg'.")

    def _get_term_df(self, term):
        """Get document frequency for a term."""
        # First try our tracked statistics
        if term in self.term_df:
            return self.term_df[term]
        
        # If we have an index builder, try its statistics
        if self.index_builder and term in self.index_builder.term_df:
            return self.index_builder.term_df[term]
        
        # Fallback to lexicon lookup
        lexicon = self.index.getLexicon()
        lex_entry = lexicon.getLexiconEntry(term)
        if lex_entry is not None:
            df = lex_entry.getValue().getDocumentFrequency()
            # Cache the result
            self.term_df[term] = df
            return df
        return 0

    def compute_scores_batch(self, queries_dict=None, method='avg'):
        """
        Compute IDF scores for multiple queries.
        
        Args:
            queries_dict (dict): Mapping from query_id to preprocessed query terms
            method (str): Aggregation method ('max' or 'avg')
            
        Returns:
            dict: Mapping from query_id to IDF score
        """
        scores_dict = {}
        for query_id, query_terms in queries_dict.items():
            scores_dict[query_id] = self.compute_score(query_terms, method=method)
        return scores_dict
