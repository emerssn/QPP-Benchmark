from ..base import PreRetrievalMethod
import numpy as np
import json

class IDF(PreRetrievalMethod):
    def __init__(self, index):
        super().__init__(index)
        self.total_docs = self.index.getCollectionStatistics().getNumberOfDocuments()
        
        # Load term frequencies from meta index
        meta_index = self.index.getMetaIndex()
        last_doc_id = self.total_docs - 1
        
        try:
            term_df_str = meta_index.getItem("term_df", last_doc_id)
            self.term_df = json.loads(term_df_str) if term_df_str else {}
        except:
            print("Warning: Could not load term_df from metadata, falling back to lexicon")
            self.term_df = {}
            # Build term_df from lexicon
            lexicon = self.index.getLexicon()
            iterator = lexicon.iterator()
            entry = iterator.next()
            while entry is not None:
                self.term_df[entry.getKey()] = entry.getDocumentFrequency()
                entry = iterator.next()

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
        if term in self.term_df:
            return self.term_df[term]
        
        # Fallback to lexicon lookup
        lexicon = self.index.getLexicon()
        lex_entry = lexicon.getLexiconEntry(term)
        if lex_entry is not None:
            df = lex_entry.getDocumentFrequency()
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
