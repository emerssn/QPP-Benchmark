from typing import Dict, Union, List, Any
from .pre_retrieval.idf import IDF
from .pre_retrieval.scq import SCQ
from .post_retrieval.wig import WIG
from .post_retrieval.nqc import NQC
from .post_retrieval.clarity import Clarity
from .post_retrieval.uef import UEF
from ..utils.text_processing import preprocess_text

class QPPMethodFactory:
    """
    Factory class for creating and managing QPP methods.
    Handles preprocessing and initialization of all QPP methods.
    """
    
    def __init__(self, index_builder, retrieval_results=None, rm_results=None, dataset_name=None):
        """
        Initialize the QPP method factory.
        
        Args:
            index_builder: IndexBuilder instance
            retrieval_results: DataFrame with retrieval results (for post-retrieval methods)
            rm_results: DataFrame with RM3 results (for UEF method)
            dataset_name: Name of the dataset (for language detection in preprocessing)
        """
        self.index = index_builder.index
        self.index_builder = index_builder
        self.retrieval_results = retrieval_results
        self.rm_results = rm_results
        self.dataset_name = dataset_name
        
        # Initialize QPP methods
        self._init_methods()
        
    def _init_methods(self):
        """Initialize all QPP methods."""
        # Pre-retrieval methods
        self.idf = IDF(self.index)
        self.scq = SCQ(self.index)
        
        # Post-retrieval methods (only if results are available)
        if self.retrieval_results is not None:
            self.wig = WIG(self.index_builder, self.retrieval_results)
            self.nqc = NQC(self.index_builder, self.retrieval_results)
            self.clarity = Clarity(self.index_builder, self.retrieval_results, dataset_name=self.dataset_name)
            
            if self.rm_results is not None:
                self.uef = UEF(self.index_builder, self.retrieval_results, self.rm_results)
    
    def preprocess_queries(self, queries: Union[str, Dict[str, str], List[str]]) -> Union[List[str], Dict[str, List[str]]]:
        """
        Preprocess queries based on their type.
        
        Args:
            queries: Can be a single query string, dictionary of queries, or list of queries
            
        Returns:
            Preprocessed queries in the same format as input
        """
        if isinstance(queries, str):
            return preprocess_text(queries, dataset_name=self.dataset_name)
        elif isinstance(queries, dict):
            return {
                qid: preprocess_text(query, dataset_name=self.dataset_name)
                for qid, query in queries.items()
            }
        elif isinstance(queries, list):
            return [preprocess_text(q, dataset_name=self.dataset_name) for q in queries]
        else:
            raise ValueError(f"Unsupported query type: {type(queries)}")
    
    def compute_all_scores(self, queries: Dict[str, str], **kwargs) -> Dict[str, Dict[str, float]]:
        """
        Compute scores for all available QPP methods.
        
        Args:
            queries: Dictionary mapping query IDs to query strings
            **kwargs: Additional arguments for specific methods
                     (e.g., list_size_param for WIG/NQC)
        
        Returns:
            Dictionary mapping query IDs to dictionaries of method scores
        """
        processed_queries = self.preprocess_queries(queries)
        scores = {}
        
        # Compute pre-retrieval scores
        idf_scores_avg = self.idf.compute_scores_batch(processed_queries, method='avg')
        idf_scores_max = self.idf.compute_scores_batch(processed_queries, method='max')
        scq_scores_avg = self.scq.compute_scores_batch(processed_queries, method='avg')
        scq_scores_max = self.scq.compute_scores_batch(processed_queries, method='max')
        
        # Initialize scores dictionary
        for qid in queries:
            scores[qid] = {
                'idf_avg': idf_scores_avg.get(qid, 0.0),
                'idf_max': idf_scores_max.get(qid, 0.0),
                'scq_avg': scq_scores_avg.get(qid, 0.0),
                'scq_max': scq_scores_max.get(qid, 0.0)
            }
        
        # Add post-retrieval scores if available
        if hasattr(self, 'wig'):
            list_size = kwargs.get('list_size_param', 10)
            wig_scores = self.wig.compute_scores_batch(processed_queries, list_size_param=list_size)
            nqc_scores = self.nqc.compute_scores_batch(processed_queries, list_size_param=list_size)
            clarity_scores = self.clarity.compute_scores_batch(processed_queries)
            
            for qid in queries:
                scores[qid].update({
                    'wig': wig_scores.get(qid, 0.0),
                    'nqc': nqc_scores.get(qid, 0.0),
                    'clarity': clarity_scores.get(qid, 0.0)
                })
            
            # Add UEF scores if available
            if hasattr(self, 'uef'):
                uef_wig_scores = self.uef.compute_scores_batch(processed_queries, wig_scores)
                uef_nqc_scores = self.uef.compute_scores_batch(processed_queries, nqc_scores)
                
                for qid in queries:
                    scores[qid].update({
                        'uef_wig': uef_wig_scores.get(qid, 0.0),
                        'uef_nqc': uef_nqc_scores.get(qid, 0.0)
                    })
        
        return scores 