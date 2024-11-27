import logging
from functools import lru_cache
from typing import Dict, Union, List, Any
from .pre_retrieval.idf import IDF
from .pre_retrieval.scq import SCQ
from .post_retrieval.wig import WIG
from .post_retrieval.nqc import NQC
from .post_retrieval.clarity import Clarity
from .post_retrieval.uef import UEF
from ..utils.text_processing import preprocess_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    
    @lru_cache(maxsize=128)
    def preprocess_queries(self, query_key: Union[str, tuple]) -> Union[List[str], Dict[str, List[str]]]:
        """
        Preprocess and cache queries based on their type.
        
        Args:
            query_key: Either a query string or a tuple of (qid, query) pairs
            
        Returns:
            Preprocessed queries in the same format as input
        """
        if isinstance(query_key, str):
            return preprocess_text(query_key, dataset_name=self.dataset_name)
        elif isinstance(query_key, tuple):
            return {
                qid: preprocess_text(query, dataset_name=self.dataset_name)
                for qid, query in query_key
            }
        else:
            raise ValueError(f"Unsupported query type: {type(query_key)}")
    
    def compute_all_scores(self, queries: Dict[str, str], **kwargs) -> Dict[str, Dict[str, float]]:
        """Compute scores for all available QPP methods."""
        # Convert queries dict to hashable tuple for caching
        query_items = tuple(queries.items())
        processed_queries = self.preprocess_queries(query_items)
        scores = {}
        
        # Compute pre-retrieval scores
        for qid in queries:
            scores[qid] = {
                'idf_avg': self.idf.compute_scores_batch(processed_queries, method='avg').get(qid, 0.0),
                'idf_max': self.idf.compute_scores_batch(processed_queries, method='max').get(qid, 0.0),
                'scq_avg': self.scq.compute_scores_batch(processed_queries, method='avg').get(qid, 0.0),
                'scq_max': self.scq.compute_scores_batch(processed_queries, method='max').get(qid, 0.0)
            }
        
        # Add post-retrieval scores if available
        if hasattr(self, 'wig'):
            list_size = kwargs.get('list_size_param', 10)
            logger.info(f"Computing post-retrieval scores with list size: {list_size}")
            
            # Compute post-retrieval scores
            post_retrieval_scores = {
                'wig': self.wig.compute_scores_batch(processed_queries, list_size_param=list_size),
                'nqc': self.nqc.compute_scores_batch(processed_queries, list_size_param=list_size),
                'clarity': self.clarity.compute_scores_batch(processed_queries)
            }
            
            # Update scores dictionary
            for qid in queries:
                scores[qid].update({
                    method: score_dict.get(qid, 0.0)
                    for method, score_dict in post_retrieval_scores.items()
                })
            
            # Add UEF scores if available
            if hasattr(self, 'uef'):
                logger.debug("Computing UEF scores")
                uef_scores = {
                    'uef_wig': self.uef.compute_scores_batch(processed_queries, post_retrieval_scores['wig'], list_size),
                    'uef_nqc': self.uef.compute_scores_batch(processed_queries, post_retrieval_scores['nqc'], list_size)
                }
                
                for qid in queries:
                    scores[qid].update({
                        method: score_dict.get(qid, 0.0)
                        for method, score_dict in uef_scores.items()
                    })
        
        return scores 