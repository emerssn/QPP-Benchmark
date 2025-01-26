import pyterrier as pt
import pandas as pd
from .iquique_dataset import IquiqueDataset
from EvaluacionQPP.utils.text_processing import preprocess_text
import warnings

class DatasetProcessor:
    def __init__(self, dataset_name: str):
        """
        Initialize the dataset processor with the specified dataset.
        
        Args:
            dataset_name (str): Name/identifier of the dataset to process
        """
        if dataset_name == "iquique_dataset":
            self.dataset = IquiqueDataset()
        elif dataset_name.startswith("irds:"):
            if not pt.started():
                pt.init()
            self.dataset = pt.get_dataset(dataset_name)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    def get_queries(self):
        """
        Get both raw and preprocessed queries from the dataset.
        
        Returns:
            dict: Dictionary mapping query IDs to query information containing:
                - 'raw': original query text
                - 'processed': list of preprocessed tokens
        """
        topics = self.dataset.get_topics()
        queries = {}
        
        if isinstance(topics, pd.DataFrame):
            raw_queries = dict(zip(topics['qid'], topics['query']))
        elif isinstance(topics, dict):
            raw_queries = topics
        elif isinstance(topics, list):
            if all(isinstance(topic, dict) for topic in topics):
                raw_queries = {topic.get('qid', topic.get('query_id')): topic.get('query', topic.get('text')) 
                             for topic in topics}
            elif all(hasattr(topic, 'query_id') and hasattr(topic, 'text') for topic in topics):
                raw_queries = {topic.query_id: topic.text for topic in topics}
            else:
                raise ValueError("Unsupported topic format")
        else:
            raise ValueError("Unsupported topic format")

        # Determine dataset language
        dataset_name = getattr(self.dataset, 'name', '')
        
        # Preprocess each query
        for qid, query_text in raw_queries.items():
            queries[qid] = {
                'raw': query_text,  # Keep original query text
                'processed': preprocess_text(query_text, dataset_name)  # Add preprocessed tokens
            }
        
        return queries

    def get_raw_queries(self):
        """Get original unprocessed queries for retrieval."""
        queries = self.get_queries()
        return {qid: query['raw'] for qid, query in queries.items()}

    def get_processed_queries(self):
        """Get preprocessed query tokens for QPP methods."""
        queries = self.get_queries()
        return {qid: query['processed'] for qid, query in queries.items()}

    def get_qrels(self):
        return self.dataset.get_qrels()

    def iter_docs(self):
        for doc in self.dataset.get_corpus_iter():
            text = doc['text']
            if isinstance(text, bytes):
                try:
                    text = text.decode('utf-8')
                except UnicodeDecodeError:
                    warnings.warn(f"UTF-8 decode failed for document {doc['docno']}. Falling back to Latin-1.")
                    try:
                        text = text.decode('latin-1')
                    except UnicodeDecodeError:
                        warnings.warn(f"Latin-1 decode failed for document {doc['docno']}. Ignoring problematic characters.")
                        text = text.decode('utf-8', errors='ignore')
            
            yield doc['docno'], text