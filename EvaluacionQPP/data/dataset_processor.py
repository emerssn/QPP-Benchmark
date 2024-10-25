import pyterrier as pt
import pandas as pd

class DatasetProcessor:
    def __init__(self, dataset_name):
        self.dataset = pt.get_dataset(dataset_name)

    def get_queries(self):
        topics = self.dataset.get_topics()
        if isinstance(topics, pd.DataFrame):
            # Assuming the DataFrame has 'qid' and 'query' columns
            return dict(zip(topics['qid'], topics['query']))
        elif isinstance(topics, dict):
            return topics
        elif isinstance(topics, list):
            if all(isinstance(topic, dict) for topic in topics):
                return {topic.get('qid', topic.get('query_id')): topic.get('query', topic.get('text')) for topic in topics}
            elif all(hasattr(topic, 'query_id') and hasattr(topic, 'text') for topic in topics):
                return {topic.query_id: topic.text for topic in topics}
        raise ValueError("Unsupported topic format")

    def get_qrels(self):
        return self.dataset.get_qrels()

    def iter_docs(self):
        for doc in self.dataset.get_corpus_iter():
            yield doc['docno'], doc['text']
