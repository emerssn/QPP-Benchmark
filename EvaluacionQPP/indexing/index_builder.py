import pyterrier as pt
from ..utils.text_processing import preprocess_text
import json
import os
import shutil
import time

class IndexBuilder:
    def __init__(self, dataset, dataset_name):
        self.dataset = dataset
        self.total_docs = 0
        self.term_df = {}
        self.term_cf = {}
        self.index = None 
        self.total_terms = 0
        self.dataset_name = dataset_name

    def build_index(self, base_index_path):
        os.makedirs(base_index_path, exist_ok=True)
        
        if not pt.started():
            pt.init()

        # Basic indexer configuration
        indexer = pt.IterDictIndexer(base_index_path)
        indexer.setProperty("terrier.index.meta.forward.keys", "docno,text")
        indexer.setProperty("terrier.index.meta.forward.keylens", "20,100000")
        
        # Track term frequencies during indexing
        term_stats = {}
        
        def doc_iterator():
            iterator = self.dataset.iter_docs()
            for doc_id, doc_text in iterator:
                # Track term frequencies using our preprocessor
                terms = preprocess_text(doc_text, dataset_name=self.dataset_name)
                
                # Track term frequencies
                for term in terms:
                    if term not in term_stats:
                        term_stats[term] = {'df': 0, 'cf': 0}
                    term_stats[term]['cf'] += 1
                    
                # Track document frequencies (unique terms per doc)
                for term in set(terms):
                    term_stats[term]['df'] += 1
                
                self.total_docs += 1
                
                # Pass preprocessed text for indexing
                processed_text = ' '.join(terms)
                yield {
                    'docno': str(doc_id),
                    'text': processed_text,
                }

        # Build index
        index_ref = indexer.index(doc_iterator())
        index = pt.IndexFactory.of(index_ref)
        self.index = index
        
        # Store our tracked statistics
        self.term_df = {term: stats['df'] for term, stats in term_stats.items()}
        self.term_cf = {term: stats['cf'] for term, stats in term_stats.items()}
        self.total_terms = sum(self.term_cf.values())
        
        return index

    def _load_statistics_from_index(self):
        """Load term statistics directly from Terrier's index"""
        # Use our tracked statistics instead of loading from index
        if not self.term_df or not self.term_cf:
            self.term_df = {}
            self.term_cf = {}
            self.total_docs = self.index.getCollectionStatistics().getNumberOfDocuments()
            self.total_terms = self.index.getCollectionStatistics().getNumberOfTokens()
            
            # Load lexicon entries
            lexicon = self.index.getLexicon()
            for entry in lexicon:
                term = entry.getKey()
                self.term_df[term] = entry.getValue().getDocumentFrequency()
                self.term_cf[term] = entry.getValue().getFrequency()

    def load_or_build_index(self, base_index_path):
        print(f"Checking for index at: {base_index_path}")
        
        if os.path.exists(base_index_path) and os.listdir(base_index_path):
            print(f"Found existing index for dataset {self.dataset_name}. Loading...")
            try:
                index = pt.IndexFactory.of(base_index_path)
                self.index = index
                
                # Load statistics from index only if we don't have our own
                if not self.term_df or not self.term_cf:
                    self._load_statistics_from_index()
                    
                print(f"Successfully loaded existing index for {self.dataset_name}")
                print(f"Total documents: {self.total_docs}")
                print(f"Total terms: {self.total_terms}")
                print(f"Unique terms: {len(self.term_df)}")
                
            except Exception as e:
                print(f"Error loading existing index: {e}")
                print("Creating new index...")
                shutil.rmtree(base_index_path)
                index = self.build_index(base_index_path)
        else:
            print(f"No existing index found for dataset {self.dataset_name}. Creating new index...")
            if os.path.exists(base_index_path):
                shutil.rmtree(base_index_path)
            index = self.build_index(base_index_path)
        
        return index

    def get_document_terms(self, doc_id):
        """
        Retrieve and preprocess terms from a document given its doc_id.
        """
        doc = self.index.getStore().getDocument(doc_id)
        text = doc['text']
        return preprocess_text(text)

    def get_vocabulary(self):
        """
        Retrieve the complete vocabulary from the collection.
        """
        lexicon = self.index.getLexicon()
        return [entry.getKey() for entry in lexicon]

    def get_term_probability(self, term):
        """
        Calculate P(w|D) for a term based on collection frequencies.
        """
        term_cf = self.term_cf.get(term, 0)
        return term_cf / self.total_terms if self.total_terms > 0 else 0.0

    def save_sample_frequencies_to_json(self, base_file_path, num_samples=100):
        """
        Save a sample of term_df and term_cf to a JSON file with dataset-specific naming.
        """
        # Create dataset-specific filename
        filename = f"sample_term_frequencies_{self.dataset_name}.json"
        file_path = os.path.join(os.path.dirname(base_file_path), filename)
        
        sample_terms = list(self.term_df.keys())[:num_samples]
        sample_data = {
            "dataset": self.dataset_name,
            "term_df": {term: self.term_df[term] for term in sample_terms},
            "term_cf": {term: self.term_cf[term] for term in sample_terms}
        }
        with open(file_path, 'w') as f:
            json.dump(sample_data, f, indent=4)
        print(f"Sample term frequencies for dataset {self.dataset_name} saved to {file_path}")
