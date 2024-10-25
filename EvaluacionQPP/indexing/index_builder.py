import pyterrier as pt
from ..utils.text_processing import preprocess_text
import json
import os
import shutil

class IndexBuilder:
    def __init__(self, dataset):
        self.dataset = dataset
        self.total_docs = 0
        self.term_df = {}
        self.term_cf = {}
        self.index = None 
        self.total_terms = 0 

    def build_index(self, index_path):
        # Ensure the directory exists
        os.makedirs(index_path, exist_ok=True)
        
        # Ensure PyTerrier is initialized
        if not pt.started():
            pt.init()

        # Create an indexer with only 'docno' and 'text' as metadata
        indexer = pt.IterDictIndexer(index_path)
        indexer.setProperty("terrier.index.meta.forward.keys", "docno,text")
        indexer.setProperty("terrier.index.meta.forward.keylens", "20,100000")

        # Prepare documents for indexing
        def doc_iterator():
            iterator = self.dataset.iter_docs()
            try:
                prev_doc = next(iterator)
            except StopIteration:
                return
            for doc_id, doc_text in iterator:
                processed_text = preprocess_text(doc_text)
                
                # Update term document frequency
                for term in set(processed_text):
                    self.term_df[term] = self.term_df.get(term, 0) + 1
                
                # Update term collection frequency
                for term in processed_text:
                    self.term_cf[term] = self.term_cf.get(term, 0) + 1
                    self.total_terms += 1  # Track total terms
                
                self.total_docs += 1
                
                yield {
                    'docno': prev_doc[0],
                    'text': ' '.join(processed_text),
                }
                
                prev_doc = (doc_id, doc_text)

            # Handle last document without adding global stats
            processed_text = preprocess_text(prev_doc[1])
            
            for term in set(processed_text):
                self.term_df[term] = self.term_df.get(term, 0) + 1
            
            for term in processed_text:
                self.term_cf[term] = self.term_cf.get(term, 0) + 1
            
            self.total_docs += 1
            
            yield {
                'docno': prev_doc[0],
                'text': ' '.join(processed_text),
                # Removed global stats from metadata
            }

        # Index the documents
        index_ref = indexer.index(doc_iterator())
        index = pt.IndexFactory.of(index_ref)
        self.index = index
        
        # No need to verify global metadata since they're not stored in documents
        return index

    def load_or_build_index(self, index_path):
        if os.path.exists(index_path) and os.listdir(index_path):
            print("Index directory exists and is not empty. Attempting to load existing index.")
            try:
                index = pt.IndexFactory.of(index_path)
                self.index = index

                # Access global statistics directly from the index
                self.total_docs = index.getCollectionStatistics().getNumberOfDocuments()
                self.total_terms = index.getCollectionStatistics().getNumberOfTokens()
                
                # Load term_df and term_cf from the index's collection statistics or lexicon
                lexicon = index.getLexicon()
                for entry in lexicon:
                    term = entry.getKey()
                    self.term_df[term] = entry.getValue().getDocumentFrequency()
                    self.term_cf[term] = entry.getValue().getFrequency()

                print(f"Loaded total_docs: {self.total_docs}")
                print(f"Loaded total_tokens: {self.total_terms}")
                print(f"Loaded term_df with {len(self.term_df)} terms.")
                print(f"Loaded term_cf with {len(self.term_cf)} terms.")

                print("Successfully loaded existing index.")
            except Exception as e:
                print(f"Error loading existing index: {e}")
                print("Deleting existing index and creating a new one.")
                shutil.rmtree(index_path)
                index = self.build_index(index_path)
        else:
            print("Index directory does not exist or is empty. Creating new index.")
            if os.path.exists(index_path):
                shutil.rmtree(index_path)
            index = self.build_index(index_path)
        
        print(f"Total documents indexed: {index.getCollectionStatistics().getNumberOfDocuments()}")
        print(f"Unique terms: {index.getCollectionStatistics().getNumberOfUniqueTerms()}")
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

    def save_sample_frequencies_to_json(self, file_path, num_samples=100):
        """
        Save a sample of term_df and term_cf to a JSON file.
        """
        sample_terms = list(self.term_df.keys())[:num_samples]
        sample_data = {
            "term_df": {term: self.term_df[term] for term in sample_terms},
            "term_cf": {term: self.term_cf[term] for term in sample_terms}
        }
        with open(file_path, 'w') as f:
            json.dump(sample_data, f, indent=4)
        print(f"Sample term frequencies saved to {file_path}")
