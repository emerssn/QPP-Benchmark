import unittest
import numpy as np
import os
import shutil
import pyterrier as pt
from EvaluacionQPP.metodos.pre_retrieval.scq import SCQ
from EvaluacionQPP.data.dataset_processor import DatasetProcessor
from EvaluacionQPP.indexing.index_builder import IndexBuilder
from EvaluacionQPP.utils.text_processing import preprocess_text

class TestSCQ(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that can be reused for all tests"""
        # Initialize PyTerrier if not already started
        if not pt.started():
            pt.init()
            
        # Create dataset processor with IquiqueDataset
        cls.dataset_processor = DatasetProcessor("iquique_dataset")
        cls.index_builder = IndexBuilder(cls.dataset_processor, "iquique_test")
        
        # Create a temporary index path for testing
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cls.test_index_path = os.path.join(script_dir, "..", "..", "..", "indices", "test_index")
        
        # Clean up any existing index
        if os.path.exists(cls.test_index_path):
            shutil.rmtree(cls.test_index_path)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(cls.test_index_path), exist_ok=True)
        
        # Build or load index
        cls.index = cls.index_builder.load_or_build_index(cls.test_index_path)
        
        # Create SCQ instance
        cls.scq = SCQ(cls.index)
        
        # Calculate term statistics for verification
        cls.term_stats = {}
        print("\nProcessed terms and their statistics:")
        for doc_id, text in cls.dataset_processor.dataset.documents.items():
            terms = preprocess_text(text)
            print(f"\nDoc {doc_id}: {terms}")
            for term in terms:
                if term not in cls.term_stats:
                    cls.term_stats[term] = {'df': 0, 'cf': 0}
                cls.term_stats[term]['cf'] += 1
                if term in set(terms):  # Count df only once per document
                    cls.term_stats[term]['df'] += 1
        
        print("\nTerm statistics:", cls.term_stats)
        cls.total_docs = len(cls.dataset_processor.dataset.documents)

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        if os.path.exists(cls.test_index_path):
            shutil.rmtree(cls.test_index_path)

    def test_compute_score_single_term(self):
        """Test SCQ score computation for single terms"""
        print("\nRunning test_compute_score_single_term...")
        
        # Get actual preprocessed terms
        sample_text = "iquique playa"
        processed_terms = preprocess_text(sample_text)
        term1, term2 = processed_terms[0], processed_terms[1]
        
        # Test common term
        score = self.scq.compute_score([term1], method='sum')
        stats = self.term_stats[term1]
        
        # Debug print
        print(f"\nTerm: {term1}")
        print(f"Stats from test: cf={stats['cf']}, df={stats['df']}")
        print(f"Total docs: {self.total_docs}")
        
        # Get actual stats from index
        lexicon = self.index.getLexicon()
        lex_entry = lexicon.getLexiconEntry(term1)
        if lex_entry:
            index_cf = lex_entry.getFrequency()
            index_df = lex_entry.getDocumentFrequency()
            print(f"Stats from index: cf={index_cf}, df={index_df}")
        
        # Use index statistics instead of calculated ones
        expected_scq = (1 + np.log(index_cf)) * np.log(1 + self.total_docs/index_df)
        self.assertAlmostEqual(score, expected_scq, places=4)
        print(f"✓ Common term test passed - Score: {score:.4f}, Expected: {expected_scq:.4f}")
        
        # Test less common term
        score = self.scq.compute_score([term2], method='sum')
        
        # Debug print
        print(f"\nTerm: {term2}")
        lex_entry = lexicon.getLexiconEntry(term2)
        if lex_entry:
            index_cf = lex_entry.getFrequency()
            index_df = lex_entry.getDocumentFrequency()
            print(f"Stats from index: cf={index_cf}, df={index_df}")
        
        # Use index statistics
        expected_scq = (1 + np.log(index_cf)) * np.log(1 + self.total_docs/index_df)
        self.assertAlmostEqual(score, expected_scq, places=4)
        print(f"✓ Less common term test passed - Score: {score:.4f}, Expected: {expected_scq:.4f}")

    def test_compute_score_multiple_terms(self):
        """Test SCQ score computation for multiple terms"""
        print("\nRunning test_compute_score_multiple_terms...")
        
        sample_text = "museo historia"
        processed_terms = preprocess_text(sample_text)
        
        # Test with different methods
        methods = ['avg', 'max', 'sum']
        for method in methods:
            score = self.scq.compute_score(processed_terms, method=method)
            raw_scores = []
            for term in processed_terms:
                stats = self.term_stats[term]
                term_score = (1 + np.log(stats['cf'])) * np.log(1 + self.total_docs/stats['df'])
                raw_scores.append(term_score)
            
            if method == 'avg':
                expected = np.mean(raw_scores)
            elif method == 'max':
                expected = np.max(raw_scores)
            else:  # sum
                expected = np.sum(raw_scores)
                
            self.assertAlmostEqual(score, expected)
            print(f"✓ {method.capitalize()} method test passed - Score: {score:.4f}, Expected: {expected:.4f}")

    def test_compute_score_unknown_term(self):
        """Test SCQ score computation for unknown terms"""
        print("\nRunning test_compute_score_unknown_term...")
        
        score = self.scq.compute_score(["unknown_term"])
        self.assertEqual(score, 0.0)
        print(f"✓ Unknown term test passed - Score: {score}")

    def test_compute_score_empty_query(self):
        """Test SCQ score computation for empty query"""
        print("\nRunning test_compute_score_empty_query...")
        
        score = self.scq.compute_score([])
        self.assertEqual(score, 0.0)
        print(f"✓ Empty query test passed - Score: {score}")

    def test_compute_score_invalid_method(self):
        """Test SCQ score computation with invalid method"""
        print("\nRunning test_compute_score_invalid_method...")
        
        with self.assertRaises(ValueError):
            self.scq.compute_score(["museo"], method='invalid')
        print("✓ Invalid method test passed - ValueError raised as expected")

    def test_compute_scores_batch(self):
        """Test batch computation of SCQ scores"""
        print("\nRunning test_compute_scores_batch...")
        
        queries = {
            "0": "playa cavancha iquique",
            "1": "zona franca zofri",
            "2": "museo historia iquique"
        }
        
        queries_dict = {
            qid: preprocess_text(query)
            for qid, query in queries.items()
        }
        
        # Test with different methods
        methods = ['avg', 'max', 'sum']
        for method in methods:
            scores = self.scq.compute_scores_batch(queries_dict, method=method)
            self.assertEqual(len(scores), len(queries))
            print(f"✓ Batch {method} scores:", scores)
            
            # Verify each score individually
            for qid, terms in queries_dict.items():
                individual_score = self.scq.compute_score(terms, method=method)
                self.assertAlmostEqual(scores[qid], individual_score)
                
        print("✓ Batch computation test passed")

if __name__ == '__main__':
    unittest.main() 