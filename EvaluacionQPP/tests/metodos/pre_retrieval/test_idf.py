import unittest
import numpy as np
import os
import shutil
import pyterrier as pt
from EvaluacionQPP.metodos.pre_retrieval.idf import IDF
from EvaluacionQPP.data.dataset_processor import DatasetProcessor
from EvaluacionQPP.indexing.index_builder import IndexBuilder
from EvaluacionQPP.utils.text_processing import preprocess_text

class TestIDF(unittest.TestCase):
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
        
        # Create IDF instance
        cls.idf = IDF(cls.index)
        
        # Count actual document frequencies after preprocessing
        cls.term_dfs = {}
        print("\nProcessed terms and their document frequencies:")
        for doc_id, text in cls.dataset_processor.dataset.documents.items():
            terms = preprocess_text(text)
            print(f"\nDoc {doc_id}: {terms}")
            for term in set(terms):  # Use set to count each term once per document
                cls.term_dfs[term] = cls.term_dfs.get(term, 0) + 1
        
        print("\nTerm frequencies:", cls.term_dfs)
        cls.total_docs = len(cls.dataset_processor.dataset.documents)

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        # Clean up the test index
        if os.path.exists(cls.test_index_path):
            shutil.rmtree(cls.test_index_path)

    def test_compute_score_single_term(self):
        """Test IDF score computation for single terms"""
        print("\nRunning test_compute_score_single_term...")
        
        # Get the actual preprocessed terms
        sample_text = "iquique playa"
        processed_terms = preprocess_text(sample_text)
        term1, term2 = processed_terms[0], processed_terms[1]
        
        # Test common term
        score = self.idf.compute_score([term1])
        expected_idf = np.log(self.total_docs/self.term_dfs[term1])
        self.assertAlmostEqual(score, expected_idf)
        print(f"✓ Common term test passed - Score: {score:.4f}, Expected: {expected_idf:.4f}")
        
        # Test rare term
        score = self.idf.compute_score([term2])
        expected_idf = np.log(self.total_docs/self.term_dfs[term2])
        self.assertAlmostEqual(score, expected_idf)
        print(f"✓ Rare term test passed - Score: {score:.4f}, Expected: {expected_idf:.4f}")

    def test_compute_score_multiple_terms(self):
        """Test IDF score computation for multiple terms"""
        print("\nRunning test_compute_score_multiple_terms...")
        
        sample_text = "museo historia"
        processed_terms = preprocess_text(sample_text)
        term1, term2 = processed_terms[0], processed_terms[1]
        
        # Test with avg method
        score_avg = self.idf.compute_score([term1, term2], method='avg')
        expected_avg = np.mean([
            np.log(self.total_docs/self.term_dfs[term1]),
            np.log(self.total_docs/self.term_dfs[term2])
        ])
        self.assertAlmostEqual(score_avg, expected_avg)
        print(f"✓ Average method test passed - Score: {score_avg:.4f}, Expected: {expected_avg:.4f}")
        
        # Test with max method
        score_max = self.idf.compute_score([term1, term2], method='max')
        expected_max = max(
            np.log(self.total_docs/self.term_dfs[term1]),
            np.log(self.total_docs/self.term_dfs[term2])
        )
        self.assertAlmostEqual(score_max, expected_max)
        print(f"✓ Maximum method test passed - Score: {score_max:.4f}, Expected: {expected_max:.4f}")

    def test_compute_score_unknown_term(self):
        """Test IDF score computation for unknown terms"""
        print("\nRunning test_compute_score_unknown_term...")
        
        score = self.idf.compute_score(["unknown_term"])
        expected_idf = np.log(self.total_docs)
        self.assertAlmostEqual(score, expected_idf)
        print(f"✓ Unknown term test passed - Score: {score:.4f}, Expected: {expected_idf:.4f}")

    def test_compute_score_empty_query(self):
        """Test IDF score computation for empty query"""
        print("\nRunning test_compute_score_empty_query...")
        
        score = self.idf.compute_score([])
        self.assertEqual(score, 0.0)
        print(f"✓ Empty query test passed - Score: {score}")

    def test_compute_score_invalid_method(self):
        """Test IDF score computation with invalid method"""
        print("\nRunning test_compute_score_invalid_method...")
        
        with self.assertRaises(ValueError):
            self.idf.compute_score(["museo"], method='invalid')
        print("✓ Invalid method test passed - ValueError raised as expected")

    def test_compute_scores_batch(self):
        """Test batch computation of IDF scores"""
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
        
        # Test with avg method
        scores_avg = self.idf.compute_scores_batch(queries_dict, method='avg')
        print(f"✓ Batch average scores: {scores_avg}")
        
        # Test with max method
        scores_max = self.idf.compute_scores_batch(queries_dict, method='max')
        print(f"✓ Batch maximum scores: {scores_max}")
        
        print("✓ Batch computation test passed")

    def test_get_term_df(self):
        """Test document frequency retrieval for terms"""
        print("\nRunning test_get_term_df...")
        
        sample_terms = preprocess_text("iquique playa historia")
        
        for term in sample_terms:
            df = self.idf._get_term_df(term)
            expected_df = self.term_dfs[term]
            self.assertEqual(df, expected_df)
            print(f"✓ Term '{term}' DF test passed - DF: {df}, Expected: {expected_df}")
        
        unknown_df = self.idf._get_term_df("unknown_term")
        self.assertEqual(unknown_df, 0)
        print("✓ Unknown term DF test passed - DF: 0")

if __name__ == '__main__':
    unittest.main() 