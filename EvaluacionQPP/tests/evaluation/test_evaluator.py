import unittest
import pandas as pd
import numpy as np
from ...evaluation.evaluator import evaluate_results
from ...data.iquique_dataset import IquiqueDataset

class TestEvaluator(unittest.TestCase):
    def setUp(self):
        """Set up test data using IquiqueDataset"""
        self.dataset = IquiqueDataset()
        self.qrels = self.dataset.get_qrels()
    
        
        # Create a sample run dataframe that matches exactly with IquiqueDataset qrels
        self.perfect_run = pd.DataFrame([
            # Query 0 - Perfect ranking for "playa cavancha iquique"
            {"qid": "0", "doc_id": "doc2", "docScore": 1.0},  # Relevant (1)
            {"qid": "0", "doc_id": "doc0", "docScore": 0.5},  # Not relevant
            {"qid": "0", "doc_id": "doc1", "docScore": 0.3},  # Not relevant
            
            # Query 1 - Perfect ranking for "zona franca zofri"
            {"qid": "1", "doc_id": "doc1", "docScore": 1.0},  # Relevant (1)
            {"qid": "1", "doc_id": "doc0", "docScore": 0.5},  # Not relevant
            
            # Query 2 - Perfect ranking for "museo historia iquique"
            {"qid": "2", "doc_id": "doc3", "docScore": 1.0},  # Relevant (1)
            {"qid": "2", "doc_id": "doc0", "docScore": 0.9},  # Relevant (1)
            {"qid": "2", "doc_id": "doc1", "docScore": 0.5},  # Not relevant
            
            # Query 3 - Perfect ranking for "historia salitre guerra pacifico"
            {"qid": "3", "doc_id": "doc6", "docScore": 1.0},  # Highly relevant (2)
            {"qid": "3", "doc_id": "doc5", "docScore": 0.9},  # Relevant (1)
            {"qid": "3", "doc_id": "doc3", "docScore": 0.8},  # Relevant (1)
            {"qid": "3", "doc_id": "doc7", "docScore": 0.7},  # Relevant (1)
        ])
        
        # Create a reversed (bad) ranking for comparison
        self.reversed_run = pd.DataFrame([
            # Query 0 - Reversed ranking
            {"qid": "0", "doc_id": "doc1", "docScore": 1.0},  # Not relevant
            {"qid": "0", "doc_id": "doc0", "docScore": 0.8},  # Not relevant
            {"qid": "0", "doc_id": "doc2", "docScore": 0.5},  # Relevant (at bottom)
            
            # Query 1 - Reversed ranking
            {"qid": "1", "doc_id": "doc0", "docScore": 1.0},  # Not relevant
            {"qid": "1", "doc_id": "doc1", "docScore": 0.5},  # Relevant (at bottom)
            
            # Query 2 - Reversed ranking
            {"qid": "2", "doc_id": "doc1", "docScore": 1.0},  # Not relevant
            {"qid": "2", "doc_id": "doc3", "docScore": 0.5},  # Relevant (at bottom)
            {"qid": "2", "doc_id": "doc0", "docScore": 0.3},  # Relevant (at bottom)
            
            # Query 3 - Reversed ranking
            {"qid": "3", "doc_id": "doc4", "docScore": 1.0},  # Not relevant
            {"qid": "3", "doc_id": "doc2", "docScore": 0.9},  # Not relevant
            {"qid": "3", "doc_id": "doc6", "docScore": 0.5},  # Highly relevant (at bottom)
            {"qid": "3", "doc_id": "doc5", "docScore": 0.4},  # Relevant (at bottom)
        ])

    def test_perfect_ndcg(self):
        """Test NDCG@10 for perfect ranking"""
        results = evaluate_results(
            self.qrels,
            self.perfect_run,
            metrics=['ndcg@10'],
            dataset_name="iquique_dataset"
        )
        
        print("\nNDCG@10 Results for perfect ranking:")
        print(results['ndcg@10'])
        
        # Perfect ranking should have high NDCG
        self.assertGreater(results['ndcg@10']['mean'], 0.8)

    def test_reversed_ndcg(self):
        """Test NDCG@10 for reversed (worst) ranking"""
        results = evaluate_results(
            self.qrels,
            self.reversed_run,
            metrics=['ndcg@10'],
            dataset_name="iquique_dataset"
        )
        
        print("\nNDCG@10 Results for reversed ranking:")
        print(results['ndcg@10'])
        
        # Reversed ranking should have lower NDCG
        self.assertLess(results['ndcg@10']['mean'], 0.6)

    def test_perfect_ap(self):
        """Test AP for perfect ranking"""
        results = evaluate_results(
            self.qrels,
            self.perfect_run,
            metrics=['ap'],
            dataset_name="iquique_dataset"
        )
        
        print("\nAP Results for perfect ranking:")
        print(results['ap'])
        
        # Perfect ranking should have high AP
        self.assertGreater(results['ap']['mean'], 0.7)

    def test_reversed_ap(self):
        """Test AP for reversed ranking"""
        results = evaluate_results(
            self.qrels,
            self.reversed_run,
            metrics=['ap'],
            dataset_name="iquique_dataset"
        )
        
        print("\nAP Results for reversed ranking:")
        print(results['ap'])
        
        # Reversed ranking should have low AP
        self.assertLess(results['ap']['mean'], 0.5)

    def test_multiple_metrics(self):
        """Test multiple metrics at once"""
        metrics = ['ndcg@10', 'ndcg@20', 'ap']
        results = evaluate_results(
            self.qrels,
            self.perfect_run,
            metrics=metrics,
            dataset_name="iquique_dataset"
        )
        
        # Check that all requested metrics are present
        for metric in metrics:
            self.assertIn(metric, results)
            self.assertIn('mean', results[metric])
            self.assertIn('per_query', results[metric])
            
        # Perfect ranking should have good scores for all metrics
        self.assertGreater(results['ndcg@10']['mean'], 0.8)
        self.assertGreater(results['ndcg@20']['mean'], 0.8)
        self.assertGreater(results['ap']['mean'], 0.7)

    def test_invalid_queries(self):
        """Test handling of queries not in qrels"""
        invalid_run = pd.DataFrame([
            {"qid": "999", "doc_id": "doc1", "docScore": 1.0}
        ])
        
        results = evaluate_results(
            self.qrels,
            invalid_run,
            metrics=['ndcg@10', 'ap'],
            dataset_name="iquique_dataset"
        )
        
        # Should return 0.0 for invalid queries
        self.assertEqual(results['ndcg@10']['mean'], 0.0)
        self.assertEqual(results['ap']['mean'], 0.0)
        self.assertEqual(len(results['ndcg@10']['per_query']), 0)
        self.assertEqual(len(results['ap']['per_query']), 0)

if __name__ == '__main__':
    unittest.main() 