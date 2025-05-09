"""
Tests for the PromptRefiner module.
"""

import unittest
from thinkml.engine.prompt_refiner import PromptRefiner

class TestPromptRefiner(unittest.TestCase):
    """Test cases for PromptRefiner class."""
    
    def setUp(self):
        self.refiner = PromptRefiner()
    
    def test_task_normalization(self):
        """Test normalization of different task phrases."""
        test_cases = [
            ("clean the data", "preprocess"),
            ("analyze the dataset", "describe"),
            ("build a model", "train"),
            ("check model performance", "evaluate"),
            ("plot the results", "visualize")
        ]
        
        for prompt, expected_task in test_cases:
            with self.subTest(prompt=prompt):
                result = self.refiner.refine(prompt)
                self.assertEqual(result['task'], expected_task)
    
    def test_model_normalization(self):
        """Test normalization of different model phrases."""
        test_cases = [
            ("train a log reg model", "logistic_regression"),
            ("use linear regression", "linear_regression"),
            ("build a decision tree", "decision_tree"),
            ("create a random forest", "random_forest")
        ]
        
        for prompt, expected_model in test_cases:
            with self.subTest(prompt=prompt):
                result = self.refiner.refine(prompt)
                self.assertEqual(result['model'], expected_model)
    
    def test_visualization_normalization(self):
        """Test normalization of different visualization phrases."""
        test_cases = [
            ("plot the ROC curve", "roc_curve"),
            ("show feature importance", "feature_importance"),
            ("display correlation heatmap", "correlation_heatmap"),
            ("plot confusion matrix", "confusion_matrix")
        ]
        
        for prompt, expected_viz in test_cases:
            with self.subTest(prompt=prompt):
                result = self.refiner.refine(prompt)
                self.assertEqual(result['visualization'], expected_viz)
                self.assertEqual(result['task'], "visualize")
    
    def test_combined_components(self):
        """Test prompts with multiple components."""
        test_cases = [
            {
                "prompt": "train a logistic regression and plot the ROC curve",
                "expected": {
                    "task": "visualize",  # Visualization takes precedence
                    "model": "logistic_regression",
                    "visualization": "roc_curve"
                }
            },
            {
                "prompt": "clean the data and show correlation heatmap",
                "expected": {
                    "task": "visualize",
                    "model": None,
                    "visualization": "correlation_heatmap"
                }
            }
        ]
        
        for test_case in test_cases:
            with self.subTest(prompt=test_case["prompt"]):
                result = self.refiner.refine(test_case["prompt"])
                for key, value in test_case["expected"].items():
                    self.assertEqual(result[key], value)
    
    def test_suggestions(self):
        """Test suggestion generation for ambiguous prompts."""
        test_cases = [
            ("do something with the data", True),  # Should give task suggestions
            ("train a model", True),  # Should give model suggestions
            ("plot something", True),  # Should give visualization suggestions
            ("train a logistic regression model", False)  # Should not give suggestions
        ]
        
        for prompt, should_have_suggestions in test_cases:
            with self.subTest(prompt=prompt):
                suggestions = self.refiner.get_suggestions(prompt)
                if should_have_suggestions:
                    self.assertTrue(len(suggestions) > 0)
                else:
                    self.assertEqual(len(suggestions), 0)
    
    def test_supported_phrases(self):
        """Test retrieval of supported phrases."""
        supported = self.refiner.get_supported_phrases()
        
        # Check that all expected categories are present
        self.assertIn('tasks', supported)
        self.assertIn('models', supported)
        self.assertIn('visualizations', supported)
        
        # Check that each category has expected items
        self.assertIn('describe', supported['tasks'])
        self.assertIn('logistic_regression', supported['models'])
        self.assertIn('roc_curve', supported['visualizations'])

if __name__ == '__main__':
    unittest.main() 