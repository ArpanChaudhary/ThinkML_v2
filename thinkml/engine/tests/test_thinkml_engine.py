"""
Tests for the main ThinkML engine module.
"""

import unittest
from thinkml.thinkml_engine import ThinkMLEngine, ExecutionError
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

class TestThinkMLEngine(unittest.TestCase):
    """Test cases for ThinkMLEngine class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = ThinkMLEngine()
        # Load sample data
        iris = load_iris()
        self.data = pd.DataFrame(iris.data, columns=iris.feature_names)
        self.data['target'] = iris.target
        # Set data in engine
        self.engine.set_data(self.data)
    
    def test_describe_task(self):
        """Test the describe task functionality."""
        try:
            result = self.engine.process("Describe the dataset")
            self.assertIsNotNone(result)
            self.assertIsInstance(result, dict)
            self.assertTrue('summary_stats' in result)
            self.assertTrue('missing_values' in result)
        except Exception as e:
            self.fail(f"Describe task failed with error: {str(e)}")
    
    def test_preprocess_task(self):
        """Test the preprocess task functionality."""
        try:
            # Test with missing values
            data_with_missing = self.data.copy()
            data_with_missing.iloc[0:10, 0] = np.nan
            self.engine.set_data(data_with_missing)
            
            result = self.engine.process("Preprocess and clean the data")
            self.assertIsNotNone(result)
            self.assertIsInstance(result, pd.DataFrame)
            self.assertTrue(result.isna().sum().sum() == 0)  # No missing values
            
            # Restore original data
            self.engine.set_data(self.data)
        except Exception as e:
            self.fail(f"Preprocess task failed with error: {str(e)}")
    
    def test_train_task(self):
        """Test the train task functionality."""
        try:
            # Test with various model types
            models = [
                "Train a logistic regression model",
                "Train a decision tree with max_depth=3",
                "Train a random forest with n_estimators=100"
            ]
            for prompt in models:
                result = self.engine.process(prompt)
                self.assertIsNotNone(result)
                self.assertTrue('model' in result)
                self.assertTrue('train_accuracy' in result)
                self.assertTrue('test_accuracy' in result)
        except Exception as e:
            self.fail(f"Train task failed with error: {str(e)}")
    
    def test_evaluate_task(self):
        """Test the evaluate task functionality."""
        try:
            # First train a model
            self.engine.process("Train a logistic regression model")
            # Test different evaluation metrics
            metrics = [
                "Evaluate the model",
                "Show model accuracy and precision",
                "Calculate model recall and f1-score"
            ]
            for prompt in metrics:
                result = self.engine.process(prompt)
                self.assertIsNotNone(result)
                self.assertTrue('metrics' in result)
        except Exception as e:
            self.fail(f"Evaluate task failed with error: {str(e)}")
    
    def test_visualize_task(self):
        """Test the visualize task functionality."""
        try:
            # First train a model
            self.engine.process("Train a logistic regression model")
            # Test different visualization types
            viz_types = [
                "Plot the ROC curve",
                "Show feature importance plot",
                "Create confusion matrix",
                "Display correlation heatmap"
            ]
            for prompt in viz_types:
                result = self.engine.process(prompt)
                self.assertIsNotNone(result)
                self.assertTrue(hasattr(result, 'figure'))
        except Exception as e:
            self.fail(f"Visualize task failed with error: {str(e)}")
    
    def test_invalid_prompt(self):
        """Test handling of invalid prompts."""
        invalid_prompts = [
            "",  # Empty prompt
            "Make the data fly",  # Nonsense prompt
            "Train model",  # Too vague
            "Plot everything",  # Too broad
            "Describe data with invalid parameter=123"  # Invalid parameters
        ]
        for prompt in invalid_prompts:
            with self.assertRaises(ExecutionError):
                self.engine.process(prompt)
    
    def test_chained_tasks(self):
        """Test chaining multiple tasks together."""
        prompts = [
            "Describe the dataset",
            "Preprocess and clean the data",
            "Train a decision tree",
            "Evaluate the model performance",
            "Plot feature importance"
        ]
        
        results = []
        for prompt in prompts:
            try:
                result = self.engine.process(prompt)
                self.assertIsNotNone(result)
                results.append(result)
            except Exception as e:
                self.fail(f"Chained task '{prompt}' failed with error: {str(e)}")
        
        # Verify chain results
        self.assertEqual(len(results), len(prompts))
        self.assertIsInstance(results[0], dict)  # Description result
        self.assertIsInstance(results[1], pd.DataFrame)  # Preprocessed data
        self.assertTrue('model' in results[2])  # Training result
        self.assertTrue('metrics' in results[3])  # Evaluation metrics
        self.assertTrue(hasattr(results[4], 'figure'))  # Visualization
    
    def test_custom_parameters(self):
        """Test tasks with custom parameters."""
        try:
            # Test various parameter combinations
            parameter_tests = [
                "Train a random forest with n_estimators=100 and max_depth=5",
                "Train a decision tree with criterion=gini and min_samples_split=5",
                "Preprocess data with scaling=standard and handle_missing=mean"
            ]
            for prompt in parameter_tests:
                result = self.engine.process(prompt)
                self.assertIsNotNone(result)
        except Exception as e:
            self.fail(f"Custom parameters task failed with error: {str(e)}")
    
    def test_error_handling(self):
        """Test error handling and recovery."""
        # Test with invalid data
        try:
            invalid_data = pd.DataFrame({'A': [1, 2, 'invalid'], 'B': [4, 5, 6]})
            self.engine.set_data(invalid_data)
            with self.assertRaises(ExecutionError) as context:
                self.engine.process("Train a logistic regression model")
            self.assertTrue("Invalid data type" in str(context.exception))
        finally:
            # Restore valid data
            self.engine.set_data(self.data)
        
        # Test with missing dependencies
        with self.assertRaises(ExecutionError) as context:
            self.engine.process("Train a xgboost model")  # XGBoost not imported
        self.assertTrue("Unsupported model type" in str(context.exception))
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        edge_cases = [
            # Empty dataset
            (pd.DataFrame(), "Describe the dataset"),
            # Single column dataset
            (pd.DataFrame({'A': [1, 2, 3]}), "Train a logistic regression model"),
            # Single row dataset
            (pd.DataFrame({'A': [1], 'B': [2]}), "Preprocess and clean the data"),
            # All missing values
            (pd.DataFrame({'A': [np.nan, np.nan], 'B': [np.nan, np.nan]}), "Preprocess and clean the data"),
            # All constant values
            (pd.DataFrame({'A': [1, 1, 1], 'B': [2, 2, 2]}), "Train a decision tree")
        ]
        
        for test_data, prompt in edge_cases:
            try:
                self.engine.set_data(test_data)
                with self.assertRaises(ExecutionError):
                    self.engine.process(prompt)
            finally:
                # Restore valid data
                self.engine.set_data(self.data)

if __name__ == '__main__':
    unittest.main() 