"""
Tests for the ThinkML Engine module.
"""

import unittest
import pandas as pd
import numpy as np
from thinkml.thinkml_engine import (
    PromptInterpreter,
    CodeGenerator,
    ExecutionManager,
    ErrorHandler,
    ExecutionError
)

class TestPromptInterpreter(unittest.TestCase):
    """Test cases for PromptInterpreter class."""
    
    def setUp(self):
        self.interpreter = PromptInterpreter()
    
    def test_describe_prompt(self):
        prompt = "Describe the dataset"
        result = self.interpreter.parse(prompt)
        self.assertEqual(result['task'], 'describe')
    
    def test_train_prompt(self):
        prompt = "Train a decision tree on my data"
        result = self.interpreter.parse(prompt)
        self.assertEqual(result['task'], 'train')
        self.assertEqual(result['model'], 'decision_tree')
    
    def test_preprocess_prompt(self):
        prompt = "Preprocess and clean the data"
        result = self.interpreter.parse(prompt)
        self.assertEqual(result['task'], 'preprocess')
    
    def test_evaluate_prompt(self):
        prompt = "Evaluate the model performance"
        result = self.interpreter.parse(prompt)
        self.assertEqual(result['task'], 'evaluate')

class TestCodeGenerator(unittest.TestCase):
    """Test cases for CodeGenerator class."""
    
    def setUp(self):
        self.generator = CodeGenerator()
    
    def test_describe_code_generation(self):
        task_dict = {'task': 'describe', 'model': None}
        code = self.generator.generate_code(task_dict)
        self.assertIn('from thinkml.describe import describe_dataset', code)
        self.assertIn('result = describe_dataset(data)', code)
    
    def test_train_code_generation(self):
        task_dict = {'task': 'train', 'model': 'decision_tree'}
        code = self.generator.generate_code(task_dict)
        self.assertIn('from thinkml.classification import ThinkMLClassifier', code)
        self.assertIn('model_type="decision_tree"', code)
    
    def test_invalid_task(self):
        task_dict = {'task': None, 'model': None}
        with self.assertRaises(ValueError):
            self.generator.generate_code(task_dict)

class TestExecutionManager(unittest.TestCase):
    """Test cases for ExecutionManager class."""
    
    def setUp(self):
        self.executor = ExecutionManager()
        # Create sample data
        self.data = pd.DataFrame({
            'A': np.random.rand(10),
            'B': np.random.rand(10)
        })
    
    def test_safe_execution(self):
        code = """
import pandas as pd
result = pd.DataFrame({'A': [1, 2, 3]})
"""
        result = self.executor.execute(code)
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_error_handling(self):
        code = "result = undefined_variable"
        with self.assertRaises(ExecutionError):
            self.executor.execute(code)

class TestErrorHandler(unittest.TestCase):
    """Test cases for ErrorHandler class."""
    
    def setUp(self):
        self.error_handler = ErrorHandler()
    
    def test_error_formatting(self):
        try:
            raise ValueError("Test error")
        except Exception as e:
            formatted_error = self.error_handler.format_error(e)
            self.assertIn("Error Type: ValueError", formatted_error)
            self.assertIn("Message: Test error", formatted_error)
            self.assertIn("Stack Trace", formatted_error)

if __name__ == '__main__':
    unittest.main() 