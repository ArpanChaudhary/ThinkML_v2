"""
ThinkML Engine - A prompt-driven ML code generation and execution system.

This module provides the core functionality for converting natural language prompts
into executable ML code using the ThinkML library.
"""

import logging
import re
from typing import Dict, Any, Optional, List, Union
import ast
import sys
from io import StringIO
import traceback
import pandas as pd
import numpy as np
from .engine.prompt_refiner import PromptRefiner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ThinkMLEngine:
    """Main engine class that coordinates all components."""
    
    def __init__(self):
        self.refiner = PromptRefiner()
        self.interpreter = PromptInterpreter()
        self.generator = CodeGenerator()
        self.executor = ExecutionManager()
        self.error_handler = ErrorHandler()
        self.data = None
        
    def set_data(self, data: pd.DataFrame):
        """Set the data to be used for analysis."""
        self.data = data
        self.executor.set_data(data)
    
    def process(self, prompt: str) -> Any:
        """
        Process a natural language prompt and execute the resulting code.
        
        Args:
            prompt (str): The natural language prompt
            
        Returns:
            Any: Result of code execution
        """
        try:
            # Validate prompt
            if not prompt or not isinstance(prompt, str):
                raise ExecutionError("Invalid prompt: prompt must be a non-empty string")
            
            # Refine the prompt
            refined = self.refiner.refine(prompt)
            
            # If no task is identified, get suggestions
            if not refined['task']:
                suggestions = self.refiner.get_suggestions(prompt)
                raise ExecutionError("Prompt could not be mapped to a valid task.\n" + "\n".join(suggestions))
            
            # If the prompt is about data, wrap set_data in try/except
            if refined['task'] in ['describe', 'preprocess', 'train', 'evaluate', 'visualize']:
                try:
                    self.executor.set_data(self.data)
                except Exception as e:
                    raise ExecutionError(str(e))
            
            # Parse the refined prompt
            command = self.interpreter.parse(refined)
            
            # Generate code
            code = self.generator.generate_code(command)
            
            # Execute code
            result = self.executor.execute(code)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing prompt: {str(e)}")
            formatted_error = self.error_handler.format_error(e)
            suggestions = self.refiner.get_suggestions(prompt)
            if suggestions:
                formatted_error += "\n\nSuggestions:\n" + "\n".join(suggestions)
            raise ExecutionError(formatted_error)

class PromptInterpreter:
    """Converts natural language prompts into structured commands."""
    
    def __init__(self):
        # Define task patterns and their corresponding actions
        self.task_patterns = {
            'describe': [
                r'describe.*data',
                r'analyze.*data',
                r'summarize.*data',
                r'statistics.*data'
            ],
            'preprocess': [
                r'preprocess.*data',
                r'clean.*data',
                r'prepare.*data',
                r'normalize.*data'
            ],
            'train': [
                r'train.*model',
                r'fit.*model',
                r'build.*model',
                r'create.*model'
            ],
            'evaluate': [
                r'evaluate.*model',
                r'assess.*model',
                r'test.*model',
                r'validate.*model'
            ],
            'visualize': [
                r'plot.*',
                r'show.*',
                r'display.*',
                r'visualize.*'
            ]
        }
        
        # Define model patterns
        self.model_patterns = {
            'logistic_regression': [
                r'logistic.*regression',
                r'logistic',
                r'classification'
            ],
            'linear_regression': [
                r'linear.*regression',
                r'linear',
                r'regression'
            ],
            'decision_tree': [
                r'decision.*tree',
                r'tree'
            ],
            'random_forest': [
                r'random.*forest',
                r'forest'
            ]
        }
        
        # Define visualization patterns
        self.visualization_patterns = {
            'roc_curve': [
                r'roc.*curve',
                r'roc'
            ],
            'feature_importance': [
                r'feature.*importance',
                r'importance.*plot'
            ],
            'correlation_heatmap': [
                r'correlation.*heatmap',
                r'correlation.*matrix'
            ],
            'confusion_matrix': [
                r'confusion.*matrix',
                r'error.*matrix'
            ]
        }

    def parse(self, refined_prompt: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse a refined prompt into a structured command.
        
        Args:
            refined_prompt (Dict[str, Any]): Refined prompt from PromptRefiner
            
        Returns:
            Dict[str, Any]: Structured command containing task and parameters
        """
        logger.info(f"Parsing refined prompt: {refined_prompt}")
        
        # Use the refined components directly
        command = {
            'task': refined_prompt['task'],
            'model': refined_prompt['model'],
            'visualization': refined_prompt['visualization'],
            'parameters': {}
        }
        
        logger.info(f"Parsed command: {command}")
        return command

class CodeGenerator:
    """Generates ThinkML code based on structured commands."""
    
    def __init__(self):
        self.imports = {
            'describe': 'from thinkml.describe import describe_dataset',
            'preprocess': 'from thinkml.preprocessing import preprocess_data',
            'train': 'from thinkml.classification import ThinkMLClassifier',
            'evaluate': 'from thinkml.evaluation import evaluate_model',
            'visualize': 'from thinkml.visualization import plot_roc_curve, plot_feature_importance, plot_correlation_heatmap, plot_confusion_matrix'
        }
        
        self.code_templates = {
            'describe': 'result = describe_dataset(data)',
            'preprocess': 'result = preprocess_data(data)',
            'train': {
                'logistic_regression': 'model = ThinkMLClassifier(model_type="logistic_regression")\nresult = model.fit(X_train, y_train)',
                'linear_regression': 'model = ThinkMLClassifier(model_type="linear_regression")\nresult = model.fit(X_train, y_train)',
                'decision_tree': 'model = ThinkMLClassifier(model_type="decision_tree")\nresult = model.fit(X_train, y_train)',
                'random_forest': 'model = ThinkMLClassifier(model_type="random_forest")\nresult = model.fit(X_train, y_train)'
            },
            'evaluate': 'result = evaluate_model(model, X_test, y_test)',
            'visualize': {
                'roc_curve': 'result = plot_roc_curve(model, X_test, y_test)',
                'feature_importance': 'result = plot_feature_importance(model, X.columns.tolist())',
                'correlation_heatmap': 'result = plot_correlation_heatmap(data)',
                'confusion_matrix': 'result = plot_confusion_matrix(model, X_test, y_test)'
            }
        }
        
        # Default model recommendations
        self.default_models = {
            'classification': 'logistic_regression',
            'regression': 'linear_regression'
        }

    def generate_code(self, command: Dict[str, Any]) -> str:
        """
        Generate code based on the command.
        
        Args:
            command (Dict[str, Any]): Structured command
            
        Returns:
            str: Generated Python code
        """
        task = command.get('task')
        if not task:
            raise ValueError("No task specified in command")
        
        # Get imports
        imports = []
        if task in self.imports:
            imports.append(self.imports[task])
        
        # Get code template
        if task == 'train':
            model_type = command.get('model', 'logistic_regression')
            if model_type not in self.code_templates['train']:
                raise ValueError(f"Unsupported model type: {model_type}")
            code = self.code_templates['train'][model_type]
        elif task == 'visualize':
            viz_type = command.get('visualization', 'roc_curve')
            if viz_type not in self.code_templates['visualize']:
                raise ValueError(f"Unsupported visualization type: {viz_type}")
            code = self.code_templates['visualize'][viz_type]
        else:
            if task not in self.code_templates:
                raise ValueError(f"Unsupported task: {task}")
            code = self.code_templates[task]
        
        # Add parameters if specified
        if 'parameters' in command and command['parameters']:
            params = []
            for key, value in command['parameters'].items():
                if isinstance(value, str):
                    params.append(f"{key}='{value}'")
                else:
                    params.append(f"{key}={value}")
            param_str = ', '.join(params)
            code = code.replace(')', f", {param_str})")
        
        # Combine imports and code
        return '\n'.join(imports + ['', code])

class ExecutionManager:
    """Manages code execution in a sandboxed environment."""
    
    def __init__(self):
        self.sandbox = {}
        self.data = None
        self.model = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def set_data(self, data: pd.DataFrame):
        """Set the data and prepare train/test splits."""
        # Robust data validation
        if data is None or not isinstance(data, pd.DataFrame):
            raise ExecutionError("Input data must be a pandas DataFrame.")
        if data.empty:
            raise ExecutionError("DataFrame is empty.")
        if data.shape[1] < 2:
            raise ExecutionError("DataFrame must have at least two columns (features and target).")
        if data.shape[0] < 2:
            raise ExecutionError("DataFrame must have at least two rows.")
        if data.isna().all().all():
            raise ExecutionError("All values in the DataFrame are missing.")
        if all(data.nunique(dropna=False) == 1):
            raise ExecutionError("All columns in the DataFrame have constant values.")
        # Check for invalid types in numeric columns
        for col in data.select_dtypes(include=[object]).columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                try:
                    pd.to_numeric(data[col])
                except Exception:
                    raise ExecutionError(f"Invalid data type in column '{col}'.")
        self.data = data
        if data is not None:
            # Prepare features and target
            if 'target' in data.columns:
                self.X = data.drop('target', axis=1)
                self.y = data['target']
            else:
                self.X = data.copy()
                self.y = None
            # Create train/test split if we have target
            if self.y is not None:
                from sklearn.model_selection import train_test_split
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    self.X, self.y, test_size=0.2, random_state=42
                )
        # Update sandbox with data variables
        self._update_sandbox()
    
    def _update_sandbox(self):
        """Update sandbox with current variables."""
        self.sandbox.update({
            'data': self.data,
            'X': self.X,
            'y': self.y,
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'model': self.model,
            'pd': pd,
            'np': np
        })
    
    def execute(self, code: str) -> Any:
        """
        Execute the generated code in a sandboxed environment.
        
        Args:
            code (str): Generated Python code to execute
            
        Returns:
            Any: Result of code execution
        """
        logger.info(f"Executing code:\n{code}")
        
        try:
            # Create a new string buffer for stdout
            stdout_buffer = StringIO()
            sys.stdout = stdout_buffer
            
            # Update sandbox before execution
            self._update_sandbox()
            
            # Execute the code in the sandbox
            exec(code, self.sandbox)
            
            # Get the result and update model if it was created
            result = self.sandbox.get('result')
            if 'model' in self.sandbox:
                self.model = self.sandbox['model']
            
            # Restore stdout
            sys.stdout = sys.__stdout__
            
            logger.info(f"Execution successful. Result type: {type(result)}")
            return result
            
        except Exception as e:
            logger.error(f"Execution failed: {str(e)}")
            raise ExecutionError(f"Code execution failed: {str(e)}")

class ErrorHandler:
    """Handles error formatting and suggestions."""
    
    @staticmethod
    def format_error(error: Exception) -> str:
        """Format an error with stack trace and suggestions."""
        error_type = type(error).__name__
        error_msg = str(error)
        stack_trace = ''.join(traceback.format_tb(error.__traceback__))
        
        formatted_error = f"""
Error Type: {error_type}
Message: {error_msg}
Stack Trace:
{stack_trace}"""
        
        return formatted_error.strip()

class ExecutionError(Exception):
    """Custom error for execution failures."""
    pass

# Example usage
if __name__ == "__main__":
    # Initialize the engine
    engine = ThinkMLEngine()
    
    # Example prompts
    prompts = [
        "Describe the dataset",
        "Train a decision tree on my data",
        "Plot the ROC curve",
        "Show feature importance"
    ]
    
    for prompt in prompts:
        try:
            print(f"\nProcessing prompt: {prompt}")
            result = engine.process(prompt)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {str(e)}") 