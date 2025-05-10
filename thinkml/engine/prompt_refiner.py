"""
Prompt Refiner - A module for normalizing and refining natural language prompts.

This module provides functionality to standardize various ways of expressing
the same ML task or command, making the prompt interpretation more robust.
"""

import re
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class PromptRefiner:
    """Refines and normalizes natural language prompts for better interpretation."""
    
    def __init__(self):
        # Define phrase mappings for task normalization
        self.task_mappings = {
            'describe': [
                'describe', 'analyze', 'summarize', 'statistics', 'overview',
                'show', 'display', 'list', 'report'
            ],
            'preprocess': [
                'preprocess', 'clean', 'prepare', 'normalize', 'standardize',
                'transform', 'process', 'handle', 'fix'
            ],
            'train': [
                'train', 'build', 'create', 'develop', 'fit',
                'learn', 'optimize', 'tune'
            ],
            'evaluate': [
                'evaluate', 'assess', 'test', 'validate', 'check',
                'measure', 'score', 'verify', 'examine',
                'accuracy', 'precision', 'recall', 'f1-score', 'f1 score', 'performance'
            ],
            'visualize': [
                'visualize', 'plot', 'graph', 'chart', 'show',
                'display', 'draw', 'illustrate', 'present'
            ]
        }
        
        # Define model name mappings
        self.model_mappings = {
            'logistic_regression': [
                'log reg', 'logistic', 'logit', 'classification',
                'binary classifier', 'logistic classifier'
            ],
            'linear_regression': [
                'linear', 'regression', 'linear model',
                'ordinary least squares', 'ols'
            ],
            'decision_tree': [
                'tree', 'decision tree', 'dt', 'cart',
                'classification tree', 'regression tree'
            ],
            'random_forest': [
                'forest', 'rf', 'random forest',
                'ensemble tree', 'bagging trees'
            ]
        }
        
        # Define visualization type mappings
        self.visualization_mappings = {
            'roc_curve': [
                'roc', 'receiver operating characteristic',
                'roc curve', 'roc plot'
            ],
            'feature_importance': [
                'feature importance', 'importance plot',
                'variable importance', 'feature ranking'
            ],
            'correlation_heatmap': [
                'correlation', 'correlation matrix',
                'heatmap', 'correlation plot'
            ],
            'confusion_matrix': [
                'confusion matrix', 'confusion plot',
                'error matrix', 'confusion chart'
            ]
        }
        
        # Compile regex patterns for faster matching
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for all mappings."""
        # Helper function to escape special characters
        def escape_pattern(pattern):
            return re.escape(pattern).replace('\\ ', '\\s*')
        
        self.task_patterns = {
            task: [re.compile(escape_pattern(phrase), re.IGNORECASE) 
                  for phrase in phrases]
            for task, phrases in self.task_mappings.items()
        }
        
        self.model_patterns = {
            model: [re.compile(escape_pattern(phrase), re.IGNORECASE)
                   for phrase in phrases]
            for model, phrases in self.model_mappings.items()
        }
        
        self.visualization_patterns = {
            viz_type: [re.compile(escape_pattern(phrase), re.IGNORECASE)
                      for phrase in phrases]
            for viz_type, phrases in self.visualization_mappings.items()
        }
    
    def refine(self, prompt: str) -> Dict[str, Optional[str]]:
        """
        Refine and normalize a natural language prompt.
        
        Args:
            prompt (str): The input prompt to refine
            
        Returns:
            Dict[str, Optional[str]]: Refined prompt components including:
                - task: The normalized task type
                - model: The normalized model type (if present)
                - visualization: The visualization type (if present)
                - original_prompt: The original prompt
        """
        logger.info(f"Refining prompt: {prompt}")
        
        # Initialize result structure
        result = {
            'task': None,
            'model': None,
            'visualization': None,
            'original_prompt': prompt
        }
        
        # Convert to lowercase for case-insensitive matching
        prompt_lower = prompt.lower()

        # Identify explicit visualization keywords (strong indicators)
        explicit_viz_keywords = [
            'plot', 'roc', 'feature importance', 'correlation heatmap', 'confusion matrix',
            'roc curve', 'roc plot', 'importance plot', 'correlation plot', 'confusion plot',
            'draw', 'graph', 'chart', 'illustrate', 'present'
        ]
        # Only consider explicit viz if the prompt starts with or is dominated by these keywords
        is_explicit_viz = any(prompt_lower.strip().startswith(kw) or f' {kw} ' in prompt_lower for kw in explicit_viz_keywords)

        # Identify task (preprocess takes precedence over visualize unless explicit viz)
        found_preprocess = False
        for task, patterns in self.task_patterns.items():
            if any(pattern.search(prompt_lower) for pattern in patterns):
                if task == 'preprocess':
                    found_preprocess = True
                result['task'] = task
                break

        # Identify model
        for model, patterns in self.model_patterns.items():
            if any(pattern.search(prompt_lower) for pattern in patterns):
                result['model'] = model
                break

        # Identify visualization type
        for viz_type, patterns in self.visualization_patterns.items():
            if any(pattern.search(prompt_lower) for pattern in patterns):
                result['visualization'] = viz_type
                # Only set task to 'visualize' if not preprocess or if explicit viz
                if not found_preprocess or is_explicit_viz:
                    result['task'] = 'visualize'
                break
        
        logger.info(f"Refined prompt components: {result}")
        return result
    
    def get_supported_phrases(self) -> Dict[str, List[str]]:
        """
        Get all supported phrases for each category.
        
        Returns:
            Dict[str, List[str]]: Dictionary of supported phrases by category
        """
        return {
            'tasks': list(self.task_mappings.keys()),
            'models': list(self.model_mappings.keys()),
            'visualizations': list(self.visualization_mappings.keys())
        }
    
    def get_suggestions(self, prompt: str) -> List[str]:
        """
        Get suggestions for unsupported or ambiguous prompts.
        
        Args:
            prompt (str): The input prompt
            
        Returns:
            List[str]: List of suggested alternative prompts
        """
        refined = self.refine(prompt)
        suggestions = []
        
        if not refined['task']:
            suggestions.extend([
                f"Try using one of these tasks: {', '.join(self.task_mappings.keys())}",
                "Example: 'describe the data' or 'train a model'"
            ])
        
        if refined['task'] == 'train' and not refined['model']:
            suggestions.extend([
                f"Available models: {', '.join(self.model_mappings.keys())}",
                "Example: 'train a logistic regression model'"
            ])
        
        if refined['task'] == 'visualize' and not refined['visualization']:
            suggestions.extend([
                f"Available visualizations: {', '.join(self.visualization_mappings.keys())}",
                "Example: 'plot the ROC curve' or 'show feature importance'"
            ])
        
        return suggestions 