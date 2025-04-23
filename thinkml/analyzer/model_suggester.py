"""
Model suggestion module for ThinkML.
This module provides functionality to suggest appropriate machine learning models
based on the dataset characteristics and problem type.
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Optional


def suggest_model(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray, List],
    problem_type: Optional[str] = None
) -> Dict[str, Union[str, List[Dict[str, str]]]]:
    """
    Suggest appropriate machine learning models based on the dataset characteristics
    and problem type.

    Args:
        X (pd.DataFrame): Feature data
        y (Union[pd.Series, np.ndarray, List]): Target variable
        problem_type (Optional[str]): Type of problem ('classification' or 'regression').
                                    If None, will be inferred from the data.

    Returns:
        Dict[str, Union[str, List[Dict[str, str]]]]: Dictionary containing:
            - 'problem_type': The inferred or provided problem type
            - 'recommended_models': List of dictionaries with model names and complexity

    Raises:
        ValueError: If X or y are empty or invalid
    """
    # Input validation
    if X is None or y is None:
        raise ValueError("Input features (X) and target (y) cannot be None")
    
    if len(X) == 0 or len(y) == 0:
        raise ValueError("Input features (X) and target (y) cannot be empty")
    
    if len(X) != len(y):
        raise ValueError("Input features (X) and target (y) must have the same length")

    # Convert y to pandas Series if it's not already
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    # If problem_type is not provided, infer it from the data
    if problem_type is None:
        # Check if y is numeric
        is_numeric = pd.api.types.is_numeric_dtype(y)
        
        if is_numeric:
            # Count unique values relative to total length
            unique_ratio = len(y.unique()) / len(y)
            
            # If more than 20% unique values, consider it regression
            problem_type = 'regression' if unique_ratio > 0.2 else 'classification'
        else:
            problem_type = 'classification'

    # Validate problem_type
    if problem_type not in ['classification', 'regression']:
        raise ValueError("problem_type must be either 'classification' or 'regression'")

    # Define model suggestions with complexity metadata
    model_suggestions = {
        'classification': [
            {'model': 'Logistic Regression', 'complexity': 'O(n * d)'},
            {'model': 'Decision Tree Classifier', 'complexity': 'O(n * log(n))'},
            {'model': 'Random Forest Classifier', 'complexity': 'O(n * log(n) * t)'}
        ],
        'regression': [
            {'model': 'Linear Regression', 'complexity': 'O(n * d^2)'},
            {'model': 'Ridge Regression', 'complexity': 'O(n * d^2)'},
            {'model': 'Decision Tree Regressor', 'complexity': 'O(n * log(n))'}
        ]
    }

    return {
        'problem_type': problem_type,
        'recommended_models': model_suggestions[problem_type]
    } 