"""
Model suggestion module for ThinkML.
This module provides functionality to suggest appropriate machine learning models
based on the dataset characteristics and problem type.
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Optional, Any
import dask.dataframe as dd
from dask.diagnostics import ProgressBar


def suggest_model(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray, List],
    problem_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Suggest appropriate machine learning models based on the dataset characteristics
    and problem type.

    Parameters
    ----------
    X : pd.DataFrame
        Feature data
    y : Union[pd.Series, np.ndarray, List]
        Target variable
    problem_type : Optional[str], default=None
        Type of problem ('classification' or 'regression').
        If None, will be inferred from the data.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'problem_type': The inferred or provided problem type
        - 'recommended_models': List of dictionaries with model recommendations
          Each recommendation contains:
          - 'model': Model name
          - 'complexity': Time complexity in Big-O notation
          - 'reason': Short explanation for the recommendation

    Raises
    ------
    ValueError
        If X or y are empty or invalid
        If problem_type is not 'classification' or 'regression'
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

    # Check if dataset is large enough to use Dask
    use_dask = len(X) > 1_000_000

    # If problem_type is not provided, infer it from the data
    if problem_type is None:
        if use_dask:
            # Use Dask for large datasets
            ddf_y = dd.from_pandas(y, npartitions=max(1, len(y) // 100000))
            with ProgressBar():
                is_numeric = pd.api.types.is_numeric_dtype(y)
                if is_numeric:
                    unique_count = ddf_y.nunique().compute()
                    unique_ratio = unique_count / len(y)
                    problem_type = 'regression' if unique_ratio > 0.2 else 'classification'
                else:
                    problem_type = 'classification'
        else:
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

    # Define model suggestions with complexity metadata and reasons
    model_suggestions = {
        'classification': [
            {
                'model': 'Logistic Regression',
                'complexity': 'O(n * d)',
                'reason': 'Efficient for binary classification.'
            },
            {
                'model': 'Decision Tree Classifier',
                'complexity': 'O(n * log(n))',
                'reason': 'Handles non-linear data well.'
            },
            {
                'model': 'Random Forest Classifier',
                'complexity': 'O(n * log(n) * t)',
                'reason': 'Reduces overfitting, robust performance.'
            }
        ],
        'regression': [
            {
                'model': 'Linear Regression',
                'complexity': 'O(n * d^2)',
                'reason': 'Simple and interpretable regression.'
            },
            {
                'model': 'Ridge Regression',
                'complexity': 'O(n * d^2)',
                'reason': 'Regularized regression to prevent overfitting.'
            },
            {
                'model': 'Decision Tree Regressor',
                'complexity': 'O(n * log(n))',
                'reason': 'Handles non-linear data well.'
            }
        ]
    }

    return {
        'problem_type': problem_type,
        'recommended_models': model_suggestions[problem_type]
    } 