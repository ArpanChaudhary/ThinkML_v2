"""
Feature selection functions for ThinkML.

This module provides functions for selecting the most relevant features from datasets
using various methods including variance threshold, correlation analysis, chi-squared test,
and mutual information.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from sklearn.feature_selection import chi2, mutual_info_classif, mutual_info_regression
from scipy import stats

def select_features(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    method: str = 'variance',
    threshold: float = 0.1,
    chunk_size: int = 100000
) -> Dict:
    """
    Select features from a dataset using various methods.

    Parameters
    ----------
    X : pd.DataFrame
        Input features DataFrame.
    y : Optional[pd.Series], default=None
        Target variable. Required for 'chi2' and 'mutual_info' methods.
    method : str, default='variance'
        Method for feature selection. Options:
        - 'variance': Remove features with variance < threshold
        - 'correlation': Remove features with correlation > threshold
        - 'chi2': Use Chi-Squared test for feature selection (classification only)
        - 'mutual_info': Use Mutual Information for feature ranking
    threshold : float, default=0.1
        Threshold for variance or correlation filtering.
    chunk_size : int, default=100000
        Size of chunks for processing large datasets.

    Returns
    -------
    Dict
        Dictionary containing:
        - 'selected_features': List of selected features
        - 'dropped_features': List of dropped features with reasons
        - 'scores': Dictionary of feature scores (if applicable)

    Raises
    ------
    ValueError
        If method is invalid or if data types are incompatible.
    """
    # Validate input
    if X.empty:
        raise ValueError("Input DataFrame X cannot be empty")
    
    # Check for non-numerical columns
    non_numeric_cols = X.select_dtypes(exclude=['int64', 'float64']).columns
    if len(non_numeric_cols) > 0:
        raise ValueError(f"Non-numerical columns found: {non_numeric_cols}. Only numerical columns are supported.")
    
    # Validate method
    if method not in ['variance', 'correlation', 'chi2', 'mutual_info']:
        raise ValueError(
            f"Invalid method: {method}. Must be one of: "
            "'variance', 'correlation', 'chi2', 'mutual_info'"
        )
    
    # Validate target for supervised methods
    if method in ['chi2', 'mutual_info'] and y is None:
        raise ValueError(f"Target variable y is required for method '{method}'")
    
    # Handle large datasets
    if len(X) > 1_000_000:
        X = dd.from_pandas(X, chunksize=chunk_size)
        is_dask = True
    else:
        is_dask = False
    
    # Select features based on method
    if method == 'variance':
        result = _select_by_variance(X, threshold, is_dask)
    elif method == 'correlation':
        result = _select_by_correlation(X, threshold, is_dask)
    elif method == 'chi2':
        result = _select_by_chi2(X, y, is_dask)
    else:  # mutual_info
        result = _select_by_mutual_info(X, y, is_dask)
    
    return result

def _select_by_variance(
    X: Union[pd.DataFrame, dd.DataFrame],
    threshold: float,
    is_dask: bool
) -> Dict:
    """Select features based on variance threshold."""
    if is_dask:
        with ProgressBar():
            variances = X.var().compute()
    else:
        variances = X.var()
    
    # Find features with variance below threshold
    low_var_features = variances[variances < threshold].index.tolist()
    selected_features = variances[variances >= threshold].index.tolist()
    
    # Create dropped features list with reasons
    dropped_features = [
        (feature, f"Variance {var:.6f} below threshold {threshold}")
        for feature, var in variances[variances < threshold].items()
    ]
    
    return {
        'selected_features': selected_features,
        'dropped_features': dropped_features,
        'scores': {feature: var for feature, var in variances.items()}
    }

def _select_by_correlation(
    X: Union[pd.DataFrame, dd.DataFrame],
    threshold: float,
    is_dask: bool
) -> Dict:
    """Select features based on correlation threshold."""
    if is_dask:
        with ProgressBar():
            corr_matrix = X.corr().compute()
    else:
        corr_matrix = X.corr()
    
    # Find highly correlated features
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column].abs() > threshold)]
    
    selected_features = [col for col in X.columns if col not in to_drop]
    
    # Create dropped features list with reasons
    dropped_features = []
    for feature in to_drop:
        corr_features = upper_tri[feature][upper_tri[feature].abs() > threshold]
        reasons = [f"Correlation {corr:.3f} with {idx}" for idx, corr in corr_features.items()]
        dropped_features.append((feature, "; ".join(reasons)))
    
    return {
        'selected_features': selected_features,
        'dropped_features': dropped_features,
        'scores': None
    }

def _select_by_chi2(
    X: Union[pd.DataFrame, dd.DataFrame],
    y: pd.Series,
    is_dask: bool
) -> Dict:
    """Select features using Chi-Squared test."""
    if is_dask:
        with ProgressBar():
            X = X.compute()
    
    # Calculate chi-squared scores
    scores, _ = chi2(X, y)
    feature_scores = dict(zip(X.columns, scores))
    
    # Sort features by score
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Select top features (keep all with non-zero scores)
    selected_features = [feature for feature, score in sorted_features if score > 0]
    dropped_features = [(feature, f"Chi-squared score {score:.3f}") 
                       for feature, score in sorted_features if score == 0]
    
    return {
        'selected_features': selected_features,
        'dropped_features': dropped_features,
        'scores': feature_scores
    }

def _select_by_mutual_info(
    X: Union[pd.DataFrame, dd.DataFrame],
    y: pd.Series,
    is_dask: bool
) -> Dict:
    """Select features using Mutual Information."""
    if is_dask:
        with ProgressBar():
            X = X.compute()
    
    # Determine if classification or regression
    if len(np.unique(y)) < 10:  # Assuming classification if less than 10 unique values
        scores = mutual_info_classif(X, y)
    else:
        scores = mutual_info_regression(X, y)
    
    feature_scores = dict(zip(X.columns, scores))
    
    # Sort features by score
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Select features with non-zero scores
    selected_features = [feature for feature, score in sorted_features if score > 0]
    dropped_features = [(feature, f"Mutual information score {score:.3f}")
                       for feature, score in sorted_features if score == 0]
    
    return {
        'selected_features': selected_features,
        'dropped_features': dropped_features,
        'scores': feature_scores
    } 