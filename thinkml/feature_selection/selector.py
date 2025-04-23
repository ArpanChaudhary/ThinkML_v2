"""
Feature selection functions for ThinkML.

This module provides functions for selecting the most relevant features from datasets
using various methods including variance threshold, correlation analysis, chi-squared test,
mutual information, and recursive feature elimination.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from sklearn.feature_selection import chi2, mutual_info_classif, mutual_info_regression, RFE
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def select_features(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    method: str = 'variance',
    threshold: float = 0.1,
    chunk_size: int = 100000,
    visualize: bool = True
) -> Dict:
    """
    Select features from a dataset using various methods.

    Parameters
    ----------
    X : pd.DataFrame
        Input features DataFrame.
    y : Optional[pd.Series], default=None
        Target variable. Required for 'chi2', 'mutual_info', and 'rfe' methods.
    method : str, default='variance'
        Method for feature selection. Options:
        - 'variance': Remove features with variance < threshold
        - 'correlation': Remove features with correlation > threshold
        - 'chi2': Use Chi-Squared test for feature selection (classification only)
        - 'mutual_info': Use Mutual Information for feature ranking
        - 'rfe': Use Recursive Feature Elimination
    threshold : float, default=0.1
        Threshold for variance or correlation filtering.
    chunk_size : int, default=100000
        Size of chunks for processing large datasets.
    visualize : bool, default=True
        If True, create interactive visualizations of feature importance.

    Returns
    -------
    Dict
        Dictionary containing:
        - 'selected_features': List of selected features
        - 'dropped_features': List of dropped features with reasons
        - 'scores': Dictionary of feature scores (if applicable)
        - 'visualization': Plotly figure object (if visualize=True)

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
    if method not in ['variance', 'correlation', 'chi2', 'mutual_info', 'rfe']:
        raise ValueError(
            f"Invalid method: {method}. Must be one of: "
            "'variance', 'correlation', 'chi2', 'mutual_info', 'rfe'"
        )
    
    # Validate target for supervised methods
    if method in ['chi2', 'mutual_info', 'rfe'] and y is None:
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
    elif method == 'mutual_info':
        result = _select_by_mutual_info(X, y, is_dask)
    else:  # rfe
        result = _select_by_rfe(X, y, is_dask)
    
    # Create visualization if requested
    if visualize and result['scores'] is not None:
        result['visualization'] = _visualize_feature_importance(result)
    
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
    
    # Calculate average correlation as score
    scores = {}
    for feature in X.columns:
        if feature in selected_features:
            # Calculate average correlation with other selected features
            other_features = [f for f in selected_features if f != feature]
            if other_features:
                scores[feature] = corr_matrix[feature][other_features].abs().mean()
            else:
                scores[feature] = 0
        else:
            scores[feature] = 0
    
    return {
        'selected_features': selected_features,
        'dropped_features': dropped_features,
        'scores': scores
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

def _select_by_rfe(
    X: Union[pd.DataFrame, dd.DataFrame],
    y: pd.Series,
    is_dask: bool
) -> Dict:
    """Select features using Recursive Feature Elimination."""
    if is_dask:
        with ProgressBar():
            X = X.compute()
    
    # Determine if classification or regression
    if len(np.unique(y)) < 10:  # Assuming classification if less than 10 unique values
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        estimator = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Select number of features to keep (half of total features)
    n_features_to_select = max(1, len(X.columns) // 2)
    
    # Fit RFE
    rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select)
    rfe.fit(X, y)
    
    # Get selected and dropped features
    selected_features = X.columns[rfe.support_].tolist()
    dropped_features = X.columns[~rfe.support_].tolist()
    
    # Get feature importances from the underlying estimator
    if hasattr(estimator, 'feature_importances_'):
        importances = estimator.feature_importances_
    else:
        # If no feature_importances_, use ranking as a proxy
        importances = 1 / rfe.ranking_
    
    # Create scores dictionary
    scores = dict(zip(X.columns, importances))
    
    # Create dropped features list with reasons
    dropped_features_with_reasons = [
        (feature, f"RFE ranking: {rfe.ranking_[list(X.columns).index(feature)]}")
        for feature in dropped_features
    ]
    
    return {
        'selected_features': selected_features,
        'dropped_features': dropped_features_with_reasons,
        'scores': scores
    }

def _visualize_feature_importance(result: Dict) -> go.Figure:
    """Create interactive visualization of feature importance."""
    # Sort features by score
    sorted_features = sorted(result['scores'].items(), key=lambda x: x[1], reverse=True)
    features = [item[0] for item in sorted_features]
    scores = [item[1] for item in sorted_features]
    
    # Create figure
    fig = go.Figure()
    
    # Add bar chart
    fig.add_trace(
        go.Bar(
            x=features,
            y=scores,
            marker_color=['green' if feat in result['selected_features'] else 'red' for feat in features],
            text=[f"{score:.4f}" for score in scores],
            textposition='auto',
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Feature Importance Scores",
        xaxis_title="Features",
        yaxis_title="Importance Score",
        xaxis_tickangle=-45,
        height=600,
        showlegend=False,
        template="plotly_white"
    )
    
    # Add annotations for dropped features
    for feature, reason in result['dropped_features']:
        idx = features.index(feature)
        fig.add_annotation(
            x=feature,
            y=scores[idx],
            text="Dropped",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40
        )
    
    return fig 