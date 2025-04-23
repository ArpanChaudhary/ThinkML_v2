"""
Outlier detection functions for ThinkML.

This module provides functions for detecting outliers in datasets using various methods
including Z-score, IQR, and Isolation Forest.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Optional, Tuple
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from scipy import stats

def detect_outliers(
    X: pd.DataFrame,
    method: str = 'zscore',
    report: bool = True,
    visualize: bool = True,
    chunk_size: int = 100000
) -> Dict:
    """
    Detect outliers in a dataset using various methods.

    Parameters
    ----------
    X : pd.DataFrame
        Input features DataFrame (numerical features only).
    method : str, default='zscore'
        Method for outlier detection. Options:
        - 'zscore': Identify outliers with |z| > 3
        - 'iqr': Identify outliers outside Q1 - 1.5*IQR and Q3 + 1.5*IQR
        - 'isolation_forest': Use sklearn's IsolationForest to detect outliers
    report : bool, default=True
        If True, print summary report.
    visualize : bool, default=True
        If True, show boxplots with highlighted outliers.
    chunk_size : int, default=100000
        Size of chunks for processing large datasets.

    Returns
    -------
    Dict
        Dictionary containing:
        - 'outlier_counts': Number of outliers per feature
        - 'outlier_percentage': Percentage of affected rows
        - 'outlier_indices': Index list of rows with at least one outlier
        - 'feature_outliers': Dictionary of outlier indices for each feature

    Raises
    ------
    ValueError
        If method is invalid or if non-numerical columns are present.
    """
    # Validate input
    if X.empty:
        raise ValueError("Input DataFrame X cannot be empty")
    
    # Check for non-numerical columns
    non_numeric_cols = X.select_dtypes(exclude=['int64', 'float64']).columns
    if len(non_numeric_cols) > 0:
        raise ValueError(f"Non-numerical columns found: {non_numeric_cols}. Only numerical columns are supported.")
    
    # Validate method
    if method not in ['zscore', 'iqr', 'isolation_forest']:
        raise ValueError(
            f"Invalid method: {method}. Must be one of: "
            "'zscore', 'iqr', 'isolation_forest'"
        )
    
    # Handle large datasets
    if len(X) > 1_000_000:
        X = dd.from_pandas(X, chunksize=chunk_size)
        is_dask = True
    else:
        is_dask = False
    
    # Get numerical columns
    if is_dask:
        numerical_cols = X.columns
    else:
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Detect outliers based on method
    if method == 'zscore':
        result = _detect_zscore_outliers(X, numerical_cols, is_dask)
    elif method == 'iqr':
        result = _detect_iqr_outliers(X, numerical_cols, is_dask)
    elif method == 'isolation_forest':
        result = _detect_isolation_forest_outliers(X, numerical_cols, is_dask)
    
    # Generate report if requested
    if report:
        _print_outlier_report(result)
    
    # Visualize outliers if requested
    if visualize:
        _visualize_outliers(X, result, numerical_cols, is_dask)
    
    return result

def _detect_zscore_outliers(X: Union[pd.DataFrame, dd.DataFrame], numerical_cols: pd.Index, is_dask: bool) -> Dict:
    """Detect outliers using Z-score method (|z| > 3)."""
    outlier_counts = {}
    feature_outliers = {}
    all_outlier_indices = set()
    
    for col in numerical_cols:
        if is_dask:
            with ProgressBar():
                data = X[col].compute()
        else:
            data = X[col]
        
        # Calculate Z-scores
        z_scores = np.abs(stats.zscore(data.dropna()))
        outlier_indices = np.where(z_scores > 3)[0]
        
        # Map back to original indices
        non_null_indices = data.dropna().index
        original_indices = non_null_indices[outlier_indices]
        
        # Store results
        outlier_counts[col] = len(original_indices)
        feature_outliers[col] = original_indices.tolist()
        all_outlier_indices.update(original_indices)
    
    # Calculate percentage of affected rows
    if is_dask:
        total_rows = len(X)
    else:
        total_rows = len(X)
    
    outlier_percentage = (len(all_outlier_indices) / total_rows) * 100
    
    return {
        'outlier_counts': outlier_counts,
        'outlier_percentage': outlier_percentage,
        'outlier_indices': sorted(list(all_outlier_indices)),
        'feature_outliers': feature_outliers
    }

def _detect_iqr_outliers(X: Union[pd.DataFrame, dd.DataFrame], numerical_cols: pd.Index, is_dask: bool) -> Dict:
    """Detect outliers using IQR method."""
    outlier_counts = {}
    feature_outliers = {}
    all_outlier_indices = set()
    
    for col in numerical_cols:
        if is_dask:
            with ProgressBar():
                data = X[col].compute()
        else:
            data = X[col]
        
        # Calculate IQR
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Find outliers
        outlier_indices = data[(data < lower_bound) | (data > upper_bound)].index
        
        # Store results
        outlier_counts[col] = len(outlier_indices)
        feature_outliers[col] = outlier_indices.tolist()
        all_outlier_indices.update(outlier_indices)
    
    # Calculate percentage of affected rows
    if is_dask:
        total_rows = len(X)
    else:
        total_rows = len(X)
    
    outlier_percentage = (len(all_outlier_indices) / total_rows) * 100
    
    return {
        'outlier_counts': outlier_counts,
        'outlier_percentage': outlier_percentage,
        'outlier_indices': sorted(list(all_outlier_indices)),
        'feature_outliers': feature_outliers
    }

def _detect_isolation_forest_outliers(X: Union[pd.DataFrame, dd.DataFrame], numerical_cols: pd.Index, is_dask: bool) -> Dict:
    """Detect outliers using Isolation Forest."""
    if is_dask:
        with ProgressBar():
            data = X.compute()
    else:
        data = X
    
    # Fit Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    predictions = iso_forest.fit_predict(data)
    
    # Get outlier indices (-1 indicates outlier)
    outlier_indices = np.where(predictions == -1)[0]
    
    # Count outliers per feature
    outlier_counts = {}
    feature_outliers = {}
    
    for col in numerical_cols:
        # For Isolation Forest, we don't have per-feature outliers
        # but we can calculate how many outliers have extreme values in each feature
        col_data = data[col]
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        extreme_indices = col_data[(col_data < lower_bound) | (col_data > upper_bound)].index
        outlier_counts[col] = len(extreme_indices)
        feature_outliers[col] = extreme_indices.tolist()
    
    # Calculate percentage of affected rows
    total_rows = len(data)
    outlier_percentage = (len(outlier_indices) / total_rows) * 100
    
    return {
        'outlier_counts': outlier_counts,
        'outlier_percentage': outlier_percentage,
        'outlier_indices': outlier_indices.tolist(),
        'feature_outliers': feature_outliers
    }

def _print_outlier_report(result: Dict) -> None:
    """Print a summary report of detected outliers."""
    print("\n===== OUTLIER DETECTION REPORT =====")
    print(f"Total outliers detected: {len(result['outlier_indices'])}")
    print(f"Percentage of affected rows: {result['outlier_percentage']:.2f}%")
    print("\nOutliers per feature:")
    for feature, count in result['outlier_counts'].items():
        print(f"  - {feature}: {count} outliers")
    print("=====================================\n")

def _visualize_outliers(X: Union[pd.DataFrame, dd.DataFrame], result: Dict, numerical_cols: pd.Index, is_dask: bool) -> None:
    """Visualize outliers using interactive boxplots."""
    if is_dask:
        with ProgressBar():
            data = X.compute()
    else:
        data = X
    
    # Create subplots
    n_cols = min(3, len(numerical_cols))
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, 
        cols=n_cols,
        subplot_titles=[f'Outliers in {col}' for col in numerical_cols],
        specs=[[{"type": "box"}] for _ in range(n_rows * n_cols)]
    )
    
    for idx, col in enumerate(numerical_cols):
        row = idx // n_cols + 1
        col_idx = idx % n_cols + 1
        
        # Add boxplot
        fig.add_trace(
            go.Box(
                y=data[col],
                name=col,
                boxpoints=False,  # Don't show default outliers
                jitter=0.3,
                pointpos=-1.8,
                hovertemplate="Feature: %{x}<br>Value: %{y}<extra></extra>"
            ),
            row=row, col=col_idx
        )
        
        # Add outliers
        outlier_indices = result['feature_outliers'][col]
        if len(outlier_indices) > 0:
            outlier_values = data.loc[outlier_indices, col]
            
            fig.add_trace(
                go.Scatter(
                    x=[col] * len(outlier_values),
                    y=outlier_values,
                    mode='markers',
                    marker=dict(
                        color='red',
                        size=10,
                        symbol='x'
                    ),
                    name=f'{col} Outliers',
                    hovertemplate="Feature: %{x}<br>Value: %{y}<br>Outlier<extra></extra>"
                ),
                row=row, col=col_idx
            )
            
            # Add annotations for extreme outliers (top 5)
            if len(outlier_values) > 0:
                sorted_outliers = outlier_values.sort_values(ascending=False)
                for i, (idx, val) in enumerate(sorted_outliers.head(5).items()):
                    fig.add_annotation(
                        x=col,
                        y=val,
                        text=f"{val:.2f}",
                        showarrow=True,
                        arrowhead=1,
                        row=row, col=col_idx
                    )
    
    fig.update_layout(
        height=300 * n_rows,
        width=1200,
        title_text="Outlier Detection Results",
        showlegend=False,
        template="plotly_white"
    )
    
    fig.show() 