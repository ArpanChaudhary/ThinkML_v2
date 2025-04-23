"""
Data description module for ThinkML.
This module provides functionality to analyze and describe datasets.
"""

import pandas as pd
import numpy as np
from typing import Union, Dict, Optional, List, Any


def describe_data(X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray, List]] = None) -> Dict[str, Any]:
    """
    Generate a comprehensive description of the dataset.

    Args:
        X (pd.DataFrame): Feature data
        y (Optional[Union[pd.Series, np.ndarray, List]]): Target variable

    Returns:
        Dict[str, Any]: Dictionary containing dataset description with the following keys:
            - 'num_samples': Total number of samples
            - 'num_features': Total number of features
            - 'feature_types': Data types for each feature
            - 'missing_values': Missing value counts
            - 'memory_usage': Memory usage in KB
            - 'duplicate_rows': Number of duplicate rows
            - 'feature_summary': Statistical summary for each feature
            - 'correlation_matrix': Correlation matrix for numerical features
            - 'target_summary': Summary of target variable (if provided)
            - 'class_balance': Class distribution (if classification)
            - 'imbalance_status': Whether the dataset is imbalanced

    Raises:
        ValueError: If X is None or empty
    """
    # Input validation
    if X is None or len(X) == 0:
        raise ValueError("Input features (X) cannot be None or empty")

    # Convert y to pandas Series if provided
    if y is not None and not isinstance(y, pd.Series):
        y = pd.Series(y)

    # Basic dataset information
    description = {
        'num_samples': len(X),
        'num_features': len(X.columns),
        'feature_types': {},
        'missing_values': {},
        'memory_usage': round(X.memory_usage(deep=True).sum() / 1024, 2),  # Convert to KB
        'duplicate_rows': X.duplicated().sum(),
        'feature_summary': {},
        'correlation_matrix': None
    }

    # Analyze each feature
    for column in X.columns:
        # Determine feature type
        if pd.api.types.is_numeric_dtype(X[column]):
            feature_type = 'numerical'
            # Calculate numerical statistics
            description['feature_summary'][column] = {
                'type': feature_type,
                'min': X[column].min(),
                'max': X[column].max(),
                'mean': X[column].mean(),
                'std': X[column].std(),
                'median': X[column].median()
            }
        else:
            feature_type = 'categorical'
            # Calculate categorical statistics
            value_counts = X[column].value_counts()
            description['feature_summary'][column] = {
                'type': feature_type,
                'unique_count': len(value_counts),
                'top_category': value_counts.index[0],
                'top_frequency': value_counts.iloc[0]
            }

        description['feature_types'][column] = feature_type
        description['missing_values'][column] = X[column].isna().sum()

    # Calculate correlation matrix for numerical features
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    if len(numerical_features) > 0:
        description['correlation_matrix'] = X[numerical_features].corr().to_dict()

    # Analyze target variable if provided
    if y is not None:
        description['missing_values']['target'] = y.isna().sum()
        
        # Target summary
        description['target_summary'] = {
            'type': 'numerical' if pd.api.types.is_numeric_dtype(y) else 'categorical',
            'unique_values': len(y.unique()),
            'sample_distribution': y.value_counts().to_dict() if not pd.api.types.is_numeric_dtype(y) else None
        }

        # Check if it's a classification problem
        if not pd.api.types.is_numeric_dtype(y) or len(y.unique()) / len(y) < 0.2:
            # Calculate class balance
            class_counts = y.value_counts()
            class_percentages = (class_counts / len(y) * 100).round(2)
            
            description['class_balance'] = {
                'counts': class_counts.to_dict(),
                'percentages': class_percentages.to_dict()
            }
            
            # Determine imbalance status
            majority_class_percentage = class_percentages.max()
            description['imbalance_status'] = 'imbalanced' if majority_class_percentage > 60 else 'balanced'

    return description 