"""
Test cases for the data_describer module.
"""

import pytest
import pandas as pd
import numpy as np
from thinkml.describer.data_describer import describe_data


def test_numerical_dataset_regression():
    """Test description of a numerical dataset for regression."""
    # Create a numerical dataset
    X = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'feature3': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    })
    
    result = describe_data(X)
    
    # Check basic information
    assert result['num_samples'] == 10
    assert result['num_features'] == 3
    assert result['duplicate_rows'] == 0
    
    # Check feature types
    assert all(feature_type == 'numerical' for feature_type in result['feature_types'].values())
    
    # Check feature summaries
    for feature in X.columns:
        assert result['feature_summary'][feature]['type'] == 'numerical'
        assert 'min' in result['feature_summary'][feature]
        assert 'max' in result['feature_summary'][feature]
        assert 'mean' in result['feature_summary'][feature]
        assert 'std' in result['feature_summary'][feature]
        assert 'median' in result['feature_summary'][feature]
        
        # Verify specific values
        assert result['feature_summary'][feature]['min'] == X[feature].min()
        assert result['feature_summary'][feature]['max'] == X[feature].max()
        assert abs(result['feature_summary'][feature]['mean'] - X[feature].mean()) < 1e-10
    
    # Check correlation matrix
    assert result['correlation_matrix'] is not None
    assert len(result['correlation_matrix']) == 3  # 3 features
    assert all(len(correlations) == 3 for correlations in result['correlation_matrix'].values())
    
    # Check memory usage
    assert isinstance(result['memory_usage'], float)
    assert result['memory_usage'] > 0


def test_categorical_dataset_classification():
    """Test description of a mixed dataset with categorical target for classification."""
    # Create a mixed dataset
    X = pd.DataFrame({
        'numeric': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'categorical': ['A', 'B', 'A', 'C', 'B', 'A', 'D', 'C', 'B', 'A']
    })
    
    # Create a categorical target with class imbalance
    y = pd.Series([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])  # 7 class 0, 3 class 1
    
    result = describe_data(X, y)
    
    # Check basic information
    assert result['num_samples'] == 10
    assert result['num_features'] == 2
    
    # Check feature types
    assert result['feature_types']['numeric'] == 'numerical'
    assert result['feature_types']['categorical'] == 'categorical'
    
    # Check feature summaries
    assert result['feature_summary']['numeric']['type'] == 'numerical'
    assert result['feature_summary']['categorical']['type'] == 'categorical'
    
    # Check categorical feature summary
    assert 'unique_count' in result['feature_summary']['categorical']
    assert 'top_category' in result['feature_summary']['categorical']
    assert 'top_frequency' in result['feature_summary']['categorical']
    assert result['feature_summary']['categorical']['unique_count'] == 4  # A, B, C, D
    assert result['feature_summary']['categorical']['top_category'] == 'A'
    assert result['feature_summary']['categorical']['top_frequency'] == 4
    
    # Check target summary
    assert 'target_summary' in result
    assert result['target_summary']['type'] == 'categorical'
    assert result['target_summary']['unique_values'] == 2
    
    # Check class balance
    assert 'class_balance' in result
    assert 'counts' in result['class_balance']
    assert 'percentages' in result['class_balance']
    assert result['class_balance']['counts'][0] == 7
    assert result['class_balance']['counts'][1] == 3
    assert result['class_balance']['percentages'][0] == 70.0
    assert result['class_balance']['percentages'][1] == 30.0
    
    # Check imbalance status
    assert result['imbalance_status'] == 'imbalanced'  # 70% is > 60%


def test_balanced_classification():
    """Test description of a dataset with balanced classes."""
    X = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    })
    
    # Create a balanced target
    y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # 5 class 0, 5 class 1
    
    result = describe_data(X, y)
    
    # Check class balance
    assert result['class_balance']['counts'][0] == 5
    assert result['class_balance']['counts'][1] == 5
    assert result['class_balance']['percentages'][0] == 50.0
    assert result['class_balance']['percentages'][1] == 50.0
    
    # Check imbalance status
    assert result['imbalance_status'] == 'balanced'  # 50% is < 60%


def test_missing_values_handling():
    """Test handling of missing values in features and target."""
    # Create a dataset with missing values
    X = pd.DataFrame({
        'feature1': [1, np.nan, 3, 4, 5, np.nan, 7, 8, 9, 10],
        'feature2': [10, 20, np.nan, 40, 50, 60, 70, np.nan, 90, 100]
    })
    
    # Create a target with missing values
    y = pd.Series([0, 1, np.nan, 0, 1, 0, 1, 0, np.nan, 1])
    
    result = describe_data(X, y)
    
    # Check missing values in features
    assert result['missing_values']['feature1'] == 2
    assert result['missing_values']['feature2'] == 2
    
    # Check missing values in target
    assert result['missing_values']['target'] == 2


def test_duplicate_rows_detection():
    """Test detection of duplicate rows."""
    # Create a dataset with duplicate rows
    X = pd.DataFrame({
        'feature1': [1, 2, 3, 1, 2, 3, 4, 5, 6, 7],
        'feature2': ['A', 'B', 'C', 'A', 'B', 'C', 'D', 'E', 'F', 'G']
    })
    
    result = describe_data(X)
    
    # Check duplicate rows count
    assert result['duplicate_rows'] == 3  # Three pairs of duplicates


def test_invalid_inputs():
    """Test handling of invalid inputs."""
    # Test with None input
    with pytest.raises(ValueError):
        describe_data(None)
    
    # Test with empty DataFrame
    with pytest.raises(ValueError):
        describe_data(pd.DataFrame())
    
    # Test with empty DataFrame but with columns
    with pytest.raises(ValueError):
        describe_data(pd.DataFrame(columns=['col1', 'col2'])) 