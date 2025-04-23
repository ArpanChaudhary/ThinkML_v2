"""
Tests for the feature selection functionality in ThinkML.
"""

import pytest
import pandas as pd
import numpy as np
from thinkml.feature_selection.selector import select_features

@pytest.fixture
def sample_data():
    """Create a sample DataFrame with various feature characteristics."""
    np.random.seed(42)
    n_samples = 1000
    
    # Create features with different characteristics
    data = {
        # Low variance feature
        'low_var': np.random.normal(0, 0.01, n_samples),
        
        # High variance feature
        'high_var': np.random.normal(0, 1, n_samples),
        
        # Highly correlated features
        'corr1': np.random.normal(0, 1, n_samples),
        'corr2': np.random.normal(0, 1, n_samples) + 0.9 * np.random.normal(0, 1, n_samples),
        
        # Independent features
        'indep1': np.random.normal(0, 1, n_samples),
        'indep2': np.random.normal(0, 1, n_samples),
        
        # Categorical target for chi2
        'cat_target': np.random.choice([0, 1, 2], n_samples),
        
        # Continuous target for mutual_info
        'cont_target': np.random.normal(0, 1, n_samples)
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def empty_data():
    """Create an empty DataFrame."""
    return pd.DataFrame()

@pytest.fixture
def non_numeric_data():
    """Create a DataFrame with non-numeric columns."""
    return pd.DataFrame({
        'numeric': [1, 2, 3],
        'string': ['a', 'b', 'c'],
        'category': pd.Categorical(['x', 'y', 'z'])
    })

def test_variance_selection(sample_data):
    """Test feature selection using variance threshold."""
    result = select_features(sample_data, method='variance', threshold=0.1)
    
    # Check result structure
    assert isinstance(result, dict)
    assert 'selected_features' in result
    assert 'dropped_features' in result
    assert 'scores' in result
    
    # Check that low variance feature was dropped
    assert 'low_var' not in result['selected_features']
    assert any('low_var' in str(feat) for feat in result['dropped_features'])
    
    # Check that high variance feature was kept
    assert 'high_var' in result['selected_features']
    
    # Check scores
    assert result['scores']['low_var'] < 0.1
    assert result['scores']['high_var'] > 0.1

def test_correlation_selection(sample_data):
    """Test feature selection using correlation threshold."""
    result = select_features(sample_data, method='correlation', threshold=0.8)
    
    # Check result structure
    assert isinstance(result, dict)
    assert 'selected_features' in result
    assert 'dropped_features' in result
    
    # Check that one of the correlated features was dropped
    corr_features = ['corr1', 'corr2']
    assert sum(feat in result['selected_features'] for feat in corr_features) == 1
    assert sum(feat in str(result['dropped_features']) for feat in corr_features) == 1
    
    # Check that independent features were kept
    assert 'indep1' in result['selected_features']
    assert 'indep2' in result['selected_features']

def test_chi2_selection(sample_data):
    """Test feature selection using Chi-Squared test."""
    X = sample_data.drop(['cat_target', 'cont_target'], axis=1)
    y = sample_data['cat_target']
    
    result = select_features(X, y, method='chi2')
    
    # Check result structure
    assert isinstance(result, dict)
    assert 'selected_features' in result
    assert 'dropped_features' in result
    assert 'scores' in result
    
    # Check that scores are calculated
    assert all(feat in result['scores'] for feat in X.columns)
    
    # Check that features are ranked by chi2 score
    scores = list(result['scores'].values())
    assert scores == sorted(scores, reverse=True)

def test_mutual_info_selection(sample_data):
    """Test feature selection using Mutual Information."""
    X = sample_data.drop(['cat_target', 'cont_target'], axis=1)
    y = sample_data['cont_target']
    
    result = select_features(X, y, method='mutual_info')
    
    # Check result structure
    assert isinstance(result, dict)
    assert 'selected_features' in result
    assert 'dropped_features' in result
    assert 'scores' in result
    
    # Check that scores are calculated
    assert all(feat in result['scores'] for feat in X.columns)
    
    # Check that features are ranked by mutual info score
    scores = list(result['scores'].values())
    assert scores == sorted(scores, reverse=True)

def test_invalid_method(sample_data):
    """Test that invalid method raises ValueError."""
    with pytest.raises(ValueError):
        select_features(sample_data, method='invalid_method')

def test_non_numeric_input(non_numeric_data):
    """Test that non-numeric data raises ValueError."""
    with pytest.raises(ValueError):
        select_features(non_numeric_data)

def test_empty_dataframe(empty_data):
    """Test that empty DataFrame raises ValueError."""
    with pytest.raises(ValueError):
        select_features(empty_data)

def test_missing_target(sample_data):
    """Test that missing target raises ValueError for supervised methods."""
    with pytest.raises(ValueError):
        select_features(sample_data, method='chi2')
    
    with pytest.raises(ValueError):
        select_features(sample_data, method='mutual_info')

def test_large_dataset_chunk_processing():
    """Test handling of large datasets with chunk processing."""
    # Create a large dataset
    n_samples = 1_100_000  # Just over the threshold for Dask
    df = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 0.01, n_samples)  # Low variance
    })
    
    result = select_features(df, method='variance', threshold=0.1, chunk_size=100000)
    
    # Check that low variance feature was dropped
    assert 'feature3' not in result['selected_features']
    assert 'feature1' in result['selected_features']
    assert 'feature2' in result['selected_features']

def test_threshold_parameter(sample_data):
    """Test that threshold parameter affects feature selection."""
    # Test with different thresholds
    result1 = select_features(sample_data, method='variance', threshold=0.01)
    result2 = select_features(sample_data, method='variance', threshold=0.5)
    
    # More features should be selected with lower threshold
    assert len(result1['selected_features']) >= len(result2['selected_features'])

def test_correlation_threshold(sample_data):
    """Test that correlation threshold affects feature selection."""
    # Test with different thresholds
    result1 = select_features(sample_data, method='correlation', threshold=0.5)
    result2 = select_features(sample_data, method='correlation', threshold=0.95)
    
    # More features should be dropped with lower threshold
    assert len(result1['dropped_features']) >= len(result2['dropped_features']) 