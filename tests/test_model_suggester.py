"""
Test cases for the model_suggester module.
"""

import pytest
import pandas as pd
import numpy as np
from thinkml.analyzer.model_suggester import suggest_model


def test_classification_with_explicit_type():
    """Test model suggestion for classification with explicit problem type."""
    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y = pd.Series([0, 1, 0])
    
    result = suggest_model(X, y, problem_type='classification')
    
    assert result['problem_type'] == 'classification'
    assert len(result['recommended_models']) == 3
    
    # Check model names and complexities
    model_names = [model['model'] for model in result['recommended_models']]
    assert 'Logistic Regression' in model_names
    assert 'Decision Tree Classifier' in model_names
    assert 'Random Forest Classifier' in model_names
    
    # Check complexities
    for model in result['recommended_models']:
        assert 'complexity' in model
        assert isinstance(model['complexity'], str)
        assert model['complexity'].startswith('O(')


def test_regression_with_explicit_type():
    """Test model suggestion for regression with explicit problem type."""
    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y = pd.Series([1.5, 2.5, 3.5])
    
    result = suggest_model(X, y, problem_type='regression')
    
    assert result['problem_type'] == 'regression'
    assert len(result['recommended_models']) == 3
    
    # Check model names and complexities
    model_names = [model['model'] for model in result['recommended_models']]
    assert 'Linear Regression' in model_names
    assert 'Ridge Regression' in model_names
    assert 'Decision Tree Regressor' in model_names
    
    # Check complexities
    for model in result['recommended_models']:
        assert 'complexity' in model
        assert isinstance(model['complexity'], str)
        assert model['complexity'].startswith('O(')


def test_classification_inference():
    """Test automatic inference of classification problem."""
    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y = pd.Series(['A', 'B', 'A'])  # Categorical data
    
    result = suggest_model(X, y)
    
    assert result['problem_type'] == 'classification'
    assert len(result['recommended_models']) == 3
    for model in result['recommended_models']:
        assert 'model' in model
        assert 'complexity' in model


def test_regression_inference():
    """Test automatic inference of regression problem."""
    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5])  # Many unique numeric values
    
    result = suggest_model(X, y)
    
    assert result['problem_type'] == 'regression'
    assert len(result['recommended_models']) == 3
    for model in result['recommended_models']:
        assert 'model' in model
        assert 'complexity' in model


def test_invalid_inputs():
    """Test handling of invalid inputs."""
    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y = pd.Series([1, 2, 3])
    
    # Test with None inputs
    with pytest.raises(ValueError):
        suggest_model(None, y)
    with pytest.raises(ValueError):
        suggest_model(X, None)
    
    # Test with empty inputs
    with pytest.raises(ValueError):
        suggest_model(pd.DataFrame(), y)
    with pytest.raises(ValueError):
        suggest_model(X, pd.Series())
    
    # Test with mismatched lengths
    with pytest.raises(ValueError):
        suggest_model(X, pd.Series([1, 2]))
    
    # Test with invalid problem type
    with pytest.raises(ValueError):
        suggest_model(X, y, problem_type='invalid_type') 