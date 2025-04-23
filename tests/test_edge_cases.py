"""
Comprehensive test suite for edge cases in all ThinkML models.
"""

import numpy as np
import pandas as pd
import pytest
import dask.dataframe as dd
from thinkml.algorithms import (
    LinearRegression,
    RidgeRegression,
    LassoRegression,
    LogisticRegression,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    KNeighborsClassifier
)

# Helper functions for data generation
def generate_empty_data():
    """Generate empty dataset."""
    return np.array([]), np.array([])

def generate_single_sample():
    """Generate single sample dataset."""
    X = np.array([[1.0, 2.0, 3.0]])
    y = np.array([1.0])
    return X, y

def generate_single_feature():
    """Generate single feature dataset."""
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    return X, y

def generate_perfect_separation():
    """Generate perfectly separable dataset."""
    X = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    y = np.array([0, 0, 1, 1])
    return X, y

def generate_no_separation():
    """Generate non-separable dataset."""
    X = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    y = np.array([0, 1, 0, 1])
    return X, y

def generate_constant_features():
    """Generate dataset with constant features."""
    X = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    return X, y

def generate_missing_values():
    """Generate dataset with missing values."""
    X = np.array([[1.0, np.nan], [2.0, 2.0], [np.nan, 3.0], [4.0, 4.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    return X, y

def generate_extreme_values():
    """Generate dataset with extreme values."""
    X = np.array([[1e10, 1e-10], [-1e10, -1e-10], [0.0, 0.0], [1e5, 1e-5]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    return X, y

def generate_dask_data():
    """Generate Dask DataFrame."""
    X = pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0, 4.0],
        'feature2': [2.0, 3.0, 4.0, 5.0]
    })
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    return dd.from_pandas(X, npartitions=2), dd.from_pandas(y, npartitions=2)

# Base test class with common test methods
class BaseModelTest:
    """Base class for model tests with common test methods."""
    
    def test_empty_data(self):
        """Test model behavior with empty dataset."""
        X, y = generate_empty_data()
        with pytest.raises(ValueError):
            self.model.fit(X, y)
    
    def test_single_sample(self):
        """Test model behavior with single sample."""
        X, y = generate_single_sample()
        self.model.fit(X, y)
        assert hasattr(self.model, 'predict')
    
    def test_single_feature(self):
        """Test model behavior with single feature."""
        X, y = generate_single_feature()
        self.model.fit(X, y)
        assert hasattr(self.model, 'predict')
    
    def test_constant_features(self):
        """Test model behavior with constant features."""
        X, y = generate_constant_features()
        self.model.fit(X, y)
        assert hasattr(self.model, 'predict')
    
    def test_missing_values(self):
        """Test model behavior with missing values."""
        X, y = generate_missing_values()
        self.model.fit(X, y)
        assert hasattr(self.model, 'predict')
    
    def test_extreme_values(self):
        """Test model behavior with extreme values."""
        X, y = generate_extreme_values()
        self.model.fit(X, y)
        assert hasattr(self.model, 'predict')
    
    def test_dask_integration(self):
        """Test model behavior with Dask DataFrames."""
        X, y = generate_dask_data()
        self.model.fit(X, y)
        assert hasattr(self.model, 'predict')

# Test classes for each model type
class TestLinearRegression(BaseModelTest):
    """Test edge cases for Linear Regression."""
    
    def setup_method(self):
        self.model = LinearRegression()
    
    def test_perfect_linear_relationship(self):
        """Test model behavior with perfect linear relationship."""
        X, y = generate_single_feature()
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        np.testing.assert_allclose(y_pred, y, rtol=1e-5)

class TestRidgeRegression(BaseModelTest):
    """Test edge cases for Ridge Regression."""
    
    def setup_method(self):
        self.model = RidgeRegression()

class TestLassoRegression(BaseModelTest):
    """Test edge cases for Lasso Regression."""
    
    def setup_method(self):
        self.model = LassoRegression()

class TestLogisticRegression(BaseModelTest):
    """Test edge cases for Logistic Regression."""
    
    def setup_method(self):
        self.model = LogisticRegression()
    
    def test_perfect_separation(self):
        """Test model behavior with perfect separation."""
        X, y = generate_perfect_separation()
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        np.testing.assert_array_equal(y_pred, y)
    
    def test_no_separation(self):
        """Test model behavior with no separation."""
        X, y = generate_no_separation()
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        assert len(np.unique(y_pred)) <= 2

class TestDecisionTreeClassifier(BaseModelTest):
    """Test edge cases for Decision Tree Classifier."""
    
    def setup_method(self):
        self.model = DecisionTreeClassifier()
    
    def test_perfect_separation(self):
        """Test model behavior with perfect separation."""
        X, y = generate_perfect_separation()
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        np.testing.assert_array_equal(y_pred, y)
    
    def test_no_separation(self):
        """Test model behavior with no separation."""
        X, y = generate_no_separation()
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        assert len(np.unique(y_pred)) <= 2

class TestDecisionTreeRegressor(BaseModelTest):
    """Test edge cases for Decision Tree Regressor."""
    
    def setup_method(self):
        self.model = DecisionTreeRegressor()
    
    def test_perfect_linear_relationship(self):
        """Test model behavior with perfect linear relationship."""
        X, y = generate_single_feature()
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        np.testing.assert_allclose(y_pred, y, rtol=1e-5)

class TestRandomForestClassifier(BaseModelTest):
    """Test edge cases for Random Forest Classifier."""
    
    def setup_method(self):
        self.model = RandomForestClassifier()
    
    def test_perfect_separation(self):
        """Test model behavior with perfect separation."""
        X, y = generate_perfect_separation()
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        np.testing.assert_array_equal(y_pred, y)
    
    def test_no_separation(self):
        """Test model behavior with no separation."""
        X, y = generate_no_separation()
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        assert len(np.unique(y_pred)) <= 2

class TestRandomForestRegressor(BaseModelTest):
    """Test edge cases for Random Forest Regressor."""
    
    def setup_method(self):
        self.model = RandomForestRegressor()
    
    def test_perfect_linear_relationship(self):
        """Test model behavior with perfect linear relationship."""
        X, y = generate_single_feature()
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        np.testing.assert_allclose(y_pred, y, rtol=1e-5)

class TestKNeighborsClassifier(BaseModelTest):
    """Test edge cases for K-Neighbors Classifier."""
    
    def setup_method(self):
        self.model = KNeighborsClassifier()
    
    def test_perfect_separation(self):
        """Test model behavior with perfect separation."""
        X, y = generate_perfect_separation()
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        np.testing.assert_array_equal(y_pred, y)
    
    def test_no_separation(self):
        """Test model behavior with no separation."""
        X, y = generate_no_separation()
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        assert len(np.unique(y_pred)) <= 2 