"""
Machine learning algorithms module for ThinkML.

This module provides implementations of various machine learning algorithms
from scratch, without relying on scikit-learn.
"""

# Regression models
from .linear_regression import LinearRegression
from .ridge_regression import RidgeRegression
from .lasso_regression import LassoRegression
from .decision_tree import DecisionTreeRegressor
from .random_forest import RandomForestRegressor

# Classification models
from .logistic_regression import LogisticRegression
from .decision_tree import DecisionTreeClassifier
from .random_forest import RandomForestClassifier
from .knn import KNeighborsClassifier

__all__ = [
    # Regression models
    'LinearRegression',
    'RidgeRegression',
    'LassoRegression',
    'DecisionTreeRegressor',
    'RandomForestRegressor',
    
    # Classification models
    'LogisticRegression',
    'DecisionTreeClassifier',
    'RandomForestClassifier',
    'KNeighborsClassifier'
] 