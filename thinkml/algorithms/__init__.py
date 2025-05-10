"""
Machine learning algorithms for ThinkML.

This module provides implementations of common machine learning algorithms
for classification and regression tasks.
"""

from .linear import (
    LogisticRegression,
    LinearRegression,
    RidgeRegression
)

from .tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    RandomForestClassifier,
    RandomForestRegressor
)

from .knn import KNeighborsClassifier, KNeighborsRegressor

from .neural_network import NeuralNetwork

from thinkml.algorithms.lasso_regression import LassoRegression

__all__ = [
    'LogisticRegression',
    'LinearRegression',
    'RidgeRegression',
    'DecisionTreeClassifier',
    'DecisionTreeRegressor',
    'RandomForestClassifier',
    'RandomForestRegressor',
    'KNeighborsClassifier',
    'KNeighborsRegressor',
    'NeuralNetwork',
    'LassoRegression',
]

__version__ = "0.1.0" 