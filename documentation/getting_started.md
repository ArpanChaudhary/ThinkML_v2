# Getting Started with ThinkML

This guide will help you get started with ThinkML, from installation to your first machine learning project.

## Installation

ThinkML can be installed using pip:

```bash
pip install thinkml
```

### Dependencies

ThinkML requires the following dependencies:
- Python 3.6+
- NumPy
- Pandas
- Scikit-learn
- Scikit-optimize

These will be automatically installed when you install ThinkML.

## Quick Start Guide

Here's a simple example to get you started:

```python
from thinkml import (
    create_features,
    select_features,
    NestedCrossValidator,
    BayesianOptimizer
)

# Create and select features
X = create_features(X, feature_types=['polynomial', 'interaction'])
X_selected = select_features(X, y, method='mutual_info', n_features=10)

# Set up validation
validator = NestedCrossValidator(
    estimator=RandomForestRegressor(),
    param_grid={
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20]
    }
)

# Optimize hyperparameters
optimizer = BayesianOptimizer(
    estimator=RandomForestRegressor(),
    param_space={
        'n_estimators': (100, 1000),
        'max_depth': (3, 30)
    },
    n_trials=50
)

# Fit and evaluate
results = validator.fit_predict(X_selected, y)
best_params = optimizer.fit(X_selected, y)
```

## Basic Usage Examples

### Feature Engineering

```python
from thinkml import create_features

# Create polynomial features
X_poly = create_features(X, feature_types=['polynomial'], polynomial_degree=2)

# Create interaction features
X_interact = create_features(X, feature_types=['interaction'])

# Create multiple feature types
X_combined = create_features(X, feature_types=['polynomial', 'interaction', 'ratio'])
```

### Feature Selection

```python
from thinkml import select_features

# Select features using mutual information
X_selected = select_features(X, y, method='mutual_info', n_features=10)

# Select features using recursive feature elimination
X_selected = select_features(X, y, method='rfe', n_features=10)
```

### Model Validation

```python
from thinkml import NestedCrossValidator

# Set up nested cross-validation
validator = NestedCrossValidator(
    estimator=RandomForestRegressor(),
    param_grid={
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20]
    }
)

# Fit and get results
results = validator.fit_predict(X, y)
print(f"Best score: {results['best_score']}")
print(f"Best parameters: {results['best_params']}")
```

### Hyperparameter Optimization

```python
from thinkml import BayesianOptimizer

# Set up Bayesian optimization
optimizer = BayesianOptimizer(
    estimator=RandomForestRegressor(),
    param_space={
        'n_estimators': (100, 1000),
        'max_depth': (3, 30)
    },
    n_trials=50
)

# Fit and get best parameters
best_params = optimizer.fit(X, y)
print(f"Best parameters: {best_params}")
```

## Next Steps

- Check out the [User Guide](user_guide.md) for more detailed information
- Explore the [API Reference](api_reference.md) for complete documentation
- Visit our [GitHub repository](https://github.com/thinkml/thinkml) for examples and updates 