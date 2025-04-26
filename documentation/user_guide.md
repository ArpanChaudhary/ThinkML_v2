# ThinkML User Guide

This guide provides detailed information about using ThinkML's features, with examples and best practices.

## Table of Contents
1. [Feature Engineering](#feature-engineering)
2. [Model Validation](#model-validation)
3. [Hyperparameter Optimization](#hyperparameter-optimization)
4. [Robust Regression](#robust-regression)
5. [Model Interpretability](#model-interpretability)

## Feature Engineering

ThinkML provides powerful feature engineering capabilities through the `create_features` and `select_features` functions.

### Creating Features

```python
import thinkml
import pandas as pd
import numpy as np

# Create sample data
X = pd.DataFrame({
    'feature1': np.random.randn(100),
    'feature2': np.random.randn(100)
})

# Create polynomial features
poly_features = thinkml.create_features(
    X,
    feature_types=['polynomial'],
    polynomial_degree=2
)

# Create interaction features
interaction_features = thinkml.create_features(
    X,
    feature_types=['interaction'],
    interaction_only=True
)

# Create multiple types of features
combined_features = thinkml.create_features(
    X,
    feature_types=['polynomial', 'interaction', 'ratio', 'log'],
    polynomial_degree=2
)
```

### Feature Selection

```python
# Select features using mutual information
selected_features, indices = thinkml.select_features(
    X,
    y,
    method='mutual_info',
    n_features=5
)

# Select features using recursive feature elimination
from sklearn.ensemble import RandomForestRegressor
estimator = RandomForestRegressor()
selected_features, indices = thinkml.select_features(
    X,
    y,
    method='rfe',
    estimator=estimator,
    n_features=5
)
```

## Model Validation

ThinkML provides robust model validation through the `NestedCrossValidator` class.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

# Create sample data
X, y = make_regression(n_samples=100, n_features=10)

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20]
}

# Create and use nested cross-validator
validator = thinkml.NestedCrossValidator(
    estimator=RandomForestRegressor(),
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error'
)

# Fit and get predictions
predictions = validator.fit_predict(X, y)

# Get best parameters and score
best_params = validator.get_best_params()
best_score = validator.get_best_score()
```

## Hyperparameter Optimization

ThinkML provides Bayesian optimization through the `BayesianOptimizer` class.

```python
# Define parameter space
param_space = {
    'n_estimators': (100, 500),
    'max_depth': (None, 50),
    'min_samples_split': (2, 20)
}

# Create and use Bayesian optimizer
optimizer = thinkml.BayesianOptimizer(
    estimator=RandomForestRegressor(),
    param_space=param_space,
    n_trials=50,
    cv=5
)

# Fit and get best parameters
best_params = optimizer.fit(X, y)

# Get best score
best_score = optimizer.get_best_score()
```

## Robust Regression

ThinkML provides robust regression through the `RobustRegressor` class.

```python
# Create robust regressor with Huber method
huber_regressor = thinkml.RobustRegressor(
    method='huber',
    epsilon=1.35
)

# Create robust regressor with RANSAC method
ransac_regressor = thinkml.RobustRegressor(
    method='ransac',
    max_iter=100
)

# Fit and make predictions
huber_regressor.fit(X, y)
predictions = huber_regressor.predict(X)

# Get score
score = huber_regressor.score(X, y)
```

## Model Interpretability

ThinkML provides model interpretability through the `explain_model` and `get_feature_importance` functions.

```python
# Train a model
model = RandomForestRegressor()
model.fit(X, y)

# Explain model predictions using SHAP
explanations = thinkml.explain_model(
    model,
    X,
    method='shap'
)

# Get feature importance using permutation importance
importance = thinkml.get_feature_importance(
    model,
    X,
    method='permutation'
)
```

## Best Practices

1. **Feature Engineering**
   - Start with basic feature transformations
   - Use domain knowledge to create meaningful interactions
   - Validate feature importance before selection

2. **Model Validation**
   - Use nested cross-validation for unbiased evaluation
   - Choose appropriate scoring metrics
   - Consider computational costs

3. **Hyperparameter Optimization**
   - Define reasonable parameter spaces
   - Use early stopping when possible
   - Monitor optimization progress

4. **Robust Regression**
   - Choose appropriate method based on data characteristics
   - Tune parameters carefully
   - Validate robustness

5. **Model Interpretability**
   - Use multiple interpretation methods
   - Validate explanations
   - Consider computational costs

## Common Pitfalls

1. **Feature Engineering**
   - Creating too many features without validation
   - Ignoring feature scaling
   - Not handling missing values properly

2. **Model Validation**
   - Using simple cross-validation for nested processes
   - Not considering data leakage
   - Ignoring computational costs

3. **Hyperparameter Optimization**
   - Defining too large parameter spaces
   - Not using early stopping
   - Ignoring optimization progress

4. **Robust Regression**
   - Not tuning parameters
   - Using inappropriate methods
   - Ignoring data characteristics

5. **Model Interpretability**
   - Relying on single interpretation method
   - Not validating explanations
   - Ignoring computational costs 