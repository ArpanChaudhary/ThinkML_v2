# ThinkML API Reference

This document provides detailed information about the ThinkML API, including all modules, classes, and functions.

## Feature Engineering

### create_features

```python
thinkml.create_features(X, feature_types=None, polynomial_degree=2, interaction_only=False, include_bias=True)
```

Creates new features from existing ones using various transformations.

**Parameters:**
- `X` (array-like or DataFrame): Input data
- `feature_types` (list of str, optional): Types of features to create. Options: 'polynomial', 'interaction', 'ratio', 'log', 'sqrt', 'exp'
- `polynomial_degree` (int, optional): Degree of polynomial features. Default: 2
- `interaction_only` (bool, optional): If True, only interaction features are created. Default: False
- `include_bias` (bool, optional): If True, include a bias column. Default: True

**Returns:**
- DataFrame: New features

### select_features

```python
thinkml.select_features(X, y, method='mutual_info', n_features=None, task='regression', estimator=None, cv=5, scoring=None, step=1, min_features_to_select=1)
```

Selects features using various methods.

**Parameters:**
- `X` (array-like or DataFrame): Input data
- `y` (array-like): Target variable
- `method` (str, optional): Feature selection method. Options: 'mutual_info', 'f_regression', 'f_classif', 'chi2', 'lasso', 'random_forest', 'rfe', 'rfecv', 'pca'
- `n_features` (int, optional): Number of features to select
- `task` (str, optional): Task type. Options: 'regression', 'classification'
- `estimator` (estimator object, optional): Estimator for methods that require one
- `cv` (int, optional): Number of cross-validation folds
- `scoring` (str, optional): Scoring metric
- `step` (int, optional): Step size for recursive feature elimination
- `min_features_to_select` (int, optional): Minimum number of features to select

**Returns:**
- tuple: (selected_features, selected_indices)

## Validation

### NestedCrossValidator

```python
thinkml.NestedCrossValidator(estimator, param_grid, cv=5, scoring=None, n_jobs=None)
```

Performs nested cross-validation with hyperparameter optimization.

**Parameters:**
- `estimator` (estimator object): Base estimator
- `param_grid` (dict): Parameter grid for optimization
- `cv` (int, optional): Number of cross-validation folds
- `scoring` (str, optional): Scoring metric
- `n_jobs` (int, optional): Number of jobs for parallel processing

**Methods:**
- `fit_predict(X, y)`: Fits the validator and returns predictions
- `get_best_params()`: Returns best parameters
- `get_best_score()`: Returns best score

## Optimization

### BayesianOptimizer

```python
thinkml.BayesianOptimizer(estimator, param_space, n_trials=50, cv=5, n_jobs=None, scoring=None)
```

Optimizes hyperparameters using Bayesian optimization.

**Parameters:**
- `estimator` (estimator object): Base estimator
- `param_space` (dict): Parameter space for optimization
- `n_trials` (int, optional): Number of optimization trials
- `cv` (int, optional): Number of cross-validation folds
- `n_jobs` (int, optional): Number of jobs for parallel processing
- `scoring` (str, optional): Scoring metric

**Methods:**
- `fit(X, y)`: Fits the optimizer and returns best parameters
- `get_best_params()`: Returns best parameters
- `get_best_score()`: Returns best score

## Regression

### RobustRegressor

```python
thinkml.RobustRegressor(method='huber', epsilon=1.35, max_iter=100, random_state=None)
```

Performs robust regression using various methods.

**Parameters:**
- `method` (str, optional): Regression method. Options: 'huber', 'ransac'
- `epsilon` (float, optional): Epsilon parameter for Huber regression
- `max_iter` (int, optional): Maximum number of iterations
- `random_state` (int, optional): Random state for reproducibility

**Methods:**
- `fit(X, y)`: Fits the regressor
- `predict(X)`: Makes predictions
- `score(X, y)`: Returns the score

## Interpretability

### explain_model

```python
thinkml.explain_model(model, X, method='shap')
```

Explains model predictions using various methods.

**Parameters:**
- `model` (estimator object): Trained model
- `X` (array-like or DataFrame): Input data
- `method` (str, optional): Explanation method. Options: 'shap', 'lime', 'permutation'

**Returns:**
- dict: Model explanations

### get_feature_importance

```python
thinkml.get_feature_importance(model, X, method='default')
```

Gets feature importance scores.

**Parameters:**
- `model` (estimator object): Trained model
- `X` (array-like or DataFrame): Input data
- `method` (str, optional): Importance method. Options: 'default', 'permutation', 'shap'

**Returns:**
- dict: Feature importance scores 