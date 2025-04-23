# ThinkML API Documentation

## Table of Contents
- [Data Description](#data-description)
- [Model Suggestion](#model-suggestion)
- [Data Preprocessing](#data-preprocessing)
- [Outlier Detection](#outlier-detection)
- [Feature Selection](#feature-selection)
- [EDA and Visualization](#eda-and-visualization)

## Data Description

### describe_data(X, y=None, chunk_size=10000)

Analyze and describe datasets, providing various statistics and summaries.

**Parameters:**
- `X`: pandas DataFrame - Input features
- `y`: pandas Series, optional - Target variable
- `chunk_size`: int, default=10000 - Size of chunks for processing large datasets

**Returns:**
Dictionary containing:
- Dataset statistics (samples, features, memory usage)
- Feature summaries (types, missing values, unique values)
- Correlation analysis
- Target variable analysis (if provided)

**Example:**
```python
from thinkml.describer import describe_data

# Load your dataset
X = pd.read_csv('data.csv')
y = X.pop('target')

# Get dataset description
description = describe_data(X, y)
print(description['feature_summary'])
```

## Model Suggestion

### suggest_model(X, y, task_type=None)

Suggest appropriate machine learning models based on dataset characteristics.

**Parameters:**
- `X`: pandas DataFrame - Input features
- `y`: pandas Series - Target variable
- `task_type`: str, optional - 'classification' or 'regression'

**Returns:**
Dictionary containing:
- Suggested models with confidence scores
- Model complexity analysis
- Dataset characteristics

**Example:**
```python
from thinkml.analyzer import suggest_model

# Get model suggestions
suggestions = suggest_model(X, y)
print(suggestions['recommended_models'])
```

## Data Preprocessing

### handle_missing_values(X, strategy='mean')

Handle missing values in the dataset.

**Parameters:**
- `X`: pandas DataFrame - Input features
- `strategy`: str, default='mean' - Strategy for handling missing values
  - 'mean': Replace with mean
  - 'median': Replace with median
  - 'mode': Replace with mode
  - 'constant': Replace with a constant value
  - 'drop': Drop rows with missing values

**Example:**
```python
from thinkml.preprocessor import handle_missing_values

# Handle missing values
X_clean = handle_missing_values(X, strategy='mean')
```

### encode_categorical(X, method='label')

Encode categorical variables.

**Parameters:**
- `X`: pandas DataFrame - Input features
- `method`: str, default='label' - Encoding method
  - 'label': Label encoding
  - 'onehot': One-hot encoding
  - 'target': Target encoding

**Example:**
```python
from thinkml.preprocessor import encode_categorical

# Encode categorical features
X_encoded = encode_categorical(X, method='onehot')
```

### scale_features(X, method='standard')

Scale numerical features.

**Parameters:**
- `X`: pandas DataFrame - Input features
- `method`: str, default='standard' - Scaling method
  - 'standard': StandardScaler
  - 'minmax': MinMaxScaler
  - 'robust': RobustScaler

**Example:**
```python
from thinkml.preprocessor import scale_features

# Scale features
X_scaled = scale_features(X, method='standard')
```

### handle_imbalance(X, y, method='smote')

Handle class imbalance in classification problems.

**Parameters:**
- `X`: pandas DataFrame - Input features
- `y`: pandas Series - Target variable
- `method`: str, default='smote' - Resampling method
  - 'smote': SMOTE oversampling
  - 'random': Random oversampling
  - 'tomek': Tomek links undersampling

**Example:**
```python
from thinkml.preprocessor import handle_imbalance

# Balance dataset
X_balanced, y_balanced = handle_imbalance(X, y, method='smote')
```

## Outlier Detection

### detect_outliers(X, method='zscore', visualize=True)

Detect outliers in the dataset using various methods.

**Parameters:**
- `X`: pandas DataFrame - Input features
- `method`: str, default='zscore' - Detection method
  - 'zscore': Z-score method
  - 'iqr': IQR method
  - 'isolation_forest': Isolation Forest
- `visualize`: bool, default=True - Create interactive visualizations

**Returns:**
Dictionary containing:
- Outlier indices
- Outlier counts per feature
- Interactive visualization

**Example:**
```python
from thinkml.outliers import detect_outliers

# Detect outliers
outliers = detect_outliers(X, method='zscore')
print(outliers['outlier_counts'])
```

## Feature Selection

### select_features(X, y=None, method='variance', threshold=0.1, visualize=True)

Select relevant features using various methods.

**Parameters:**
- `X`: pandas DataFrame - Input features
- `y`: pandas Series, optional - Target variable
- `method`: str, default='variance' - Selection method
  - 'variance': Variance threshold
  - 'correlation': Correlation analysis
  - 'chi2': Chi-squared test
  - 'mutual_info': Mutual information
  - 'rfe': Recursive feature elimination
- `threshold`: float, default=0.1 - Threshold for filtering
- `visualize`: bool, default=True - Create interactive visualizations

**Returns:**
Dictionary containing:
- Selected features
- Feature importance scores
- Interactive visualization

**Example:**
```python
from thinkml.feature_selection import select_features

# Select features
selection = select_features(X, y, method='mutual_info')
print(selection['selected_features'])
```

## EDA and Visualization

### plot_feature_distributions(X, y=None)

Create interactive visualizations of feature distributions.

**Parameters:**
- `X`: pandas DataFrame - Input features
- `y`: pandas Series, optional - Target variable

**Returns:**
Plotly figure object with feature distributions

**Example:**
```python
from thinkml.eda import plot_feature_distributions

# Plot distributions
fig = plot_feature_distributions(X)
fig.show()
```

### plot_correlations(X, method='pearson')

Create interactive correlation heatmap.

**Parameters:**
- `X`: pandas DataFrame - Input features
- `method`: str, default='pearson' - Correlation method

**Returns:**
Plotly figure object with correlation heatmap

**Example:**
```python
from thinkml.eda import plot_correlations

# Plot correlations
fig = plot_correlations(X)
fig.show()
``` 