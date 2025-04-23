# ThinkML API Documentation

## Table of Contents
1. [Data Description](#data-description)
2. [Model Suggestion](#model-suggestion)
3. [Data Preprocessing](#data-preprocessing)

## Data Description

### `describe_data(X, y=None, chunk_size=10000)`

Analyzes and describes a dataset, providing comprehensive statistics and insights.

#### Parameters:
- `X` (pd.DataFrame): Input features
- `y` (pd.Series, optional): Target variable
- `chunk_size` (int, optional): Size of chunks for processing large datasets. Defaults to 10000.

#### Returns:
- `dict`: A dictionary containing:
  - `n_samples`: Number of samples
  - `n_features`: Number of features
  - `feature_types`: Dictionary of feature types
  - `missing_values`: Dictionary of missing value counts
  - `memory_usage`: Memory usage in bytes
  - `duplicate_rows`: Number of duplicate rows
  - `feature_summaries`: Dictionary of feature summaries
  - `correlation_matrix`: Correlation matrix for numerical features
  - `target_summary`: Dictionary of target variable statistics (if y is provided)

#### Example:
```python
import pandas as pd
from thinkml.describer import describe_data

# Create sample data
X = pd.DataFrame({
    'numeric': [1, 2, 3, 4, 5],
    'categorical': ['A', 'B', 'A', 'C', 'B']
})
y = pd.Series([0, 1, 0, 1, 0])

# Describe the data
description = describe_data(X, y)
print(description)
```

## Model Suggestion

### `suggest_model(X, y, task_type=None)`

Suggests appropriate machine learning models based on dataset characteristics.

#### Parameters:
- `X` (pd.DataFrame): Input features
- `y` (pd.Series): Target variable
- `task_type` (str, optional): Type of task ('classification' or 'regression'). If None, inferred from data.

#### Returns:
- `dict`: A dictionary containing:
  - `suggested_models`: List of suggested models with their parameters
  - `task_type`: Inferred or specified task type
  - `complexity`: Model complexity assessment
  - `reasoning`: Explanation for model suggestions

#### Example:
```python
import pandas as pd
from thinkml.analyzer import suggest_model

# Create sample data
X = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [0.1, 0.2, 0.3, 0.4, 0.5]
})
y = pd.Series([0, 1, 0, 1, 0])

# Get model suggestions
suggestions = suggest_model(X, y)
print(suggestions)
```

## Data Preprocessing

### Missing Value Handling

#### `handle_missing_values(X, strategy='mean', fill_value=None)`

Handles missing values in the dataset using various strategies.

#### Parameters:
- `X` (pd.DataFrame): Input features
- `strategy` (str): Strategy for handling missing values ('mean', 'median', 'mode', 'constant', 'drop')
- `fill_value` (any, optional): Value to use when strategy is 'constant'

#### Returns:
- `pd.DataFrame`: DataFrame with handled missing values

#### Example:
```python
import pandas as pd
from thinkml.preprocessor import handle_missing_values

# Create sample data with missing values
X = pd.DataFrame({
    'feature1': [1, None, 3, None, 5],
    'feature2': [0.1, 0.2, None, 0.4, 0.5]
})

# Handle missing values
X_cleaned = handle_missing_values(X, strategy='mean')
print(X_cleaned)
```

### Categorical Encoding

#### `encode_categorical(X, method='onehot', columns=None)`

Encodes categorical features using various encoding methods.

#### Parameters:
- `X` (pd.DataFrame): Input features
- `method` (str): Encoding method ('onehot', 'label', 'target', 'frequency')
- `columns` (list, optional): Columns to encode. If None, all categorical columns are encoded.

#### Returns:
- `pd.DataFrame`: DataFrame with encoded categorical features

#### Example:
```python
import pandas as pd
from thinkml.preprocessor import encode_categorical

# Create sample data with categorical features
X = pd.DataFrame({
    'category1': ['A', 'B', 'A', 'C', 'B'],
    'category2': ['X', 'Y', 'X', 'Z', 'Y']
})

# Encode categorical features
X_encoded = encode_categorical(X, method='onehot')
print(X_encoded)
```

### Feature Scaling

#### `scale_features(X, method='standard', columns=None)`

Scales numerical features using various scaling methods.

#### Parameters:
- `X` (pd.DataFrame): Input features
- `method` (str): Scaling method ('standard', 'minmax', 'robust', 'normalizer')
- `columns` (list, optional): Columns to scale. If None, all numerical columns are scaled.

#### Returns:
- `pd.DataFrame`: DataFrame with scaled features

#### Example:
```python
import pandas as pd
from thinkml.preprocessor import scale_features

# Create sample data with numerical features
X = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [0.1, 0.2, 0.3, 0.4, 0.5]
})

# Scale features
X_scaled = scale_features(X, method='standard')
print(X_scaled)
```

### Imbalance Handling

#### `handle_imbalance(X, y, method='smote', sampling_strategy='auto')`

Handles class imbalance in classification datasets.

#### Parameters:
- `X` (pd.DataFrame): Input features
- `y` (pd.Series): Target variable
- `method` (str): Method for handling imbalance ('smote', 'random_oversample', 'random_undersample')
- `sampling_strategy` (str or dict): Sampling strategy for resampling

#### Returns:
- `tuple`: (X_resampled, y_resampled) - Resampled features and target

#### Example:
```python
import pandas as pd
from thinkml.preprocessor import handle_imbalance

# Create sample imbalanced data
X = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'feature2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
})
y = pd.Series([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])

# Handle class imbalance
X_resampled, y_resampled = handle_imbalance(X, y, method='smote')
print(X_resampled.shape, y_resampled.shape)
``` 