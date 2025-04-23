# ThinkML User Guide

## Introduction

ThinkML is a Python library designed to simplify and automate common machine learning tasks. This guide will help you get started with using ThinkML effectively in your machine learning projects.

## Installation

```bash
pip install thinkml
```

## Quick Start

Here's a simple example of how to use ThinkML for a basic machine learning workflow:

```python
import pandas as pd
from thinkml.describer import describe_data
from thinkml.analyzer import suggest_model
from thinkml.preprocessor import (
    handle_missing_values,
    encode_categorical,
    scale_features,
    handle_imbalance
)

# Load your data
X = pd.read_csv('your_data.csv')
y = X.pop('target')  # Assuming 'target' is your target column

# 1. Analyze your data
description = describe_data(X, y)
print("Data Description:", description)

# 2. Get model suggestions
suggestions = suggest_model(X, y)
print("Model Suggestions:", suggestions)

# 3. Preprocess your data
# Handle missing values
X_cleaned = handle_missing_values(X, strategy='mean')

# Encode categorical features
X_encoded = encode_categorical(X_cleaned, method='onehot')

# Scale numerical features
X_scaled = scale_features(X_encoded, method='standard')

# Handle class imbalance (if needed)
X_resampled, y_resampled = handle_imbalance(X_scaled, y, method='smote')
```

## Data Description

The `describe_data` function provides comprehensive insights about your dataset:

```python
from thinkml.describer import describe_data

# Basic usage
description = describe_data(X)

# With target variable
description = describe_data(X, y)

# For large datasets, use chunking
description = describe_data(X, chunk_size=50000)
```

Key features of the data description:
- Dataset size and memory usage
- Feature types and missing values
- Statistical summaries for numerical features
- Value counts for categorical features
- Correlation analysis
- Target variable analysis (if provided)

## Model Suggestion

The `suggest_model` function helps you choose appropriate models:

```python
from thinkml.analyzer import suggest_model

# Automatic task type inference
suggestions = suggest_model(X, y)

# Specify task type
suggestions = suggest_model(X, y, task_type='classification')
```

The function considers:
- Dataset size and complexity
- Feature types and distributions
- Target variable characteristics
- Task type (classification/regression)

## Data Preprocessing

### 1. Handling Missing Values

```python
from thinkml.preprocessor import handle_missing_values

# Using mean imputation
X_cleaned = handle_missing_values(X, strategy='mean')

# Using median imputation
X_cleaned = handle_missing_values(X, strategy='median')

# Using mode imputation
X_cleaned = handle_missing_values(X, strategy='mode')

# Using constant value
X_cleaned = handle_missing_values(X, strategy='constant', fill_value=0)

# Dropping rows with missing values
X_cleaned = handle_missing_values(X, strategy='drop')
```

### 2. Categorical Encoding

```python
from thinkml.preprocessor import encode_categorical

# One-hot encoding
X_encoded = encode_categorical(X, method='onehot')

# Label encoding
X_encoded = encode_categorical(X, method='label')

# Target encoding
X_encoded = encode_categorical(X, method='target')

# Frequency encoding
X_encoded = encode_categorical(X, method='frequency')

# Encode specific columns
X_encoded = encode_categorical(X, method='onehot', columns=['cat1', 'cat2'])
```

### 3. Feature Scaling

```python
from thinkml.preprocessor import scale_features

# Standard scaling
X_scaled = scale_features(X, method='standard')

# Min-Max scaling
X_scaled = scale_features(X, method='minmax')

# Robust scaling
X_scaled = scale_features(X, method='robust')

# Normalizer scaling
X_scaled = scale_features(X, method='normalizer')

# Scale specific columns
X_scaled = scale_features(X, method='standard', columns=['num1', 'num2'])
```

### 4. Handling Class Imbalance

```python
from thinkml.preprocessor import handle_imbalance

# Using SMOTE
X_resampled, y_resampled = handle_imbalance(X, y, method='smote')

# Random oversampling
X_resampled, y_resampled = handle_imbalance(X, y, method='random_oversample')

# Random undersampling
X_resampled, y_resampled = handle_imbalance(X, y, method='random_undersample')

# Custom sampling strategy
X_resampled, y_resampled = handle_imbalance(
    X, y, 
    method='smote',
    sampling_strategy={0: 1000, 1: 1000}  # Balance classes to 1000 samples each
)
```

## Best Practices

1. **Data Analysis**
   - Always start with `describe_data` to understand your dataset
   - Use the insights to guide your preprocessing steps

2. **Model Selection**
   - Let `suggest_model` guide your initial model choices
   - Consider the suggested models' complexity and requirements

3. **Preprocessing**
   - Handle missing values before encoding categorical features
   - Scale features after encoding categorical variables
   - Handle class imbalance only if necessary for your task

4. **Memory Management**
   - Use chunking for large datasets
   - Monitor memory usage through the data description

## Common Issues and Solutions

1. **Memory Errors**
   - Use smaller chunk sizes with `describe_data`
   - Process data in batches
   - Consider using Dask for very large datasets

2. **Categorical Encoding Issues**
   - Check for missing values before encoding
   - Handle rare categories appropriately
   - Consider using target encoding for high-cardinality features

3. **Scaling Problems**
   - Handle outliers before scaling
   - Choose appropriate scaling method for your data distribution
   - Scale features consistently across train and test sets

4. **Imbalance Handling**
   - Verify class distribution before resampling
   - Choose appropriate resampling method based on dataset size
   - Consider using stratified sampling for validation

## Contributing

We welcome contributions to ThinkML! Please see our contributing guidelines for more information.

## License

ThinkML is licensed under the MIT License. See the LICENSE file for details. 