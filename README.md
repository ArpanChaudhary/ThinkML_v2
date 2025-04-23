# ThinkML

ThinkML is a Python library designed to simplify machine learning workflows by providing intelligent data analysis, model suggestions, and preprocessing capabilities.

## Features

### Data Description
- Comprehensive dataset analysis with `describe_data()`
- Automatic detection of feature types (numerical, categorical)
- Statistical summaries for both features and target variables
- Memory usage analysis and optimization for large datasets
- Correlation analysis and visualization

### Model Suggestion
- Intelligent model recommendations based on dataset characteristics
- Automatic problem type inference (classification/regression)
- Model complexity analysis with Big-O notation
- Detailed reasoning for each recommended model
- Support for large datasets using Dask

### Data Preprocessing
- Missing value handling with multiple strategies
- Categorical feature encoding
- Feature scaling and normalization
- Imbalanced dataset handling

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/thinkml.git
cd thinkml

# Install the package
pip install -e .
```

## Usage Examples

### Data Description

```python
import pandas as pd
from thinkml.describer.data_describer import describe_data

# Load your dataset
X = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': ['A', 'B', 'A', 'C', 'B']
})
y = pd.Series([0, 1, 0, 1, 0])

# Get a comprehensive description of your dataset
description = describe_data(X, y)
print(description)
```

### Model Suggestion

```python
import pandas as pd
from thinkml.analyzer.model_suggester import suggest_model

# Load your dataset
X = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [10, 20, 30, 40, 50]
})
y = pd.Series([0, 1, 0, 1, 0])

# Get model suggestions
suggestions = suggest_model(X, y)
print(suggestions)

# Or specify the problem type explicitly
suggestions = suggest_model(X, y, problem_type='classification')
print(suggestions)
```

### Data Preprocessing

```python
import pandas as pd
from thinkml.preprocessor.missing_handler import handle_missing_values
from thinkml.preprocessor.encoder import encode_categorical
from thinkml.preprocessor.scaler import scale_features
from thinkml.preprocessor.imbalance_handler import handle_imbalance

# Load your dataset
X = pd.DataFrame({
    'feature1': [1, 2, None, 4, 5],
    'feature2': ['A', 'B', 'A', None, 'B']
})
y = pd.Series([0, 1, 0, 1, 0])

# Handle missing values
X_clean = handle_missing_values(X, strategy='mean')

# Encode categorical features
X_encoded = encode_categorical(X_clean, method='onehot')

# Scale features
X_scaled = scale_features(X_encoded, method='standard')

# Handle class imbalance
X_balanced, y_balanced = handle_imbalance(X_scaled, y, method='smote')
```

## Documentation

For detailed documentation, please refer to the [documentation](docs/README.md) directory.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 