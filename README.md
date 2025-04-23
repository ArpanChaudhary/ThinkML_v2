# ThinkML

ThinkML is a comprehensive machine learning library implemented from scratch, designed to provide a deep understanding of machine learning algorithms without relying on external libraries like scikit-learn.

## Features

### Algorithms

ThinkML implements the following machine learning algorithms from scratch:

#### Regression Models
- **Linear Regression**: Basic linear regression with gradient descent optimization
- **Ridge Regression**: L2-regularized linear regression
- **Lasso Regression**: L1-regularized linear regression
- **Decision Tree Regressor**: Regression using decision trees
- **Random Forest Regressor**: Ensemble of decision trees for regression

#### Classification Models
- **Logistic Regression**: Binary and multi-class classification
- **Decision Tree Classifier**: Classification using decision trees
- **Random Forest Classifier**: Ensemble of decision trees for classification
- **K-Nearest Neighbors**: Classification based on nearest neighbors

### Preprocessing

ThinkML includes comprehensive data preprocessing capabilities:

- **Missing Value Handling**: Various strategies for handling missing data
- **Categorical Feature Encoding**: One-hot encoding, label encoding, and more
- **Feature Scaling**: Standardization, normalization, and robust scaling
- **Class Imbalance Handling**: Techniques to address imbalanced datasets
- **Outlier Detection**: Methods to identify and handle outliers

### Utilities

- **Data Description**: Comprehensive dataset analysis and visualization
- **Feature Selection**: Methods to select the most important features
- **Model Selection**: Tools to suggest the best model for a given dataset
- **Cross-Validation**: K-fold cross-validation for model evaluation
- **Hyperparameter Tuning**: Grid search and random search for parameter optimization

## Implementation Details

### Base Model

All models in ThinkML inherit from the `BaseModel` class, which provides:

- Data preprocessing capabilities
- Support for both small and large datasets
- Chunk-based processing for memory efficiency
- Dask integration for distributed computing

### Edge Case Handling

All models have been tested for various edge cases:

- Empty datasets
- Single sample datasets
- Single feature datasets
- Perfect separation (for classification)
- No separation (for classification)
- Constant features
- Missing values
- Extreme values
- Dask integration

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/thinkml.git
cd thinkml

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from thinkml.algorithms import LinearRegression
from thinkml.preprocessing import StandardScaler, MissingHandler

# Load and preprocess data
X, y = load_data()  # Your data loading function
X = MissingHandler().fit_transform(X)
X = StandardScaler().fit_transform(X)

# Train a model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Evaluate the model
score = model.score(X, y)
print(f"R² Score: {score}")
```

## Testing

ThinkML includes comprehensive test suites for all components:

```bash
# Run all tests
pytest

# Run specific test files
pytest tests/test_models.py
pytest tests/test_edge_cases.py
```

## Documentation

Detailed documentation is available in the `docs` directory:

- [API Documentation](docs/API.md)
- [User Guide](docs/UserGuide.md)
- [Examples](docs/Examples.md)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This library was created for educational purposes to understand machine learning algorithms from first principles.
- Special thanks to all contributors who have helped improve the library. 