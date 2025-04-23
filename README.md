# ThinkML

ThinkML is a comprehensive machine learning library that helps data scientists and ML engineers streamline their workflow by providing intelligent model suggestions, automated data preprocessing, and insightful exploratory data analysis.

## Features

- **Intelligent Model Suggestion**: Automatically suggests suitable ML models based on dataset characteristics
- **Comprehensive Data Description**: Provides detailed dataset analysis and statistics
- **Automated Preprocessing**: Handles missing values, encoding, scaling, and class imbalance
- **Outlier Detection**: Multiple methods for detecting and visualizing outliers
- **Feature Selection**: Various methods for selecting the most relevant features
- **Interactive Visualizations**: Beautiful and informative plots for data exploration

## Installation

```bash
pip install thinkml
```

## Quick Start

```python
import pandas as pd
from thinkml.describer import describe_data
from thinkml.analyzer import suggest_model
from thinkml.preprocessor import handle_missing_values, encode_categorical, scale_features

# Load your dataset
X = pd.read_csv('data.csv')
y = X.pop('target')

# Get dataset description
description = describe_data(X, y)

# Get model suggestions
suggestions = suggest_model(X, y)

# Preprocess data
X = handle_missing_values(X)
X = encode_categorical(X)
X = scale_features(X)
```

## Documentation

- [API Documentation](docs/API.md)
- [User Guide](docs/user_guide.md)
- [Examples](examples/)

## Contributing

We welcome contributions! Here's how you can help:

### Setting Up Development Environment

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/ThinkML.git
   cd ThinkML
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

### Development Guidelines

1. **Code Style**
   - Follow PEP 8 guidelines
   - Use type hints
   - Write comprehensive docstrings
   - Keep functions focused and modular

2. **Testing**
   - Write unit tests for new features
   - Ensure all tests pass before submitting PR
   - Run tests with pytest:
     ```bash
     pytest tests/
     ```

3. **Documentation**
   - Update API documentation for new features
   - Include examples in docstrings
   - Update user guide if necessary

4. **Pull Request Process**
   - Create a new branch for your feature
   - Write clear commit messages
   - Update documentation
   - Add tests
   - Submit PR with description of changes

### Project Structure

```
ThinkML/
├── docs/
│   ├── API.md
│   └── user_guide.md
├── examples/
│   └── notebooks/
├── tests/
│   ├── test_analyzer/
│   ├── test_describer/
│   └── test_preprocessor/
├── thinkml/
│   ├── analyzer/
│   ├── describer/
│   ├── preprocessor/
│   ├── outliers/
│   ├── feature_selection/
│   └── eda/
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use ThinkML in your research, please cite:

```bibtex
@software{thinkml2024,
  title = {ThinkML: Intelligent Machine Learning Workflow Library},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/ThinkML}
}
``` 