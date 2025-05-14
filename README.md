# ThinkML: End-to-End Machine Learning Workflow Library

ThinkML is a comprehensive machine learning library designed to streamline your ML workflow while maintaining flexibility and ease of use.

## 🚀 Quick Installation

### Basic Installation (Core Features)
```bash
pip install thinkml
```

### Installation with Optional Components

Choose the components you need:

```bash
# For all features
pip install "thinkml[all]"

# For visualization tools
pip install "thinkml[viz]"

# For boosting algorithms
pip install "thinkml[boost]"

# For deployment tools
pip install "thinkml[deploy]"

# For deep learning support
pip install "thinkml[deep]"

# For full ML stack (excluding deep learning)
pip install "thinkml[full]"
```

## 📦 Components

### Core Features (Included in Basic Installation)
- Data preprocessing and analysis
- Basic model training and evaluation
- Standard visualization tools
- Performance metrics and validation

### Optional Components

#### Visualization Tools [viz]
- Interactive plotting with Plotly
- Dashboard creation with Dash

#### Boosting Libraries [boost]
- XGBoost
- LightGBM
- (Note: CatBoost available for non-Windows platforms)

#### Deployment Tools [deploy]
- FastAPI integration
- ONNX export
- Model serving utilities

#### Deep Learning [deep]
- PyTorch integration
- TensorFlow support

#### Full ML Stack [full]
- All ML-related tools
- Advanced optimization
- Distributed computing support

## 🔧 Troubleshooting

### Common Installation Issues

1. **Missing C++ Build Tools (Windows)**
   ```bash
   # Solution 1: Install pre-built wheels
   pip install --only-binary :all: thinkml[boost]
   
   # Solution 2: Install Build Tools
   # Download and install from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   ```

2. **Memory Issues**
   ```bash
   # Install with minimal dependencies first
   pip install thinkml
   
   # Then add components as needed
   pip install "thinkml[boost]"
   ```

3. **Version Conflicts**
   ```bash
   # Create a fresh virtual environment
   python -m venv thinkml_env
   source thinkml_env/bin/activate  # Linux/Mac
   thinkml_env\Scripts\activate     # Windows
   
   # Install ThinkML
   pip install "thinkml[all]"
   ```

## 📚 Documentation

For detailed documentation, visit: [Documentation Link]

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Features

- Modular design for flexible usage
- Comprehensive ML workflow support
- Easy integration with popular ML frameworks
- Extensive visualization capabilities
- Production-ready deployment tools

## 🔗 Links

- [Documentation](docs/)
- [Examples](examples/)
- [Issue Tracker](https://github.com/ArpanChaudhary/ThinkML/issues)
- [Source Code](https://github.com/ArpanChaudhary/ThinkML)

## Key Features

- **Natural Language Processing of ML Tasks:** Converts user prompts into structured commands.
- **Automatic Code Generation:** Generates Python code for common ML workflows (data description, preprocessing, model training, evaluation, visualization).
- **Support for Various ML Models:** Includes logistic regression, linear regression, decision trees, random forests, and more.
- **Built-in Visualization:** Supports ROC curves, feature importance plots, correlation heatmaps, and confusion matrices.
- **Comprehensive Model Evaluation:** Provides metrics like accuracy, F1 score, MSE, and R².
- **Dataset Description and Analysis:** Tools for summarizing and analyzing datasets.

## Project Structure

- **`thinkml/`**: Core source code.
  - **`thinkml_engine.py`**: Main engine that coordinates prompt processing, code generation, and execution.
  - **`engine/`**: Contains the `PromptRefiner` for normalizing and refining natural language prompts.
  - **`model/`**: Includes `trainer.py` for training and evaluating multiple models.
  - **`algorithms/`**: Contains implementations of various ML algorithms (logistic regression, linear regression, decision trees, random forests, etc.).
  - **`preprocessing/`, `visualization/`, `evaluation/`, etc.**: Additional modules for data preprocessing, visualization, and model evaluation.
- **`tests/`**: Test suite for the project.
- **`docs/`, `documentation/`**: Documentation files.
- **`examples/`, `book/`**: Example code and tutorials.
- **`.github/`**: GitHub workflows for CI/CD.
- **`requirements.txt`**: Python dependencies.
- **`setup.py`**: Project setup for packaging.
- **`README.md`**: Project overview and usage.

## Dependencies

- **Core ML Libraries:** `numpy`, `pandas`, `scikit-learn`, `dask`, `dask-ml`, `vaex`, `pyarrow`, `fastparquet`.
- **Advanced ML and Optimization:** `scikit-optimize`, `shap`, `lime`, `xgboost`, `lightgbm`, `catboost`, `optuna`, `statsmodels`, `scipy`.
- **Visualization:** `matplotlib`, `seaborn`, `plotly`, `dash`.
- **Deep Learning (Optional):** `torch`, `tensorflow`.
- **Model Deployment:** `fastapi`, `uvicorn`, `onnx`, `pmml2json`.
- **Monitoring and Logging:** `prometheus-client`, `python-json-logger`, `mlflow`.
- **Development and Testing:** `pytest`, `pytest-cov`, `black`, `flake8`, `mypy`, `isort`.

## Quick Start

```python
from thinkml import ThinkMLEngine
import pandas as pd

# Initialize the engine
engine = ThinkMLEngine()

# Load your data
data = pd.read_csv('your_data.csv')
engine.set_data(data)

# Process natural language prompts
result = engine.process("Describe the dataset")
result = engine.process("Train a decision tree model")
result = engine.process("Plot feature importance")
```

## Supported Tasks

- Dataset description and analysis
- Data preprocessing and cleaning
- Model training (various algorithms)
- Model evaluation
- Visualization (ROC curves, feature importance, etc.)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use ThinkML in your research, please cite:

```bibtex
@software{thinkml2025,
  title = {ThinkML: Advanced Validation and Feature Engineering for Machine Learning},
  author = {Arpan Chaudhary},
  year = {2025},
  version = {1.0},
  url = {https://github.com/ArpanChaudhary/ThinkML}
}
```

## Acknowledgments

- Inspired by various open-source machine learning libraries
- Special thanks to the Python data science community

## Contact

Arpan Chaudhary - [@ArpanChaudhary](https://github.com/ArpanChaudhary)

Project Link: [https://github.com/ArpanChaudhary/ThinkML](https://github.com/ArpanChaudhary/ThinkML)

## Support

If you encounter any issues or have questions, please:

1. Check the [documentation](https://github.com/ArpanChaudhary/ThinkML#readme)
2. Search existing [issues](https://github.com/ArpanChaudhary/ThinkML/issues)
3. Create a new issue if needed

## Roadmap

- [ ] Enhanced deep learning support
- [ ] Automated hyperparameter tuning
- [ ] Distributed computing support
- [ ] Model deployment tools
- [ ] Web API interface
- [ ] GUI for interactive model building 