# ThinkML Engine Documentation

The ThinkML Engine is a powerful, prompt-driven machine learning system that converts natural language instructions into executable ML code. This document provides a comprehensive guide to using the engine effectively.

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Supported Prompts](#supported-prompts)
5. [Components](#components)
6. [Error Handling](#error-handling)
7. [Examples](#examples)

## Overview

The ThinkML Engine consists of several components that work together to provide a seamless ML coding experience:

- **PromptRefiner**: Normalizes and standardizes natural language prompts
- **PromptInterpreter**: Converts refined prompts into structured commands
- **CodeGenerator**: Generates ThinkML code based on commands
- **ExecutionManager**: Safely executes generated code
- **ErrorHandler**: Provides helpful error messages and suggestions

## Installation

The ThinkML Engine is included in the ThinkML package. Install it using pip:

```bash
pip install thinkml
```

## Basic Usage

Here's a simple example of using the ThinkML Engine:

```python
from thinkml.engine import ThinkMLEngine

# Initialize the engine
engine = ThinkMLEngine()

# Process a prompt
result = engine.process("Train a logistic regression model on my data")
```

## Supported Prompts

### Data Description
- "Describe the dataset"
- "Analyze the data"
- "Show data statistics"
- "Summarize the data"

### Data Preprocessing
- "Clean the data"
- "Preprocess the dataset"
- "Normalize the features"
- "Prepare the data"

### Model Training
- "Train a logistic regression model"
- "Build a decision tree"
- "Create a random forest"
- "Fit a linear regression"

### Model Evaluation
- "Evaluate the model"
- "Check model performance"
- "Test the model"
- "Validate the results"

### Visualization
- "Plot the ROC curve"
- "Show feature importance"
- "Display correlation heatmap"
- "Plot confusion matrix"

## Components

### PromptRefiner

The PromptRefiner normalizes various ways of expressing the same ML task:

```python
from thinkml.engine import PromptRefiner

refiner = PromptRefiner()
refined = refiner.refine("clean and prepare the data")
# Returns: {'task': 'preprocess', 'model': None, 'visualization': None}
```

### PromptInterpreter

The PromptInterpreter converts refined prompts into structured commands:

```python
from thinkml.engine import PromptInterpreter

interpreter = PromptInterpreter()
command = interpreter.parse("Train a decision tree")
# Returns: {'task': 'train', 'model': 'decision_tree', 'parameters': {}}
```

### CodeGenerator

The CodeGenerator creates ThinkML code based on commands:

```python
from thinkml.engine import CodeGenerator

generator = CodeGenerator()
code = generator.generate_code({
    'task': 'train',
    'model': 'decision_tree'
})
```

### ExecutionManager

The ExecutionManager safely executes generated code:

```python
from thinkml.engine import ExecutionManager

executor = ExecutionManager()
result = executor.execute(code)
```

## Error Handling

The engine provides helpful error messages and suggestions when:

1. The prompt is ambiguous
2. The requested model is not supported
3. The task is not recognized
4. The visualization type is not available

Example:
```python
try:
    result = engine.process("do something with the data")
except Exception as e:
    print(engine.get_suggestions())
    # Outputs: "Try using one of these tasks: describe, preprocess, train, evaluate, visualize"
```

## Examples

### Basic Data Analysis
```python
engine.process("Describe the dataset")
```

### Model Training with Visualization
```python
engine.process("Train a random forest and show feature importance")
```

### Complete ML Pipeline
```python
engine.process("""
1. Clean the data
2. Train a logistic regression
3. Plot the ROC curve
4. Show the confusion matrix
""")
```

### Custom Parameters
```python
engine.process("Train a decision tree with max_depth=5 and min_samples_split=10")
```

## Best Practices

1. **Be Specific**: Use clear, specific prompts for better results
2. **Use Standard Terms**: Stick to common ML terminology
3. **Check Suggestions**: Use the suggestion system when unsure
4. **Combine Tasks**: You can combine multiple tasks in one prompt
5. **Handle Errors**: Always wrap engine calls in try-except blocks

## Advanced Usage

### Custom Model Parameters
```python
engine.process("Train a random forest with n_estimators=100 and max_depth=10")
```

### Multiple Visualizations
```python
engine.process("Show both ROC curve and feature importance")
```

### Pipeline Chaining
```python
engine.process("""
1. Preprocess the data
2. Train a model
3. Evaluate performance
4. Show all relevant plots
""")
```

## Troubleshooting

### Common Issues

1. **Ambiguous Prompts**
   - Solution: Use more specific language
   - Example: Instead of "train a model", use "train a logistic regression model"

2. **Unsupported Models**
   - Solution: Check available models using `engine.get_supported_models()`
   - Use one of the supported model types

3. **Missing Parameters**
   - Solution: Provide all required parameters
   - Use the suggestion system to see required parameters

4. **Execution Errors**
   - Solution: Check the error message and suggestions
   - Ensure data is properly loaded and formatted

## Contributing

To contribute to the ThinkML Engine:

1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Write tests
5. Submit a pull request

## License

ThinkML is licensed under the MIT License. See the LICENSE file for details. 