"""
Neural Network implementation for ThinkML.

This module provides Neural Network models for both regression and classification tasks,
implemented from scratch using backpropagation and gradient descent.
"""

import numpy as np
import pandas as pd
import dask.dataframe as dd
from typing import Union, Dict, Any, Optional, List, Tuple, Callable
import dask.array as da
from .base import BaseModel

class NeuralNetwork(BaseModel):
    """
    Base Neural Network class implemented from scratch.
    
    This class provides the foundation for both regression and classification
    neural networks, with customizable architecture and activation functions.
    """
    
    def __init__(
        self,
        hidden_layers: List[int] = [10],
        activation: str = 'relu',
        output_activation: str = 'linear',
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        batch_size: int = 32,
        chunk_size: int = 10000,
        tol: float = 1e-4,
        verbose: bool = False
    ):
        """
        Initialize the Neural Network.
        
        Parameters
        ----------
        hidden_layers : List[int], default=[10]
            List of integers representing the number of neurons in each hidden layer
        activation : str, default='relu'
            Activation function for hidden layers
        output_activation : str, default='linear'
            Activation function for output layer
        learning_rate : float, default=0.01
            Learning rate for gradient descent
        n_iterations : int, default=1000
            Maximum number of iterations for gradient descent
        batch_size : int, default=32
            Size of mini-batches for training
        chunk_size : int, default=10000
            Size of chunks for processing large datasets
        tol : float, default=1e-4
            Tolerance for stopping criterion
        verbose : bool, default=False
            Whether to print progress during training
        """
        super().__init__(chunk_size=chunk_size)
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.tol = tol
        self.verbose = verbose
        self.weights = []
        self.biases = []
        self.loss_history = []
        
        # Set activation functions
        self._set_activation_functions()
    
    def _set_activation_functions(self):
        """Set the activation functions based on the specified strings."""
        # Hidden layer activation
        if self.activation == 'relu':
            self.activation_fn = self._relu
            self.activation_derivative = self._relu_derivative
        elif self.activation == 'sigmoid':
            self.activation_fn = self._sigmoid
            self.activation_derivative = self._sigmoid_derivative
        elif self.activation == 'tanh':
            self.activation_fn = self._tanh
            self.activation_derivative = self._tanh_derivative
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")
        
        # Output layer activation
        if self.output_activation == 'linear':
            self.output_activation_fn = self._linear
            self.output_activation_derivative = self._linear_derivative
        elif self.output_activation == 'sigmoid':
            self.output_activation_fn = self._sigmoid
            self.output_activation_derivative = self._sigmoid_derivative
        elif self.output_activation == 'softmax':
            self.output_activation_fn = self._softmax
            self.output_activation_derivative = self._softmax_derivative
        else:
            raise ValueError(f"Unsupported output activation function: {self.output_activation}")
    
    # Activation functions
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU activation function."""
        return np.where(x > 0, 1, 0)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))
    
    def _sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid activation function."""
        sigmoid_x = self._sigmoid(x)
        return sigmoid_x * (1 - sigmoid_x)
    
    def _tanh(self, x: np.ndarray) -> np.ndarray:
        """Tanh activation function."""
        return np.tanh(x)
    
    def _tanh_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of tanh activation function."""
        return 1 - np.tanh(x) ** 2
    
    def _linear(self, x: np.ndarray) -> np.ndarray:
        """Linear activation function (identity)."""
        return x
    
    def _linear_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of linear activation function."""
        return np.ones_like(x)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation function."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _softmax_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of softmax activation function."""
        # This is a simplified version, as the full derivative is more complex
        # and depends on the specific output being differentiated
        softmax_x = self._softmax(x)
        return softmax_x * (1 - softmax_x)
    
    def _initialize_parameters(self, n_features: int, n_outputs: int):
        """
        Initialize the network parameters (weights and biases).
        
        Parameters
        ----------
        n_features : int
            Number of input features
        n_outputs : int
            Number of output neurons
        """
        self.weights = []
        self.biases = []
        
        # Input layer to first hidden layer
        self.weights.append(np.random.randn(n_features, self.hidden_layers[0]) * 0.01)
        self.biases.append(np.zeros((1, self.hidden_layers[0])))
        
        # Hidden layers
        for i in range(len(self.hidden_layers) - 1):
            self.weights.append(np.random.randn(self.hidden_layers[i], self.hidden_layers[i+1]) * 0.01)
            self.biases.append(np.zeros((1, self.hidden_layers[i+1])))
        
        # Last hidden layer to output layer
        self.weights.append(np.random.randn(self.hidden_layers[-1], n_outputs) * 0.01)
        self.biases.append(np.zeros((1, n_outputs)))
    
    def _forward_propagation(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Perform forward propagation through the network.
        
        Parameters
        ----------
        X : np.ndarray
            Input features
            
        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray]]
            Activations and weighted sums for each layer
        """
        activations = [X]
        z_values = []
        
        # Hidden layers
        for i in range(len(self.hidden_layers)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)
            activations.append(self.activation_fn(z))
        
        # Output layer
        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        z_values.append(z)
        activations.append(this.output_activation_fn(z))
        
        return activations, z_values
    
    def _backward_propagation(self, X: np.ndarray, y: np.ndarray, activations: List[np.ndarray], z_values: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Perform backward propagation to compute gradients.
        
        Parameters
        ----------
        X : np.ndarray
            Input features
        y : np.ndarray
            Target values
        activations : List[np.ndarray]
            Activations for each layer
        z_values : List[np.ndarray]
            Weighted sums for each layer
            
        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray]]
            Gradients for weights and biases
        """
        m = X.shape[0]
        n_layers = len(this.weights)
        
        # Initialize gradients
        dW = [np.zeros_like(w) for w in this.weights]
        db = [np.zeros_like(b) for b in this.biases]
        
        # Compute output layer error
        # This will be overridden by subclasses
        delta = np.zeros_like(activations[-1])
        
        # Backpropagate the error
        for i in range(n_layers - 1, -1, -1):
            if i == n_layers - 1:
                # Output layer
                dW[i] = np.dot(activations[i].T, delta) / m
                db[i] = np.sum(delta, axis=0, keepdims=True) / m
            else:
                # Hidden layers
                delta = np.dot(delta, this.weights[i+1].T) * this.activation_derivative(z_values[i])
                dW[i] = np.dot(activations[i].T, delta) / m
                db[i] = np.sum(delta, axis=0, keepdims=True) / m
        
        return dW, db
    
    def _update_parameters(self, dW: List[np.ndarray], db: List[np.ndarray]):
        """
        Update network parameters using gradient descent.
        
        Parameters
        ----------
        dW : List[np.ndarray]
            Gradients for weights
        db : List[np.ndarray]
            Gradients for biases
        """
        for i in range(len(this.weights)):
            this.weights[i] -= this.learning_rate * dW[i]
            this.biases[i] -= this.learning_rate * db[i]
    
    def _compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute the loss function.
        
        Parameters
        ----------
        y_pred : np.ndarray
            Predicted values
        y_true : np.ndarray
            True values
            
        Returns
        -------
        float
            Loss value
        """
        # This will be overridden by subclasses
        return 0.0
    
    def _create_mini_batches(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create mini-batches for training.
        
        Parameters
        ----------
        X : np.ndarray
            Input features
        y : np.ndarray
            Target values
            
        Returns
        -------
        List[Tuple[np.ndarray, np.ndarray]]
            List of mini-batches
        """
        m = X.shape[0]
        mini_batches = []
        
        # Shuffle the data
        permutation = np.random.permutation(m)
        shuffled_X = X[permutation]
        shuffled_y = y[permutation]
        
        # Create mini-batches
        num_complete_batches = m // this.batch_size
        for i in range(num_complete_batches):
            mini_batch_X = shuffled_X[i * this.batch_size:(i + 1) * this.batch_size]
            mini_batch_y = shuffled_y[i * this.batch_size:(i + 1) * this.batch_size]
            mini_batches.append((mini_batch_X, mini_batch_y))
        
        # Handle the last mini-batch if it's smaller than batch_size
        if m % this.batch_size != 0:
            mini_batch_X = shuffled_X[num_complete_batches * this.batch_size:]
            mini_batch_y = shuffled_y[num_complete_batches * this.batch_size:]
            mini_batches.append((mini_batch_X, mini_batch_y))
        
        return mini_batches
    
    def fit(self, X, y):
        """
        Fit the Neural Network to the data.
        
        Parameters
        ----------
        X : Union[pd.DataFrame, dd.DataFrame, np.ndarray]
            Training features
        y : Union[pd.Series, np.ndarray, dd.Series]
            Target values
            
        Returns
        -------
        self : object
            Returns self
        """
        # This will be overridden by subclasses
        return self
    
    def predict(self, X):
        """
        Make predictions for X.
        
        Parameters
        ----------
        X : Union[pd.DataFrame, dd.DataFrame, np.ndarray]
            Samples
            
        Returns
        -------
        Union[np.ndarray, pd.Series, dd.Series]
            Predicted values
        """
        # This will be overridden by subclasses
        return None
    
    def score(self, X, y):
        """
        Return the score of the model on the given test data and labels.
        
        Parameters
        ----------
        X : Union[pd.DataFrame, dd.DataFrame, np.ndarray]
            Test samples
        y : Union[pd.Series, np.ndarray, dd.Series]
            True labels for X
            
        Returns
        -------
        float
            Score of the model
        """
        # This will be overridden by subclasses
        return 0.0


class NeuralNetworkRegressor(NeuralNetwork):
    """
    Neural Network Regressor implemented from scratch.
    
    This model uses backpropagation and gradient descent to learn the parameters
    for regression tasks.
    """
    
    def __init__(
        self,
        hidden_layers: List[int] = [10],
        activation: str = 'relu',
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        batch_size: int = 32,
        chunk_size: int = 10000,
        tol: float = 1e-4,
        verbose: bool = False
    ):
        """
        Initialize the Neural Network Regressor.
        
        Parameters
        ----------
        hidden_layers : List[int], default=[10]
            List of integers representing the number of neurons in each hidden layer
        activation : str, default='relu'
            Activation function for hidden layers
        learning_rate : float, default=0.01
            Learning rate for gradient descent
        n_iterations : int, default=1000
            Maximum number of iterations for gradient descent
        batch_size : int, default=32
            Size of mini-batches for training
        chunk_size : int, default=10000
            Size of chunks for processing large datasets
        tol : float, default=1e-4
            Tolerance for stopping criterion
        verbose : bool, default=False
            Whether to print progress during training
        """
        super().__init__(
            hidden_layers=hidden_layers,
            activation=activation,
            output_activation='linear',
            learning_rate=learning_rate,
            n_iterations=n_iterations,
            batch_size=batch_size,
            chunk_size=chunk_size,
            tol=tol,
            verbose=verbose
        )
    
    def _compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute the mean squared error loss.
        
        Parameters
        ----------
        y_pred : np.ndarray
            Predicted values
        y_true : np.ndarray
            True values
            
        Returns
        -------
        float
            Mean squared error
        """
        return np.mean((y_pred - y_true) ** 2)
    
    def _backward_propagation(self, X: np.ndarray, y: np.ndarray, activations: List[np.ndarray], z_values: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Perform backward propagation to compute gradients.
        
        Parameters
        ----------
        X : np.ndarray
            Input features
        y : np.ndarray
            Target values
        activations : List[np.ndarray]
            Activations for each layer
        z_values : List[np.ndarray]
            Weighted sums for each layer
            
        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray]]
            Gradients for weights and biases
        """
        m = X.shape[0]
        n_layers = len(this.weights)
        
        # Initialize gradients
        dW = [np.zeros_like(w) for w in this.weights]
        db = [np.zeros_like(b) for b in this.biases]
        
        # Compute output layer error
        delta = 2 * (activations[-1] - y) / m
        
        # Backpropagate the error
        for i in range(n_layers - 1, -1, -1):
            if i == n_layers - 1:
                # Output layer
                dW[i] = np.dot(activations[i].T, delta)
                db[i] = np.sum(delta, axis=0, keepdims=True)
            else:
                # Hidden layers
                delta = np.dot(delta, this.weights[i+1].T) * this.activation_derivative(z_values[i])
                dW[i] = np.dot(activations[i].T, delta)
                db[i] = np.sum(delta, axis=0, keepdims=True)
        
        return dW, db
    
    def fit(self, X, y):
        """
        Fit the Neural Network Regressor to the data.
        
        Parameters
        ----------
        X : Union[pd.DataFrame, dd.DataFrame, np.ndarray]
            Training features
        y : Union[pd.Series, np.ndarray, dd.Series]
            Target values
            
        Returns
        -------
        self : object
            Returns self
        """
        # Preprocess data
        X, y = this._preprocess_data(X, y)
        
        # Check if we're working with Dask DataFrames
        if isinstance(X, dd.DataFrame) or isinstance(y, dd.Series):
            # Convert to numpy arrays
            X = X.compute() if isinstance(X, dd.DataFrame) else X
            y = y.compute() if isinstance(y, dd.Series) else y
        
        # Initialize parameters
        n_features = X.shape[1]
        n_outputs = 1 if len(y.shape) == 1 else y.shape[1]
        this._initialize_parameters(n_features, n_outputs)
        
        # Training loop
        for i in range(this.n_iterations):
            # Create mini-batches
            mini_batches = this._create_mini_batches(X, y)
            
            # Process each mini-batch
            for mini_batch_X, mini_batch_y in mini_batches:
                # Forward propagation
                activations, z_values = this._forward_propagation(mini_batch_X)
                
                # Backward propagation
                dW, db = this._backward_propagation(mini_batch_X, mini_batch_y, activations, z_values)
                
                # Update parameters
                this._update_parameters(dW, db)
            
            # Compute loss on the entire dataset
            activations, _ = this._forward_propagation(X)
            loss = this._compute_loss(activations[-1], y)
            this.loss_history.append(loss)
            
            # Check for convergence
            if i > 0 and abs(this.loss_history[-1] - this.loss_history[-2]) < this.tol:
                if this.verbose:
                    print(f"Converged at iteration {i+1}")
                break
            
            if this.verbose and (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{this.n_iterations}, Loss: {this.loss_history[-1]}")
        
        this.is_fitted = True
        return this
    
    def predict(self, X):
        """
        Make predictions for X.
        
        Parameters
        ----------
        X : Union[pd.DataFrame, dd.DataFrame, np.ndarray]
            Samples
            
        Returns
        -------
        Union[np.ndarray, pd.Series, dd.Series]
            Predicted values
        """
        this._check_is_fitted()
        
        # Preprocess data
        X = this._preprocess_data(X)
        
        # Check if we're working with Dask DataFrames
        is_dask = isinstance(X, dd.DataFrame)
        
        if is_dask:
            # Convert to numpy array
            X = X.compute()
        
        # Forward propagation
        activations, _ = this._forward_propagation(X)
        y_pred = activations[-1]
        
        # Convert back to Dask Series if input was a DataFrame
        if is_dask:
            y_pred = dd.from_array(y_pred)
        
        return y_pred
    
    def score(self, X, y):
        """
        Return the R² score of the model on the given test data and labels.
        
        Parameters
        ----------
        X : Union[pd.DataFrame, dd.DataFrame, np.ndarray]
            Test samples
        y : Union[pd.Series, np.ndarray, dd.Series]
            True labels for X
            
        Returns
        -------
        float
            R² score of the model
        """
        this._check_is_fitted()
        
        # Preprocess data
        X, y = this._preprocess_data(X, y)
        
        # Make predictions
        y_pred = this.predict(X)
        
        # Check if we're working with Dask DataFrames
        is_dask = isinstance(y, dd.Series) or isinstance(y_pred, dd.Series)
        
        if is_dask:
            # Convert to numpy arrays
            y = y.compute() if isinstance(y, dd.Series) else y
            y_pred = y_pred.compute() if isinstance(y_pred, dd.Series) else y_pred
        
        # Compute R² score
        y_mean = np.mean(y)
        ss_tot = np.sum((y - y_mean) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        return r2


class NeuralNetworkClassifier(NeuralNetwork):
    """
    Neural Network Classifier implemented from scratch.
    
    This model uses backpropagation and gradient descent to learn the parameters
    for classification tasks.
    """
    
    def __init__(
        self,
        hidden_layers: List[int] = [10],
        activation: str = 'relu',
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        batch_size: int = 32,
        chunk_size: int = 10000,
        tol: float = 1e-4,
        verbose: bool = False
    ):
        """
        Initialize the Neural Network Classifier.
        
        Parameters
        ----------
        hidden_layers : List[int], default=[10]
            List of integers representing the number of neurons in each hidden layer
        activation : str, default='relu'
            Activation function for hidden layers
        learning_rate : float, default=0.01
            Learning rate for gradient descent
        n_iterations : int, default=1000
            Maximum number of iterations for gradient descent
        batch_size : int, default=32
            Size of mini-batches for training
        chunk_size : int, default=10000
            Size of chunks for processing large datasets
        tol : float, default=1e-4
            Tolerance for stopping criterion
        verbose : bool, default=False
            Whether to print progress during training
        """
        super().__init__(
            hidden_layers=hidden_layers,
            activation=activation,
            output_activation='sigmoid' if hidden_layers[-1] == 1 else 'softmax',
            learning_rate=learning_rate,
            n_iterations=n_iterations,
            batch_size=batch_size,
            chunk_size=chunk_size,
            tol=tol,
            verbose=verbose
        )
        this.classes = None
    
    def _compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute the cross-entropy loss.
        
        Parameters
        ----------
        y_pred : np.ndarray
            Predicted probabilities
        y_true : np.ndarray
            True labels
            
        Returns
        -------
        float
            Cross-entropy loss
        """
        # Add a small epsilon to avoid log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        if len(this.classes) == 2:
            # Binary classification
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            # Multiclass classification
            return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    def _backward_propagation(self, X: np.ndarray, y: np.ndarray, activations: List[np.ndarray], z_values: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Perform backward propagation to compute gradients.
        
        Parameters
        ----------
        X : np.ndarray
            Input features
        y : np.ndarray
            Target values
        activations : List[np.ndarray]
            Activations for each layer
        z_values : List[np.ndarray]
            Weighted sums for each layer
            
        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray]]
            Gradients for weights and biases
        """
        m = X.shape[0]
        n_layers = len(this.weights)
        
        # Initialize gradients
        dW = [np.zeros_like(w) for w in this.weights]
        db = [np.zeros_like(b) for b in this.biases]
        
        # Compute output layer error
        if len(this.classes) == 2:
            # Binary classification
            delta = activations[-1] - y
        else:
            # Multiclass classification
            delta = activations[-1] - y
        
        # Backpropagate the error
        for i in range(n_layers - 1, -1, -1):
            if i == n_layers - 1:
                # Output layer
                dW[i] = np.dot(activations[i].T, delta) / m
                db[i] = np.sum(delta, axis=0, keepdims=True) / m
            else:
                # Hidden layers
                delta = np.dot(delta, this.weights[i+1].T) * this.activation_derivative(z_values[i])
                dW[i] = np.dot(activations[i].T, delta) / m
                db[i] = np.sum(delta, axis=0, keepdims=True) / m
        
        return dW, db
    
    def fit(self, X, y):
        """
        Fit the Neural Network Classifier to the data.
        
        Parameters
        ----------
        X : Union[pd.DataFrame, dd.DataFrame, np.ndarray]
            Training features
        y : Union[pd.Series, np.ndarray, dd.Series]
            Target values
            
        Returns
        -------
        self : object
            Returns self
        """
        # Preprocess data
        X, y = this._preprocess_data(X, y)
        
        # Check if we're working with Dask DataFrames
        if isinstance(X, dd.DataFrame) or isinstance(y, dd.Series):
            # Convert to numpy arrays
            X = X.compute() if isinstance(X, dd.DataFrame) else X
            y = y.compute() if isinstance(y, dd.Series) else y
        
        # Store classes
        this.classes = np.unique(y)
        n_classes = len(this.classes)
        
        # Convert labels to one-hot encoding for multiclass classification
        if n_classes > 2:
            y_one_hot = np.zeros((len(y), n_classes))
            for i, label in enumerate(this.classes):
                y_one_hot[y == label, i] = 1
            y = y_one_hot
        
        # Initialize parameters
        n_features = X.shape[1]
        n_outputs = 1 if n_classes == 2 else n_classes
        this._initialize_parameters(n_features, n_outputs)
        
        # Training loop
        for i in range(this.n_iterations):
            # Create mini-batches
            mini_batches = this._create_mini_batches(X, y)
            
            # Process each mini-batch
            for mini_batch_X, mini_batch_y in mini_batches:
                # Forward propagation
                activations, z_values = this._forward_propagation(mini_batch_X)
                
                # Backward propagation
                dW, db = this._backward_propagation(mini_batch_X, mini_batch_y, activations, z_values)
                
                # Update parameters
                this._update_parameters(dW, db)
            
            # Compute loss on the entire dataset
            activations, _ = this._forward_propagation(X)
            loss = this._compute_loss(activations[-1], y)
            this.loss_history.append(loss)
            
            # Check for convergence
            if i > 0 and abs(this.loss_history[-1] - this.loss_history[-2]) < this.tol:
                if this.verbose:
                    print(f"Converged at iteration {i+1}")
                break
            
            if this.verbose and (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{this.n_iterations}, Loss: {this.loss_history[-1]}")
        
        this.is_fitted = True
        return this
    
    def predict(self, X):
        """
        Make predictions for X.
        
        Parameters
        ----------
        X : Union[pd.DataFrame, dd.DataFrame, np.ndarray]
            Samples
            
        Returns
        -------
        Union[np.ndarray, pd.Series, dd.Series]
            Predicted classes
        """
        this._check_is_fitted()
        
        # Preprocess data
        X = this._preprocess_data(X)
        
        # Check if we're working with Dask DataFrames
        is_dask = isinstance(X, dd.DataFrame)
        
        if is_dask:
            # Convert to numpy array
            X = X.compute()
        
        # Forward propagation
        activations, _ = this._forward_propagation(X)
        y_pred_proba = activations[-1]
        
        # Convert probabilities to class labels
        if len(this.classes) == 2:
            y_pred = (y_pred_proba >= 0.5).astype(int)
            y_pred = this.classes[y_pred.flatten()]
        else:
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_pred = this.classes[y_pred]
        
        # Convert back to Dask Series if input was a DataFrame
        if is_dask:
            y_pred = dd.from_array(y_pred)
        
        return y_pred
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Parameters
        ----------
        X : Union[pd.DataFrame, dd.DataFrame, np.ndarray]
            Samples
            
        Returns
        -------
        Union[np.ndarray, pd.Series, dd.Series]
            Predicted class probabilities
        """
        this._check_is_fitted()
        
        # Preprocess data
        X = this._preprocess_data(X)
        
        # Check if we're working with Dask DataFrames
        is_dask = isinstance(X, dd.DataFrame)
        
        if is_dask:
            # Convert to numpy array
            X = X.compute()
        
        # Forward propagation
        activations, _ = this._forward_propagation(X)
        y_pred_proba = activations[-1]
        
        # Convert back to Dask DataFrame if input was a DataFrame
        if is_dask:
            y_pred_proba = dd.from_array(y_pred_proba)
        
        return y_pred_proba
    
    def score(self, X, y):
        """
        Return the accuracy score of the model on the given test data and labels.
        
        Parameters
        ----------
        X : Union[pd.DataFrame, dd.DataFrame, np.ndarray]
            Test samples
        y : Union[pd.Series, np.ndarray, dd.Series]
            True labels for X
            
        Returns
        -------
        float
            Accuracy score of the model
        """
        this._check_is_fitted()
        
        # Preprocess data
        X, y = this._preprocess_data(X, y)
        
        # Make predictions
        y_pred = this.predict(X)
        
        # Check if we're working with Dask DataFrames
        is_dask = isinstance(y, dd.Series) or isinstance(y_pred, dd.Series)
        
        if is_dask:
            # Convert to numpy arrays
            y = y.compute() if isinstance(y, dd.Series) else y
            y_pred = y_pred.compute() if isinstance(y_pred, dd.Series) else y_pred
        
        # Compute accuracy
        accuracy = np.mean(y == y_pred)
        
        return accuracy 