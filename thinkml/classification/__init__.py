"""
Advanced classification models for ThinkML.
"""

from thinkml.classification.multi_label import MultiLabelClassifier
from thinkml.classification.cost_sensitive import CostSensitiveClassifier
from thinkml.classification.ordinal_classification import OrdinalClassifier

__version__ = "1.0.0"
__all__ = [
    "MultiLabelClassifier",
    "CostSensitiveClassifier",
    "OrdinalClassifier"
]

from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class ThinkMLClassifier:
    """A wrapper class for various classification models."""
    
    def __init__(self, model_type: str = "logistic_regression", **kwargs):
        """
        Initialize the classifier.
        
        Args:
            model_type (str): Type of model to use
            **kwargs: Additional model parameters
        """
        self.model_type = model_type
        self.model = self._create_model(**kwargs)
        self.is_fitted = False
    
    def _create_model(self, **kwargs) -> Any:
        """
        Create the specified model.
        
        Args:
            **kwargs: Model parameters
            
        Returns:
            Any: The created model
        """
        if self.model_type == "logistic_regression":
            return LogisticRegression(**kwargs)
        elif self.model_type == "decision_tree":
            return DecisionTreeClassifier(**kwargs)
        elif self.model_type == "random_forest":
            return RandomForestClassifier(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Fit the model to the data.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            
        Returns:
            Dict[str, Any]: Training results
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Fit the model
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Calculate training accuracy
        train_accuracy = self.model.score(X_train, y_train)
        test_accuracy = self.model.score(X_test, y_test)
        
        return {
            "model_type": self.model_type,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "model": self.model
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Args:
            X (pd.DataFrame): Features to predict
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get probability estimates.
        
        Args:
            X (pd.DataFrame): Features to predict
            
        Returns:
            np.ndarray: Probability estimates
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict_proba(X) 