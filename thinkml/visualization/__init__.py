"""
Visualization module for ThinkML.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import pandas as pd
from typing import Any, Optional

class PlotResult:
    """Wrapper class for plot results."""
    def __init__(self, figure: plt.Figure, data: Optional[dict] = None):
        self.figure = figure
        self.data = data or {}

def plot_roc_curve(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> PlotResult:
    """
    Plot ROC curve for a trained model.
    
    Args:
        model: Trained model with predict_proba method
        X_test: Test features
        y_test: Test labels
        
    Returns:
        PlotResult: Plot result with figure and metrics
    """
    if model is None:
        raise ValueError("Model must be trained before plotting")
    
    # Check if binary or multiclass
    unique_classes = np.unique(y_test)
    n_classes = len(unique_classes)
    
    if n_classes == 2:
        # Binary classification
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        
        return PlotResult(fig, {'auc': roc_auc, 'fpr': fpr, 'tpr': tpr})
    else:
        # Multiclass ROC curve
        y_pred_proba = model.predict_proba(X_test)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot ROC curve for each class
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
        
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Multiclass ROC Curves (One-vs-Rest)')
        ax.legend(loc="lower right")
        
        return PlotResult(fig)

def plot_feature_importance(model: Any, feature_names: list) -> PlotResult:
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        
    Returns:
        PlotResult: Plot result with figure and importance scores
    """
    if model is None:
        raise ValueError("Model must be trained before plotting")
        
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not support feature importance")
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(importances)), importances[indices])
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    ax.set_title('Feature Importance')
    plt.tight_layout()
    
    return PlotResult(fig, {'importance_scores': dict(zip(feature_names, importances))})

def plot_correlation_heatmap(data: pd.DataFrame) -> PlotResult:
    """
    Plot correlation heatmap for numerical features.
    
    Args:
        data: Input DataFrame
        
    Returns:
        PlotResult: Plot result with figure and correlation matrix
    """
    numeric_data = data.select_dtypes(include=[np.number])
    if numeric_data.empty:
        raise ValueError("No numeric columns found in data")
        
    corr_matrix = numeric_data.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Feature Correlation Heatmap')
    plt.tight_layout()
    
    return PlotResult(fig, {'correlation_matrix': corr_matrix.to_dict()})

def plot_confusion_matrix(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> PlotResult:
    """
    Plot confusion matrix for a trained model.
    
    Args:
        model: Trained model with predict method
        X_test: Test features
        y_test: Test labels
        
    Returns:
        PlotResult: Plot result with figure and confusion matrix
    """
    if model is None:
        raise ValueError("Model must be trained before plotting")
    
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    # Get class labels
    unique_classes = np.unique(y_test)
    n_classes = len(unique_classes)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    
    # Add class labels if available
    if hasattr(model, 'classes_'):
        ax.set_xticklabels(model.classes_)
        ax.set_yticklabels(model.classes_)
    
    return PlotResult(fig, {'confusion_matrix': cm.tolist()}) 