"""
Model evaluation module for ThinkML.

This module provides functionality for evaluating machine learning models
with various metrics for both classification and regression tasks.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
from typing import Dict, Any
from thinkml.evaluation.metrics import mean_squared_error

def evaluate_model(model: Any, X_test: Any, y_test: Any) -> Dict[str, Any]:
    """
    Evaluate a trained model using various metrics.
    
    Args:
        model: Trained model with predict and predict_proba methods
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dict[str, Any]: Dictionary containing evaluation metrics
    """
    try:
        if model is None:
            raise ValueError("Model must be trained before evaluation")
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Handle multiclass vs binary case
        unique_classes = np.unique(y_test)
        is_binary = len(unique_classes) == 2
        
        if is_binary:
            # Binary classification metrics
            metrics.update({
                'precision': precision_score(y_test, y_pred, average='binary'),
                'recall': recall_score(y_test, y_pred, average='binary'),
                'f1': f1_score(y_test, y_pred, average='binary')
            })
            
            # ROC AUC if predict_proba is available
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        else:
            # Multiclass metrics
            metrics.update({
                'precision_macro': precision_score(y_test, y_pred, average='macro'),
                'recall_macro': recall_score(y_test, y_pred, average='macro'),
                'f1_macro': f1_score(y_test, y_pred, average='macro'),
                'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
                'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
                'f1_weighted': f1_score(y_test, y_pred, average='weighted')
            })
            
            # Per-class metrics
            per_class_metrics = {}
            for class_idx in unique_classes:
                binary_y_test = (y_test == class_idx)
                binary_y_pred = (y_pred == class_idx)
                per_class_metrics[f'class_{class_idx}'] = {
                    'precision': precision_score(binary_y_test, binary_y_pred),
                    'recall': recall_score(binary_y_test, binary_y_pred),
                    'f1': f1_score(binary_y_test, binary_y_pred)
                }
            metrics['per_class'] = per_class_metrics
        return {'metrics': metrics}
    except Exception as e:
        return {'metrics': {'error': str(e)}}

__all__ = [
    'evaluate_model',
    'accuracy_score',
    'precision_score',
    'recall_score',
    'f1_score',
    'confusion_matrix',
    'mean_squared_error',
] 