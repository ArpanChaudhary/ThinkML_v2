"""
Dataset description module for ThinkML.
"""

import pandas as pd
import numpy as np

def describe_dataset(data: pd.DataFrame) -> dict:
    """
    Generate a comprehensive description of the dataset.
    
    Args:
        data (pd.DataFrame): Input dataset
        
    Returns:
        dict: Dictionary containing various dataset statistics
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
        
    if data.empty:
        raise ValueError("DataFrame is empty")
    
    description = {
        'summary_stats': data.describe().to_dict(),
        'missing_values': data.isna().sum().to_dict(),
        'data_types': data.dtypes.astype(str).to_dict(),
        'shape': {
            'rows': data.shape[0],
            'columns': data.shape[1]
        },
        'column_info': {
            col: {
                'unique_values': data[col].nunique(),
                'memory_usage': data[col].memory_usage(deep=True) / 1024,  # KB
            } for col in data.columns
        }
    }
    
    # Add correlation matrix for numerical columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        description['correlations'] = data[numeric_cols].corr().to_dict()
    
    return description 