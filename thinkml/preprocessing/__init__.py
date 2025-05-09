"""
Preprocessing module for ThinkML.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the input data.
    
    Args:
        data (pd.DataFrame): Input data to preprocess
        
    Returns:
        pd.DataFrame: Preprocessed data
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Handle missing values
    df = df.fillna(df.mean())
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Encode categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    return df 