"""
Categorical encoding module for ThinkML.
This module provides functionality to encode categorical features.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Union
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from category_encoders import BinaryEncoder


def encode_categorical(
    X: pd.DataFrame, 
    method: str = 'onehot',
    columns: Optional[List[str]] = None,
    drop_first: bool = False,
    handle_unknown: str = 'ignore',
    chunk_size: int = 100000
) -> pd.DataFrame:
    """
    Encode categorical features using various encoding methods.

    Args:
        X (pd.DataFrame): Input DataFrame containing categorical features.
        method (str, optional): Encoding method to use.
            Options: 'onehot', 'label', 'ordinal', 'binary'.
            Default is 'onehot'.
        columns (Optional[List[str]], optional): List of column names to encode.
            If None, all categorical columns will be encoded. Default is None.
        drop_first (bool, optional): Whether to drop one category when using one-hot encoding.
            Default is False.
        handle_unknown (str, optional): How to handle unknown categories.
            Options: 'error', 'ignore', 'use_encoded_value'.
            Default is 'ignore'.
        chunk_size (int, optional): Number of rows to process at a time for memory efficiency.
            Default is 100000.

    Returns:
        pd.DataFrame: DataFrame with categorical features encoded according to the specified method.

    Raises:
        ValueError: If method is not one of the supported options.
        ValueError: If columns is provided but contains non-existent column names.
    """
    # Input validation
    if not isinstance(X, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    if X.empty:
        return X
    
    valid_methods = ['onehot', 'label', 'ordinal', 'binary']
    if method not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}")
    
    # Determine which columns to encode
    if columns is None:
        # Automatically detect categorical columns
        categorical_columns = [col for col in X.columns 
                              if not pd.api.types.is_numeric_dtype(X[col])]
    else:
        # Validate that all specified columns exist
        missing_columns = [col for col in columns if col not in X.columns]
        if missing_columns:
            raise ValueError(f"Columns not found in DataFrame: {missing_columns}")
        categorical_columns = columns
    
    # If no categorical columns to encode, return the original DataFrame
    if not categorical_columns:
        return X
    
    # Check if dataset is large enough to use Dask
    if len(X) > 1_000_000:
        return _encode_categorical_dask(X, method, categorical_columns, drop_first)
    
    # Process in chunks for memory efficiency
    if len(X) > chunk_size:
        return _encode_categorical_chunks(X, method, categorical_columns, drop_first, chunk_size)
    
    # For small datasets, process directly
    return _encode_categorical_direct(X, method, categorical_columns, drop_first)


def _encode_categorical_direct(
    X: pd.DataFrame, 
    method: str, 
    categorical_columns: List[str],
    drop_first: bool
) -> pd.DataFrame:
    """
    Encode categorical features directly for small datasets.
    
    Parameters
    ----------
    X : pd.DataFrame
        Input DataFrame.
    method : str
        Encoding method to use.
    categorical_columns : List[str]
        List of column names to encode.
    drop_first : bool
        Whether to drop one category for each categorical feature when using one-hot encoding.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with categorical features encoded.
    """
    result = X.copy()
    
    if method == 'onehot':
        # Use pandas get_dummies for one-hot encoding
        encoded = pd.get_dummies(result[categorical_columns], 
                                prefix=categorical_columns,
                                drop_first=drop_first)
        
        # Drop original columns and add encoded columns
        result = result.drop(columns=categorical_columns)
        result = pd.concat([result, encoded], axis=1)
        
    elif method == 'label':
        # Use LabelEncoder for each column
        for col in categorical_columns:
            result[col] = LabelEncoder().fit_transform(result[col].astype(str))
            
    elif method == 'ordinal':
        # Use OrdinalEncoder for all columns at once
        encoder = OrdinalEncoder()
        result[categorical_columns] = encoder.fit_transform(result[categorical_columns])
        
    elif method == 'binary':
        # Use BinaryEncoder for each column
        for col in categorical_columns:
            encoder = BinaryEncoder(cols=[col])
            encoded = encoder.fit_transform(result)
            
            # Drop original column and add encoded columns
            result = result.drop(columns=[col])
            for new_col in encoded.columns:
                if new_col != col:  # Skip the original column name
                    result[new_col] = encoded[new_col]
    
    return result


def _encode_categorical_chunks(
    X: pd.DataFrame, 
    method: str, 
    categorical_columns: List[str],
    drop_first: bool,
    chunk_size: int
) -> pd.DataFrame:
    """
    Encode categorical features in chunks for medium-sized datasets.
    
    Parameters
    ----------
    X : pd.DataFrame
        Input DataFrame.
    method : str
        Encoding method to use.
    categorical_columns : List[str]
        List of column names to encode.
    drop_first : bool
        Whether to drop one category for each categorical feature when using one-hot encoding.
    chunk_size : int
        Number of rows to process at a time.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with categorical features encoded.
    """
    # For label and ordinal encoding, we need to fit the encoders on the entire dataset first
    if method in ['label', 'ordinal']:
        # Fit encoders on the entire dataset
        if method == 'label':
            encoders = {}
            for col in categorical_columns:
                encoders[col] = LabelEncoder()
                encoders[col].fit(X[col].astype(str))
        else:  # ordinal
            encoder = OrdinalEncoder()
            encoder.fit(X[categorical_columns])
        
        # Process in chunks
        result_chunks = []
        for i in range(0, len(X), chunk_size):
            chunk = X.iloc[i:i+chunk_size].copy()
            
            if method == 'label':
                for col in categorical_columns:
                    chunk[col] = encoders[col].transform(chunk[col].astype(str))
            else:  # ordinal
                chunk[categorical_columns] = encoder.transform(chunk[categorical_columns])
            
            result_chunks.append(chunk)
        
        return pd.concat(result_chunks, axis=0, ignore_index=True)
    
    # For one-hot and binary encoding, we can process each chunk independently
    elif method == 'onehot':
        # Get all possible categories for each column
        categories = {}
        for col in categorical_columns:
            categories[col] = X[col].unique()
        
        # Process in chunks
        result_chunks = []
        for i in range(0, len(X), chunk_size):
            chunk = X.iloc[i:i+chunk_size].copy()
            
            # One-hot encode
            encoded = pd.get_dummies(chunk[categorical_columns], 
                                    prefix=categorical_columns,
                                    drop_first=drop_first)
            
            # Drop original columns and add encoded columns
            chunk = chunk.drop(columns=categorical_columns)
            chunk = pd.concat([chunk, encoded], axis=1)
            
            result_chunks.append(chunk)
        
        return pd.concat(result_chunks, axis=0, ignore_index=True)
    
    elif method == 'binary':
        # For binary encoding, we need to fit the encoders on the entire dataset first
        encoders = {}
        for col in categorical_columns:
            encoders[col] = BinaryEncoder(cols=[col])
            encoders[col].fit(X)
        
        # Process in chunks
        result_chunks = []
        for i in range(0, len(X), chunk_size):
            chunk = X.iloc[i:i+chunk_size].copy()
            
            # Apply binary encoding
            for col in categorical_columns:
                encoded = encoders[col].transform(chunk)
                
                # Drop original column and add encoded columns
                chunk = chunk.drop(columns=[col])
                for new_col in encoded.columns:
                    if new_col != col:  # Skip the original column name
                        chunk[new_col] = encoded[new_col]
            
            result_chunks.append(chunk)
        
        return pd.concat(result_chunks, axis=0, ignore_index=True)
    
    return X  # Fallback (should not reach here)


def _encode_categorical_dask(
    X: pd.DataFrame, 
    method: str, 
    categorical_columns: List[str],
    drop_first: bool
) -> pd.DataFrame:
    """
    Encode categorical features using Dask for large datasets.
    
    Parameters
    ----------
    X : pd.DataFrame
        Input DataFrame.
    method : str
        Encoding method to use.
    categorical_columns : List[str]
        List of column names to encode.
    drop_first : bool
        Whether to drop one category for each categorical feature when using one-hot encoding.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with categorical features encoded.
    """
    # Convert to Dask DataFrame
    ddf = dd.from_pandas(X, npartitions=max(1, len(X) // 100000))
    
    if method == 'onehot':
        # For one-hot encoding, we need to get all possible categories first
        # This is done on the pandas DataFrame
        categories = {}
        for col in categorical_columns:
            categories[col] = X[col].unique()
        
        # Define a function to apply one-hot encoding to each partition
        def onehot_encode_partition(df):
            encoded = pd.get_dummies(df[categorical_columns], 
                                    prefix=categorical_columns,
                                    drop_first=drop_first)
            df = df.drop(columns=categorical_columns)
            return pd.concat([df, encoded], axis=1)
        
        # Apply the function to each partition
        result_ddf = ddf.map_partitions(onehot_encode_partition)
    
    elif method == 'label':
        # For label encoding, we need to fit the encoders on the entire dataset first
        encoders = {}
        for col in categorical_columns:
            encoders[col] = LabelEncoder()
            encoders[col].fit(X[col].astype(str))
        
        # Define a function to apply label encoding to each partition
        def label_encode_partition(df):
            for col in categorical_columns:
                df[col] = encoders[col].transform(df[col].astype(str))
            return df
        
        # Apply the function to each partition
        result_ddf = ddf.map_partitions(label_encode_partition)
    
    elif method == 'ordinal':
        # For ordinal encoding, we need to fit the encoder on the entire dataset first
        encoder = OrdinalEncoder()
        encoder.fit(X[categorical_columns])
        
        # Define a function to apply ordinal encoding to each partition
        def ordinal_encode_partition(df):
            df[categorical_columns] = encoder.transform(df[categorical_columns])
            return df
        
        # Apply the function to each partition
        result_ddf = ddf.map_partitions(ordinal_encode_partition)
    
    elif method == 'binary':
        # For binary encoding, we need to fit the encoders on the entire dataset first
        encoders = {}
        for col in categorical_columns:
            encoders[col] = BinaryEncoder(cols=[col])
            encoders[col].fit(X)
        
        # Define a function to apply binary encoding to each partition
        def binary_encode_partition(df):
            for col in categorical_columns:
                encoded = encoders[col].transform(df)
                
                # Drop original column and add encoded columns
                df = df.drop(columns=[col])
                for new_col in encoded.columns:
                    if new_col != col:  # Skip the original column name
                        df[new_col] = encoded[new_col]
            return df
        
        # Apply the function to each partition
        result_ddf = ddf.map_partitions(binary_encode_partition)
    
    else:
        # Fallback (should not reach here)
        result_ddf = ddf
    
    # Convert back to pandas DataFrame
    with ProgressBar():
        result = result_ddf.compute()
    
    return result 