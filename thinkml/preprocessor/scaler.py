"""
Feature scaling module for ThinkML.
This module provides functionality to scale or normalize numerical features.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Union
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer


def scale_features(
    X: pd.DataFrame, 
    method: str = 'standard',
    columns: Optional[List[str]] = None,
    with_mean: bool = True,
    with_std: bool = True,
    feature_range: tuple = (0, 1),
    chunk_size: int = 100000
) -> pd.DataFrame:
    """
    Scale or normalize numerical features using various methods.

    Args:
        X (pd.DataFrame): Input DataFrame containing numerical features.
        method (str, optional): Scaling method to use.
            Options: 'standard', 'minmax', 'robust', 'normalize'.
            Default is 'standard'.
        columns (Optional[List[str]], optional): List of column names to scale.
            If None, all numerical columns will be scaled. Default is None.
        with_mean (bool, optional): Whether to center the data before scaling.
            Only used with 'standard' method. Default is True.
        with_std (bool, optional): Whether to scale the data to unit variance.
            Only used with 'standard' method. Default is True.
        feature_range (tuple, optional): Desired range of transformed data.
            Only used with 'minmax' method. Default is (0, 1).
        chunk_size (int, optional): Number of rows to process at a time for memory efficiency.
            Default is 100000.

    Returns:
        pd.DataFrame: DataFrame with numerical features scaled according to the specified method.

    Raises:
        ValueError: If method is not one of the supported options.
        ValueError: If columns is provided but contains non-existent column names.
    """
    # Input validation
    if not isinstance(X, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    if X.empty:
        return X
    
    valid_methods = ['standard', 'minmax', 'robust', 'normalize']
    if method not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}")
    
    # Determine which columns to scale
    if columns is None:
        # Automatically detect numeric columns
        numeric_columns = [col for col in X.columns 
                          if pd.api.types.is_numeric_dtype(X[col])]
    else:
        # Validate that all specified columns exist and are numeric
        missing_columns = [col for col in columns if col not in X.columns]
        if missing_columns:
            raise ValueError(f"Columns not found in DataFrame: {missing_columns}")
        
        non_numeric_columns = [col for col in columns 
                              if not pd.api.types.is_numeric_dtype(X[col])]
        if non_numeric_columns:
            raise ValueError(f"Non-numeric columns found: {non_numeric_columns}")
        
        numeric_columns = columns
    
    # If no numeric columns to scale, return the original DataFrame
    if not numeric_columns:
        return X
    
    # Check if dataset is large enough to use Dask
    if len(X) > 1_000_000:
        return _scale_features_dask(X, method, numeric_columns)
    
    # Process in chunks for memory efficiency
    if len(X) > chunk_size:
        return _scale_features_chunks(X, method, numeric_columns, chunk_size)
    
    # For small datasets, process directly
    return _scale_features_direct(X, method, numeric_columns)


def _scale_features_direct(
    X: pd.DataFrame, 
    method: str, 
    numeric_columns: List[str]
) -> pd.DataFrame:
    """
    Scale numerical features directly for small datasets.
    
    Parameters
    ----------
    X : pd.DataFrame
        Input DataFrame.
    method : str
        Scaling method to use.
    numeric_columns : List[str]
        List of column names to scale.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with numerical features scaled.
    """
    result = X.copy()
    
    if method == 'standard':
        scaler = StandardScaler()
        result[numeric_columns] = scaler.fit_transform(result[numeric_columns])
        
    elif method == 'minmax':
        scaler = MinMaxScaler()
        result[numeric_columns] = scaler.fit_transform(result[numeric_columns])
        
    elif method == 'robust':
        scaler = RobustScaler()
        result[numeric_columns] = scaler.fit_transform(result[numeric_columns])
        
    elif method == 'normalize':
        scaler = Normalizer()
        result[numeric_columns] = scaler.fit_transform(result[numeric_columns])
    
    return result


def _scale_features_chunks(
    X: pd.DataFrame, 
    method: str, 
    numeric_columns: List[str],
    chunk_size: int
) -> pd.DataFrame:
    """
    Scale numerical features in chunks for medium-sized datasets.
    
    Parameters
    ----------
    X : pd.DataFrame
        Input DataFrame.
    method : str
        Scaling method to use.
    numeric_columns : List[str]
        List of column names to scale.
    chunk_size : int
        Number of rows to process at a time.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with numerical features scaled.
    """
    # For all scaling methods except normalization, we need to fit on the entire dataset first
    if method != 'normalize':
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:  # robust
            scaler = RobustScaler()
        
        # Fit scaler on the entire dataset
        scaler.fit(X[numeric_columns])
        
        # Process in chunks
        result_chunks = []
        for i in range(0, len(X), chunk_size):
            chunk = X.iloc[i:i+chunk_size].copy()
            chunk[numeric_columns] = scaler.transform(chunk[numeric_columns])
            result_chunks.append(chunk)
        
        return pd.concat(result_chunks, axis=0, ignore_index=True)
    
    else:  # normalize
        # For normalization, we can process each chunk independently
        scaler = Normalizer()
        result_chunks = []
        for i in range(0, len(X), chunk_size):
            chunk = X.iloc[i:i+chunk_size].copy()
            chunk[numeric_columns] = scaler.fit_transform(chunk[numeric_columns])
            result_chunks.append(chunk)
        
        return pd.concat(result_chunks, axis=0, ignore_index=True)


def _scale_features_dask(
    X: pd.DataFrame, 
    method: str, 
    numeric_columns: List[str]
) -> pd.DataFrame:
    """
    Scale numerical features using Dask for large datasets.
    
    Parameters
    ----------
    X : pd.DataFrame
        Input DataFrame.
    method : str
        Scaling method to use.
    numeric_columns : List[str]
        List of column names to scale.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with numerical features scaled.
    """
    # Convert to Dask DataFrame
    ddf = dd.from_pandas(X, npartitions=max(1, len(X) // 100000))
    
    # For all scaling methods except normalization, we need to fit on the entire dataset first
    if method != 'normalize':
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:  # robust
            scaler = RobustScaler()
        
        # Fit scaler on the entire dataset
        scaler.fit(X[numeric_columns])
        
        # Define a function to apply scaling to each partition
        def scale_partition(df):
            df = df.copy()
            df[numeric_columns] = scaler.transform(df[numeric_columns])
            return df
        
        # Apply the function to each partition
        result_ddf = ddf.map_partitions(scale_partition)
    
    else:  # normalize
        # For normalization, we can process each partition independently
        scaler = Normalizer()
        
        # Define a function to apply normalization to each partition
        def normalize_partition(df):
            df = df.copy()
            df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
            return df
        
        # Apply the function to each partition
        result_ddf = ddf.map_partitions(normalize_partition)
    
    # Convert back to pandas DataFrame
    with ProgressBar():
        result = result_ddf.compute()
    
    return result 