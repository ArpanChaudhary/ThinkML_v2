"""
Data splitting functionality for ThinkML.

This module provides functions for standardizing and splitting datasets
into training and testing sets with various scaling options.
"""

from typing import Union, Tuple, Optional, Literal
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import dask.dataframe as dd
from dask_ml.preprocessing import StandardScaler as DaskStandardScaler
from dask_ml.preprocessing import MinMaxScaler as DaskMinMaxScaler
from dask_ml.preprocessing import RobustScaler as DaskRobustScaler

def standardize_and_split(
    X: Union[pd.DataFrame, dd.DataFrame],
    y: Optional[Union[pd.Series, np.ndarray, dd.Series]] = None,
    scaler: Optional[Literal['standard', 'minmax', 'robust']] = 'standard',
    test_size: float = 0.2,
    random_state: Optional[int] = None
) -> Tuple[Union[pd.DataFrame, dd.DataFrame], Union[pd.DataFrame, dd.DataFrame], 
          Optional[Union[pd.Series, np.ndarray, dd.Series]], 
          Optional[Union[pd.Series, np.ndarray, dd.Series]]]:
    """
    Split and standardize a dataset into training and testing sets.

    Parameters
    ----------
    X : Union[pd.DataFrame, dd.DataFrame]
        Features dataset
    y : Optional[Union[pd.Series, np.ndarray, dd.Series]], default=None
        Target variable
    scaler : Optional[Literal['standard', 'minmax', 'robust']], default='standard'
        Type of scaler to use. Options are:
        - 'standard': StandardScaler (zero mean, unit variance)
        - 'minmax': MinMaxScaler (scale to range [0,1])
        - 'robust': RobustScaler (robust to outliers)
        - None: No scaling
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split
    random_state : Optional[int], default=None
        Controls the shuffling applied to the data before splitting

    Returns
    -------
    Tuple containing:
        - X_train: Training features
        - X_test: Test features
        - y_train: Training target (if y is provided)
        - y_test: Test target (if y is provided)

    Raises
    ------
    ValueError
        If scaler is not one of the supported options
    """
    # Input validation
    if scaler not in [None, 'standard', 'minmax', 'robust']:
        raise ValueError("scaler must be one of: None, 'standard', 'minmax', 'robust'")

    # Determine if we're working with Dask DataFrames
    is_dask = isinstance(X, dd.DataFrame)

    # Split the data
    if is_dask:
        X_train, X_test = X.random_split([1 - test_size, test_size], random_state=random_state)
        if y is not None:
            y_train, y_test = y.random_split([1 - test_size, test_size], random_state=random_state)
        else:
            y_train, y_test = None, None
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    # Apply scaling if requested
    if scaler is not None:
        if is_dask:
            if scaler == 'standard':
                scaler_obj = DaskStandardScaler()
            elif scaler == 'minmax':
                scaler_obj = DaskMinMaxScaler()
            else:  # robust
                scaler_obj = DaskRobustScaler()
            
            # Fit on training data only
            scaler_obj.fit(X_train)
            
            # Transform both training and test data
            X_train = scaler_obj.transform(X_train)
            X_test = scaler_obj.transform(X_test)
        else:
            if scaler == 'standard':
                scaler_obj = StandardScaler()
            elif scaler == 'minmax':
                scaler_obj = MinMaxScaler()
            else:  # robust
                scaler_obj = RobustScaler()
            
            # Fit on training data only
            scaler_obj.fit(X_train)
            
            # Transform both training and test data
            X_train = pd.DataFrame(
                scaler_obj.transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test = pd.DataFrame(
                scaler_obj.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )

    return X_train, X_test, y_train, y_test 