"""
Test script for ThinkML's preprocessor.scaler and validation.advanced_validation modules
on edge cases and big data scenarios.
"""

import numpy as np
import pandas as pd
import time
import sys
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Import ThinkML modules
from thinkml.preprocessor.scaler import scale_features
from thinkml.validation.cross_validation import NestedCrossValidator
from thinkml.validation.advanced_validation import (
    TimeSeriesValidator,
    StratifiedGroupValidator,
    BootstrapValidator
)

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Prepare datasets
def prepare_datasets():
    """Prepare edge case datasets and big data dataset."""
    edge_datasets = {
        "empty": (pd.DataFrame(), pd.Series(dtype=int)),
        "single_row": (pd.DataFrame({"feature1": [5]}), pd.Series([1])),
        "all_missing": (pd.DataFrame({"feature1": [None]*10}), pd.Series([1]*10)),
        "extreme_values": (pd.DataFrame({"feature1": [1e10, -1e10]}), pd.Series([1, 0])),
        "highly_correlated": (
            pd.DataFrame({"X1": np.random.rand(100), "X2": np.random.rand(100)}).assign(X2=lambda df: df.X1 + 0.001),
            pd.Series(np.random.randint(0, 2, size=100))
        )
    }
    
    # Create big data dataset
    logger.info("Creating big data dataset (1M rows)...")
    big_data_X = pd.DataFrame(np.random.rand(1_000_000, 10))
    big_data_y = pd.Series(np.random.randint(0, 2, size=1_000_000))
    
    return edge_datasets, (big_data_X, big_data_y)

# Test scaler on edge cases
def test_scaler_edge_cases(edge_datasets):
    """Test scaler on edge cases."""
    logger.info("Testing scaler on edge cases...")
    
    results = {}
    
    for name, (X, y) in edge_datasets.items():
        logger.info(f"Testing scaler on {name} dataset...")
        try:
            start_time = time.time()
            
            # Skip empty dataset
            if X.empty:
                logger.info(f"Scaler | {name} | Skipped (empty or all missing)")
                results[name] = "Skipped (empty or all missing)"
                continue
                
            # Check if all values are missing
            if X.isna().all().all():
                logger.info(f"Scaler | {name} | Skipped (empty or all missing)")
                results[name] = "Skipped (empty or all missing)"
                continue
                
            # Scale features
            X_scaled = scale_features(X, method='standard')
            
            # Check if scaling was successful
            if not X_scaled.empty and not X_scaled.isna().all().all():
                logger.info(f"Scaler | {name} | Scaling Success")
                results[name] = "Scaling Success"
            else:
                logger.warning(f"Scaler | {name} | Scaling Failed")
                results[name] = "Scaling Failed"
                
            end_time = time.time()
            logger.info(f"Time taken: {end_time - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Scaler | {name} | Error: {str(e)}")
            results[name] = f"Error: {str(e)}"
    
    return results

# Test scaler + validation on big data
def test_scaler_validation_big_data(big_data):
    """Test scaler + validation on big data."""
    logger.info("Testing scaler + validation on big data...")
    
    X, y = big_data
    results = {}
    
    try:
        # Scale features
        start_time = time.time()
        X_scaled = scale_features(X, method='standard')
        scaling_time = time.time() - start_time
        logger.info(f"Scaling completed in {scaling_time:.2f} seconds")
        
        # Create a simple model for validation
        model = LogisticRegression(max_iter=100)
        
        # Try different validation methods
        validators = {
            "NestedCrossValidator": NestedCrossValidator(
                estimator=model,
                param_grid={'C': [0.1, 1.0]},
                inner_cv=2,
                outer_cv=2
            ),
            "TimeSeriesValidator": TimeSeriesValidator(
                n_splits=2,
                test_size=0.2
            ),
            "StratifiedGroupValidator": StratifiedGroupValidator(
                n_splits=2
            ),
            "BootstrapValidator": BootstrapValidator(
                n_iterations=2,
                sample_size=0.8
            )
        }
        
        for name, validator in validators.items():
            try:
                logger.info(f"Testing {name} on big data...")
                start_time = time.time()
                
                if name == "NestedCrossValidator":
                    result = validator.fit(X_scaled, y)
                else:
                    result = validator.fit_predict(X_scaled, y, model)
                
                end_time = time.time()
                logger.info(f"{name} completed in {end_time - start_time:.2f} seconds")
                results[name] = "Validation Success"
                
            except Exception as e:
                logger.error(f"{name} | Error: {str(e)}")
                results[name] = f"Error: {str(e)}"
        
    except Exception as e:
        logger.error(f"Scaler + Validation | Error: {str(e)}")
        results["Overall"] = f"Error: {str(e)}"
    
    return results

# Main function
def main():
    """Main function to run all tests."""
    logger.info("Starting ThinkML edge case and big data tests...")
    
    # Prepare datasets
    edge_datasets, big_data = prepare_datasets()
    
    # Test scaler on edge cases
    scaler_results = test_scaler_edge_cases(edge_datasets)
    
    # Test scaler + validation on big data
    validation_results = test_scaler_validation_big_data(big_data)
    
    # Print results
    logger.info("\n=== Test Results ===")
    logger.info("\nScaler Edge Cases:")
    for name, result in scaler_results.items():
        logger.info(f"Scaler | {name} | {result}")
    
    logger.info("\nScaler + Validation Big Data:")
    for name, result in validation_results.items():
        logger.info(f"Scaler + Validation | {name} | {result}")
    
    logger.info("\nTests completed.")

if __name__ == "__main__":
    main() 