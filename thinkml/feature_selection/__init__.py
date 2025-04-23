"""
Feature Selection module for ThinkML.

This module provides functions for selecting the most relevant features from datasets
using various methods including variance threshold, correlation analysis, chi-squared test,
and mutual information.
"""

from thinkml.feature_selection.selector import select_features

__all__ = ['select_features'] 