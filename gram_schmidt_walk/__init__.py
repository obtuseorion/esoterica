"""
Gram-Schmidt Walk Algorithm Implementation

A Python implementation of the Gram-Schmidt Walk algorithm for finding balanced 
±1 colorings of vectors with low discrepancy.

Based on "The Gram-Schmidt Walk: A Cure for the Banaszczyk Blues" by
Bansal, Dadush, Garg, and Lovett.
"""

from .core.gram_schmidt_walk import GramSchmidtWalk
from .data.data_processor import DataProcessor, prepare_dataset
from .data.balanced_data_splitter import (
    BalancedDataSplitter, 
    quick_balanced_split, 
    quick_balanced_split_indices
)

__version__ = "0.1.0"
__author__ = "Urbas"

__all__ = [
    "GramSchmidtWalk",
    "DataProcessor", 
    "prepare_dataset",
    "BalancedDataSplitter",
    "quick_balanced_split",
    "quick_balanced_split_indices"
]