"""
TTT: Pair-Set Transformer Package

A PyTorch-based implementation of the Pair-Set Transformer, a permutation-invariant 
model that processes two sets simultaneously via intra- and cross-attention.

Main components:
- PairSetTransformer: The core transformer model
- train_model: Training function with CPU optimization
- Utility functions for data generation and processing
"""

from .model import PairSetTransformer
from .train import train_model, evaluate_model
from .utils import (
    synthetic_set_pair_generator,
    normalize_data,
    denormalize_data,
    compute_rmse,
    compute_mae,
    EarlyStopping,
    mean_pooling,
    masked_mean_pooling,
    pad_sequences,
    batch_generator
)
import os
import warnings

def _load_pretrained_model():
    """
    Load a pretrained TTT model if available, otherwise return None.
    """
    # Look for pretrained model in package directory
    package_dir = os.path.dirname(__file__)
    pretrained_path = os.path.join(package_dir, 'pretrained_ttt_model.pth')
    
    if os.path.exists(pretrained_path):
        try:
            return PairSetTransformer.load_model(pretrained_path)
        except Exception as e:
            warnings.warn(f"Failed to load pretrained model: {e}")
            return None
    else:
        # No pretrained model found
        return None

# Load pretrained model if available
ttt = _load_pretrained_model()

__version__ = "0.1.0"
__author__ = "TTT Package"
__email__ = "ttt@example.com"

__all__ = [
    "PairSetTransformer",
    "train_model",
    "evaluate_model",
    "synthetic_set_pair_generator",
    "normalize_data",
    "denormalize_data",
    "compute_rmse", 
    "compute_mae",
    "EarlyStopping",
    "mean_pooling",
    "masked_mean_pooling",
    "pad_sequences",
    "batch_generator",
    "ttt"
]