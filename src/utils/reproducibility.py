"""
Reproducibility utilities.

Ensures all experiments are reproducible by setting seeds
for all random number generators.
"""

import os
import random
from typing import Optional

import numpy as np


def set_all_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Integer seed value.

    Sets seeds for:
        - Python's random module
        - NumPy
        - PyTorch (if available)
        - TensorFlow (if available)
        - Environment variables for hash seed

    Example:
        >>> set_all_seeds(42)
        >>> np.random.rand()  # Always same value with seed 42
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Hash seed for reproducible hashing
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch (if available)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Deterministic algorithms (may slow down training)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS doesn't have seed setting yet, but manual_seed covers it
            pass
    except ImportError:
        pass

    # TensorFlow (if available)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        # Ensure operation-level determinism
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
    except ImportError:
        pass


def get_device() -> str:
    """
    Detect and return the best available compute device.

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'.

    Example:
        >>> device = get_device()
        >>> print(f"Using device: {device}")
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass

    return "cpu"


def get_environment_info() -> dict:
    """
    Collect environment information for reproducibility logging.

    Returns:
        Dictionary with library versions and hardware info.
    """
    import platform
    import sys

    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }

    # NumPy version
    info["numpy_version"] = np.__version__

    # PyTorch version and CUDA info
    try:
        import torch

        info["pytorch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["pytorch_version"] = "not installed"

    # Scikit-learn version
    try:
        import sklearn

        info["sklearn_version"] = sklearn.__version__
    except ImportError:
        info["sklearn_version"] = "not installed"

    # Polars version
    try:
        import polars

        info["polars_version"] = polars.__version__
    except ImportError:
        info["polars_version"] = "not installed"

    return info
