"""
Utility functions for wave simulation.

Provides:
    - Tensor/array conversion functions
    - Assessment metrics for model comparison
    - Convolution operations

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

# Tensor operations
from .tensor_ops import (
    numpy2tensor,
    tensor2numpy,
    gpu2cpu,
    list2numpy,
    numpy2list,
)

# Metrics
from .metrics import (
    mse,
    rmse,
    mape,
    snr,
)

# Convolution operations
from .signal import (
    create_conv_matrix,
    convolve_trace,
    convolve_gather,
    apply_wavelet,
)

__all__ = [
    # Tensor operations
    'numpy2tensor',
    'tensor2numpy',
    'gpu2cpu',
    'list2numpy',
    'numpy2list',
    # Metrics
    'mse',
    'rmse',
    'mape',
    'snr',
    # Convolution
    'create_conv_matrix',
    'convolve_trace',
    'convolve_gather',
    'apply_wavelet',
]