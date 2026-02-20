"""
Signal processing utilities.

Provides convolution operations for seismic data processing.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
from torch import Tensor
from typing import Tuple


# =============================================================================
# Convolution Operations
# =============================================================================

def create_conv_matrix(
    wavelet: Tensor,
    n_samples: int,
    mode: str = 'full',
) -> Tensor:
    """
    Create convolution matrix from wavelet.

    The convolution matrix W allows computing convolution as
    matrix multiplication: seismic = W @ reflectivity

    Args:
        wavelet: Source wavelet tensor with shape (nw,).
        n_samples: Number of output samples (reflectivity length).
        mode: Convolution mode.
            - ``'full'``: output shape ``(n_samples + nw - 1, n_samples)``.
            - ``'same'``: output shape ``(n_samples, n_samples)`` by trimming
              ``nw // 2`` rows from each end.

    Returns:
        Convolution matrix.

    Example:
        >>> from geobrain.physics.wave import RickerWavelet
        >>> wavelet, _ = RickerWavelet()(f0=25.0, dt=0.001)
        >>> W = create_conv_matrix(wavelet, 200, mode='same')
        >>> seismic = W @ reflectivity   # same length as reflectivity
    """
    nw = len(wavelet)
    n_out = n_samples + nw - 1

    # Create Toeplitz-like convolution matrix
    W = torch.zeros(
        (n_out, n_samples),
        device=wavelet.device,
        dtype=wavelet.dtype
    )

    for i in range(n_samples):
        W[i:i + nw, i] = wavelet

    if mode == 'same':
        start = nw // 2
        W = W[start:start + n_samples]
    elif mode != 'full':
        raise ValueError(f"mode must be 'full' or 'same', got '{mode}'")

    return W


def convolve_trace(
    trace: Tensor,
    wavelet: Tensor,
    mode: str = 'same',
) -> Tensor:
    """
    Convolve a trace with a wavelet.

    Args:
        trace: Input trace with shape (nt,) or (batch, nt).
        wavelet: Wavelet with shape (nw,).
        mode: Output mode. Default: 'same'.
            Options:
                - 'full': Full convolution output
                - 'same': Same length as input
                - 'valid': Only valid (non-padded) output

    Returns:
        Convolved trace with same batch dimensions as input.

    Raises:
        ValueError: If mode is unknown.

    Example:
        >>> trace = torch.randn(1000)
        >>> wavelet, _ = RickerWavelet()(f0=25.0, dt=0.001)
        >>> result = convolve_trace(trace, wavelet, mode='same')
    """
    # Handle dimensions
    squeeze = False
    if trace.ndim == 1:
        trace = trace.unsqueeze(0)
        squeeze = True

    # Prepare for conv1d: [batch, 1, nt]
    trace_3d = trace.unsqueeze(1)
    wavelet_3d = wavelet.flip(0).unsqueeze(0).unsqueeze(0)

    nw = len(wavelet)

    if mode == 'full':
        padding = nw - 1
    elif mode == 'same':
        padding = nw // 2
    elif mode == 'valid':
        padding = 0
    else:
        raise ValueError(f"Unknown mode: '{mode}'")

    result = torch.nn.functional.conv1d(trace_3d, wavelet_3d, padding=padding)
    result = result.squeeze(1)

    if squeeze:
        result = result.squeeze(0)

    return result


def convolve_gather(
    gather: Tensor,
    wavelet: Tensor,
    mode: str = 'same',
) -> Tensor:
    """
    Convolve a shot gather with a wavelet.

    Applies wavelet convolution to all traces in a gather.

    Args:
        gather: Input gather with shape (n_traces, nt) or
            (batch, n_traces, nt).
        wavelet: Wavelet with shape (nw,).
        mode: Output mode ('full', 'same', 'valid'). Default: 'same'.

    Returns:
        Convolved gather with same shape as input (except time dimension
        may change depending on mode).

    Example:
        >>> gather = torch.randn(100, 1000)  # 100 traces, 1000 samples
        >>> result = convolve_gather(gather, wavelet)
    """
    original_shape = gather.shape

    # Flatten to [n_all_traces, nt]
    if gather.ndim == 2:
        flat = gather
    else:
        flat = gather.reshape(-1, gather.shape[-1])

    # Convolve
    result = convolve_trace(flat, wavelet, mode)

    # Reshape back
    if gather.ndim > 2:
        new_shape = list(original_shape[:-1]) + [result.shape[-1]]
        result = result.reshape(new_shape)

    return result


def apply_wavelet(reflectivity: Tensor, wavelet: Tensor) -> Tensor:
    """
    Apply wavelet to reflectivity series using convolution matrix.

    This is the forward modeling operation: seismic = W @ reflectivity

    Args:
        reflectivity: Reflectivity series with shape (..., nt).
        wavelet: Source wavelet with shape (nw,).

    Returns:
        Synthetic seismic with same shape as reflectivity.

    Example:
        >>> reflectivity = torch.randn(10, 500)  # 10 traces
        >>> seismic = apply_wavelet(reflectivity, wavelet)
    """
    nw = len(wavelet)
    half_nw = nw // 2

    # Create convolution matrix
    if reflectivity.ndim == 1:
        W = create_conv_matrix(wavelet, len(reflectivity))
        if half_nw > 0:
            W = W[half_nw:-half_nw].T
        else:
            W = W.T
        return W @ reflectivity

    # Multi-dimensional case
    nt = reflectivity.shape[-1]
    W = create_conv_matrix(wavelet, nt)
    if half_nw > 0:
        W = W[half_nw:-half_nw].T
    else:
        W = W.T

    # Use einsum for batch operation
    return torch.einsum('...i,ji->...j', reflectivity, W)