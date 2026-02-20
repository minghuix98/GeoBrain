"""
Type aliases and constants for rock physics.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
from typing import Union, List

Tensor = torch.Tensor
TensorLike = Union[float, int, List[float], Tensor]

PI = 3.141592653589793
EPS = 1e-10


def as_tensor(x: TensorLike, dtype: torch.dtype = torch.float32) -> Tensor:
    """
    Convert input to PyTorch tensor.

    Ensures consistent tensor format across all rock physics calculations.
    Scalar inputs are converted to 1D tensors for broadcasting compatibility.

    Args:
        x: Input value (scalar, list, or tensor)
        dtype: Target data type. Default: torch.float32

    Returns:
        PyTorch tensor of at least 1 dimension
    """
    if isinstance(x, Tensor):
        return x.to(dtype)
    return torch.atleast_1d(torch.as_tensor(x, dtype=dtype))


def ensure_same_device(*tensors: Tensor):
    """
    Move all tensors to the same device, preferring CUDA over CPU.

    When mixing scalar parameters (CPU) with data tensors (CUDA),
    this ensures all operands are on the same device.

    Args:
        *tensors: Variable number of tensors

    Returns:
        Tuple of tensors, all on the same device
    """
    device = None
    for t in tensors:
        if isinstance(t, Tensor) and t.device.type != 'cpu':
            device = t.device
            break
    if device is None:
        return tensors
    return tuple(t.to(device) if isinstance(t, Tensor) else t for t in tensors)
