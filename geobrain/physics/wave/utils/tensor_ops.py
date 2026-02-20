"""
Tensor and array conversion utilities.

Provides functions for converting between numpy arrays,
PyTorch tensors, and Python lists.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
import numpy as np
from torch import Tensor
from typing import Union, List


def numpy2tensor(
    a: Union[np.ndarray, Tensor],
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Convert numpy array to PyTorch tensor.
    
    Args:
        a: Input array or tensor.
        dtype: Target dtype. Default: torch.float32.
    
    Returns:
        Tensor with specified dtype.
    
    Example:
        >>> arr = np.random.randn(100, 100)
        >>> tensor = numpy2tensor(arr, torch.float64)
    """
    if torch.is_tensor(a):
        return a.to(dtype)
    return torch.tensor(a, requires_grad=False, dtype=dtype)


def tensor2numpy(a: Union[np.ndarray, Tensor]) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy array.
    
    Automatically detaches gradients and moves to CPU if necessary.

    Args:
        a: Input tensor or array.

    Returns:
        Numpy array.

    Example:
        >>> tensor = torch.randn(100, 100, device='cuda')
        >>> arr = tensor2numpy(tensor)
    """
    if not torch.is_tensor(a):
        return np.asarray(a)
    return a.detach().cpu().numpy()


def gpu2cpu(a: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
    """
    Move tensor from GPU to CPU.

    Detaches gradients if present and moves tensor to CPU.
    Returns numpy arrays unchanged.

    Args:
        a: Input tensor or array.

    Returns:
        Tensor on CPU or original array.

    Example:
        >>> gpu_tensor = torch.randn(100, device='cuda')
        >>> cpu_tensor = gpu2cpu(gpu_tensor)
    """
    if torch.is_tensor(a):
        if a.requires_grad:
            a = a.detach()
        if a.device.type != 'cpu':
            a = a.cpu()
    return a


def list2numpy(a: Union[List, np.ndarray]) -> np.ndarray:
    """
    Convert Python list to numpy array.

    Args:
        a: Input list or array.

    Returns:
        Numpy array.

    Example:
        >>> data = [[1, 2, 3], [4, 5, 6]]
        >>> arr = list2numpy(data)
    """
    if isinstance(a, list):
        return np.array(a)
    return a


def numpy2list(a: Union[List, np.ndarray]) -> List:
    """
    Convert numpy array to Python list.

    Args:
        a: Input array or list.

    Returns:
        Python list.

    Example:
        >>> arr = np.array([1, 2, 3])
        >>> lst = numpy2list(arr)
    """
    if not isinstance(a, list):
        return a.tolist()
    return a