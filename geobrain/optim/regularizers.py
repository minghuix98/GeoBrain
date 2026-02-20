"""
Regularization functions for inversion.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
import torch.nn as nn
from typing import Union


def l2_regularizer(
    x: Union[torch.Tensor, nn.Module],
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    L2 norm (Tikhonov) regularization.
    
    Penalizes large parameter values, encouraging smoother solutions.
    
    Args:
        x: Input tensor or nn.Module.
        reduction: 'mean' or 'sum'.
        
    Returns:
        L2 regularization term.
    
    Example:
        >>> reg = l2_regularizer(model_params)
        >>> loss = data_loss + 0.01 * reg
    """
    if isinstance(x, nn.Module):
        total = sum(torch.sum(p ** 2) for p in x.parameters() if p.requires_grad)
        n = sum(p.numel() for p in x.parameters() if p.requires_grad)
    else:
        total = torch.sum(x ** 2)
        n = x.numel()
    
    if reduction == 'mean' and n > 0:
        return total / n
    return total


def l1_regularizer(
    x: Union[torch.Tensor, nn.Module],
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    L1 norm (sparsity) regularization.
    
    Encourages sparse solutions with many zero values.
    
    Args:
        x: Input tensor or nn.Module.
        reduction: 'mean' or 'sum'.
        
    Returns:
        L1 regularization term.
    
    Example:
        >>> reg = l1_regularizer(model_params)
        >>> loss = data_loss + 0.01 * reg
    """
    if isinstance(x, nn.Module):
        total = sum(torch.sum(torch.abs(p)) for p in x.parameters() if p.requires_grad)
        n = sum(p.numel() for p in x.parameters() if p.requires_grad)
    else:
        total = torch.sum(torch.abs(x))
        n = x.numel()
    
    if reduction == 'mean' and n > 0:
        return total / n
    return total


def get_regularizer(name: str):
    """
    Get regularizer function by name.
    
    Args:
        name: Regularizer name ('l1' or 'l2').
        
    Returns:
        Regularizer function.
        
    Raises:
        ValueError: If regularizer name is unknown.
    
    Example:
        >>> reg_fn = get_regularizer('l2')
        >>> reg_term = reg_fn(model_params)
    """
    regularizers = {
        'l1': l1_regularizer,
        'l2': l2_regularizer,
    }
    
    name = name.lower()
    if name not in regularizers:
        raise ValueError(f"Unknown regularizer: {name}. Available: {list(regularizers.keys())}")
    
    return regularizers[name]
