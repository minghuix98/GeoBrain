"""
Loss functions for inversion.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
import torch.nn.functional as F
from typing import Optional


def mse_loss(
    predictions: torch.Tensor, 
    targets: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Mean Squared Error loss.
    
    Args:
        predictions: Predicted values.
        targets: Target values.
        weights: Optional sample weights.
        reduction: 'none', 'mean', or 'sum'.
        
    Returns:
        Loss value.
    
    Example:
        >>> loss = mse_loss(pred, obs)
        >>> loss = mse_loss(pred, obs, weights=uncertainty**(-2))
    """
    squared_diff = (predictions - targets) ** 2
    
    if weights is not None:
        squared_diff = squared_diff * weights
    
    if reduction == 'none':
        return squared_diff
    elif reduction == 'sum':
        return torch.sum(squared_diff)
    else:
        return torch.mean(squared_diff)


def nrmse_loss(
    predictions: torch.Tensor, 
    targets: torch.Tensor,
    normalization: str = 'std',
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Normalized Root Mean Squared Error loss.
    
    A dimensionless error metric useful for comparing across different scales.
    
    Args:
        predictions: Predicted values.
        targets: Target values.
        normalization: 'std', 'range', or 'mean'.
        eps: Small constant for numerical stability.
        
    Returns:
        NRMSE value.
    
    Example:
        >>> loss = nrmse_loss(pred, obs, normalization='std')
    """
    mse = torch.mean((predictions - targets) ** 2)
    rmse = torch.sqrt(mse + eps)
    
    if normalization == 'std':
        norm = torch.std(targets) + eps
    elif normalization == 'range':
        norm = torch.max(targets) - torch.min(targets) + eps
    elif normalization == 'mean':
        norm = torch.mean(torch.abs(targets)) + eps
    else:
        raise ValueError(f"Invalid normalization: {normalization}")
    
    return rmse / norm


def l1_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    L1 (Mean Absolute Error) loss.
    
    More robust to outliers than MSE.
    
    Args:
        predictions: Predicted values.
        targets: Target values.
        weights: Optional sample weights.
        reduction: 'none', 'mean', or 'sum'.
        
    Returns:
        Loss value.
    """
    abs_diff = torch.abs(predictions - targets)
    
    if weights is not None:
        abs_diff = abs_diff * weights
    
    if reduction == 'none':
        return abs_diff
    elif reduction == 'sum':
        return torch.sum(abs_diff)
    else:
        return torch.mean(abs_diff)


def huber_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    delta: float = 1.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Huber loss (smooth L1).
    
    Quadratic for small errors, linear for large errors.
    Combines advantages of L1 and L2 loss.
    
    Args:
        predictions: Predicted values.
        targets: Target values.
        delta: Threshold for switching between L1 and L2.
        reduction: 'none', 'mean', or 'sum'.
        
    Returns:
        Loss value.
    """
    return F.smooth_l1_loss(
        predictions, targets, 
        reduction=reduction, 
        beta=delta
    )


def get_loss_fn(name: str):
    """
    Get loss function by name.
    
    Args:
        name: Loss function name ('mse', 'nrmse', 'l1', 'huber').
        
    Returns:
        Loss function.
        
    Raises:
        ValueError: If loss name is unknown.
    """
    loss_fns = {
        'mse': mse_loss,
        'nrmse': nrmse_loss,
        'l1': l1_loss,
        'mae': l1_loss,
        'huber': huber_loss,
    }
    
    name = name.lower()
    if name not in loss_fns:
        raise ValueError(f"Unknown loss: {name}. Available: {list(loss_fns.keys())}")
    
    return loss_fns[name]
