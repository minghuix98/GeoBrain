"""
Utility functions for Bayesian inference.

Contains distance metrics, kernel utilities, and statistical measures.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
from typing import Optional


def pairwise_distances(
    x: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    squared: bool = False
) -> torch.Tensor:
    """
    Compute pairwise Euclidean distances efficiently.
    
    Uses the formula: ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
    
    Args:
        x: First tensor of shape [n, d].
        y: Second tensor of shape [m, d]. If None, use y = x.
        squared: If True, return squared distances.
        
    Returns:
        Distance matrix of shape [n, m].
    
    Example:
        >>> x = torch.randn(100, 5)
        >>> dists = pairwise_distances(x)  # [100, 100]
    """
    if y is None:
        y = x
    
    x_sq = (x ** 2).sum(dim=1, keepdim=True)  # [n, 1]
    y_sq = (y ** 2).sum(dim=1, keepdim=True)  # [m, 1]
    xy = torch.matmul(x, y.T)  # [n, m]
    
    dist_sq = torch.clamp(x_sq + y_sq.T - 2.0 * xy, min=0.0)
    
    if squared:
        return dist_sq
    return torch.sqrt(dist_sq + 1e-8)


def median_heuristic(distances: torch.Tensor, factor: float = 1.0) -> torch.Tensor:
    """
    Compute median heuristic for kernel bandwidth selection.
    
    The median of pairwise distances is often a good choice
    for kernel bandwidth in practice.
    
    Args:
        distances: Pairwise distance matrix of shape [n, m].
        factor: Scaling factor for the median.
        
    Returns:
        Bandwidth value (scalar tensor).
    
    Example:
        >>> dists = pairwise_distances(x)
        >>> bandwidth = median_heuristic(dists)
    """
    n, m = distances.shape
    
    if n == m:
        # Use upper triangular part for self-distances
        mask = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
        dists = distances[mask]
    else:
        dists = distances.flatten()
    
    # Remove zeros
    dists = dists[dists > 0]
    
    if len(dists) == 0:
        return torch.tensor(1.0, device=distances.device, dtype=distances.dtype)
    
    return factor * torch.median(dists)


def mmd(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: str = 'rbf',
    bandwidth: Optional[float] = None
) -> torch.Tensor:
    """
    Compute Maximum Mean Discrepancy between two sample sets.
    
    MMD^2 = E[k(X,X')] + E[k(Y,Y')] - 2E[k(X,Y)]
    
    This is useful for comparing how well samples approximate
    a target distribution.
    
    Args:
        x: Samples from first distribution of shape [n, d].
        y: Samples from second distribution of shape [m, d].
        kernel: Kernel type ('rbf' or 'imq').
        bandwidth: Kernel bandwidth. If None, use median heuristic.
        
    Returns:
        MMD value (scalar).
    
    Example:
        >>> x = torch.randn(100, 2)
        >>> y = torch.randn(100, 2) + 1
        >>> dist = mmd(x, y)
    """
    n, m = x.shape[0], y.shape[0]
    
    # Compute bandwidth if not provided
    if bandwidth is None:
        combined = torch.cat([x, y], dim=0)
        dists = pairwise_distances(combined)
        bandwidth = median_heuristic(dists).item()
    
    # Define kernel functions
    def rbf_kernel(a, b, h):
        dist_sq = pairwise_distances(a, b, squared=True)
        return torch.exp(-dist_sq / (2 * h ** 2))
    
    def imq_kernel(a, b, c):
        dist_sq = pairwise_distances(a, b, squared=True)
        return (c ** 2 + dist_sq) ** (-0.5)
    
    # Select kernel
    if kernel == 'rbf':
        k_fn = lambda a, b: rbf_kernel(a, b, bandwidth)
    elif kernel == 'imq':
        k_fn = lambda a, b: imq_kernel(a, b, bandwidth)
    else:
        raise ValueError(f"Unknown kernel: {kernel}. Use 'rbf' or 'imq'.")
    
    # Compute kernel matrices
    Kxx = k_fn(x, x)
    Kyy = k_fn(y, y)
    Kxy = k_fn(x, y)
    
    # Unbiased MMD^2 estimator
    mmd_sq = (
        (Kxx.sum() - Kxx.trace()) / (n * (n - 1)) +
        (Kyy.sum() - Kyy.trace()) / (m * (m - 1)) -
        2 * Kxy.mean()
    )
    
    return torch.sqrt(torch.clamp(mmd_sq, min=0.0))


def energy_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute energy distance between two sample sets.
    
    E = 2E[||X-Y||] - E[||X-X'||] - E[||Y-Y'||]
    
    Energy distance is a metric that equals zero if and only if
    the two distributions are identical.
    
    Args:
        x: Samples from first distribution of shape [n, d].
        y: Samples from second distribution of shape [m, d].
        
    Returns:
        Energy distance (scalar).
    
    Example:
        >>> x = torch.randn(100, 2)
        >>> y = torch.randn(100, 2)
        >>> dist = energy_distance(x, y)
    """
    d_xy = pairwise_distances(x, y).mean()
    d_xx = pairwise_distances(x, x).mean()
    d_yy = pairwise_distances(y, y).mean()
    
    return 2 * d_xy - d_xx - d_yy
