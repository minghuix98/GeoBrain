"""
Kernel functions for particle-based inference.

Provides RBF and IMQ kernels with adaptive bandwidth support.
These kernels are primarily used in SVGD for measuring particle similarity.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
import math
from typing import Optional

from .core import Kernel
from .utils import pairwise_distances, median_heuristic

# Minimum bandwidth to prevent numerical issues
MIN_BANDWIDTH = 1e-8


class RBFKernel(Kernel):
    """
    Radial Basis Function (Gaussian) kernel.
    
    k(x, y) = exp(-||x-y||^2 / (2h^2))
    
    The RBF kernel is the most commonly used kernel for SVGD
    due to its smoothness and universal approximation properties.
    
    Args:
        bandwidth: Fixed bandwidth h. If None, use adaptive bandwidth.
        adaptive: Whether to use median heuristic for bandwidth.
        bandwidth_factor: Scaling factor for adaptive bandwidth.
    
    Example:
        >>> kernel = RBFKernel(adaptive=True)
        >>> K = kernel(x)         # Kernel matrix [n, n]
        >>> grad_K = kernel.grad_x(x)  # Gradient [n, n, d]
    """
    
    def __init__(
        self,
        bandwidth: Optional[float] = None,
        adaptive: bool = True,
        bandwidth_factor: float = 1.0
    ):
        super().__init__()
        self._fixed_bandwidth = bandwidth
        self._bandwidth = bandwidth
        self.adaptive = adaptive
        self.bandwidth_factor = bandwidth_factor
    
    def __call__(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute RBF kernel matrix.
        
        Args:
            x: First input of shape [n, d].
            y: Second input of shape [m, d]. If None, use y = x.
            
        Returns:
            Kernel matrix of shape [n, m].
        """
        if y is None:
            y = x
        
        if self.adaptive and self._fixed_bandwidth is None:
            self.update_bandwidth(x, y)
        
        dist_sq = pairwise_distances(x, y, squared=True)
        h = self._bandwidth if self._bandwidth is not None else 1.0
        # Ensure bandwidth is not too small to prevent numerical issues
        h = max(h, MIN_BANDWIDTH)
        gamma = 1.0 / (2.0 * h ** 2)

        return torch.exp(-gamma * dist_sq)
    
    def grad_x(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute gradient: grad_x k(x, y).
        
        For RBF: grad_x k(x,y) = -k(x,y) * (x-y) / h^2
        
        Args:
            x: First input of shape [n, d].
            y: Second input of shape [m, d]. If None, use y = x.
            
        Returns:
            Gradient tensor of shape [n, m, d].
        """
        if y is None:
            y = x
        
        K = self(x, y)  # [n, m]
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # [n, m, d]
        h = self._bandwidth if self._bandwidth is not None else 1.0
        h = max(h, MIN_BANDWIDTH)
        h_sq = h ** 2

        return -K.unsqueeze(-1) * diff / h_sq  # [n, m, d]
    
    def update_bandwidth(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> None:
        """
        Update bandwidth using median heuristic.
        
        The bandwidth is scaled by 1/sqrt(log(n+1)) as suggested
        in the original SVGD paper for better performance.
        
        Args:
            x: First dataset of shape [n, d].
            y: Second dataset of shape [m, d]. If None, use x.
        """
        if y is None:
            y = x
        
        with torch.no_grad():
            distances = pairwise_distances(x, y)
            h = median_heuristic(distances)

            # Scale by log(n) factor for SVGD
            n = x.shape[0]
            bandwidth = self.bandwidth_factor * h.item() / math.sqrt(math.log(n + 1))
            # Ensure minimum bandwidth to prevent numerical issues
            self._bandwidth = max(bandwidth, MIN_BANDWIDTH)
    
    def __repr__(self) -> str:
        bw = f"{self._bandwidth:.4f}" if self._bandwidth else "None"
        return f"RBFKernel(bandwidth={bw}, adaptive={self.adaptive})"


class IMQKernel(Kernel):
    """
    Inverse MultiQuadric (IMQ) kernel.
    
    k(x, y) = (c^2 + ||x-y||^2)^beta
    
    The IMQ kernel has heavier tails compared to RBF, which can
    help maintain particle diversity in SVGD and avoid mode collapse.
    
    Args:
        c: Scale parameter.
        beta: Power parameter (typically -0.5 for inverse).
        adaptive: Whether to adapt c based on data.
        bandwidth_factor: Scaling factor for adaptive c.
    
    Example:
        >>> kernel = IMQKernel(c=1.0, beta=-0.5)
        >>> K = kernel(x)
    """
    
    def __init__(
        self,
        c: float = 1.0,
        beta: float = -0.5,
        adaptive: bool = True,
        bandwidth_factor: float = 1.0
    ):
        super().__init__()
        self._c = c
        self._fixed_c = None if adaptive else c
        self.beta = beta
        self.adaptive = adaptive
        self.bandwidth_factor = bandwidth_factor
        self._bandwidth = c
    
    def __call__(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute IMQ kernel matrix.
        
        Args:
            x: First input of shape [n, d].
            y: Second input of shape [m, d]. If None, use y = x.
            
        Returns:
            Kernel matrix of shape [n, m].
        """
        if y is None:
            y = x
        
        if self.adaptive and self._fixed_c is None:
            self.update_bandwidth(x, y)
        
        dist_sq = pairwise_distances(x, y, squared=True)
        
        return (self._c ** 2 + dist_sq) ** self.beta
    
    def grad_x(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute gradient: grad_x k(x, y).
        
        For IMQ: grad_x k(x,y) = 2*beta*(c^2 + ||x-y||^2)^(beta-1) * (x-y)
        
        Args:
            x: First input of shape [n, d].
            y: Second input of shape [m, d]. If None, use y = x.
            
        Returns:
            Gradient tensor of shape [n, m, d].
        """
        if y is None:
            y = x
        
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # [n, m, d]
        dist_sq = (diff ** 2).sum(dim=2)  # [n, m]
        
        base = self._c ** 2 + dist_sq
        coeff = 2.0 * self.beta * (base ** (self.beta - 1.0))  # [n, m]
        
        return coeff.unsqueeze(-1) * diff  # [n, m, d]
    
    def update_bandwidth(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> None:
        """
        Update c parameter using median heuristic.
        
        Args:
            x: First dataset of shape [n, d].
            y: Second dataset of shape [m, d]. If None, use x.
        """
        if y is None:
            y = x
        
        with torch.no_grad():
            distances = pairwise_distances(x, y)
            h = median_heuristic(distances)
            c = self.bandwidth_factor * h.item()
            # Ensure minimum c parameter to prevent numerical issues
            self._c = max(c, MIN_BANDWIDTH)
            self._bandwidth = self._c
    
    @property
    def c(self) -> float:
        """Get current c parameter."""
        return self._c
    
    def __repr__(self) -> str:
        return f"IMQKernel(c={self._c:.4f}, beta={self.beta}, adaptive={self.adaptive})"


def create_kernel(name: str, **kwargs) -> Kernel:
    """
    Factory function to create kernel by name.
    
    Args:
        name: Kernel type ('rbf' or 'imq').
        **kwargs: Kernel-specific parameters.
        
    Returns:
        Kernel instance.
        
    Raises:
        ValueError: If kernel name is unknown.
    
    Example:
        >>> kernel = create_kernel('rbf', adaptive=True)
        >>> kernel = create_kernel('imq', c=1.0, beta=-0.5)
    """
    kernels = {
        'rbf': RBFKernel,
        'imq': IMQKernel,
    }
    
    name = name.lower()
    if name not in kernels:
        raise ValueError(f"Unknown kernel: {name}. Available: {list(kernels.keys())}")
    
    return kernels[name](**kwargs)
