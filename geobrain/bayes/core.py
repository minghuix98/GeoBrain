"""
Core abstract interfaces for Bayesian inference.

This module defines the fundamental abstract classes that all
distributions, samplers, and kernels must implement.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
import torch.autograd as autograd
from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple

from .sampling import SamplingResult


class Distribution(ABC):
    """
    Abstract base class for probability distributions.

    All target distributions must implement log_prob(). The score function
    (gradient of log probability) has a default autodiff implementation
    but can be overridden for efficiency.

    Args:
        dim: Dimensionality of the distribution.

    Example:
        >>> class MyDistribution(Distribution):
        ...     def __init__(self, dim):
        ...         super().__init__(dim=dim)
        ...
        ...     def log_prob(self, x):
        ...         return -0.5 * torch.sum(x**2, dim=-1, keepdim=True)
    """

    def __init__(self, dim: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)  # Support cooperative multiple inheritance
        self._dim = dim
    
    @abstractmethod
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability density.
        
        Args:
            x: Input tensor of shape [batch_size, dim].
            
        Returns:
            Log probability of shape [batch_size, 1].
        """
        pass
    
    def score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute score function: grad_x log p(x).
        
        Default implementation uses automatic differentiation.
        Override for analytical gradients (more efficient).
        
        Args:
            x: Input tensor of shape [batch_size, dim].
            
        Returns:
            Score (gradient) of shape [batch_size, dim].
        """
        x = x.detach().requires_grad_(True)
        log_p = self.log_prob(x)
        score = autograd.grad(
            outputs=log_p.sum(),
            inputs=x,
            create_graph=True
        )[0]
        return score
    
    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Generate samples from the distribution.
        
        Args:
            n_samples: Number of samples to generate.
            
        Returns:
            Samples of shape [n_samples, dim].
            
        Raises:
            NotImplementedError: If direct sampling is not supported.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support direct sampling. "
            "Use a sampler (SVGD, HMC, etc.) to generate samples."
        )
    
    @property
    def dim(self) -> Optional[int]:
        """Get dimensionality of the distribution."""
        return self._dim
    
    @dim.setter
    def dim(self, value: int):
        """Set dimensionality of the distribution."""
        self._dim = value


class Kernel(ABC):
    """
    Abstract base class for kernel functions.
    
    Kernels measure similarity between particles and are used in
    particle-based methods like SVGD. They must be positive definite.
    
    Example:
        >>> kernel = RBFKernel(bandwidth=1.0, adaptive=True)
        >>> K = kernel(x, y)              # Kernel matrix [n, m]
        >>> grad_K = kernel.grad_x(x, y)  # Gradient [n, m, d]
    """
    
    def __init__(self):
        self._bandwidth = None
    
    @abstractmethod
    def __call__(
        self, 
        x: torch.Tensor, 
        y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute kernel matrix K[i,j] = k(x_i, y_j).
        
        Args:
            x: First input of shape [n, d].
            y: Second input of shape [m, d]. If None, use y = x.
            
        Returns:
            Kernel matrix of shape [n, m].
        """
        pass
    
    @abstractmethod
    def grad_x(
        self, 
        x: torch.Tensor, 
        y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute gradient: grad_x k(x, y).
        
        Args:
            x: First input of shape [n, d].
            y: Second input of shape [m, d]. If None, use y = x.
            
        Returns:
            Gradient tensor of shape [n, m, d].
        """
        pass
    
    def update_bandwidth(
        self, 
        x: torch.Tensor, 
        y: Optional[torch.Tensor] = None
    ) -> None:
        """
        Update bandwidth based on data (for adaptive kernels).
        
        Args:
            x: First dataset of shape [n, d].
            y: Second dataset of shape [m, d]. If None, use x.
        """
        pass
    
    @property
    def bandwidth(self) -> Optional[float]:
        """Get current bandwidth value."""
        return self._bandwidth
    
    @bandwidth.setter
    def bandwidth(self, value: float):
        """Set bandwidth value."""
        self._bandwidth = value


class BaseSampler(ABC):
    """
    Abstract base class for all sampling algorithms.
    
    Defines the unified interface that all samplers must implement.
    Supports two usage patterns:
    
    1. Simple: Use run() for complete sampling workflow.
    2. Advanced: Use step() for fine-grained control.
    
    Args:
        target: Target distribution to sample from.
        device: Computation device. Defaults to CUDA if available.
    
    Example:
        >>> # Simple usage
        >>> sampler = SVGD(target=distribution)
        >>> result = sampler.run(n_samples=100, n_steps=500)
        
        >>> # Advanced usage
        >>> samples = torch.randn(100, dim)
        >>> for step in range(500):
        ...     samples, info = sampler.step(samples)
        ...     # custom logging, checkpointing, etc.
    """
    
    def __init__(
        self,
        target: Optional[Distribution] = None,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        # Ensure target distribution is on the same device
        if target is not None and hasattr(target, 'to'):
            target = target.to(self.device)
        self.target = target
        self._samples = None
    
    @abstractmethod
    def step(
        self, 
        samples: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Perform a single sampling step.
        
        Args:
            samples: Current sample positions of shape [n_samples, dim].
            
        Returns:
            Tuple of (updated_samples, info_dict).
            info_dict contains metrics like 'log_prob', 'bandwidth', etc.
        """
        pass
    
    @abstractmethod
    def run(
        self,
        n_samples: int,
        n_steps: int,
        initial_samples: Optional[torch.Tensor] = None,
        verbose: bool = True,
        print_every: int = 50,
    ) -> SamplingResult:
        """
        Run the complete sampling algorithm.
        
        Args:
            n_samples: Number of samples/particles to generate.
            n_steps: Number of sampling iterations.
            initial_samples: Starting positions of shape [n_samples, dim].
            verbose: Whether to print progress.
            print_every: Print frequency when verbose=True.
            
        Returns:
            SamplingResult containing samples and diagnostics.
        """
        pass
    
    @property
    def samples(self) -> Optional[torch.Tensor]:
        """Get current/final samples."""
        return self._samples
    
    def get_samples(self) -> torch.Tensor:
        """
        Get samples, raising error if none available.
        
        Returns:
            Current samples tensor.
            
        Raises:
            ValueError: If no samples available.
        """
        if self._samples is None:
            raise ValueError("No samples available. Run sampling first.")
        return self._samples
    
    def reset(self) -> None:
        """Reset sampler state."""
        self._samples = None

    def __call__(
        self,
        n_samples: int,
        n_steps: int,
        **kwargs
    ) -> SamplingResult:
        """
        Shortcut for run() - enables sampler(n_samples, n_steps) syntax.

        Args:
            n_samples: Number of samples/particles to generate.
            n_steps: Number of sampling iterations.
            **kwargs: Additional arguments passed to run().

        Returns:
            SamplingResult containing samples and diagnostics.

        Example:
            >>> sampler = SVGD(target=distribution)
            >>> result = sampler(n_samples=100, n_steps=500)
        """
        return self.run(n_samples, n_steps, **kwargs)
