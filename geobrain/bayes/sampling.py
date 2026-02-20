"""
Sampling result container.

Provides an immutable container for sampling results with
summary and serialization utilities.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class SamplingResult:
    """
    Immutable result from completed sampling.
    
    Contains final samples, convergence information, and history
    of the sampling process.
    
    Attributes:
        samples: Final sample positions of shape [n_samples, dim].
        log_prob_history: History of mean log probabilities.
        algorithm: Name of the sampling algorithm.
        n_steps: Number of steps completed.
        converged: Whether sampling converged (early stopping).
        total_time: Total sampling time in seconds.
        trajectory: Optional full trajectory of shape [n_snapshots, n_samples, dim].
        config: Configuration dictionary for reproducibility.
    
    Example:
        >>> result = svgd.run(n_samples=100, n_steps=500)
        >>> print(result.summary())
        >>> samples = result.samples
    """
    
    # Core results
    samples: torch.Tensor
    log_prob_history: List[float]
    algorithm: str
    n_steps: int
    
    # Status
    converged: bool = False
    total_time: float = 0.0
    
    # Optional trajectory
    trajectory: Optional[torch.Tensor] = None
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def n_samples(self) -> int:
        """Number of samples."""
        if self.samples is None or self.samples.numel() == 0:
            return 0
        return self.samples.shape[0]
    
    @property
    def dim(self) -> int:
        """Dimensionality of samples."""
        if self.samples is None or self.samples.numel() == 0:
            return 0
        return self.samples.shape[1]
    
    @property
    def final_log_prob(self) -> float:
        """Final mean log probability."""
        if self.log_prob_history:
            return self.log_prob_history[-1]
        return float('-inf')
    
    def summary(self) -> str:
        """
        Generate a text summary of results.
        
        Returns:
            Formatted summary string.
        """
        lines = [
            "=" * 50,
            f"Sampling Result: {self.algorithm}",
            "=" * 50,
            f"Samples: {self.n_samples} x {self.dim}",
            f"Steps: {self.n_steps}",
            f"Converged: {self.converged}",
            f"Final log prob: {self.final_log_prob:.6e}",
            f"Time: {self.total_time:.2f}s",
            "=" * 50,
        ]
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary with all result data.
        """
        result = {
            'samples': self.samples.cpu().numpy() if self.samples is not None else None,
            'log_prob_history': self.log_prob_history,
            'algorithm': self.algorithm,
            'n_steps': self.n_steps,
            'n_samples': self.n_samples,
            'dim': self.dim,
            'converged': self.converged,
            'total_time': self.total_time,
            'final_log_prob': self.final_log_prob,
            'config': self.config,
        }
        if self.trajectory is not None:
            result['trajectory'] = self.trajectory.cpu().numpy()
        return result
    
    def to_numpy(self) -> np.ndarray:
        """
        Get samples as numpy array.
        
        Returns:
            Numpy array of shape [n_samples, dim].
        """
        return self.samples.cpu().numpy()
    
    def __repr__(self) -> str:
        return (
            f"SamplingResult(algorithm='{self.algorithm}', "
            f"n_samples={self.n_samples}, dim={self.dim}, "
            f"n_steps={self.n_steps}, converged={self.converged})"
        )
