"""
Stein Variational Gradient Descent (SVGD) sampler.

SVGD is a particle-based variational inference algorithm that
iteratively transports particles to approximate a target distribution.

Reference:
    Liu & Wang (2016). Stein Variational Gradient Descent:
    A General Purpose Bayesian Inference Algorithm. NeurIPS.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import logging
import time
from typing import Optional, Dict, Any, Union, Tuple, List

import torch
import torch.optim as optim

from ..core import Distribution, BaseSampler, Kernel
from ..sampling import SamplingResult
from ..kernels import create_kernel

logger = logging.getLogger(__name__)


class SVGD(BaseSampler):
    """
    Stein Variational Gradient Descent.
    
    SVGD iteratively updates particles to approximate the target
    distribution using the following update rule:
    
        x <- x + eps * phi(x)
    
    where:
        phi(x) = (1/n) * sum_j [k(x_j, x) * grad_log_p(x_j) + grad_x k(x_j, x)]
    
    The first term (attractive) drives particles toward high probability regions.
    The second term (repulsive) maintains particle diversity.
    
    Args:
        target: Target distribution to sample from.
        kernel: Kernel instance or name ('rbf', 'imq'). Default: 'rbf'.
        lr: Learning rate. Default: 0.01.
        optimizer: Optimizer name ('adam', 'sgd', 'rmsprop'). Default: 'adam'.
        device: Computation device. Default: CUDA if available.
    
    Example:
        Simple usage:
        >>> target = Gaussian(torch.zeros(2), torch.eye(2))
        >>> svgd = SVGD(target=target, lr=0.01)
        >>> result = svgd.run(n_samples=100, n_steps=500)
        >>> samples = result.samples
        
        Advanced usage (manual loop):
        >>> svgd = SVGD(target=target)
        >>> samples = torch.randn(100, 2, requires_grad=True)
        >>> optimizer = torch.optim.Adam([samples], lr=0.01)
        >>> for step in range(500):
        ...     samples, info = svgd.step(samples, optimizer)
        ...     print(f"Step {step}: log_prob={info['log_prob']:.4f}")
    """
    
    def __init__(
        self,
        target: Optional[Distribution] = None,
        kernel: Union[str, Kernel] = 'rbf',
        lr: float = 0.01,
        optimizer: str = 'adam',
        device: Optional[torch.device] = None,
    ):
        super().__init__(target, device)
        
        # Setup kernel
        if isinstance(kernel, str):
            self.kernel = create_kernel(kernel)
        else:
            self.kernel = kernel
        
        # Store optimizer config
        self.lr = lr
        self.optimizer_name = optimizer.lower()
        
        # Internal state
        self._optimizer = None
    
    def step(
        self,
        samples: torch.Tensor,
        optimizer: Optional[torch.optim.Optimizer] = None,
        target: Optional[Distribution] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Perform a single SVGD update step.
        
        This method can be used for fine-grained control over the
        sampling process, allowing custom logging, checkpointing, etc.
        
        Args:
            samples: Current particles of shape [n_samples, dim].
                    Must have requires_grad=True.
            optimizer: PyTorch optimizer. If None, uses internal optimizer.
            target: Target distribution. If None, uses self.target.
            
        Returns:
            Tuple of (updated_samples, info_dict).
            info_dict contains:
                - log_prob: Mean log probability
                - bandwidth: Current kernel bandwidth
                - attractive_norm: Norm of attractive term
                - repulsive_norm: Norm of repulsive term
        
        Example:
            >>> samples = torch.randn(100, dim, requires_grad=True)
            >>> optimizer = torch.optim.Adam([samples], lr=0.01)
            >>> for step in range(n_steps):
            ...     samples, info = svgd.step(samples, optimizer)
        """
        target = target or self.target
        if target is None:
            raise ValueError("Target distribution must be provided")
        if hasattr(target, 'to'):
            target = target.to(self.device)

        if optimizer is None:
            if self._optimizer is None:
                self._optimizer = self._create_optimizer(samples)
            optimizer = self._optimizer
        
        n = samples.shape[0]
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Detach samples for kernel computation (no gradient through kernel)
        samples_detached = samples.detach()
        
        # Compute score function: grad_x log p(x)
        # We need samples with grad for score computation
        samples_for_score = samples_detached.clone().requires_grad_(True)
        log_prob = target.log_prob(samples_for_score)
        log_prob_mean = log_prob.mean().item()
        
        # Compute score function (analytical or autodiff via base class)
        score = target.score(samples_for_score).detach()  # [n, d]
        
        # Update kernel bandwidth
        self.kernel.update_bandwidth(samples_detached)
        
        # Compute kernel matrix and gradient
        K = self.kernel(samples_detached)  # [n, n]
        grad_K = self.kernel.grad_x(samples_detached)  # [n, n, d]
        
        # SVGD update direction:
        # phi(x_i) = (1/n) * sum_j [k(x_j, x_i) * grad_log_p(x_j) + grad_{x_i} k(x_j, x_i)]
        
        # Attractive term: K @ score / n
        attractive = torch.matmul(K, score) / n  # [n, d]
        
        # Repulsive term: mean of kernel gradients
        repulsive = grad_K.mean(dim=0)  # [n, d]
        
        phi = attractive + repulsive
        
        # Set gradient directly for optimizer
        # We want to move in the direction of phi (gradient ascent on log prob)
        # Optimizer does: x = x - lr * grad, so we set grad = -phi
        if samples.grad is None:
            samples.grad = -phi
        else:
            samples.grad.copy_(-phi)
        
        # Optimizer step
        optimizer.step()
        
        # Collect diagnostics
        info = {
            'log_prob': log_prob_mean,
            'bandwidth': self.kernel.bandwidth or 0.0,
            'attractive_norm': attractive.norm(dim=1).mean().item(),
            'repulsive_norm': repulsive.norm(dim=1).mean().item(),
        }
        
        return samples, info
    
    def run(
        self,
        n_samples: int,
        n_steps: int,
        initial_samples: Optional[torch.Tensor] = None,
        *,
        dim: Optional[int] = None,
        target: Optional[Distribution] = None,
        verbose: bool = True,
        print_every: int = 50,
        patience: Optional[int] = None,
        min_delta: float = 1e-6,
        return_trajectory: bool = False,
        trajectory_every: int = 10,
    ) -> SamplingResult:
        """
        Run complete SVGD sampling.
        
        Args:
            n_samples: Number of particles.
            n_steps: Number of update iterations.
            initial_samples: Starting positions of shape [n_samples, dim].
                           If None, samples from standard Gaussian.
            dim: Dimension (required if initial_samples is None).
            target: Target distribution (overrides self.target).
            verbose: Whether to print progress. Default: True.
            print_every: Print frequency. Default: 50.
            patience: Early stopping patience. If None, no early stopping.
            min_delta: Minimum improvement for early stopping.
            return_trajectory: Whether to save particle trajectory.
            trajectory_every: Save trajectory every N steps.
            
        Returns:
            SamplingResult containing samples and diagnostics.
        
        Example:
            >>> result = svgd.run(n_samples=100, n_steps=500, verbose=True)
            >>> print(result.summary())
            >>> samples = result.samples
        """
        # Get target and ensure device consistency
        target = target or self.target
        if target is None:
            raise ValueError("Target distribution must be provided")
        if hasattr(target, 'to'):
            target = target.to(self.device)

        # Reset optimizer state to avoid carryover from previous runs
        # (Adam's moments, RMSprop's averages, etc. would otherwise accumulate)
        self._optimizer = None

        # Initialize particles
        if initial_samples is not None:
            samples = initial_samples.to(self.device)
            dim = samples.shape[1]
        else:
            dim = dim or getattr(target, 'dim', None)
            if dim is None:
                raise ValueError("Specify dim or provide initial_samples")
            samples = torch.randn(n_samples, dim, device=self.device)
        
        if not samples.requires_grad:
            samples = samples.requires_grad_(True)
        
        # Setup optimizer
        optimizer = self._create_optimizer(samples)
        
        # Initialize tracking
        log_prob_history: List[float] = []
        trajectory: List[torch.Tensor] = [] if return_trajectory else None
        
        # Early stopping state
        best_log_prob = float('-inf')
        patience_counter = 0
        converged = False
        
        # Timing
        start_time = time.time()
        
        # Print header
        if verbose:
            logger.info(f"SVGD: {n_samples} particles, {dim} dims, {n_steps} steps")
            logger.info("-" * 50)
        
        # Main loop
        for step in range(n_steps):
            # SVGD step
            samples, info = self.step(samples, optimizer, target)
            
            # Record history
            log_prob_history.append(info['log_prob'])
            
            # Record trajectory
            if return_trajectory and step % trajectory_every == 0:
                trajectory.append(samples.detach().cpu().clone())
            
            # Print progress
            if verbose and (step % print_every == 0 or step == n_steps - 1):
                elapsed = time.time() - start_time
                logger.info(
                    f"[{step+1:5d}/{n_steps}] "
                    f"log_prob: {info['log_prob']:.4e} | "
                    f"bandwidth: {info['bandwidth']:.4f} | "
                    f"time: {elapsed:.1f}s"
                )
            
            # Early stopping check
            if patience is not None:
                if info['log_prob'] > best_log_prob + min_delta:
                    best_log_prob = info['log_prob']
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    converged = True
                    if verbose:
                        logger.info(f"Early stopping at step {step+1}")
                    break
            
            # Check for divergence
            if not torch.isfinite(torch.tensor(info['log_prob'])):
                if verbose:
                    logger.warning(f"Diverged at step {step+1}")
                break
        
        # Store final samples
        self._samples = samples.detach()
        
        # Build trajectory tensor
        trajectory_tensor = None
        if return_trajectory and trajectory:
            trajectory_tensor = torch.stack(trajectory, dim=0)
        
        # Create result
        total_time = time.time() - start_time
        
        result = SamplingResult(
            samples=self._samples,
            log_prob_history=log_prob_history,
            algorithm='SVGD',
            n_steps=step + 1,
            converged=converged,
            total_time=total_time,
            trajectory=trajectory_tensor,
            config={
                'n_samples': n_samples,
                'dim': dim,
                'lr': self.lr,
                'optimizer': self.optimizer_name,
                'kernel': str(self.kernel),
            }
        )
        
        if verbose:
            logger.info("-" * 50)
            logger.info(f"Completed in {total_time:.2f}s")
        
        return result
    
    def _create_optimizer(self, samples: torch.Tensor) -> torch.optim.Optimizer:
        """Create optimizer for particles."""
        optimizer_map = {
            'adam': optim.Adam,
            'sgd': optim.SGD,
            'rmsprop': optim.RMSprop,
            'adagrad': optim.Adagrad,
            'adamw': optim.AdamW,
        }
        
        opt_cls = optimizer_map.get(self.optimizer_name, optim.Adam)
        
        kwargs = {'lr': self.lr}
        if self.optimizer_name == 'sgd':
            kwargs['momentum'] = 0.9
        
        return opt_cls([samples], **kwargs)
    
    def __repr__(self) -> str:
        return (
            f"SVGD(kernel={self.kernel}, lr={self.lr}, "
            f"optimizer='{self.optimizer_name}')"
        )
