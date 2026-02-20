"""
Probability distributions for Bayesian inference.

Provides Gaussian, Gaussian Mixture, and Posterior distributions
that can be used as targets for sampling algorithms.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
import torch.nn as nn
import math
from typing import Optional, List, Callable

from .core import Distribution


class Gaussian(Distribution, nn.Module):
    """
    Multivariate Gaussian distribution.
    
    p(x) = N(x | mu, Sigma)
         = (2*pi)^(-d/2) |Sigma|^(-1/2) exp(-0.5 (x-mu)^T Sigma^{-1} (x-mu))
    
    This distribution supports both covariance and precision matrix
    parameterizations. It provides analytical score function for efficiency.
    
    Args:
        mean: Mean vector of shape [dim] or [1, dim].
        cov: Covariance matrix of shape [dim, dim].
        precision: Precision matrix (alternative to cov).
    
    Note:
        Provide either cov or precision, not both.
    
    Example:
        >>> mean = torch.zeros(2)
        >>> cov = torch.eye(2)
        >>> dist = Gaussian(mean, cov)
        >>> log_prob = dist.log_prob(samples)
        >>> samples = dist.sample(100)
    """
    
    def __init__(
        self,
        mean: torch.Tensor,
        cov: Optional[torch.Tensor] = None,
        precision: Optional[torch.Tensor] = None
    ):
        # Handle 1D mean
        if mean.dim() == 1:
            mean = mean.unsqueeze(0)

        super().__init__(dim=mean.shape[-1])
        
        if cov is None and precision is None:
            raise ValueError("Must provide either cov or precision")
        if cov is not None and precision is not None:
            raise ValueError("Cannot provide both cov and precision")
        
        self.register_buffer('mean', mean)
        
        if cov is not None:
            self.register_buffer('cov', cov)
            self.register_buffer('precision', torch.linalg.inv(cov))
            self.register_buffer('chol', torch.linalg.cholesky(cov))
        else:
            self.register_buffer('precision', precision)
            self.register_buffer('cov', torch.linalg.inv(precision))
            self.register_buffer('chol', torch.linalg.cholesky(self.cov))
        
        # Precompute log normalization constant
        d = self._dim
        log_det = torch.logdet(self.cov)
        self.register_buffer('_log_norm', -0.5 * (d * math.log(2 * math.pi) + log_det))
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability.
        
        Args:
            x: Input of shape [batch_size, dim].
            
        Returns:
            Log probability of shape [batch_size, 1].
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        diff = x - self.mean
        # Mahalanobis distance: (x-mu)^T Sigma^{-1} (x-mu)
        mahal = torch.einsum('bi,ij,bj->b', diff, self.precision, diff)
        log_prob = self._log_norm - 0.5 * mahal
        
        return log_prob.unsqueeze(-1)
    
    def score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute score function (analytical).
        
        For Gaussian: grad log p(x) = -Sigma^{-1}(x-mu)
        
        Args:
            x: Input of shape [batch_size, dim].
            
        Returns:
            Score of shape [batch_size, dim].
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        diff = x - self.mean
        return -torch.matmul(diff, self.precision)
    
    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Generate samples using reparameterization trick.
        
        Args:
            n_samples: Number of samples.
            
        Returns:
            Samples of shape [n_samples, dim].
        """
        eps = torch.randn(
            n_samples, self._dim,
            device=self.mean.device,
            dtype=self.mean.dtype
        )
        return self.mean + torch.matmul(eps, self.chol.T)
    
    def entropy(self) -> torch.Tensor:
        """
        Compute differential entropy.
        
        Returns:
            Entropy value (scalar).
        """
        d = self._dim
        log_det = torch.logdet(self.cov)
        return 0.5 * (d * math.log(2 * math.pi * math.e) + log_det)
    
    def kl_divergence(self, other: 'Gaussian') -> torch.Tensor:
        """
        Compute KL divergence: KL(self || other).
        
        Args:
            other: Another Gaussian distribution.
            
        Returns:
            KL divergence value (scalar).
            
        Raises:
            ValueError: If other is not a Gaussian.
        """
        if not isinstance(other, Gaussian):
            raise ValueError("KL divergence only defined between Gaussians")
        
        d = self._dim
        trace_term = torch.trace(torch.matmul(other.precision, self.cov))
        diff = other.mean - self.mean
        mean_term = torch.einsum('bi,ij,bj->', diff, other.precision, diff)
        log_det_term = torch.logdet(other.cov) - torch.logdet(self.cov)
        
        return 0.5 * (trace_term + mean_term - d + log_det_term)
    
    def __repr__(self) -> str:
        return f"Gaussian(dim={self._dim})"


class GaussianMixture(Distribution, nn.Module):
    """
    Gaussian Mixture Model.
    
    p(x) = sum_k w_k N(x | mu_k, Sigma_k)
    
    This is useful for testing sampling algorithms on multimodal
    distributions where direct sampling is available for comparison.
    
    Args:
        means: List of mean vectors.
        covs: List of covariance matrices.
        weights: Component weights (will be normalized).
    
    Example:
        >>> means = [torch.zeros(2), torch.ones(2) * 3]
        >>> covs = [torch.eye(2), 0.5 * torch.eye(2)]
        >>> weights = torch.tensor([0.3, 0.7])
        >>> gmm = GaussianMixture(means, covs, weights)
    """
    
    def __init__(
        self,
        means: List[torch.Tensor],
        covs: List[torch.Tensor],
        weights: Optional[torch.Tensor] = None
    ):
        # Infer dimension
        dim = means[0].shape[-1] if means[0].dim() > 0 else 1
        super().__init__(dim=dim)
        
        self.n_components = len(means)
        if len(covs) != self.n_components:
            raise ValueError("Number of means and covariances must match")
        
        # Create component distributions
        self.components = nn.ModuleList([
            Gaussian(mean, cov) for mean, cov in zip(means, covs)
        ])
        
        # Set weights
        if weights is None:
            weights = torch.ones(self.n_components) / self.n_components
        else:
            weights = weights / weights.sum()
        
        self.register_buffer('weights', weights)
        self.register_buffer('log_weights', torch.log(weights))
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability using log-sum-exp for numerical stability.
        
        Args:
            x: Input of shape [batch_size, dim].
            
        Returns:
            Log probability of shape [batch_size, 1].
        """
        log_probs = torch.stack([
            comp.log_prob(x).squeeze(-1) + self.log_weights[k]
            for k, comp in enumerate(self.components)
        ], dim=1)  # [batch, n_components]
        
        return torch.logsumexp(log_probs, dim=1, keepdim=True)
    
    def score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute score function.
        
        grad log p(x) = sum_k p(k|x) * grad log p_k(x)
        
        Args:
            x: Input of shape [batch_size, dim].
            
        Returns:
            Score of shape [batch_size, dim].
        """
        # Compute posteriors p(k|x)
        log_probs = torch.stack([
            comp.log_prob(x).squeeze(-1) + self.log_weights[k]
            for k, comp in enumerate(self.components)
        ], dim=1)  # [batch, k]
        posteriors = torch.softmax(log_probs, dim=1).unsqueeze(-1)  # [batch, k, 1]
        
        # Compute component scores
        scores = torch.stack([
            comp.score(x) for comp in self.components
        ], dim=1)  # [batch, k, d]
        
        # Weighted sum
        return (posteriors * scores).sum(dim=1)
    
    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Generate samples by first sampling component, then from component.
        
        Args:
            n_samples: Number of samples.
            
        Returns:
            Samples of shape [n_samples, dim].
        """
        # Sample component indices
        indices = torch.multinomial(self.weights, n_samples, replacement=True)
        
        # Sample from each component
        samples_list = []
        for k in range(self.n_components):
            n_k = (indices == k).sum().item()
            if n_k > 0:
                samples_list.append(self.components[k].sample(n_k))
        
        # Concatenate and shuffle
        samples = torch.cat(samples_list, dim=0)
        perm = torch.randperm(n_samples)
        return samples[perm]
    
    def component_posteriors(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute posterior probabilities p(k|x).
        
        Args:
            x: Input of shape [batch_size, dim].
            
        Returns:
            Posteriors of shape [batch_size, n_components].
        """
        log_probs = torch.stack([
            comp.log_prob(x).squeeze(-1) + self.log_weights[k]
            for k, comp in enumerate(self.components)
        ], dim=1)
        return torch.softmax(log_probs, dim=1)
    
    def __repr__(self) -> str:
        return f"GaussianMixture(n_components={self.n_components}, dim={self._dim})"


class Posterior(Distribution, nn.Module):
    """
    Bayesian posterior distribution.

    p(theta|D) propto p(D|theta) p(theta)

    where:
        - p(D|theta) is the likelihood
        - p(theta) is the prior
        - D is observed data

    This is the primary distribution class for Bayesian inference
    in geophysical inverse problems.

    Args:
        log_likelihood: Function computing log p(D|theta).
                       Signature: (theta, data=None, **kwargs) -> Tensor
        log_prior: Function computing log p(theta). Optional.
                  Signature: (theta, **kwargs) -> Tensor
        data: Observed data tensor.
        dim: Parameter dimension.

    Example:
        >>> def log_likelihood(theta, data):
        ...     pred = forward_model(theta)
        ...     return -0.5 * ((pred - data)**2).sum(dim=-1)
        ...
        >>> def log_prior(theta):
        ...     return -0.5 * (theta**2).sum(dim=-1)  # Standard normal
        ...
        >>> posterior = Posterior(log_likelihood, log_prior, data=obs, dim=10)
        >>> svgd = SVGD(target=posterior)
    """

    def __init__(
        self,
        log_likelihood: Callable,
        log_prior: Optional[Callable] = None,
        data: Optional[torch.Tensor] = None,
        dim: Optional[int] = None
    ):
        super().__init__(dim=dim)

        if log_likelihood is None:
            raise ValueError("log_likelihood function is required")

        self.log_likelihood_fn = log_likelihood
        self.log_prior_fn = log_prior

        # Register data as buffer so it moves with .to(device)
        if data is not None:
            self.register_buffer('data', data)
        else:
            self.data = None
    
    def set_data(self, data: torch.Tensor) -> 'Posterior':
        """
        Set or update observed data.

        This is useful for sequential inference where data
        arrives over time.

        Args:
            data: Observed data tensor.

        Returns:
            Self for method chaining.
        """
        # Use register_buffer to ensure data moves with .to(device)
        self.register_buffer('data', data)
        return self
    
    def log_prob(self, theta: torch.Tensor, data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute log posterior (up to normalizing constant).
        
        log p(theta|D) = log p(D|theta) + log p(theta) + const
        
        Args:
            theta: Parameters of shape [batch_size, dim].
            data: Optional data override.
            
        Returns:
            Log posterior of shape [batch_size, 1].
        """
        # Use provided data or stored data
        if data is None:
            data = self.data
        
        # Compute log likelihood
        log_lik = self.log_likelihood_fn(theta, data=data)
        if log_lik.dim() == 1:
            log_lik = log_lik.unsqueeze(-1)
        
        # Compute log prior
        if self.log_prior_fn is not None:
            log_prior = self.log_prior_fn(theta)
            if log_prior.dim() == 1:
                log_prior = log_prior.unsqueeze(-1)
        else:
            # Uniform (improper) prior
            log_prior = torch.zeros_like(log_lik)
        
        return log_lik + log_prior
    
    def log_likelihood(self, theta: torch.Tensor, data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute log likelihood only.
        
        Args:
            theta: Parameters of shape [batch_size, dim].
            data: Optional data override.
            
        Returns:
            Log likelihood of shape [batch_size, 1].
        """
        if data is None:
            data = self.data
        
        log_lik = self.log_likelihood_fn(theta, data=data)
        if log_lik.dim() == 1:
            log_lik = log_lik.unsqueeze(-1)
        return log_lik
    
    def log_prior_value(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Compute log prior only.
        
        Args:
            theta: Parameters of shape [batch_size, dim].
            
        Returns:
            Log prior of shape [batch_size, 1].
        """
        if self.log_prior_fn is None:
            return torch.zeros(theta.shape[0], 1, device=theta.device)
        
        log_prior = self.log_prior_fn(theta)
        if log_prior.dim() == 1:
            log_prior = log_prior.unsqueeze(-1)
        return log_prior
    
    def __repr__(self) -> str:
        has_prior = self.log_prior_fn is not None
        has_data = self.data is not None
        return f"Posterior(dim={self._dim}, has_prior={has_prior}, has_data={has_data})"
