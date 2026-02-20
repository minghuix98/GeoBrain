"""
GeoBrain Bayesian Inference Module.

A lightweight framework for Bayesian inference and uncertainty quantification
using particle-based and MCMC sampling methods.

Implemented Samplers:
    - SVGD: Stein Variational Gradient Descent
    - HMC: Hamiltonian Monte Carlo
    - NUTS: No-U-Turn Sampler
    - LDS: Langevin Dynamics Sampler (ULA/MALA)

Features:
    - Multiple kernel functions (RBF, IMQ) with adaptive bandwidth
    - Flexible distribution framework (Gaussian, GMM, Posterior)
    - Simple API with both high-level run() and low-level step() interfaces
    - Mass matrix preconditioning (HMC/NUTS)
    - Adaptive step size with dual averaging (NUTS)
    - Box constraints with reflection (HMC/NUTS)

Example:
    Simple usage:

    >>> from geobrain.bayes import SVGD, Gaussian
    >>> import torch
    >>>
    >>> # Define target distribution
    >>> target = Gaussian(mean=torch.zeros(2), cov=torch.eye(2))
    >>>
    >>> # Create sampler and run
    >>> svgd = SVGD(target=target, lr=0.01)
    >>> result = svgd.run(n_samples=100, n_steps=500)
    >>>
    >>> # Access results
    >>> print(result.summary())
    >>> samples = result.samples

    HMC sampling:

    >>> from geobrain.bayes import HMC
    >>> hmc = HMC(target=target, step_size=0.1, n_leapfrog=10)
    >>> result = hmc.run(n_samples=10, n_steps=500)

    NUTS sampling (auto-tuned):

    >>> from geobrain.bayes import NUTS
    >>> nuts = NUTS(target=target, target_accept_rate=0.8)
    >>> result = nuts.run(n_samples=1, n_steps=1000)

    Bayesian inference for inverse problems:

    >>> from geobrain.bayes import SVGD, Posterior
    >>>
    >>> def log_likelihood(theta, data):
    ...     pred = forward_model(theta)
    ...     return -0.5 * ((pred - data)**2).sum(dim=-1)
    >>>
    >>> def log_prior(theta):
    ...     return -0.5 * (theta**2).sum(dim=-1)
    >>>
    >>> posterior = Posterior(log_likelihood, log_prior, data=observations)
    >>> svgd = SVGD(target=posterior)
    >>> result = svgd.run(n_samples=200, n_steps=1000)

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "Mingliang Liu"
__email__ = "mingliangliu@sdu.edu.cn"

# Core interfaces
from .core import (
    Distribution,
    BaseSampler,
    Kernel,
)

# Result container
from .sampling import SamplingResult

# Utility functions
from .utils import (
    pairwise_distances,
    median_heuristic,
    mmd,
    energy_distance,
)

# Kernels
from .kernels import (
    RBFKernel,
    IMQKernel,
    create_kernel,
)

# Distributions
from .distributions import (
    Gaussian,
    GaussianMixture,
    Posterior,
)

# Samplers
from .samplers import (
    SVGD,
    HMC,
    NUTS,
    LDS,
)


__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__email__',

    # Core interfaces
    'Distribution',
    'BaseSampler',
    'Kernel',

    # Result
    'SamplingResult',

    # Utilities
    'pairwise_distances',
    'median_heuristic',
    'mmd',
    'energy_distance',

    # Kernels
    'RBFKernel',
    'IMQKernel',
    'create_kernel',

    # Distributions
    'Gaussian',
    'GaussianMixture',
    'Posterior',

    # Samplers
    'SVGD',
    'HMC',
    'NUTS',
    'LDS',
]
