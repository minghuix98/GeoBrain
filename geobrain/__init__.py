"""
GeoBrain: A Framework for Geophysical Modeling and Inversion.

Modules:
    - bayes: Bayesian inference (SVGD, distributions)
    - optim: Deterministic inversion (Inverter)
    - core: Shared infrastructure (InverseProblem, registry)
    - geomodel: Geological model generation (FFT-MA, etc.)
    - physics: Physics simulations (flow, wave, rock)
    - nn: Neural network components
    - vis: Visualization utilities

Quick Start:
    >>> from geobrain import InverseProblem, SVGD, Inverter
    >>>
    >>> # Define problem once
    >>> problem = InverseProblem(forward_fn=forward, observed=data, noise_std=0.1)
    >>>
    >>> # Deterministic inversion
    >>> inverter = problem.create_inverter(initial_model=model)
    >>> result = inverter.run(problem.observed, max_epochs=100)
    >>>
    >>> # Bayesian inference
    >>> posterior = problem.as_posterior(log_prior=my_prior)
    >>> svgd = SVGD(target=posterior)
    >>> samples = svgd.run(n_samples=200, n_steps=1000)

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "Mingliang Liu"
__email__ = "mingliangliu@sdu.edu.cn"

# =============================================================================
# Shared Inverse Problem
# =============================================================================
from .core import InverseProblem

# =============================================================================
# Bayesian Inference
# =============================================================================
from .bayes import (
    # Samplers
    SVGD,
    # Distributions
    Gaussian,
    GaussianMixture,
    Posterior,
    # Kernels
    RBFKernel,
    IMQKernel,
    # Result
    SamplingResult,
)

# =============================================================================
# Deterministic Inversion
# =============================================================================
from .optim import (
    Inverter,
    InversionResult,
    # Parameterization
    ExplicitParameterization,
    LatentParameterization,
    NetworkParameterization,
    # Constraints
    bound_constraint,
    clip_constraint,
)

# =============================================================================
# Geological Modeling
# =============================================================================
from .geomodel import (
    Simulator,
    SimulationConfig,
    CoSimConfig,
)

# =============================================================================
# Decision Support
# =============================================================================
from .decision import ValueOfInformation, ClosedLoopManager

# =============================================================================
# Registry Hub
# =============================================================================
from .core import registries

# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # Version
    "__version__",
    "__author__",
    "__email__",
    # Inverse Problem
    "InverseProblem",
    # Bayes
    "SVGD",
    "Gaussian",
    "GaussianMixture",
    "Posterior",
    "RBFKernel",
    "IMQKernel",
    "SamplingResult",
    # Optim
    "Inverter",
    "InversionResult",
    "ExplicitParameterization",
    "LatentParameterization",
    "NetworkParameterization",
    "bound_constraint",
    "clip_constraint",
    # Geomodel
    "Simulator",
    "SimulationConfig",
    "CoSimConfig",
    # Decision
    "ValueOfInformation",
    "ClosedLoopManager",
    # Registry
    "registries",
]
