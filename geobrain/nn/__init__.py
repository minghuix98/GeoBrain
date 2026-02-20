"""
Neural network building blocks for GeoBrain.

Provides general-purpose layers for building geophysical neural networks.
Domain-specific architectures (decoders, autoencoders) live in their
respective modules (e.g., geomodel.geogen).

Components:
    - Bayesian layers (Flipout): LinearFlipout, Conv2dFlipout, Conv3dFlipout
    - Utilities: Reshape, get_kl_loss, count_variational_parameters
    - Activations: ReLU, ClippedLinearActivation

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

from .layers import (
    Reshape,
    BaseVariationalLayer,
    LinearFlipout,
    Conv2dFlipout,
    Conv3dFlipout,
    get_kl_loss,
    count_variational_parameters,
)
from .activations import (
    ReLU,
    ClippedLinearActivation,
)

__all__ = [
    # Layers
    'Reshape',
    # Bayesian layers
    'BaseVariationalLayer',
    'LinearFlipout',
    'Conv2dFlipout',
    'Conv3dFlipout',
    # Utility functions
    'get_kl_loss',
    'count_variational_parameters',
    # Activations
    'ReLU',
    'ClippedLinearActivation',
]