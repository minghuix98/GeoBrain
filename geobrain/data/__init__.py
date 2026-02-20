"""
Data utilities for geological properties.

This module provides:
    - Transform functions for normalizing and constraining geological data
    - Transform classes for composable data pipelines
    - Dataset classes for PyTorch data loading

Example:
    >>> from geobrain.data import Compose, Normalize, Clamp
    >>> transform = Compose([Normalize(), Clamp()])
    >>> normalized = transform(raw_data)
"""

from .transforms import (
    # Functions
    clamp_properties,
    sigmoid_transform,
    normalize_properties,
    denormalize_properties,
    logit_transform,
    exp_transform,
    log_transform,
    compute_property_stats,
    # Classes
    PropertyBounds,
    Transform,
    Compose,
    Normalize,
    Denormalize,
    Clamp,
    Sigmoid,
    Logit,
    Log,
    Exp,
)

from .dataset import (
    SimpleTensorDataset,
    GeoDataset,
)

__all__ = [
    # Functions
    'clamp_properties',
    'sigmoid_transform',
    'normalize_properties',
    'denormalize_properties',
    'logit_transform',
    'exp_transform',
    'log_transform',
    'compute_property_stats',
    # Classes
    'PropertyBounds',
    'Transform',
    'Compose',
    'Normalize',
    'Denormalize',
    'Clamp',
    'Sigmoid',
    'Logit',
    'Log',
    'Exp',
    # Datasets
    'SimpleTensorDataset',
    'GeoDataset',
]