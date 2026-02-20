"""
Core infrastructure for rock physics modeling.

This package provides the foundational classes and utilities for building
differentiable rock physics models using PyTorch.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

from .types import Tensor, TensorLike, PI, EPS, as_tensor, ensure_same_device
from .base import BaseModel, ComponentModel, CompositeModel
from .categories import (
    EffectiveModel,
    FluidModel,
    GranularModel,
    EmpiricalModel,
    AnisotropyModel,
    AVOModel,
    PermeabilityModel,
)
from .registry import ModelRegistry, register

__all__ = [
    # Types
    'Tensor', 'TensorLike', 'PI', 'EPS', 'as_tensor', 'ensure_same_device',
    # Base classes
    'BaseModel', 'ComponentModel', 'CompositeModel',
    # Category classes
    'EffectiveModel', 'FluidModel', 'GranularModel',
    'EmpiricalModel', 'AnisotropyModel', 'AVOModel', 'PermeabilityModel',
    # Registry
    'ModelRegistry', 'register',
]
