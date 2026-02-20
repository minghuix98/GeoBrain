"""
Core abstractions and configurations for wave simulation.

This module provides the foundational building blocks:
    - Abstract base classes for models and propagators
    - Configuration dataclasses for simulation parameters

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

from .config import (
    GridConfig,
    BoundaryConfig,
    TimeConfig,
    WaveletConfig,
    SolverConfig,
)

from .base import (
    AbstractModel,
    AbstractPropagator,
    UNITS,
)

from .result import WaveResult

__all__ = [
    # Configuration
    'GridConfig',
    'BoundaryConfig',
    'TimeConfig',
    'WaveletConfig',
    'SolverConfig',
    # Base classes
    'AbstractModel',
    'AbstractPropagator',
    'UNITS',
    # Result
    'WaveResult',
]