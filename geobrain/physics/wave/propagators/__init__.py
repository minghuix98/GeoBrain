"""
Wave equation propagators.

High-level interfaces for wave simulation combining velocity
models, survey geometry, and FDTD computational kernels.

Available propagators:
    - AcousticPropagator: Acoustic wave equation solver
    - ElasticPropagator: Elastic wave equation solver

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

from .acoustic import AcousticPropagator
from .elastic import ElasticPropagator
from ..core import WaveResult

__all__ = [
    'AcousticPropagator',
    'ElasticPropagator',
    'WaveResult',
]