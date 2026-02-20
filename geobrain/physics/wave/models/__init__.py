"""
Velocity models for wave simulation.

Provides model classes for different wave equation formulations:
    - AcousticModel: P-wave velocity and density
    - IsotropicElasticModel: P-wave, S-wave velocities and density

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

from .acoustic import AcousticModel
from .elastic import IsotropicElasticModel

__all__ = [
    'AcousticModel',
    'IsotropicElasticModel',
]