"""
FDTD kernels for wave propagation.

Provides low-level finite-difference time-domain computational
kernels for acoustic and elastic wave equations.

Available kernels:
    - Acoustic: 4th-order staggered grid scheme
    - Elastic: 4th, 6th, 8th order staggered grid schemes

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

# Acoustic kernels (2D)
from .acoustic_2d import (
    pad_model,
    pad_model_3d,
    acoustic_step_forward,
    acoustic_forward_kernel,
)

# Acoustic kernels (3D)
from .acoustic_3d import (
    acoustic_step_forward_3d,
    acoustic_forward_kernel_3d,
)

# Elastic kernels (2D)
from .elastic_2d import (
    fd_coefficients,
    Dxf_4, Dzf_4, Dxb_4, Dzb_4,
    Dxf_6, Dzf_6, Dxb_6, Dzb_6,
    Dxf_8, Dzf_8, Dxb_8, Dzb_8,
    pad_model_elastic,
    thomsen_to_elastic,
    lame_from_velocities,
    staggered_grid_parameters,
    elastic_step_forward,
    elastic_forward_kernel,
)

# Elastic kernels (3D)
from .elastic_3d import (
    elastic_step_forward_3d,
    elastic_forward_kernel_3d,
)

__all__ = [
    # Acoustic (2D)
    'pad_model',
    'pad_model_3d',
    'acoustic_step_forward',
    'acoustic_forward_kernel',
    # Acoustic (3D)
    'acoustic_step_forward_3d',
    'acoustic_forward_kernel_3d',
    # Elastic FD operators
    'fd_coefficients',
    'Dxf_4', 'Dzf_4', 'Dxb_4', 'Dzb_4',
    'Dxf_6', 'Dzf_6', 'Dxb_6', 'Dzb_6',
    'Dxf_8', 'Dzf_8', 'Dxb_8', 'Dzb_8',
    # Elastic utilities
    'pad_model_elastic',
    'thomsen_to_elastic',
    'lame_from_velocities',
    'staggered_grid_parameters',
    # Elastic (2D)
    'elastic_step_forward',
    'elastic_forward_kernel',
    # Elastic (3D)
    'elastic_step_forward_3d',
    'elastic_forward_kernel_3d',
]
