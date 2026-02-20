"""
GeoBrain Wave Physics Module.

A comprehensive module for seismic wave propagation simulation,
including acoustic and elastic wave equations, AVO modeling,
wavelets, and boundary conditions.

Architecture:
    wave/
    ├── core/           # Abstract bases and configurations
    ├── models/         # Velocity models (acoustic, elastic)
    ├── wavelets/       # Source wavelet generators
    ├── boundary/       # Absorbing boundary conditions
    ├── avo/            # AVO reflectivity models
    ├── survey/         # Acquisition geometry
    ├── propagators/    # Wave equation solvers
    ├── kernels/        # Low-level FDTD kernels
    └── utils/          # Utility functions

Example:
    >>> from wave import (
    ...     GridConfig, BoundaryConfig,
    ...     AcousticModel, Survey, Source, Receiver,
    ...     AcousticPropagator,
    ...     RickerWavelet, Shuey
    ... )
    >>>
    >>> # Create configuration
    >>> grid = GridConfig(nx=200, nz=100, dx=10, dz=10)
    >>> boundary = BoundaryConfig(type='pml', n_layers=20)
    >>>
    >>> # Create model
    >>> model = AcousticModel(
    ...     grid=grid, boundary=boundary,
    ...     vp=vp_array, rho=rho_array,
    ...     vp_grad=True
    ... )
    >>>
    >>> # Create survey
    >>> wavelet, _ = RickerWavelet()(f0=15.0, dt=0.001)
    >>> source = Source(nt=1000, dt=0.001, f0=15)
    >>> source.add_sources(src_x, src_z, wavelet.numpy())
    >>> receiver = Receiver(nt=1000, dt=0.001)
    >>> receiver.add_receivers(rcv_x, rcv_z, 'pr')
    >>> survey = Survey(source, receiver)
    >>>
    >>> # Run simulation
    >>> propagator = AcousticPropagator(model, survey, device='cuda')
    >>> result = propagator.forward(checkpoint_segments=4)
    >>>
    >>> # Access results
    >>> pressure = result['p']  # (n_shots, nt, n_receivers)

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "Mingliang Liu"
__email__ = "mingliangliu@sdu.edu.cn"

# =============================================================================
# Core: Configurations and Base Classes
# =============================================================================

from .core import (
    # Configuration
    GridConfig,
    BoundaryConfig,
    TimeConfig,
    WaveletConfig,
    SolverConfig,
    # Base classes
    AbstractModel,
    AbstractPropagator,
    # Result
    WaveResult,
)

# =============================================================================
# Models: Velocity Models
# =============================================================================

from .models import (
    AcousticModel,
    IsotropicElasticModel,
)

# =============================================================================
# Wavelets: Source Wavelet Generators
# =============================================================================

from .wavelets import (
    WaveletGenerator,
    RickerWavelet,
    GaussianWavelet,
    OrmsbyWavelet,
    KlauderWavelet,
    create_wavelet,
)

# =============================================================================
# Boundary: Absorbing Boundary Conditions
# =============================================================================

from .boundary import (
    bc_pml,
    bc_pml_xz,
    bc_gerjan,
    bc_sincos,
    create_boundary,
)

# =============================================================================
# AVO: Reflectivity Models
# =============================================================================

from .avo import (
    AVOModel,
    Zoeppritz,
    AkiRichards,
    Shuey,
    compute_reflectivity,
    normal_incidence_rc,
    as_tensor,
)

# =============================================================================
# Survey: Acquisition Geometry
# =============================================================================

from .survey import (
    Source,
    Receiver,
    Survey,
)

# =============================================================================
# Propagators: Wave Equation Solvers
# =============================================================================

from .propagators import (
    AcousticPropagator,
    ElasticPropagator,
)

# =============================================================================
# Utils: Utility Functions
# =============================================================================

from .utils import (
    # Tensor operations
    numpy2tensor,
    tensor2numpy,
    gpu2cpu,
    list2numpy,
    numpy2list,
    # Metrics
    mse,
    rmse,
    mape,
    snr,
    # Convolution
    create_conv_matrix,
    convolve_trace,
    convolve_gather,
    apply_wavelet,
)

# =============================================================================
# Kernels: Low-level FDTD (for advanced users)
# =============================================================================

from . import kernels

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Version
    '__version__',
    '__author__',
    '__email__',

    # Configuration
    'GridConfig',
    'BoundaryConfig',
    'TimeConfig',
    'WaveletConfig',
    'SolverConfig',

    # Base classes
    'AbstractModel',
    'AbstractPropagator',

    # Result
    'WaveResult',

    # Velocity models
    'AcousticModel',
    'IsotropicElasticModel',

    # Wavelets
    'WaveletGenerator',
    'RickerWavelet',
    'GaussianWavelet',
    'OrmsbyWavelet',
    'KlauderWavelet',
    'create_wavelet',

    # Boundary conditions
    'bc_pml',
    'bc_pml_xz',
    'bc_gerjan',
    'bc_sincos',
    'create_boundary',

    # AVO models
    'AVOModel',
    'Zoeppritz',
    'AkiRichards',
    'Shuey',
    'compute_reflectivity',
    'normal_incidence_rc',
    'as_tensor',

    # Survey
    'Source',
    'Receiver',
    'Survey',

    # Propagators
    'AcousticPropagator',
    'ElasticPropagator',

    # Utils - tensor
    'numpy2tensor',
    'tensor2numpy',
    'gpu2cpu',
    'list2numpy',
    'numpy2list',

    # Utils - metrics
    'mse',
    'rmse',
    'mape',
    'snr',

    # Utils - convolution
    'create_conv_matrix',
    'convolve_trace',
    'convolve_gather',
    'apply_wavelet',

    # Kernels module
    'kernels',
]