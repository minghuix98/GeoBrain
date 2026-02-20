"""
GeoBrain Physics Module.

Provides differentiable multiphysics simulation for subsurface modeling:

- **wave**: Seismic wave propagation (acoustic, elastic), AVO modeling
- **rock**: Rock physics modeling (effective medium, fluid substitution)
- **flow**: Reservoir flow simulation (two-phase, fully-implicit)

All physics modules are built on PyTorch and support automatic differentiation
for gradient-based inversion and optimization.

Quick Start:
    Wave propagation:
    >>> from geobrain.physics import AcousticPropagator, AcousticModel
    >>> model = AcousticModel(grid=grid, boundary=boundary, vp=velocity)
    >>> propagator = AcousticPropagator(model, survey)
    >>> seismic = propagator.forward()

    Rock physics:
    >>> from geobrain.physics import RockPhysicsWorkflow
    >>> workflow = RockPhysicsWorkflow.from_preset('shaly_sand')
    >>> Vp, Vs, rho = workflow(phi=0.2, Sw=0.8)

    Flow simulation:
    >>> from geobrain.physics import ReservoirModel, FlowPropagator
    >>> model = ReservoirModel(nx=30, ny=15, nz=1)
    >>> propagator = FlowPropagator(model)
    >>> result = propagator(t_end=365.0)

Submodule Access:
    For full API access, import submodules directly:
    >>> from geobrain.physics import wave, rock, flow
    >>> wave.RickerWavelet()(f0=25.0, dt=0.001)

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "Mingliang Liu"
__email__ = "mingliangliu@sdu.edu.cn"

# =============================================================================
# Submodules (for full API access)
# =============================================================================

from . import wave
from . import rock
from . import flow

# =============================================================================
# Wave: Seismic Propagation
# =============================================================================

from .wave import (
    # Propagators
    AcousticPropagator,
    # Models
    AcousticModel,
    # Configuration
    GridConfig,
    BoundaryConfig,
    # Survey
    Source,
    Receiver,
    Survey,
    # Wavelets
    RickerWavelet,
    # AVO
    AkiRichards,
    Shuey,
)

# =============================================================================
# Rock: Rock Physics
# =============================================================================

from .rock import (
    # Workflow
    RockPhysicsWorkflow,
    RockPhysicsConfig,
    # Effective medium
    VRH,
    HashinShtrikman,
    DEM,
    SelfConsistent,
    CriticalPorosity,
    # Granular
    HertzMindlin,
    SoftSand,
    StiffSand,
    ContactCement,
    # Fluid
    Gassmann,
    Wood,
    Brie,
    BatzleWang,
    # Empirical
    Gardner,
    Han,
    CastagnaMudrock,
    # Anisotropy
    Thomsen,
    Backus,
    # Data
    get_mineral,
    get_fluid,
    MINERALS,
    FLUIDS,
    # Utils
    v_from_moduli,
    moduli_from_v,
)

# =============================================================================
# Flow: Reservoir Simulation
# =============================================================================

from .flow import (
    # Main interfaces
    ReservoirModel,
    FlowPropagator,
    SimulationResult,
    # Well
    Well,
    WellGroup,
    # PVT
    PVTTable,
    PVTAnalytic,
    # Relative permeability
    RelPermCorey,
    RelPermTable,
)

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Version
    "__version__",
    "__author__",
    "__email__",
    # Submodules
    "wave",
    "rock",
    "flow",
    # Wave - Propagators
    "AcousticPropagator",
    # Wave - Models
    "AcousticModel",
    # Wave - Config
    "GridConfig",
    "BoundaryConfig",
    # Wave - Survey
    "Source",
    "Receiver",
    "Survey",
    # Wave - Wavelets
    "RickerWavelet",
    # Wave - AVO
    "AkiRichards",
    "Shuey",
    # Rock - Workflow
    "RockPhysicsWorkflow",
    "RockPhysicsConfig",
    # Rock - Effective medium
    "VRH",
    "HashinShtrikman",
    "DEM",
    "SelfConsistent",
    "CriticalPorosity",
    # Rock - Granular
    "HertzMindlin",
    "SoftSand",
    "StiffSand",
    "ContactCement",
    # Rock - Fluid
    "Gassmann",
    "Wood",
    "Brie",
    "BatzleWang",
    # Rock - Empirical
    "Gardner",
    "Han",
    "CastagnaMudrock",
    # Rock - Anisotropy
    "Thomsen",
    "Backus",
    # Rock - Data
    "get_mineral",
    "get_fluid",
    "MINERALS",
    "FLUIDS",
    # Rock - Utils
    "v_from_moduli",
    "moduli_from_v",
    # Flow - Main
    "ReservoirModel",
    "FlowPropagator",
    "SimulationResult",
    # Flow - Well
    "Well",
    "WellGroup",
    # Flow - PVT
    "PVTTable",
    "PVTAnalytic",
    # Flow - RelPerm
    "RelPermCorey",
    "RelPermTable",
]
