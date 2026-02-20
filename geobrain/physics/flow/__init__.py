"""
Flow simulation module for GeoBrain.

Provides oil-water two-phase reservoir simulation with fully-implicit
discretization and Newton-Raphson solver.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0

Example:
    >>> from geobrain.physics.flow import ReservoirModel, FlowPropagator
    >>> from geobrain.physics.flow import PVTTable, PVTAnalytic, RelPermCorey
    >>> from geobrain.physics.flow import Well
    >>>
    >>> # Create and configure model
    >>> model = ReservoirModel(nx=30, ny=15, nz=1)
    >>> model.set_grid(dx=100.0, dy=100.0, dz=20.0)
    >>> model.set_rock(perm=100.0, poro=0.2)
    >>> model.set_pvt(
    ...     pvdo=[[100, 1.05, 2.0], [4000, 1.02, 1.5]],
    ...     pvtw=[4000, 1.0, 3e-6, 0.5, 0]
    ... )
    >>> model.set_relperm(corey={'swc': 0.2, 'sor': 0.2, 'nw': 2, 'no': 2})
    >>>
    >>> # Add wells
    >>> prod = Well('P1', well_type='PROD')
    >>> prod.add_perforation(cell_idx=449, wi=5.0)
    >>> prod.set_control('RATE', target=-500.0, phase='LIQ')
    >>> model.add_well(prod)
    >>>
    >>> # Initialize
    >>> model.initialize(po=4000.0, sw=0.2)
    >>>
    >>> # Run simulation
    >>> propagator = FlowPropagator(model)
    >>> result = propagator(t_end=365.0, verbose=True)
"""

__version__ = "0.1.0"
__author__ = "Mingliang Liu"
__email__ = "mingliangliu@sdu.edu.cn"


# Constants
from .constants import (
    DEVICE,
    DTYPE,
    ALPHA,
    M,
    BETA,
    G,
    GC,
    EPS,
    MAX_NEWTON_ITER,
    NEWTON_TOL
)

# Grid
from .grid import (
    CartGrid,
    ConnList
)

# Rock
from .rock import Rock

# PVT
from .pvt import (
    PVT,
    PVTTable,
    PVTAnalytic
)

# Relative Permeability
from .relperm import (
    RelPerm,
    RelPermTable,
    RelPermCorey
)

# Fluid
from .fluid import (
    PhaseState,
    OilWaterFluid
)

# Well
from .well import (
    Well,
    WellGroup,
    WellPerforation,
    WellHistory,
    compute_well_index
)

# Solver
from .solver import (
    NewtonSolver,
    TimeStepScheduler,
    SolverResult
)

# Model and Propagator (main interfaces)
from .model import ReservoirModel
from .propagator import FlowPropagator, SimulationResult


__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__email__',

    # Constants
    'DEVICE',
    'DTYPE',
    'ALPHA',
    'M',
    'BETA',
    'G',
    'GC',
    'EPS',
    'MAX_NEWTON_ITER',
    'NEWTON_TOL',

    # Grid
    'CartGrid',
    'ConnList',

    # Rock
    'Rock',

    # PVT
    'PVT',
    'PVTTable',
    'PVTAnalytic',

    # Relative Permeability
    'RelPerm',
    'RelPermTable',
    'RelPermCorey',

    # Fluid
    'PhaseState',
    'OilWaterFluid',

    # Well
    'Well',
    'WellGroup',
    'WellPerforation',
    'WellHistory',
    'compute_well_index',

    # Solver
    'NewtonSolver',
    'TimeStepScheduler',
    'SolverResult',

    # Main Interfaces
    'ReservoirModel',
    'FlowPropagator',
    'SimulationResult',
]
