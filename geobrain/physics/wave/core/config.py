"""
Configuration classes for wave propagation simulation.

Provides dataclasses for organizing simulation parameters including
grid specification, boundary conditions, time stepping, and solver options.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Literal


@dataclass
class GridConfig:
    """
    Grid configuration for spatial discretization.

    Defines the computational domain geometry. Supports both 2-D
    (x, z) and 3-D (x, y, z) grids.  When ``ny`` and ``dy`` are
    omitted (or ``ny`` is ``None``), the grid is treated as 2-D.

    Args:
        nx: Number of grid points in x-direction.
        nz: Number of grid points in z-direction.
        dx: Grid spacing in x-direction (m).
        dz: Grid spacing in z-direction (m).
        ox: Origin x-coordinate (m). Default: 0.0.
        oz: Origin z-coordinate (m). Default: 0.0.
        ny: Number of grid points in y-direction (3-D only).
        dy: Grid spacing in y-direction (m) (3-D only).
        oy: Origin y-coordinate (m). Default: 0.0.

    Attributes:
        ndim: Number of spatial dimensions (2 or 3).
        shape: Grid shape — ``(nz, nx)`` for 2-D, ``(nz, ny, nx)`` for 3-D.
        x: X-coordinate array.
        z: Z-coordinate array.
        y: Y-coordinate array (None in 2-D).
        extent: Spatial extent for plotting (x_min, x_max, z_max, z_min).

    Example:
        2-D grid:

        >>> grid = GridConfig(nx=200, nz=100, dx=10.0, dz=10.0)
        >>> print(grid.shape)  # (100, 200)

        3-D grid:

        >>> grid = GridConfig(nx=200, nz=100, dx=10.0, dz=10.0,
        ...                   ny=80, dy=10.0)
        >>> print(grid.shape)  # (100, 80, 200)

    Note:
        All provided grid spacings must be equal for numerical stability.
    """
    nx: int
    nz: int
    dx: float
    dz: float
    ox: float = 0.0
    oz: float = 0.0
    # Optional Y-dimension for 3-D grids
    ny: Optional[int] = None
    dy: Optional[float] = None
    oy: float = 0.0

    def __post_init__(self):
        if self.nx <= 0 or self.nz <= 0:
            raise ValueError(f"Grid dimensions must be positive: nx={self.nx}, nz={self.nz}")
        if self.dx <= 0 or self.dz <= 0:
            raise ValueError(f"Grid spacing must be positive: dx={self.dx}, dz={self.dz}")
        if self.dx != self.dz:
            raise ValueError(f"Grid spacing must be equal: dx={self.dx}, dz={self.dz}")

        # Validate Y-dimension consistency
        if (self.ny is None) != (self.dy is None):
            raise ValueError("ny and dy must both be provided for 3-D grids")
        if self.ny is not None:
            if self.ny <= 0:
                raise ValueError(f"ny must be positive: {self.ny}")
            if self.dy <= 0:
                raise ValueError(f"dy must be positive: {self.dy}")
            if self.dy != self.dx:
                raise ValueError(f"Grid spacing must be equal: dx={self.dx}, dy={self.dy}")

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions (2 or 3)."""
        return 3 if self.ny is not None else 2

    @property
    def shape(self) -> tuple:
        """Grid shape — ``(nz, nx)`` for 2-D, ``(nz, ny, nx)`` for 3-D."""
        if self.ndim == 3:
            return (self.nz, self.ny, self.nx)
        return (self.nz, self.nx)

    @property
    def x(self) -> np.ndarray:
        """X-coordinate array."""
        return np.arange(self.nx) * self.dx + self.ox

    @property
    def z(self) -> np.ndarray:
        """Z-coordinate array."""
        return np.arange(self.nz) * self.dz + self.oz

    @property
    def y(self) -> Optional[np.ndarray]:
        """Y-coordinate array (None in 2-D)."""
        if self.ny is None:
            return None
        return np.arange(self.ny) * self.dy + self.oy

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        """Spatial extent for plotting: (x_min, x_max, z_max, z_min)."""
        return (
            self.ox,
            self.ox + (self.nx - 1) * self.dx,
            self.oz + (self.nz - 1) * self.dz,
            self.oz,
        )


@dataclass
class BoundaryConfig:
    """
    Boundary condition configuration.

    Specifies absorbing boundary layer parameters for suppressing
    artificial reflections at domain edges.

    Args:
        type: Boundary type. Default: 'pml'.
            Options:
                - 'pml': Perfectly Matched Layer (recommended)
                - 'gerjan': Exponential damping
                - 'sincos': Trigonometric damping
        n_layers: Number of absorbing boundary layers. Default: 20.
        free_surface: Whether to use free surface at top. Default: False.
        alpha: Attenuation factor for Gerjan boundary. Default: 0.0053.

    Example:
        >>> bc = BoundaryConfig(type='pml', n_layers=30, free_surface=True)
    """
    type: Literal['pml', 'gerjan', 'sincos'] = 'pml'
    n_layers: int = 20
    free_surface: bool = False
    alpha: float = 0.0053

    def __post_init__(self):
        self.type = self.type.lower()
        if self.type not in ('pml', 'gerjan', 'sincos'):
            raise ValueError(
                f"Unknown boundary type: '{self.type}'. "
                f"Choose from: 'pml', 'gerjan', 'sincos'"
            )
        if self.n_layers <= 0:
            raise ValueError(f"Number of layers must be positive: {self.n_layers}")


@dataclass
class TimeConfig:
    """
    Time stepping configuration.

    Defines temporal discretization for wave simulation.

    Args:
        nt: Number of time steps.
        dt: Time step size (s).
        t0: Start time (s). Default: 0.0.

    Attributes:
        duration: Total simulation duration (s).
        t: Time array.

    Example:
        >>> time = TimeConfig(nt=1000, dt=0.001)
        >>> print(time.duration)  # 1.0 second
        >>> print(time.t[:5])  # [0.000, 0.001, 0.002, 0.003, 0.004]
    """
    nt: int
    dt: float
    t0: float = 0.0

    def __post_init__(self):
        if self.nt <= 0:
            raise ValueError(f"Number of time steps must be positive: {self.nt}")
        if self.dt <= 0:
            raise ValueError(f"Time step must be positive: {self.dt}")

    @property
    def duration(self) -> float:
        """Total simulation duration (s)."""
        return self.nt * self.dt

    @property
    def t(self) -> np.ndarray:
        """Time array."""
        return np.arange(self.nt) * self.dt + self.t0


@dataclass
class WaveletConfig:
    """
    Source wavelet configuration.

    Specifies wavelet parameters for seismic source.

    Args:
        f0: Dominant/peak frequency (Hz).
        type: Wavelet type. Default: 'ricker'.
            Options: 'ricker', 'gaussian', 'ormsby', 'klauder'.
        amp: Amplitude scaling factor. Default: 1.0.
        delay: Time delay (s). Default: 1.2/f0.

    Example:
        >>> wavelet = WaveletConfig(f0=25.0, type='ricker')
        >>> print(wavelet.delay)  # 0.048
    """
    f0: float
    type: Literal['ricker', 'gaussian', 'ormsby', 'klauder'] = 'ricker'
    amp: float = 1.0
    delay: Optional[float] = None

    def __post_init__(self):
        self.type = self.type.lower()
        if self.f0 <= 0:
            raise ValueError(f"Frequency must be positive: {self.f0}")
        if self.delay is None:
            self.delay = 1.2 / self.f0


@dataclass
class SolverConfig:
    """
    Numerical solver configuration.

    Specifies finite difference scheme and computation parameters.

    Args:
        fd_order: Finite difference order. Default: 4.
            Options: 4, 6, 8, 10.
        device: Computation device. Default: 'cpu'.
            Options: 'cpu', 'cuda', 'cuda:0', etc.
        dtype: Data type. Default: 'float32'.
            Options: 'float32', 'float64'.
        checkpoint_segments: Number of segments for gradient checkpointing.
            Higher values reduce memory but increase computation. Default: 1.

    Attributes:
        torch_device: Converted torch.device object.
        torch_dtype: Converted torch.dtype object.

    Example:
        >>> solver = SolverConfig(fd_order=8, device='cuda')
        >>> print(solver.torch_device)  # device(type='cuda')
    """
    fd_order: int = 4
    device: str = 'cpu'
    dtype: str = 'float32'
    checkpoint_segments: int = 1

    def __post_init__(self):
        if self.fd_order not in (4, 6, 8, 10):
            raise ValueError(
                f"FD order must be 4, 6, 8, or 10, got {self.fd_order}"
            )

        valid_devices = ('cpu', 'cuda')
        if self.device not in valid_devices and not self.device.startswith('cuda:'):
            raise ValueError(f"Unknown device: '{self.device}'")

        if self.dtype not in ('float32', 'float64'):
            raise ValueError(f"Unknown dtype: '{self.dtype}'")

        if self.checkpoint_segments < 1:
            raise ValueError(
                f"Checkpoint segments must be >= 1, got {self.checkpoint_segments}"
            )

    @property
    def torch_device(self) -> torch.device:
        """Convert to torch.device."""
        return torch.device(self.device)

    @property
    def torch_dtype(self) -> torch.dtype:
        """Convert to torch.dtype."""
        return torch.float32 if self.dtype == 'float32' else torch.float64