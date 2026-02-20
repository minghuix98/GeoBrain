"""
Data structures for differentiable implicit geological modeling.

Defines input data types for Universal Cokriging with gradient constraints
(Lajaunie et al. 1997 / GemPy algorithm).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
"""

import torch
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class StackRelation(Enum):
    """Relation between geological series for multi-series stacking."""
    ERODE = "erode"
    ONLAP = "onlap"
    FAULT = "fault"
    BASEMENT = "basement"


@dataclass
class SurfacePointData:
    """Surface contact points defining geological interfaces.

    Args:
        coords: Point coordinates, shape (N, D) where D=2 or 3.
        surface_id: Integer surface index for each point, shape (N,).
        nugget: Nugget effect for regularization.
    """
    coords: torch.Tensor
    surface_id: torch.Tensor
    nugget: float = 1e-6


@dataclass
class OrientationData:
    """Orientation measurements (gradient constraints).

    Args:
        coords: Measurement locations, shape (M, D).
        gradients: Gradient directions, shape (M, D). Internally normalized.
        nugget: Nugget effect for regularization.
    """
    coords: torch.Tensor
    gradients: torch.Tensor
    nugget: float = 0.01


@dataclass
class SeriesDefinition:
    """Definition of a geological series (a group of related surfaces).

    Args:
        name: Human-readable name for the series.
        surface_points: Surface contact points.
        orientations: Orientation / gradient measurements.
        relation: How this series interacts with older series.
        surface_names: Optional names for each surface in this series.
    """
    name: str
    surface_points: SurfacePointData
    orientations: OrientationData
    relation: StackRelation = StackRelation.ERODE
    surface_names: Optional[List[str]] = None


@dataclass
class FaultDefinition:
    """Definition of a fault surface.

    Args:
        name: Fault name.
        surface_points: Contact points on the fault surface.
        orientations: Orientation measurements on the fault.
        affected_series_indices: Which series indices are offset by this fault.
            None means all series are affected.
        displacement: Displacement magnitude (can be learned via gradient descent).
    """
    name: str
    surface_points: SurfacePointData
    orientations: OrientationData
    affected_series_indices: Optional[List[int]] = None
    displacement: float = 1.0


@dataclass
class InterpolationInput:
    """Packed input for a single Cokriging interpolation.

    Created from a SeriesDefinition with gradients normalized.
    """
    sp_coords: torch.Tensor       # (N, D)
    sp_surface_id: torch.Tensor   # (N,) int
    ori_coords: torch.Tensor      # (M, D)
    ori_gradients: torch.Tensor   # (M, D) unit vectors
    sp_nugget: float
    ori_nugget: float
    ndim: int
    n_surfaces: int

    @staticmethod
    def from_series(series: SeriesDefinition) -> 'InterpolationInput':
        """Create InterpolationInput from a SeriesDefinition.

        Normalizes gradient vectors to unit length.
        """
        sp = series.surface_points
        ori = series.orientations

        # Normalize gradients
        g = ori.gradients.clone()
        norms = g.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        g = g / norms

        ndim = sp.coords.shape[-1]
        n_surfaces = int(sp.surface_id.max().item()) + 1

        return InterpolationInput(
            sp_coords=sp.coords,
            sp_surface_id=sp.surface_id,
            ori_coords=ori.coords,
            ori_gradients=g,
            sp_nugget=sp.nugget,
            ori_nugget=ori.nugget,
            ndim=ndim,
            n_surfaces=n_surfaces,
        )


@dataclass
class ImplicitModelConfig:
    """Configuration for implicit geological model.

    Args:
        extent: Spatial extent as (x0, x1, y0, y1) for 2D
            or (x0, x1, y0, y1, z0, z1) for 3D.
        resolution: Grid resolution as (nx, ny) or (nx, ny, nz).
        kernel: Covariance kernel type: "cubic" or "gaussian".
        range: Kernel range parameter. Default sqrt(3), GemPy convention.
        c_o: Sill / variance. If None, auto-computed as range^2 / (14/3).
        drift_degree: Polynomial drift degree. 0=constant, 1=linear.
        device: Compute device ("cpu", "cuda", "auto").
        dtype: Data type string. "float64" recommended for Cokriging stability.
    """
    extent: Tuple[float, ...] = (0.0, 1.0, 0.0, 1.0)
    resolution: Tuple[int, ...] = (50, 50)
    kernel: str = "cubic"
    range: float = 1.7320508075688772  # sqrt(3)
    c_o: Optional[float] = None
    drift_degree: int = 1
    device: str = "auto"
    dtype: str = "float64"

    @property
    def ndim(self) -> int:
        """Spatial dimensionality (2 or 3)."""
        return len(self.resolution)

    @property
    def is_3d(self) -> bool:
        return self.ndim == 3

    @property
    def torch_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    @property
    def torch_dtype(self) -> torch.dtype:
        return getattr(torch, self.dtype)

    @property
    def computed_c_o(self) -> float:
        """Sill value, auto-computed if not set."""
        if self.c_o is not None:
            return self.c_o
        return self.range ** 2 / (14.0 / 3.0)

    def make_grid(self) -> torch.Tensor:
        """Create a regular evaluation grid from extent and resolution.

        Returns:
            Tensor of shape (n_points, ndim) with grid coordinates.
        """
        ndim = self.ndim
        slices = []
        for i in range(ndim):
            lo = self.extent[2 * i]
            hi = self.extent[2 * i + 1]
            slices.append(torch.linspace(lo, hi, self.resolution[i]))

        grids = torch.meshgrid(*slices, indexing='ij')
        grid = torch.stack([g.reshape(-1) for g in grids], dim=-1)
        return grid.to(device=self.torch_device, dtype=self.torch_dtype)
