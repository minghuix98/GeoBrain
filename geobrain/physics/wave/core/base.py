"""
Abstract base classes for wave propagation simulation.

Provides foundational interfaces for velocity models and propagators
that all concrete implementations must follow.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod
from typing import Optional, Dict, Tuple, Union, Any

from .config import GridConfig, BoundaryConfig


# Units for model parameters
UNITS = {
    "vp": "m/s",
    "vs": "m/s",
    "rho": "kg/m³",
    "lam": "Pa",
    "mu": "Pa",
    "eps": "",
    "gamma": "",
    "delta": "",
}


class AbstractModel(nn.Module):
    """
    Abstract base class for velocity models.
    
    Manages model parameters, gradients, and bounds for wave simulation.
    All concrete velocity model classes (acoustic, elastic) must inherit
    from this class.
    
    Args:
        grid: Grid configuration specifying spatial discretization.
        boundary: Boundary condition configuration.
        device: Computation device. Default: 'cpu'.
        dtype: Data type for tensors. Default: torch.float32.

    Attributes:
        pars: List of parameter names managed by this model.
        requires_grad: Dict mapping parameter names to gradient flags.
        lower_bound: Dict mapping parameter names to lower bounds.
        upper_bound: Dict mapping parameter names to upper bounds.
        shape: Model shape (nz, nx).

    Example:
        Subclass implementation:
        >>> class AcousticModel(AbstractModel):
        ...     def __init__(self, grid, boundary, vp, **kwargs):
        ...         super().__init__(grid, boundary, **kwargs)
        ...         self.pars = ['vp']
        ...         self.vp = nn.Parameter(vp, requires_grad=False)
        ...
        ...     def forward(self):
        ...         self.clip_params()
    """

    def __init__(
        self,
        grid: GridConfig,
        boundary: BoundaryConfig,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        # Grid configuration
        self.grid = grid
        self.ox, self.oz = grid.ox, grid.oz
        self.dx, self.dz = grid.dx, grid.dz
        self.nx, self.nz = grid.nx, grid.nz

        # Optional Y-dimension (None for 2-D)
        self.ny = grid.ny
        self.dy = grid.dy
        self.oy = grid.oy

        # Boundary configuration
        self.boundary = boundary
        self.free_surface = boundary.free_surface
        self.abc_type = boundary.type
        self.abc_alpha = boundary.alpha
        self.nabc = boundary.n_layers

        # Device and dtype
        self.device = device
        self.dtype = dtype

        # Coordinate arrays
        self.x = grid.x
        self.z = grid.z
        self.y = grid.y  # None for 2-D

        # Model parameters and bounds (to be set by subclasses)
        self.pars: list = []
        self.requires_grad: Dict[str, bool] = {}
        self.lower_bound: Dict[str, Optional[float]] = {}
        self.upper_bound: Dict[str, Optional[float]] = {}

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions (2 or 3)."""
        return self.grid.ndim

    def __repr__(self) -> str:
        """String representation with parameter statistics."""
        info = f"{self.__class__.__name__} with parameters {self.pars}:\n"

        for par in self.pars:
            par_data = self.get_model(par)
            par_min, par_max = par_data.min(), par_data.max()
            req_grad = self.requires_grad.get(par, False)
            lb = self.lower_bound.get(par)
            ub = self.upper_bound.get(par)
            unit = UNITS.get(par, "")
            info += (
                f"  {par:4s}: {par_min:8.2f} - {par_max:8.2f} {unit:6s}, "
                f"grad={req_grad}, bounds=[{lb}, {ub}]\n"
            )

        if self.ndim == 3:
            info += f"  Grid: {self.nx} x {self.ny} x {self.nz}, dx={self.dx:.2f}m\n"
            info += f"  Origin: ({self.ox:.2f}, {self.oy:.2f}, {self.oz:.2f})m\n"
        else:
            info += f"  Grid: {self.nx} x {self.nz}, dx={self.dx:.2f}m\n"
            info += f"  Origin: ({self.ox:.2f}, {self.oz:.2f})m\n"
        info += f"  Boundary: {self.abc_type}, {self.nabc} layers\n"
        info += f"  Free surface: {self.free_surface}\n"

        return info

    @abstractmethod
    def forward(self) -> None:
        """
        Prepare model for wave propagation.

        This method is called before each forward simulation to:
            - Update derived parameters (e.g., density from velocity)
            - Clip parameters to valid bounds
            - Perform any necessary preprocessing

        Must be implemented by all subclasses.
        """
        raise NotImplementedError

    def get_model(self, par: str) -> np.ndarray:
        """
        Get model parameter as numpy array.

        Args:
            par: Parameter name (e.g., 'vp', 'rho').

        Returns:
            Model array with shape (nz, nx).

        Raises:
            ValueError: If parameter not found.

        Example:
            >>> vp = model.get_model('vp')
            >>> print(vp.shape)  # (100, 200)
        """
        if par not in self.pars:
            raise ValueError(f"Parameter '{par}' not in model. Available: {self.pars}")

        data = getattr(self, par)
        if torch.is_tensor(data):
            return data.detach().cpu().numpy().copy()
        return np.array(data)

    def set_model(self, par: str, model: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Set model parameter.

        Args:
            par: Parameter name.
            model: Model array with shape (nz, nx).

        Raises:
            ValueError: If parameter not found or shape mismatch.

        Example:
            >>> new_vp = np.ones((100, 200)) * 2500.0
            >>> model.set_model('vp', new_vp)
        """
        if par not in self.pars:
            raise ValueError(f"Parameter '{par}' not in model. Available: {self.pars}")

        if model.shape != self.shape:
            raise ValueError(
                f"Shape mismatch: expected {self.shape}, got {model.shape}"
            )

        # Convert to tensor
        if not torch.is_tensor(model):
            model = torch.tensor(model, dtype=self.dtype, device=self.device)
        else:
            model = model.to(dtype=self.dtype, device=self.device)

        # Set as parameter
        setattr(
            self, par,
            nn.Parameter(model, requires_grad=self.requires_grad.get(par, False))
        )

    def get_grad(self, par: str) -> np.ndarray:
        """
        Get parameter gradient.

        Args:
            par: Parameter name.

        Returns:
            Gradient array with shape (nz, nx), or zeros if no gradient.

        Raises:
            ValueError: If parameter not found.
        """
        if par not in self.pars:
            raise ValueError(f"Parameter '{par}' not in model")

        m = getattr(self, par)
        if m.grad is None:
            return np.zeros(self.shape)
        return m.grad.detach().cpu().numpy()

    def get_bound(self, par: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Get parameter bounds.

        Args:
            par: Parameter name.

        Returns:
            Tuple of (lower_bound, upper_bound). None indicates no bound.

        Raises:
            ValueError: If parameter not found.
        """
        if par not in self.pars:
            raise ValueError(f"Parameter '{par}' not in model")
        return (self.lower_bound.get(par), self.upper_bound.get(par))

    def clip_params(self) -> None:
        """Clip all parameters to their specified bounds."""
        for par in self.pars:
            lb = self.lower_bound.get(par)
            ub = self.upper_bound.get(par)
            if lb is not None or ub is not None:
                m = getattr(self, par)
                m.data.clamp_(lb, ub)

    def save(self, filename: str) -> None:
        """
        Save model to npz file.

        Args:
            filename: Output file path (should end with .npz).

        Example:
            >>> model.save('velocity_model.npz')
        """
        data = {
            'ox': self.ox, 'oz': self.oz,
            'dx': self.dx, 'dz': self.dz,
            'nx': self.nx, 'nz': self.nz,
            'ndim': self.ndim,
            'free_surface': self.free_surface,
            'abc_type': self.abc_type,
            'nabc': self.nabc,
        }
        if self.ndim == 3:
            data.update({'oy': self.oy, 'dy': self.dy, 'ny': self.ny})

        for par in self.pars:
            data[par] = self.get_model(par)
            data[f'{par}_bound'] = self.get_bound(par)
            data[f'{par}_grad'] = self.requires_grad.get(par, False)

        np.savez(filename, **data)

    @property
    def shape(self) -> tuple:
        """Model shape — ``(nz, nx)`` for 2-D, ``(nz, ny, nx)`` for 3-D."""
        return self.grid.shape


class AbstractPropagator(nn.Module):
    """
    Abstract base class for wave propagators.

    Defines the interface for wave equation solvers. All concrete
    propagator implementations must inherit from this class.

    Args:
        model: Velocity model instance.
        device: Computation device. Default: 'cpu'.
        dtype: Data type for tensors. Default: torch.float32.

    Example:
        Subclass implementation:
        >>> class AcousticPropagator(AbstractPropagator):
        ...     def forward(self, source, receiver, wavelet, **kwargs):
        ...         # Implement wave propagation
        ...         return {'pressure': seismogram}
    """

    def __init__(
        self,
        model: AbstractModel,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.model = model
        self.device = device
        self.dtype = dtype

        # Grid parameters from model
        self.ox, self.oz = model.ox, model.oz
        self.dx, self.dz = model.dx, model.dz
        self.nx, self.nz = model.nx, model.nz
        self.ny, self.dy, self.oy = model.ny, model.dy, model.oy

        # Boundary parameters
        self.abc_type = model.abc_type
        self.nabc = model.nabc
        self.free_surface = model.free_surface

    @abstractmethod
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Run forward wave propagation.

        Returns:
            Dictionary containing recorded seismograms and optional
            wavefields. Keys depend on implementation.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  model={self.model.__class__.__name__},\n"
            f"  grid=({self.nx}, {self.nz}),\n"
            f"  device='{self.device}'\n"
            f")"
        )