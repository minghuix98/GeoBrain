"""
Base class and utilities for AVO models.

Provides abstract interface and common decorators for
reflection coefficient calculations.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from abc import abstractmethod
from functools import wraps
from typing import Union

# Constants
EPS = 1e-10


def as_tensor(
    x: Union[Tensor, np.ndarray, float],
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> Tensor:
    """
    Convert input to tensor.
    
    Args:
        x: Input value (tensor, array, or scalar).
        dtype: Target dtype. Default: float32.
        device: Target device. Default: None.

    Returns:
        Tensor on specified device with specified dtype.

    Example:
        >>> t = as_tensor([1.0, 2.0, 3.0])
        >>> t = as_tensor(np.array([1, 2, 3]), dtype=torch.float64)
    """
    if isinstance(x, Tensor):
        if dtype is not None:
            x = x.to(dtype)
        if device is not None:
            x = x.to(device)
        return x

    if isinstance(x, np.ndarray):
        return torch.from_numpy(x.astype(np.float32))

    return torch.tensor(x, dtype=dtype or torch.float32, device=device)


def deg2rad(x: Union[Tensor, float]) -> Union[Tensor, float]:
    """Convert degrees to radians."""
    return x * (np.pi / 180.0)


def vectorize_angles(func):
    """
    Decorator that handles angle conversion and broadcasting.

    Responsibilities:
        - Convert all inputs to tensors
        - Convert angles from degrees to radians
        - Validate angle range (0-90 degrees)
        - Reshape angles for broadcasting: [n_angles, 1, 1, ...]

    The decorated function receives theta in radians with shape
    suitable for broadcasting against the velocity arrays.

    Example:
        >>> @vectorize_angles
        ... def forward(self, vp1, vs1, rho1, vp2, vs2, rho2, theta):
        ...     # theta is now in radians with shape [n_angles, 1, ...]
        ...     return some_computation(theta)
    """
    @wraps(func)
    def wrapper(
        self,
        vp1, vs1, rho1,
        vp2, vs2, rho2,
        theta,
        **kwargs
    ):
        # Convert all inputs to tensors
        vp1 = as_tensor(vp1)
        vs1 = as_tensor(vs1)
        rho1 = as_tensor(rho1)
        vp2 = as_tensor(vp2)
        vs2 = as_tensor(vs2)
        rho2 = as_tensor(rho2)

        # Handle theta conversion
        if not isinstance(theta, Tensor):
            if isinstance(theta, (list, tuple)):
                theta = torch.tensor(theta, dtype=vp1.dtype, device=vp1.device)
            else:
                theta = torch.tensor([theta], dtype=vp1.dtype, device=vp1.device)

        # Ensure at least 1D
        if theta.ndim == 0:
            theta = theta.unsqueeze(0)

        # Validate angle range (assumes degrees)
        if (theta > 90.0).any():
            raise ValueError("Incidence angle must be <= 90 degrees")
        if (theta < 0).any():
            raise ValueError("Incidence angle must be >= 0 degrees")

        # Convert to radians
        theta_rad = deg2rad(theta)

        # Reshape for broadcasting: [n_angles, 1, 1, ...]
        n_dims = vp1.ndim
        reshape = [len(theta)] + [1] * n_dims
        theta_rad = theta_rad.reshape(*reshape)

        return func(self, vp1, vs1, rho1, vp2, vs2, rho2, theta_rad, **kwargs)

    return wrapper


class AVOModel(nn.Module):
    """
    Abstract base class for AVO reflectivity models.

    All AVO implementations should inherit from this class
    and implement the forward method.

    Attributes:
        _name: Model identifier string.
        _valid_angles: Tuple of (min, max) valid angles in degrees.

    Example:
        >>> class MyAVOModel(AVOModel):
        ...     def forward(self, vp1, vs1, rho1, vp2, vs2, rho2, theta):
        ...         # Custom implementation
        ...         return reflection_coefficients
    """

    def __init__(self):
        super().__init__()
        self._name = 'base'
        self._valid_angles = (0, 90)

    @abstractmethod
    def forward(
        self,
        vp1: Tensor,
        vs1: Tensor,
        rho1: Tensor,
        vp2: Tensor,
        vs2: Tensor,
        rho2: Tensor,
        theta: Tensor,
    ) -> Tensor:
        """
        Compute reflection coefficients.

        Args:
            vp1: P-wave velocity of upper layer (m/s).
            vs1: S-wave velocity of upper layer (m/s).
            rho1: Density of upper layer (kg/m³).
            vp2: P-wave velocity of lower layer (m/s).
            vs2: S-wave velocity of lower layer (m/s).
            rho2: Density of lower layer (kg/m³).
            theta: Incidence angles (radians).

        Returns:
            Reflection coefficients with shape [n_angles, *input_shape].
        """
        raise NotImplementedError

    def __call__(
        self,
        vp1, vs1, rho1,
        vp2, vs2, rho2,
        theta,
    ) -> Tensor:
        """Compute reflection coefficients (callable interface)."""
        return self.forward(vp1, vs1, rho1, vp2, vs2, rho2, theta)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(valid_angles={self._valid_angles})"

    @property
    def name(self) -> str:
        """Model name."""
        return self._name


def normal_incidence_rc(
    vp1: Union[Tensor, np.ndarray, float],
    rho1: Union[Tensor, np.ndarray, float],
    vp2: Union[Tensor, np.ndarray, float],
    rho2: Union[Tensor, np.ndarray, float],
) -> Tensor:
    """
    Compute normal incidence reflection coefficient.

    Simple impedance contrast formula:

        R = (Z₂ - Z₁) / (Z₂ + Z₁)

    where Z = vp × ρ is acoustic impedance.

    Args:
        vp1: P-wave velocity of upper layer (m/s).
        rho1: Density of upper layer (kg/m³).
        vp2: P-wave velocity of lower layer (m/s).
        rho2: Density of lower layer (kg/m³).

    Returns:
        Normal incidence reflection coefficient (scalar or tensor).

    Example:
        >>> rc = normal_incidence_rc(
        ...     vp1=2000, rho1=2.0,
        ...     vp2=2500, rho2=2.2
        ... )
        >>> print(f"R0 = {rc.item():.4f}")
    """
    vp1 = as_tensor(vp1)
    rho1 = as_tensor(rho1)
    vp2 = as_tensor(vp2)
    rho2 = as_tensor(rho2)

    # Acoustic impedance
    Z1 = vp1 * rho1
    Z2 = vp2 * rho2

    return (Z2 - Z1) / (Z2 + Z1 + EPS)