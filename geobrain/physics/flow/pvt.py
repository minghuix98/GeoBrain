"""
PVT (Pressure-Volume-Temperature) module for reservoir simulation.

Provides PVT property models including tabulated and analytical
formulations for fluid properties.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Optional, Union

from .constants import DEVICE, DTYPE


# =============================================================================
# Abstract Base Class
# =============================================================================

class PVT(ABC, nn.Module):
    """
    Abstract base class for PVT models.

    Defines interface for computing formation volume factor (B)
    and viscosity (μ) as functions of pressure.

    Args:
        device: Torch device.
        dtype: Torch dtype.
    """

    def __init__(self, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.device = device
        self.dtype = dtype

    @abstractmethod
    def get_b(self, p: torch.Tensor) -> torch.Tensor:
        """
        Compute formation volume factor.

        Args:
            p: Pressure (psi), shape [n].

        Returns:
            Formation volume factor B (rb/stb), shape [n].
        """
        pass

    @abstractmethod
    def get_mu(self, p: torch.Tensor) -> torch.Tensor:
        """
        Compute viscosity.

        Args:
            p: Pressure (psi), shape [n].

        Returns:
            Viscosity μ (cp), shape [n].
        """
        pass

    def get_b_derivative(self, p: torch.Tensor) -> torch.Tensor:
        """
        Compute derivative of B with respect to pressure.

        Default implementation uses autograd.

        Args:
            p: Pressure (psi), shape [n].

        Returns:
            dB/dp (rb/stb/psi), shape [n].
        """
        p_grad = p.clone().requires_grad_(True)
        b = self.get_b(p_grad)
        db_dp = torch.autograd.grad(
            b.sum(), p_grad, create_graph=True
        )[0]
        return db_dp


# =============================================================================
# Tabulated PVT (PVDO-style)
# =============================================================================

class PVTTable(PVT):
    """
    Tabulated PVT properties using linear interpolation.

    Interpolates formation volume factor and viscosity from
    pressure-dependent tables (PVDO format).

    Args:
        p_table: Pressure values (psi).
        b_table: Formation volume factor values (rb/stb).
        mu_table: Viscosity values (cp).
        device: Torch device.
        dtype: Torch dtype.

    Example:
        >>> pvt = PVTTable(
        ...     p_table=[100, 500, 1000, 2000, 4000],
        ...     b_table=[1.05, 1.04, 1.03, 1.02, 1.01],
        ...     mu_table=[2.0, 1.8, 1.5, 1.2, 1.0]
        ... )
        >>> b = pvt.get_b(pressure)
    """

    def __init__(
        self,
        p_table: Union[List[float], torch.Tensor],
        b_table: Union[List[float], torch.Tensor],
        mu_table: Union[List[float], torch.Tensor],
        device: torch.device = DEVICE,
        dtype: torch.dtype = DTYPE
    ):
        super().__init__(device, dtype)

        # Register tables as buffers (not parameters)
        self.register_buffer('p_table', self._to_tensor(p_table))
        self.register_buffer('b_table', self._to_tensor(b_table))
        self.register_buffer('mu_table', self._to_tensor(mu_table))

    @classmethod
    def from_data(
        cls,
        data: List[List[float]],
        device: torch.device = DEVICE,
        dtype: torch.dtype = DTYPE
    ) -> 'PVTTable':
        """
        Create PVTTable from PVDO-style data.

        Args:
            data: List of [pressure, B, viscosity] rows.
            device: Torch device.
            dtype: Torch dtype.

        Returns:
            PVTTable instance.
        """
        p_table = [row[0] for row in data]
        b_table = [row[1] for row in data]
        mu_table = [row[2] for row in data]
        return cls(p_table, b_table, mu_table, device, dtype)

    def get_b(self, p: torch.Tensor) -> torch.Tensor:
        """
        Compute formation volume factor by interpolation.

        Args:
            p: Pressure (psi), shape [n].

        Returns:
            Formation volume factor B (rb/stb), shape [n].
        """
        return self._interpolate(p, self.b_table)

    def get_mu(self, p: torch.Tensor) -> torch.Tensor:
        """
        Compute viscosity by interpolation.

        Args:
            p: Pressure (psi), shape [n].

        Returns:
            Viscosity μ (cp), shape [n].
        """
        return self._interpolate(p, self.mu_table)

    def _interpolate(self, p: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        """
        Linear interpolation with extrapolation clamping.

        Uses searchsorted with right=True and subtracts 1.

        Args:
            p: Pressure values to interpolate.
            table: Table of values to interpolate from.

        Returns:
            Interpolated values.
        """
        # Clamp pressure to table range
        p_clamped = torch.clamp(p, self.p_table[0], self.p_table[-1])

        # Find interpolation indices
        idx = torch.searchsorted(self.p_table, p_clamped.contiguous(), right=True) - 1
        idx = torch.clamp(idx, 0, len(self.p_table) - 2)

        # Get bracketing values
        p_lo = self.p_table[idx]
        p_hi = self.p_table[idx + 1]
        v_lo = table[idx]
        v_hi = table[idx + 1]

        # Linear interpolation
        weight = (p_clamped - p_lo) / (p_hi - p_lo + 1e-10)
        return v_lo + weight * (v_hi - v_lo)

    def _to_tensor(self, value: Union[List[float], torch.Tensor]) -> torch.Tensor:
        """Convert list to tensor if needed."""
        if isinstance(value, list):
            return torch.tensor(value, device=self.device, dtype=self.dtype)
        return value.to(device=self.device, dtype=self.dtype)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returns formation volume factor.

        Args:
            p: Pressure (psi), shape [n].

        Returns:
            Formation volume factor B (rb/stb), shape [n].
        """
        return self.get_b(p)


# =============================================================================
# Analytical PVT (Constant Compressibility)
# =============================================================================

class PVTAnalytic(PVT):
    """
    Analytical PVT model with constant compressibility.

    Uses rational model for slightly compressible fluids:
        B(p) = B_ref / (1 + c*(p-p_ref) + 0.5*c²*(p-p_ref)²)
        μ(p) = μ_ref / (1 + cμ*(p-p_ref) + 0.5*cμ²*(p-p_ref)²)

    For incompressible fluids (c=0):
        B(p) = B_ref (constant)
        μ(p) = μ_ref (constant)

    Args:
        b_ref: Reference formation volume factor (rb/stb).
        mu_ref: Reference viscosity (cp).
        comp: Fluid compressibility (1/psi).
        p_ref: Reference pressure (psi).
        mu_comp: Viscosity compressibility (1/psi), defaults to 0.
        device: Torch device.
        dtype: Torch dtype.

    Example:
        >>> # Incompressible water
        >>> pvt_w = PVTAnalytic(b_ref=1.0, mu_ref=0.5, comp=0.0)
        >>>
        >>> # Slightly compressible oil
        >>> pvt_o = PVTAnalytic(
        ...     b_ref=1.2, mu_ref=2.0, comp=1e-5, p_ref=4000
        ... )
    """

    def __init__(
        self,
        b_ref: float,
        mu_ref: float,
        comp: float = 0.0,
        p_ref: float = 0.0,
        mu_comp: float = 0.0,
        device: torch.device = DEVICE,
        dtype: torch.dtype = DTYPE
    ):
        super().__init__(device, dtype)

        self.b_ref = b_ref
        self.mu_ref = mu_ref
        self.comp = comp
        self.p_ref = p_ref
        self.mu_comp = mu_comp

    @classmethod
    def from_data(
        cls,
        data: List[float],
        device: torch.device = DEVICE,
        dtype: torch.dtype = DTYPE
    ) -> 'PVTAnalytic':
        """
        Create PVTAnalytic from PVTW-style data.

        PVTW format: [p_ref, B_ref, comp, μ_ref, μ_comp]

        Args:
            data: PVTW data list.
            device: Torch device.
            dtype: Torch dtype.

        Returns:
            PVTAnalytic instance.
        """
        # Handle both [[...]] and [...] formats
        if isinstance(data[0], (list, tuple)):
            row = data[0]
        else:
            row = data

        p_ref = row[0] if len(row) > 0 else 0.0
        b_ref = row[1] if len(row) > 1 else 1.0
        comp = row[2] if len(row) > 2 else 0.0
        mu_ref = row[3] if len(row) > 3 else 1.0
        mu_comp = row[4] if len(row) > 4 else 0.0

        return cls(b_ref, mu_ref, comp, p_ref, mu_comp, device, dtype)

    def get_b(self, p: torch.Tensor) -> torch.Tensor:
        """
        Compute formation volume factor.

        B = B_ref / (1 + c*(p-p_ref) + 0.5*c²*(p-p_ref)²)

        Args:
            p: Pressure (psi), shape [n].

        Returns:
            Formation volume factor B (rb/stb), shape [n].
        """
        if self.comp == 0.0:
            return torch.full_like(p, self.b_ref)

        dp = p - self.p_ref
        return self.b_ref / (1.0 + self.comp * dp + 0.5 * self.comp**2 * dp**2)

    def get_mu(self, p: torch.Tensor) -> torch.Tensor:
        """
        Compute viscosity.

        μ = μ_ref / (1 + cμ*(p-p_ref) + 0.5*cμ²*(p-p_ref)²)

        Args:
            p: Pressure (psi), shape [n].

        Returns:
            Viscosity μ (cp), shape [n].
        """
        if self.mu_comp == 0.0:
            return torch.full_like(p, self.mu_ref)

        dp = p - self.p_ref
        return self.mu_ref / (1.0 + self.mu_comp * dp + 0.5 * self.mu_comp**2 * dp**2)

    def get_b_derivative(self, p: torch.Tensor) -> torch.Tensor:
        """
        Compute dB/dp analytically.

        Args:
            p: Pressure (psi), shape [n].

        Returns:
            dB/dp (rb/stb/psi), shape [n].
        """
        if self.comp == 0.0:
            return torch.zeros_like(p)

        dp = p - self.p_ref
        denom = 1.0 + self.comp * dp + 0.5 * self.comp**2 * dp**2
        return -self.b_ref * (self.comp + self.comp**2 * dp) / (denom**2)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returns formation volume factor.

        Args:
            p: Pressure (psi), shape [n].

        Returns:
            Formation volume factor B (rb/stb), shape [n].
        """
        return self.get_b(p)
