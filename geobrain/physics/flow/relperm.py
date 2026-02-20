"""
Relative permeability module for reservoir simulation.

Provides relative permeability models including tabulated (SWOF)
and analytical (Corey) formulations.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

from .constants import DEVICE, DTYPE, SMALL_SAT


# =============================================================================
# Abstract Base Class
# =============================================================================

class RelPerm(ABC, nn.Module):
    """
    Abstract base class for relative permeability models.

    Defines interface for computing oil and water relative
    permeabilities as functions of water saturation.

    Args:
        device: Torch device.
        dtype: Torch dtype.
    """

    def __init__(self, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.device = device
        self.dtype = dtype

    @abstractmethod
    def get_kro(self, sw: torch.Tensor) -> torch.Tensor:
        """
        Compute oil relative permeability.

        Args:
            sw: Water saturation (fraction), shape [n].

        Returns:
            Oil relative permeability kro, shape [n].
        """
        pass

    @abstractmethod
    def get_krw(self, sw: torch.Tensor) -> torch.Tensor:
        """
        Compute water relative permeability.

        Args:
            sw: Water saturation (fraction), shape [n].

        Returns:
            Water relative permeability krw, shape [n].
        """
        pass

    def get_kr(self, sw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute both oil and water relative permeabilities.

        Args:
            sw: Water saturation (fraction), shape [n].

        Returns:
            Tuple of (kro, krw), each shape [n].
        """
        return self.get_kro(sw), self.get_krw(sw)

    @property
    @abstractmethod
    def swc(self) -> float:
        """Connate water saturation."""
        pass

    @property
    @abstractmethod
    def sor(self) -> float:
        """Residual oil saturation."""
        pass


# =============================================================================
# Tabulated Relative Permeability (SWOF-style)
# =============================================================================

class RelPermTable(RelPerm):
    """
    Tabulated relative permeability using linear interpolation.

    Interpolates relative permeabilities from saturation-dependent
    tables (SWOF format). Oil relative permeability is given as
    function of Sw in SWOF format.

    Args:
        sw_table: Water saturation values.
        krw_table: Water relative permeability values.
        kro_table: Oil relative permeability values.
        device: Torch device.
        dtype: Torch dtype.

    Example:
        >>> relperm = RelPermTable(
        ...     sw_table=[0.2, 0.3, 0.5, 0.7, 0.8],
        ...     krw_table=[0.0, 0.05, 0.2, 0.5, 0.7],
        ...     kro_table=[1.0, 0.7, 0.3, 0.05, 0.0]
        ... )
        >>> kro, krw = relperm.get_kr(sw)
    """

    def __init__(
        self,
        sw_table: Union[List[float], torch.Tensor],
        krw_table: Union[List[float], torch.Tensor],
        kro_table: Union[List[float], torch.Tensor],
        device: torch.device = DEVICE,
        dtype: torch.dtype = DTYPE
    ):
        super().__init__(device, dtype)

        # Register tables as buffers
        self.register_buffer('sw_table', self._to_tensor(sw_table))
        self.register_buffer('krw_table', self._to_tensor(krw_table))
        self.register_buffer('kro_table', self._to_tensor(kro_table))

        # Extract endpoint saturations
        self._swc = float(self.sw_table[0])
        self._sor = 1.0 - float(self.sw_table[-1])

        # For oil kr lookup, we need So table sorted ascending
        self.register_buffer('so_table', 1.0 - self.sw_table)

    @classmethod
    def from_data(
        cls,
        data: List[List[float]],
        device: torch.device = DEVICE,
        dtype: torch.dtype = DTYPE
    ) -> 'RelPermTable':
        """
        Create RelPermTable from SWOF-style data.

        Args:
            data: List of [Sw, Krw, Kro, Pc] rows. Pc is ignored.
            device: Torch device.
            dtype: Torch dtype.

        Returns:
            RelPermTable instance.
        """
        sw_table = [row[0] for row in data]
        krw_table = [row[1] for row in data]
        kro_table = [row[2] for row in data]
        return cls(sw_table, krw_table, kro_table, device, dtype)

    @property
    def swc(self) -> float:
        """Connate water saturation."""
        return self._swc

    @property
    def sor(self) -> float:
        """Residual oil saturation."""
        return self._sor

    def get_kro(self, sw: torch.Tensor) -> torch.Tensor:
        """
        Compute oil relative permeability by interpolation.

        kro is given vs Sw in SWOF table, so we interpolate
        directly using Sw.

        Args:
            sw: Water saturation (fraction), shape [n].

        Returns:
            Oil relative permeability kro, shape [n].
        """
        return self._interpolate(sw, self.sw_table, self.kro_table)

    def get_krw(self, sw: torch.Tensor) -> torch.Tensor:
        """
        Compute water relative permeability by interpolation.

        Args:
            sw: Water saturation (fraction), shape [n].

        Returns:
            Water relative permeability krw, shape [n].
        """
        return self._interpolate(sw, self.sw_table, self.krw_table)

    def _interpolate(
        self,
        x: torch.Tensor,
        x_table: torch.Tensor,
        y_table: torch.Tensor
    ) -> torch.Tensor:
        """
        Linear interpolation with clamping.

        Args:
            x: Values to interpolate.
            x_table: Independent variable table.
            y_table: Dependent variable table.

        Returns:
            Interpolated values.
        """
        # Clamp to table range
        x_clamped = torch.clamp(x, x_table[0], x_table[-1])

        # Find interpolation indices
        idx = torch.searchsorted(x_table, x_clamped.contiguous(), right=True) - 1
        idx = torch.clamp(idx, 0, len(x_table) - 2)

        # Get bracketing values
        x_lo = x_table[idx]
        x_hi = x_table[idx + 1]
        y_lo = y_table[idx]
        y_hi = y_table[idx + 1]

        # Linear interpolation
        weight = (x_clamped - x_lo) / (x_hi - x_lo + 1e-30)
        return y_lo + weight * (y_hi - y_lo)

    def _to_tensor(self, value: Union[List[float], torch.Tensor]) -> torch.Tensor:
        """Convert list to tensor if needed."""
        if isinstance(value, list):
            return torch.tensor(value, device=self.device, dtype=self.dtype)
        return value.to(device=self.device, dtype=self.dtype)

    def forward(self, sw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returns (kro, krw).

        Args:
            sw: Water saturation (fraction), shape [n].

        Returns:
            Tuple of (kro, krw), each shape [n].
        """
        return self.get_kr(sw)


# =============================================================================
# Corey Relative Permeability (Analytical)
# =============================================================================

class RelPermCorey(RelPerm):
    """
    Corey relative permeability model.

    Uses power-law relationships:
        krw = krw_max * Se^nw
        kro = kro_max * (1 - Se)^no

    where Se = (Sw - Swc) / (1 - Swc - Sor) is normalized saturation.

    Args:
        swc: Connate water saturation.
        sor: Residual oil saturation.
        krw_max: Maximum water relative permeability (default 1.0).
        kro_max: Maximum oil relative permeability (default 1.0).
        nw: Water Corey exponent (default 2.0).
        no: Oil Corey exponent (default 2.0).
        device: Torch device.
        dtype: Torch dtype.

    Example:
        >>> relperm = RelPermCorey(swc=0.2, sor=0.2, nw=2.0, no=2.0)
        >>> kro, krw = relperm.get_kr(sw)
    """

    def __init__(
        self,
        swc: float,
        sor: float,
        krw_max: float = 1.0,
        kro_max: float = 1.0,
        nw: float = 2.0,
        no: float = 2.0,
        device: torch.device = DEVICE,
        dtype: torch.dtype = DTYPE
    ):
        super().__init__(device, dtype)

        self._swc = swc
        self._sor = sor
        self.krw_max = krw_max
        self.kro_max = kro_max
        self.nw = nw
        self.no = no

        # Normalized saturation range
        self._delta_s = 1.0 - swc - sor

    @property
    def swc(self) -> float:
        """Connate water saturation."""
        return self._swc

    @property
    def sor(self) -> float:
        """Residual oil saturation."""
        return self._sor

    def get_kro(self, sw: torch.Tensor) -> torch.Tensor:
        """
        Compute oil relative permeability.

        kro = kro_max * ((1 - Sw - Sor) / (1 - Swc - Sor))^no
            = kro_max * (1 - Se)^no

        Args:
            sw: Water saturation (fraction), shape [n].

        Returns:
            Oil relative permeability kro, shape [n].
        """
        se = self._normalized_saturation(sw)
        return self.kro_max * torch.pow(1.0 - se, self.no)

    def get_krw(self, sw: torch.Tensor) -> torch.Tensor:
        """
        Compute water relative permeability.

        krw = krw_max * ((Sw - Swc) / (1 - Swc - Sor))^nw
            = krw_max * Se^nw

        Args:
            sw: Water saturation (fraction), shape [n].

        Returns:
            Water relative permeability krw, shape [n].
        """
        se = self._normalized_saturation(sw)
        return self.krw_max * torch.pow(se, self.nw)

    def get_kro_derivative(self, sw: torch.Tensor) -> torch.Tensor:
        """
        Compute dkro/dSw analytically.

        Args:
            sw: Water saturation (fraction), shape [n].

        Returns:
            dkro/dSw, shape [n].
        """
        se = self._normalized_saturation(sw)
        se_clamped = torch.clamp(se, SMALL_SAT, 1.0 - SMALL_SAT)
        return -self.kro_max * self.no * torch.pow(
            1.0 - se_clamped, self.no - 1
        ) / self._delta_s

    def get_krw_derivative(self, sw: torch.Tensor) -> torch.Tensor:
        """
        Compute dkrw/dSw analytically.

        Args:
            sw: Water saturation (fraction), shape [n].

        Returns:
            dkrw/dSw, shape [n].
        """
        se = self._normalized_saturation(sw)
        se_clamped = torch.clamp(se, SMALL_SAT, 1.0 - SMALL_SAT)
        return self.krw_max * self.nw * torch.pow(
            se_clamped, self.nw - 1
        ) / self._delta_s

    def _normalized_saturation(self, sw: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized saturation Se.

        Args:
            sw: Water saturation (fraction), shape [n].

        Returns:
            Normalized saturation Se, shape [n].
        """
        se = (sw - self._swc) / (self._delta_s + 1e-30)
        return torch.clamp(se, 0.0, 1.0)

    def forward(self, sw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returns (kro, krw).

        Args:
            sw: Water saturation (fraction), shape [n].

        Returns:
            Tuple of (kro, krw), each shape [n].
        """
        return self.get_kr(sw)
