"""
Rock properties module for reservoir simulation.

Provides rock property classes including permeability,
porosity, and compressibility for reservoir simulation.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
import torch.nn as nn
from typing import Union, Optional

from .constants import DEVICE, DTYPE


# =============================================================================
# Rock Properties
# =============================================================================

class Rock(nn.Module):
    """
    Rock properties container for reservoir simulation.

    Stores permeability, porosity, and compressibility. Supports
    both isotropic and anisotropic permeability.

    Args:
        nc: Number of cells.
        device: Torch device.
        dtype: Torch dtype.

    Attributes:
        perm_x: Permeability in x-direction (md).
        perm_y: Permeability in y-direction (md).
        perm_z: Permeability in z-direction (md).
        poro: Porosity (fraction).
        comp: Rock compressibility (1/psi).
        pref: Reference pressure for compressibility (psi).

    Example:
        >>> rock = Rock(nc=450)
        >>> rock.set_permeability(kx=100.0, ky=100.0, kz=10.0)
        >>> rock.set_porosity(0.2)
        >>> rock.set_compressibility(1e-6, pref=4000.0)
    """

    def __init__(
        self,
        nc: int,
        device: torch.device = DEVICE,
        dtype: torch.dtype = DTYPE
    ):
        super().__init__()
        self.nc = nc
        self.device = device
        self.dtype = dtype

        # Permeability (md)
        self._perm_x: Optional[torch.Tensor] = None
        self._perm_y: Optional[torch.Tensor] = None
        self._perm_z: Optional[torch.Tensor] = None

        # Porosity
        self._poro: Optional[torch.Tensor] = None

        # Compressibility
        self._comp: float = 0.0
        self._pref: float = 0.0

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def perm_x(self) -> torch.Tensor:
        """Permeability in x-direction (md)."""
        if self._perm_x is None:
            raise ValueError("Permeability not set. Call set_permeability first.")
        return self._perm_x

    @perm_x.setter
    def perm_x(self, value: torch.Tensor) -> None:
        """Set x-direction permeability."""
        self._perm_x = value.to(device=self.device, dtype=self.dtype)

    @property
    def perm_y(self) -> torch.Tensor:
        """Permeability in y-direction (md)."""
        if self._perm_y is None:
            return self.perm_x  # Default to isotropic
        return self._perm_y

    @perm_y.setter
    def perm_y(self, value: torch.Tensor) -> None:
        """Set y-direction permeability."""
        self._perm_y = value.to(device=self.device, dtype=self.dtype)

    @property
    def perm_z(self) -> torch.Tensor:
        """Permeability in z-direction (md)."""
        if self._perm_z is None:
            return self.perm_x  # Default to isotropic
        return self._perm_z

    @perm_z.setter
    def perm_z(self, value: torch.Tensor) -> None:
        """Set z-direction permeability."""
        self._perm_z = value.to(device=self.device, dtype=self.dtype)

    @property
    def poro(self) -> torch.Tensor:
        """Porosity (fraction)."""
        if self._poro is None:
            raise ValueError("Porosity not set. Call set_porosity first.")
        return self._poro

    @poro.setter
    def poro(self, value: torch.Tensor) -> None:
        """Set porosity."""
        self._poro = value.to(device=self.device, dtype=self.dtype)

    @property
    def comp(self) -> float:
        """Rock compressibility (1/psi)."""
        return self._comp

    @property
    def pref(self) -> float:
        """Reference pressure for compressibility (psi)."""
        return self._pref

    # Aliases for compatibility
    @property
    def kx(self) -> torch.Tensor:
        """Alias for perm_x."""
        return self.perm_x

    @property
    def ky(self) -> torch.Tensor:
        """Alias for perm_y."""
        return self.perm_y

    @property
    def kz(self) -> torch.Tensor:
        """Alias for perm_z."""
        return self.perm_z

    # -------------------------------------------------------------------------
    # Configuration Methods
    # -------------------------------------------------------------------------

    def set_permeability(
        self,
        kx: Union[float, torch.Tensor],
        ky: Optional[Union[float, torch.Tensor]] = None,
        kz: Optional[Union[float, torch.Tensor]] = None
    ) -> 'Rock':
        """
        Set permeability values.

        Args:
            kx: Permeability in x-direction (md).
            ky: Permeability in y-direction (md), defaults to kx.
            kz: Permeability in z-direction (md), defaults to kx.

        Returns:
            Self for method chaining.
        """
        self._perm_x = self._to_tensor(kx)
        self._perm_y = self._to_tensor(ky) if ky is not None else None
        self._perm_z = self._to_tensor(kz) if kz is not None else None
        return self

    def set_perm(
        self,
        kx: Union[float, torch.Tensor],
        ky: Optional[Union[float, torch.Tensor]] = None,
        kz: Optional[Union[float, torch.Tensor]] = None
    ) -> 'Rock':
        """
        Alias for set_permeability.

        Args:
            kx: Permeability in x-direction (md).
            ky: Permeability in y-direction (md), defaults to kx.
            kz: Permeability in z-direction (md), defaults to kx.

        Returns:
            Self for method chaining.
        """
        return self.set_permeability(kx, ky, kz)

    def set_porosity(self, poro: Union[float, torch.Tensor]) -> 'Rock':
        """
        Set porosity values.

        Args:
            poro: Porosity (fraction, 0-1).

        Returns:
            Self for method chaining.
        """
        self._poro = self._to_tensor(poro)
        return self

    def set_poro(self, poro: Union[float, torch.Tensor]) -> 'Rock':
        """
        Alias for set_porosity.

        Args:
            poro: Porosity (fraction, 0-1).

        Returns:
            Self for method chaining.
        """
        return self.set_porosity(poro)

    def set_compressibility(self, comp: float, pref: float = 0.0) -> 'Rock':
        """
        Set rock compressibility.

        Args:
            comp: Rock compressibility (1/psi).
            pref: Reference pressure (psi).

        Returns:
            Self for method chaining.
        """
        self._comp = comp
        self._pref = pref
        return self

    # -------------------------------------------------------------------------
    # Computation Methods
    # -------------------------------------------------------------------------

    def get_pore_volume(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Compute pore volume.

        Args:
            volume: Cell bulk volume (ft³).

        Returns:
            Pore volume (ft³).
        """
        return self.poro * volume

    def get_porosity_at_pressure(self, pressure: torch.Tensor) -> torch.Tensor:
        """
        Compute pressure-dependent porosity.

        Uses linear compressibility model:
            φ(p) = φ_ref * (1 + c_r * (p - p_ref))

        Args:
            pressure: Current pressure (psi).

        Returns:
            Porosity at given pressure.
        """
        if self._comp == 0.0:
            return self.poro

        return self.poro * (1.0 + self._comp * (pressure - self._pref))

    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------

    def _to_tensor(self, value: Union[float, torch.Tensor]) -> torch.Tensor:
        """Convert scalar to tensor if needed."""
        if isinstance(value, (int, float)):
            return torch.full(
                (self.nc,), float(value),
                device=self.device, dtype=self.dtype
            )
        return value.to(device=self.device, dtype=self.dtype)

    def forward(self) -> None:
        """Forward pass (placeholder for nn.Module compatibility)."""
        pass

    def __repr__(self) -> str:
        return (
            f"Rock(nc={self.nc}, "
            f"kx_mean={self.perm_x.mean():.2f}, "
            f"poro_mean={self.poro.mean():.4f})"
        )
