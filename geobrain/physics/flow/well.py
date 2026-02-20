"""
Well module for reservoir simulation.

Provides well models for production and injection operations
in reservoir simulation, including rate and BHP controls.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
import torch.nn as nn
import math
from typing import Dict, List, Optional, Literal, Tuple
from dataclasses import dataclass, field

from .constants import DEVICE, DTYPE, ALPHA


# =============================================================================
# Type Aliases
# =============================================================================

ControlType = Literal['RATE', 'BHP']
WellType = Literal['PROD', 'INJ']
PhaseType = Literal['OIL', 'WATER', 'LIQ']


# =============================================================================
# Well Data Classes
# =============================================================================

@dataclass
class WellPerforation:
    """
    Well perforation (connection) data.

    Attributes:
        cell_idx: Global cell index.
        wi: Well index (md·ft).
        depth: Perforation depth (ft).
    """
    cell_idx: int
    wi: float
    depth: float = 0.0


@dataclass
class WellHistory:
    """
    Well production/injection history.

    Stores time series of rates and pressures.

    Attributes:
        time: Time points (days).
        qo: Oil rate history (STB/D).
        qw: Water rate history (STB/D).
        bhp: Bottom-hole pressure history (psi).
        wct: Water cut history (fraction).
    """
    time: List[float] = field(default_factory=list)
    qo: List[float] = field(default_factory=list)
    qw: List[float] = field(default_factory=list)
    bhp: List[float] = field(default_factory=list)
    wct: List[float] = field(default_factory=list)

    def append(self, t: float, qo: float, qw: float, bhp: float) -> None:
        """
        Append a timestep record.

        Args:
            t: Time (days).
            qo: Oil rate (STB/D).
            qw: Water rate (STB/D).
            bhp: Bottom-hole pressure (psi).
        """
        self.time.append(t)
        self.qo.append(qo)
        self.qw.append(qw)
        self.bhp.append(bhp)
        total = abs(qo) + abs(qw)
        self.wct.append(abs(qw) / total if total > 0 else 0.0)

    def to_dict(self) -> Dict[str, List[float]]:
        """
        Convert to dictionary.

        Returns:
            Dictionary with keys: time, qo, qw, bhp, wct.
        """
        return {
            'time': self.time,
            'qo': self.qo,
            'qw': self.qw,
            'bhp': self.bhp,
            'wct': self.wct
        }


# =============================================================================
# Well Class
# =============================================================================

class Well(nn.Module):
    """
    Well model for reservoir simulation.

    Supports production and injection wells with rate or BHP control.
    Uses Peaceman well model for well-reservoir coupling.

    Args:
        name: Well name.
        well_type: 'PROD' for producer, 'INJ' for injector.
        device: Torch device.
        dtype: Torch dtype.

    Example:
        >>> prod = Well('P1', well_type='PROD')
        >>> prod.add_perforation(cell_idx=225, wi=5.0)
        >>> prod.set_control('RATE', target=-500.0, phase='LIQ')
    """

    def __init__(
        self,
        name: str,
        well_type: WellType = 'PROD',
        device: torch.device = DEVICE,
        dtype: torch.dtype = DTYPE
    ):
        super().__init__()
        self.name = name
        self.well_type = well_type
        self.device = device
        self.dtype = dtype

        # Perforations
        self._perforations: List[WellPerforation] = []

        # Control
        self._control_type: ControlType = 'RATE'
        self._target: float = 0.0
        self._control_phase: PhaseType = 'LIQ'
        self._bhp_limit: Optional[float] = None

        # State
        self._bhp: float = 0.0
        self._qo: torch.Tensor = None
        self._qw: torch.Tensor = None

        # History
        self.history = WellHistory()

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def nperf(self) -> int:
        """Number of perforations."""
        return len(self._perforations)

    @property
    def cell_indices(self) -> List[int]:
        """Cell indices of all perforations."""
        return [p.cell_idx for p in self._perforations]

    @property
    def well_indices(self) -> torch.Tensor:
        """Well indices tensor (md·ft)."""
        wi = [p.wi for p in self._perforations]
        return torch.tensor(wi, device=self.device, dtype=self.dtype)

    @property
    def control_type(self) -> ControlType:
        """Current control type."""
        return self._control_type

    @property
    def target(self) -> float:
        """Control target value."""
        return self._target

    @property
    def bhp(self) -> float:
        """Bottom-hole pressure (psi)."""
        return self._bhp

    @bhp.setter
    def bhp(self, value: float) -> None:
        """Set bottom-hole pressure."""
        self._bhp = value

    @property
    def qo(self) -> float:
        """Total oil rate (STB/D), positive for production."""
        if self._qo is None:
            return 0.0
        return self._qo.sum().item()

    @property
    def qw(self) -> float:
        """Total water rate (STB/D), positive for production."""
        if self._qw is None:
            return 0.0
        return self._qw.sum().item()

    @property
    def qliq(self) -> float:
        """Liquid rate (STB/D)."""
        return self.qo + self.qw

    # -------------------------------------------------------------------------
    # Configuration Methods
    # -------------------------------------------------------------------------

    def add_perforation(
        self,
        cell_idx: int,
        wi: float,
        depth: float = 0.0
    ) -> 'Well':
        """
        Add a perforation.

        Args:
            cell_idx: Global cell index.
            wi: Well index (md·ft).
            depth: Perforation depth (ft).

        Returns:
            Self for method chaining.
        """
        self._perforations.append(WellPerforation(cell_idx, wi, depth))
        return self

    def set_control(
        self,
        control_type: ControlType,
        target: float,
        phase: PhaseType = 'LIQ',
        bhp_limit: Optional[float] = None
    ) -> 'Well':
        """
        Set well control.

        Args:
            control_type: 'RATE' or 'BHP'.
            target: Target value (STB/D or psi).
            phase: Control phase for rate control.
            bhp_limit: BHP limit for rate control.

        Returns:
            Self for method chaining.
        """
        self._control_type = control_type
        self._target = target
        self._control_phase = phase
        self._bhp_limit = bhp_limit
        return self

    # -------------------------------------------------------------------------
    # Computation Methods
    # -------------------------------------------------------------------------

    def compute_rates(
        self,
        po: torch.Tensor,
        sw: torch.Tensor,
        bo: torch.Tensor,
        bw: torch.Tensor,
        muo: torch.Tensor,
        muw: torch.Tensor,
        kro: torch.Tensor,
        krw: torch.Tensor,
        kx: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute well rates for current state.

        Uses ALPHA constant and Peaceman well model.

        Args:
            po: Oil pressure (psi), shape (nc,).
            sw: Water saturation, shape (nc,).
            bo: Oil FVF, shape (nc,).
            bw: Water FVF, shape (nc,).
            muo: Oil viscosity (cp), shape (nc,).
            muw: Water viscosity (cp), shape (nc,).
            kro: Oil relative permeability, shape (nc,).
            krw: Water relative permeability, shape (nc,).
            kx: X-permeability (md), shape (nc,).

        Returns:
            Dict with 'qo', 'qw' arrays (nc,) and 'bhp' (float).
        """
        nc = po.shape[0]
        qo_total = torch.zeros(nc, device=self.device, dtype=self.dtype)
        qw_total = torch.zeros(nc, device=self.device, dtype=self.dtype)

        if self.nperf == 0:
            self._qo = torch.zeros(0, device=self.device, dtype=self.dtype)
            self._qw = torch.zeros(0, device=self.device, dtype=self.dtype)
            return {'qo': qo_total, 'qw': qw_total, 'bhp': self._bhp}

        wi_tensor = self.well_indices
        cell_idx = self.cell_indices

        # Mobility calculation: lam = kr / (mu * B)
        lam_o = kro[cell_idx] / (muo[cell_idx] * bo[cell_idx] + 1e-30)
        lam_w = krw[cell_idx] / (muw[cell_idx] * bw[cell_idx] + 1e-30)

        if self._control_type == 'BHP':
            pwf = self._target
            self._bhp = pwf

            if self.well_type == 'PROD':
                # Producer: positive rate out of reservoir
                dp = po[cell_idx] - pwf
                dp = torch.clamp(dp, min=0.0)
                qo_perf = wi_tensor * lam_o * dp
                qw_perf = wi_tensor * lam_w * dp
            else:
                # Injector: negative dp means injection
                dp = po[cell_idx] - pwf
                dp = torch.clamp(dp, max=0.0)
                qo_perf = torch.zeros_like(dp)
                qw_perf = wi_tensor * (lam_o + lam_w) * dp

        else:  # Rate control
            target_rate = abs(self._target)
            total_lam = lam_o + lam_w

            if self._control_phase == 'OIL':
                qo_perf = torch.full_like(lam_o, target_rate / len(cell_idx))
                qw_perf = (lam_w / (lam_o + 1e-10)) * qo_perf
            elif self._control_phase == 'WATER':
                if self.well_type == 'INJ':
                    qo_perf = torch.zeros_like(lam_o)
                    qw_perf = torch.full_like(lam_w, -target_rate / len(cell_idx))
                else:
                    qw_perf = torch.full_like(lam_w, target_rate / len(cell_idx))
                    qo_perf = (lam_o / (lam_w + 1e-10)) * qw_perf
            else:  # LIQ
                qo_perf = (lam_o / (total_lam + 1e-10)) * target_rate / len(cell_idx)
                qw_perf = (lam_w / (total_lam + 1e-10)) * target_rate / len(cell_idx)

            # Compute BHP from rate
            total_rate = qo_perf.sum() + qw_perf.sum()
            pi_total = (wi_tensor * total_lam).sum()
            p_avg = po[cell_idx].mean()

            if pi_total > 1e-30:
                self._bhp = p_avg - total_rate / pi_total
            else:
                self._bhp = p_avg

            # Apply BHP limit
            if self._bhp_limit is not None:
                if self.well_type == 'PROD' and self._bhp < self._bhp_limit:
                    self._bhp = self._bhp_limit
                    dp = po[cell_idx] - self._bhp
                    dp = torch.clamp(dp, min=0.0)
                    qo_perf = wi_tensor * lam_o * dp
                    qw_perf = wi_tensor * lam_w * dp
                elif self.well_type == 'INJ' and self._bhp > self._bhp_limit:
                    self._bhp = self._bhp_limit
                    dp = po[cell_idx] - self._bhp
                    dp = torch.clamp(dp, max=0.0)
                    qo_perf = torch.zeros_like(dp)
                    qw_perf = wi_tensor * (lam_o + lam_w) * dp

        # Store per-perforation rates
        self._qo = qo_perf
        self._qw = qw_perf

        # Scatter to cell arrays
        for i, idx in enumerate(cell_idx):
            qo_total[idx] += qo_perf[i]
            qw_total[idx] += qw_perf[i]

        return {'qo': qo_total, 'qw': qw_total, 'bhp': self._bhp}

    def get_rates_tensor(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get per-perforation rate tensors.

        Returns:
            Tuple of (qo, qw), each shape (nperf,).
        """
        if self._qo is None:
            return (
                torch.zeros(self.nperf, device=self.device, dtype=self.dtype),
                torch.zeros(self.nperf, device=self.device, dtype=self.dtype)
            )
        return self._qo, self._qw

    def record_state(self, time: float) -> None:
        """
        Record current state to history.

        Args:
            time: Current simulation time (days).
        """
        self.history.append(time, self.qo, self.qw, self._bhp)

    def reset_history(self) -> None:
        """Clear history."""
        self.history = WellHistory()

    def forward(self) -> None:
        """Forward pass (placeholder for nn.Module compatibility)."""
        pass

    def __repr__(self) -> str:
        return (
            f"Well(name='{self.name}', type={self.well_type}, "
            f"nperf={self.nperf}, control={self._control_type}, "
            f"target={self._target:.1f})"
        )


# =============================================================================
# Well Group
# =============================================================================

class WellGroup(nn.Module):
    """
    Collection of wells.

    Args:
        device: Torch device.
        dtype: Torch dtype.

    Example:
        >>> wells = WellGroup()
        >>> wells.add_well(prod)
        >>> wells.add_well(inj)
    """

    def __init__(
        self,
        device: torch.device = DEVICE,
        dtype: torch.dtype = DTYPE
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self._wells: Dict[str, Well] = {}

    @property
    def wells(self) -> Dict[str, Well]:
        """Dictionary of wells."""
        return self._wells

    @property
    def nwell(self) -> int:
        """Number of wells."""
        return len(self._wells)

    def add_well(self, well: Well) -> 'WellGroup':
        """
        Add a well.

        Args:
            well: Well to add.

        Returns:
            Self for method chaining.
        """
        self._wells[well.name] = well
        return self

    def get_well(self, name: str) -> Well:
        """
        Get well by name.

        Args:
            name: Well name.

        Returns:
            Well instance.

        Raises:
            KeyError: If well not found.
        """
        return self._wells[name]

    def compute_all_rates(
        self,
        po: torch.Tensor,
        sw: torch.Tensor,
        bo: torch.Tensor,
        bw: torch.Tensor,
        muo: torch.Tensor,
        muw: torch.Tensor,
        kro: torch.Tensor,
        krw: torch.Tensor,
        kx: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute rates for all wells.

        Args:
            po: Oil pressure (psi), shape (nc,).
            sw: Water saturation, shape (nc,).
            bo: Oil FVF, shape (nc,).
            bw: Water FVF, shape (nc,).
            muo: Oil viscosity (cp), shape (nc,).
            muw: Water viscosity (cp), shape (nc,).
            kro: Oil relative permeability, shape (nc,).
            krw: Water relative permeability, shape (nc,).
            kx: X-permeability (md), shape (nc,).

        Returns:
            Dict with total 'qo', 'qw' arrays (nc,).
        """
        nc = po.shape[0]
        total_qo = torch.zeros(nc, device=self.device, dtype=self.dtype)
        total_qw = torch.zeros(nc, device=self.device, dtype=self.dtype)

        for well in self._wells.values():
            rates = well.compute_rates(po, sw, bo, bw, muo, muw, kro, krw, kx)
            total_qo += rates['qo']
            total_qw += rates['qw']

        return {'qo': total_qo, 'qw': total_qw}

    def record_all_states(self, time: float) -> None:
        """
        Record state for all wells.

        Args:
            time: Current simulation time (days).
        """
        for well in self._wells.values():
            well.record_state(time)

    def get_all_histories(self) -> Dict[str, Dict]:
        """
        Get histories for all wells.

        Returns:
            Dictionary mapping well names to history dicts.
        """
        return {name: well.history.to_dict() for name, well in self._wells.items()}

    def forward(self) -> None:
        """Forward pass (placeholder for nn.Module compatibility)."""
        pass

    def __repr__(self) -> str:
        well_names = list(self._wells.keys())
        return f"WellGroup(nwell={self.nwell}, wells={well_names})"


# =============================================================================
# Well Index Calculation (Peaceman formula)
# =============================================================================

def compute_well_index(
    kx: float,
    ky: float,
    dx: float,
    dy: float,
    dz: float,
    rw: float,
    skin: float = 0.0
) -> float:
    """
    Compute well index using Peaceman formula.

    WI = 2 * π * ALPHA * sqrt(kx * ky) * dz / (ln(r0/rw) + skin)

    where r0 = 0.28 * sqrt(sqrt(ky/kx)*dx² + sqrt(kx/ky)*dy²) /
               (pow(ky/kx, 0.25) + pow(kx/ky, 0.25))

    Args:
        kx: X-permeability (md).
        ky: Y-permeability (md).
        dx: Cell size in x (ft).
        dy: Cell size in y (ft).
        dz: Cell size in z / perforation length (ft).
        rw: Wellbore radius (ft).
        skin: Skin factor.

    Returns:
        Well index (md·ft).

    Example:
        >>> wi = compute_well_index(kx=100, ky=100, dx=100, dy=100, dz=20, rw=0.5)
    """
    # Peaceman equivalent radius
    r0 = 0.28 * math.sqrt(
        math.sqrt(ky/kx) * dx**2 + math.sqrt(kx/ky) * dy**2
    ) / (math.pow(ky/kx, 0.25) + math.pow(kx/ky, 0.25))

    # Well index
    wi = 2.0 * ALPHA * math.pi * math.sqrt(kx * ky) * dz / (math.log(r0 / rw) + skin)

    return wi
