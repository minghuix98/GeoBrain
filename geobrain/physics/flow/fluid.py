"""
Fluid module for reservoir simulation.

Provides fluid state containers and oil-water two-phase
fluid system for reservoir simulation.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Dict

from .constants import DEVICE, DTYPE, M, BETA, G, GC
from .grid import ConnList
from .pvt import PVT
from .relperm import RelPerm


# =============================================================================
# Phase State Container
# =============================================================================

@dataclass
class PhaseState:
    """
    State container for a single phase.

    Stores all phase-dependent properties including pressure,
    saturation, PVT properties, and flux terms.

    Attributes:
        p: Phase pressure (psi).
        s: Phase saturation (fraction).
        b: Formation volume factor (rb/stb).
        mu: Viscosity (cp).
        kr: Relative permeability.
        rho: Density (lbm/ft³).
        mobility: Phase mobility kr/(μB).
        flux: Inter-cell flux (stb/day).
        upstream: Upstream cell indices for each connection.
        p_n: Previous timestep pressure.
        s_n: Previous timestep saturation.
        b_n: Previous timestep formation volume factor.
        rho_s: Standard condition density (lbm/ft³).
    """
    p: torch.Tensor
    s: torch.Tensor
    b: torch.Tensor = field(default=None)
    mu: torch.Tensor = field(default=None)
    kr: torch.Tensor = field(default=None)
    rho: torch.Tensor = field(default=None)
    mobility: torch.Tensor = field(default=None)
    flux: torch.Tensor = field(default=None)
    upstream: torch.Tensor = field(default=None)

    # Previous timestep values
    p_n: torch.Tensor = field(default=None)
    s_n: torch.Tensor = field(default=None)
    b_n: torch.Tensor = field(default=None)

    # Standard condition density
    rho_s: float = field(default=0.0)

    def clone(self) -> 'PhaseState':
        """
        Create a deep copy of the phase state.

        Returns:
            Deep copy of PhaseState.
        """
        return PhaseState(
            p=self.p.clone(),
            s=self.s.clone(),
            b=self.b.clone() if self.b is not None else None,
            mu=self.mu.clone() if self.mu is not None else None,
            kr=self.kr.clone() if self.kr is not None else None,
            rho=self.rho.clone() if self.rho is not None else None,
            mobility=self.mobility.clone() if self.mobility is not None else None,
            flux=self.flux.clone() if self.flux is not None else None,
            upstream=self.upstream.clone() if self.upstream is not None else None,
            p_n=self.p_n.clone() if self.p_n is not None else None,
            s_n=self.s_n.clone() if self.s_n is not None else None,
            b_n=self.b_n.clone() if self.b_n is not None else None,
            rho_s=self.rho_s
        )


# =============================================================================
# Oil-Water Two-Phase Fluid System
# =============================================================================

class OilWaterFluid(nn.Module):
    """
    Oil-water two-phase fluid system.

    Manages fluid states and computes phase properties for
    immiscible oil-water flow simulation.

    Args:
        nc: Number of cells.
        nconn: Number of connections.
        rho_o: Oil density at standard conditions (lbm/ft³).
        rho_w: Water density at standard conditions (lbm/ft³).
        pvt_o: Oil PVT model.
        pvt_w: Water PVT model.
        relperm: Relative permeability model.
        device: Torch device.
        dtype: Torch dtype.

    Example:
        >>> fluid = OilWaterFluid(
        ...     nc=450, nconn=800,
        ...     rho_o=45.0, rho_w=64.0,
        ...     pvt_o=pvt_oil, pvt_w=pvt_water,
        ...     relperm=relperm_model
        ... )
        >>> fluid.initialize(p_init=4000.0, sw_init=0.2)
    """

    def __init__(
        self,
        nc: int,
        nconn: int,
        rho_o: float,
        rho_w: float,
        pvt_o: PVT,
        pvt_w: PVT,
        relperm: RelPerm,
        device: torch.device = DEVICE,
        dtype: torch.dtype = DTYPE
    ):
        super().__init__()

        self.nc = nc
        self.nconn = nconn
        self.rho_o_std = rho_o
        self.rho_w_std = rho_w
        self.device = device
        self.dtype = dtype

        # PVT and relative permeability models (as submodules)
        self.pvt_o = pvt_o
        self.pvt_w = pvt_w
        self.relperm = relperm

        # Initialize phase states
        self.oil: Optional[PhaseState] = None
        self.water: Optional[PhaseState] = None

        # Residuals
        self.res_o: Optional[torch.Tensor] = None
        self.res_w: Optional[torch.Tensor] = None

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def initialize(
        self,
        p_init: torch.Tensor,
        sw_init: torch.Tensor
    ) -> None:
        """
        Initialize fluid states.

        Args:
            p_init: Initial pressure (psi), shape (nc,).
            sw_init: Initial water saturation, shape (nc,).
        """
        p_init = self._to_tensor(p_init)
        sw_init = self._to_tensor(sw_init)
        so_init = 1.0 - sw_init

        # Initialize oil phase
        self.oil = PhaseState(
            p=p_init.clone(),
            s=so_init.clone(),
            b=torch.zeros(self.nc, device=self.device, dtype=self.dtype),
            mu=torch.zeros(self.nc, device=self.device, dtype=self.dtype),
            kr=torch.zeros(self.nc, device=self.device, dtype=self.dtype),
            rho=torch.zeros(self.nc, device=self.device, dtype=self.dtype),
            mobility=torch.zeros(self.nc, device=self.device, dtype=self.dtype),
            flux=torch.zeros(self.nconn, device=self.device, dtype=self.dtype),
            upstream=torch.zeros(self.nconn, device=self.device, dtype=torch.long),
            p_n=p_init.clone(),
            s_n=so_init.clone(),
            b_n=torch.zeros(self.nc, device=self.device, dtype=self.dtype),
            rho_s=self.rho_o_std
        )

        # Initialize water phase
        self.water = PhaseState(
            p=p_init.clone(),
            s=sw_init.clone(),
            b=torch.zeros(self.nc, device=self.device, dtype=self.dtype),
            mu=torch.zeros(self.nc, device=self.device, dtype=self.dtype),
            kr=torch.zeros(self.nc, device=self.device, dtype=self.dtype),
            rho=torch.zeros(self.nc, device=self.device, dtype=self.dtype),
            mobility=torch.zeros(self.nc, device=self.device, dtype=self.dtype),
            flux=torch.zeros(self.nconn, device=self.device, dtype=self.dtype),
            upstream=torch.zeros(self.nconn, device=self.device, dtype=torch.long),
            p_n=p_init.clone(),
            s_n=sw_init.clone(),
            b_n=torch.zeros(self.nc, device=self.device, dtype=self.dtype),
            rho_s=self.rho_w_std
        )

        # Initialize residuals
        self.res_o = torch.zeros(self.nc, device=self.device, dtype=self.dtype)
        self.res_w = torch.zeros(self.nc, device=self.device, dtype=self.dtype)

        # Compute initial B values for previous timestep
        with torch.no_grad():
            self.oil.b_n = self.pvt_o.get_b(p_init)
            self.water.b_n = self.pvt_w.get_b(p_init)

    # -------------------------------------------------------------------------
    # Property Computation
    # -------------------------------------------------------------------------

    def compute_properties(self, connlist: ConnList) -> None:
        """
        Compute all phase properties.

        Updates PVT properties, relative permeabilities,
        mobilities, and fluxes for both phases.

        Args:
            connlist: Connection list for flux calculation.
        """
        # Oil phase properties
        self._compute_phase_pvt(self.oil, self.pvt_o, self.rho_o_std)
        # kro is function of Sw (water saturation)
        self.oil.kr = self.relperm.get_kro(self.water.s)
        # mobility = kr / (mu * B)
        self.oil.mobility = self.oil.kr / (self.oil.mu * self.oil.b + 1e-30)

        # Water phase properties
        self._compute_phase_pvt(self.water, self.pvt_w, self.rho_w_std)
        self.water.kr = self.relperm.get_krw(self.water.s)
        self.water.mobility = self.water.kr / (self.water.mu * self.water.b + 1e-30)

        # Compute fluxes
        self._compute_phase_flux(self.oil, connlist)
        self._compute_phase_flux(self.water, connlist)

    def _compute_phase_pvt(
        self,
        phase: PhaseState,
        pvt: PVT,
        rho_std: float
    ) -> None:
        """
        Compute PVT properties for a phase.

        Args:
            phase: Phase state to update.
            pvt: PVT model for this phase.
            rho_std: Standard condition density.
        """
        phase.b = pvt.get_b(phase.p)
        phase.mu = pvt.get_mu(phase.p)
        phase.rho = rho_std / phase.b

    def _compute_phase_flux(
        self,
        phase: PhaseState,
        connlist: ConnList
    ) -> None:
        """
        Compute inter-cell flux for a phase.

        Args:
            phase: Phase state to update.
            connlist: Connection list with transmissibilities.
        """
        l = connlist.left
        r = connlist.right

        # Average density for gravity
        rho_l = phase.rho[l]
        rho_r = phase.rho[r]
        gamma = BETA * G / GC * (rho_l + rho_r) / 2.0

        # Potential difference
        delta_psi = phase.p[l] - phase.p[r] - gamma * connlist.delta_depth

        # Upstream weighting
        delta_psi_val = delta_psi.detach()
        phase.upstream = torch.where(delta_psi_val > 0, l, r)

        # Flux = mobility_upstream * transmissibility * Δψ
        mob_up = phase.mobility[phase.upstream]
        phase.flux = mob_up * connlist.trans * delta_psi

    # -------------------------------------------------------------------------
    # State Management
    # -------------------------------------------------------------------------

    def update_timestep(self) -> None:
        """Store current state as previous timestep values."""
        # Oil phase
        self.oil.p_n = self.oil.p.detach().clone()
        self.oil.s_n = self.oil.s.detach().clone()
        self.oil.b_n = self.oil.b.detach().clone()

        # Water phase
        self.water.p_n = self.water.p.detach().clone()
        self.water.s_n = self.water.s.detach().clone()
        self.water.b_n = self.water.b.detach().clone()

    def restore_previous(self) -> None:
        """Restore state from previous timestep (for failed iterations)."""
        # Oil phase
        self.oil.p = self.oil.p_n.clone()
        self.oil.s = self.oil.s_n.clone()
        self.oil.b = self.oil.b_n.clone()

        # Water phase
        self.water.p = self.water.p_n.clone()
        self.water.s = self.water.s_n.clone()
        self.water.b = self.water.b_n.clone()

    def get_state(self) -> Dict[str, torch.Tensor]:
        """
        Get current fluid state as dictionary.

        Returns:
            Dictionary with keys: po, so, sw, bo, bw, muo, muw.
        """
        return {
            'po': self.oil.p.clone(),
            'so': self.oil.s.clone(),
            'sw': self.water.s.clone(),
            'bo': self.oil.b.clone(),
            'bw': self.water.b.clone(),
            'muo': self.oil.mu.clone(),
            'muw': self.water.mu.clone()
        }

    def set_state(self, po: torch.Tensor, sw: torch.Tensor) -> None:
        """
        Set current fluid state.

        Args:
            po: Oil pressure (psi), shape (nc,).
            sw: Water saturation, shape (nc,).
        """
        self.oil.p = po.clone()
        self.water.p = po.clone()  # Assume no capillary pressure
        self.water.s = sw.clone()
        self.oil.s = 1.0 - sw

    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------

    def _to_tensor(self, value) -> torch.Tensor:
        """Convert scalar or array to tensor."""
        if isinstance(value, (int, float)):
            return torch.full(
                (self.nc,), float(value),
                device=self.device, dtype=self.dtype
            )
        if isinstance(value, torch.Tensor):
            return value.to(device=self.device, dtype=self.dtype)
        return torch.tensor(value, device=self.device, dtype=self.dtype)

    def forward(self) -> None:
        """Forward pass (placeholder for nn.Module compatibility)."""
        pass

    def __repr__(self) -> str:
        initialized = self.oil is not None and self.water is not None
        return (
            f"OilWaterFluid(nc={self.nc}, nconn={self.nconn}, "
            f"rho_o={self.rho_o_std:.1f}, rho_w={self.rho_w_std:.1f}, "
            f"initialized={initialized})"
        )
