"""
Solver module for reservoir simulation.

Provides Newton-Raphson nonlinear solver with numerical Jacobian
for fully-implicit oil-water reservoir simulation.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

from .constants import (
    DEVICE, DTYPE, M,
    MAX_NEWTON_ITER, NEWTON_TOL
)
from .grid import CartGrid, ConnList
from .rock import Rock
from .fluid import OilWaterFluid
from .well import WellGroup


# =============================================================================
# Solver Result
# =============================================================================

@dataclass
class SolverResult:
    """
    Result of Newton iteration.

    Attributes:
        converged: Whether solver converged.
        iterations: Number of iterations taken.
        residual_norm: Final residual norm.
        po: Final oil pressure (psi).
        sw: Final water saturation.
    """
    converged: bool
    iterations: int
    residual_norm: float
    po: torch.Tensor
    sw: torch.Tensor


# =============================================================================
# Newton-Raphson Solver
# =============================================================================

class NewtonSolver(nn.Module):
    """
    Newton-Raphson solver for fully-implicit reservoir simulation.

    Solves the coupled pressure-saturation system using numerical
    Jacobian and direct linear solve.

    Args:
        grid: Cartesian grid.
        rock: Rock properties.
        fluid: Oil-water fluid system.
        wells: Well group (optional).
        max_iter: Maximum Newton iterations.
        tol: Convergence tolerance.
        device: Torch device.
        dtype: Torch dtype.

    Example:
        >>> solver = NewtonSolver(grid, rock, fluid, wells)
        >>> result = solver.solve(dt=1.0)
        >>> if result.converged:
        ...     fluid.update_timestep()
    """

    def __init__(
        self,
        grid: CartGrid,
        rock: Rock,
        fluid: OilWaterFluid,
        wells: Optional[WellGroup] = None,
        max_iter: int = MAX_NEWTON_ITER,
        tol: float = NEWTON_TOL,
        device: torch.device = DEVICE,
        dtype: torch.dtype = DTYPE
    ):
        super().__init__()

        self.grid = grid
        self.rock = rock
        self.fluid = fluid
        self.wells = wells
        self.max_iter = max_iter
        self.tol = tol
        self.device = device
        self.dtype = dtype

        # Number of cells and unknowns
        self.nc = grid.nc
        self.n_unknowns = 2 * self.nc  # po and sw for each cell

        # Statistics tracking
        self.num_iter = []

    # -------------------------------------------------------------------------
    # Main Solve Method
    # -------------------------------------------------------------------------

    def solve(self, dt: float) -> SolverResult:
        """
        Solve for one timestep.

        Uses Newton-Raphson iteration to solve the fully-implicit
        discretization of the oil-water flow equations.

        Args:
            dt: Timestep size (days).

        Returns:
            SolverResult with convergence info and solution.
        """
        # Get initial guess from current state
        po = self.fluid.oil.p.clone()
        sw = self.fluid.water.s.clone()

        res_norm = float('inf')

        # Newton iteration
        for iteration in range(self.max_iter):
            # Update fluid properties
            self.fluid.set_state(po, sw)
            self.fluid.compute_properties(self.grid.connlist)

            # Update well rates
            if self.wells is not None:
                self.wells.compute_all_rates(
                    po, sw,
                    self.fluid.oil.b, self.fluid.water.b,
                    self.fluid.oil.mu, self.fluid.water.mu,
                    self.fluid.oil.kr, self.fluid.water.kr,
                    self.rock.perm_x
                )

            # Compute residual
            res_o, res_w = self._compute_residual(dt)

            # Residual ordering: [res_w, res_o] interleaved
            residual = torch.zeros(2 * self.nc, dtype=self.dtype, device=self.device)
            residual[0::2] = res_w
            residual[1::2] = res_o

            res_norm = self._compute_error(res_o, res_w, dt)

            # Check convergence
            if res_norm < self.tol:
                self.num_iter.append(iteration + 1)
                return SolverResult(
                    converged=True,
                    iterations=iteration + 1,
                    residual_norm=res_norm,
                    po=po,
                    sw=sw
                )

            # Compute Jacobian
            jacobian = self._compute_jacobian_numerical(po, sw, dt)

            # Solve linear system: J * dx = residual
            try:
                dx = torch.linalg.solve(jacobian, residual)
            except RuntimeError:
                # Singular matrix - try regularization
                jacobian_reg = jacobian + 1e-8 * torch.eye(
                    self.n_unknowns, device=self.device, dtype=self.dtype
                )
                dx = torch.linalg.solve(jacobian_reg, residual)

            # Update solution
            dpo = dx[0::2]
            dsw = dx[1::2]

            po = po - dpo
            sw = sw - dsw

            # Clamp saturation
            sw = torch.clamp(sw, 0.0, 1.0)

        # Did not converge
        self.num_iter.append(self.max_iter)
        return SolverResult(
            converged=False,
            iterations=self.max_iter,
            residual_norm=res_norm,
            po=po,
            sw=sw
        )

    # -------------------------------------------------------------------------
    # Residual Computation
    # -------------------------------------------------------------------------

    def _compute_residual(self, dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute residual for oil and water equations.

        Fully-implicit discretization:
            R_o = V*φ/B_o * S_o - V*φ/B_o_n * S_o_n - dt * (flux_o + well_o) / M
            R_w = V*φ/B_w * S_w - V*φ/B_w_n * S_w_n - dt * (flux_w + well_w) / M

        Args:
            dt: Timestep size (days).

        Returns:
            Tuple of (res_o, res_w) residual tensors, each shape (nc,).
        """
        v = self.grid.volume
        poro = self.rock.poro

        oil = self.fluid.oil
        water = self.fluid.water
        connlist = self.grid.connlist
        l, r = connlist.left, connlist.right

        # Accumulation terms with M factor
        acc_o = v * poro * (oil.s / oil.b - oil.s_n / oil.b_n) / dt / M
        acc_w = v * poro * (water.s / water.b - water.s_n / water.b_n) / dt / M

        # Initialize residuals
        res_o = -acc_o.clone()
        res_w = -acc_w.clone()

        # Flux divergence using scatter_add
        res_o = res_o.scatter_add(0, l, -oil.flux)
        res_o = res_o.scatter_add(0, r, oil.flux)
        res_w = res_w.scatter_add(0, l, -water.flux)
        res_w = res_w.scatter_add(0, r, water.flux)

        # Well source terms
        if self.wells is not None:
            for well in self.wells.wells.values():
                if well.nperf == 0:
                    continue

                ind = torch.tensor(
                    well.cell_indices, dtype=torch.long, device=self.device
                )
                qo, qw = well.get_rates_tensor()

                res_o = res_o.scatter_add(0, ind, -qo)
                res_w = res_w.scatter_add(0, ind, -qw)

        return res_o, res_w

    def _compute_error(
        self,
        res_o: torch.Tensor,
        res_w: torch.Tensor,
        dt: float
    ) -> float:
        """
        Compute maximum normalized residual error.

        Args:
            res_o: Oil residual, shape (nc,).
            res_w: Water residual, shape (nc,).
            dt: Timestep size (days).

        Returns:
            Maximum normalized error.
        """
        a = dt * M / (self.grid.volume * self.rock.poro)

        err_o = (a * self.fluid.oil.b.detach() * res_o).abs().max().item()
        err_w = (a * self.fluid.water.b.detach() * res_w).abs().max().item()

        return max(err_o, err_w)

    # -------------------------------------------------------------------------
    # Jacobian Computation
    # -------------------------------------------------------------------------

    def _compute_jacobian_numerical(
        self,
        po: torch.Tensor,
        sw: torch.Tensor,
        dt: float,
        eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Compute Jacobian matrix using numerical differentiation.

        Unknowns are ordered as [p0, sw0, p1, sw1, ...].
        Residuals are ordered as [res_w0, res_o0, res_w1, res_o1, ...].

        Args:
            po: Current oil pressure, shape (nc,).
            sw: Current water saturation, shape (nc,).
            dt: Timestep size (days).
            eps: Perturbation size.

        Returns:
            Jacobian matrix of shape (2*nc, 2*nc).
        """
        jacobian = torch.zeros(
            (self.n_unknowns, self.n_unknowns),
            device=self.device, dtype=self.dtype
        )

        # Base residual
        self.fluid.set_state(po, sw)
        self.fluid.compute_properties(self.grid.connlist)
        if self.wells is not None:
            self.wells.compute_all_rates(
                po, sw,
                self.fluid.oil.b, self.fluid.water.b,
                self.fluid.oil.mu, self.fluid.water.mu,
                self.fluid.oil.kr, self.fluid.water.kr,
                self.rock.perm_x
            )
        res_o_base, res_w_base = self._compute_residual(dt)

        # Derivatives w.r.t. each cell
        for i in range(self.nc):
            # Perturb pressure at cell i
            po_pert = po.clone()
            po_pert[i] += eps

            self.fluid.set_state(po_pert, sw)
            self.fluid.compute_properties(self.grid.connlist)
            if self.wells is not None:
                self.wells.compute_all_rates(
                    po_pert, sw,
                    self.fluid.oil.b, self.fluid.water.b,
                    self.fluid.oil.mu, self.fluid.water.mu,
                    self.fluid.oil.kr, self.fluid.water.kr,
                    self.rock.perm_x
                )
            res_o_pert, res_w_pert = self._compute_residual(dt)

            d_res_w = (res_w_pert - res_w_base) / eps
            d_res_o = (res_o_pert - res_o_base) / eps

            # Column 2*i corresponds to p_i
            jacobian[0::2, 2*i] = d_res_w
            jacobian[1::2, 2*i] = d_res_o

            # Perturb saturation at cell i
            sw_pert = sw.clone()
            sw_pert[i] += eps

            self.fluid.set_state(po, sw_pert)
            self.fluid.compute_properties(self.grid.connlist)
            if self.wells is not None:
                self.wells.compute_all_rates(
                    po, sw_pert,
                    self.fluid.oil.b, self.fluid.water.b,
                    self.fluid.oil.mu, self.fluid.water.mu,
                    self.fluid.oil.kr, self.fluid.water.kr,
                    self.rock.perm_x
                )
            res_o_pert, res_w_pert = self._compute_residual(dt)

            d_res_w = (res_w_pert - res_w_base) / eps
            d_res_o = (res_o_pert - res_o_base) / eps

            # Column 2*i+1 corresponds to sw_i
            jacobian[0::2, 2*i+1] = d_res_w
            jacobian[1::2, 2*i+1] = d_res_o

        # Restore original state
        self.fluid.set_state(po, sw)
        self.fluid.compute_properties(self.grid.connlist)

        return jacobian

    def forward(self, dt: float) -> SolverResult:
        """
        Forward pass calls solve.

        Args:
            dt: Timestep size (days).

        Returns:
            SolverResult.
        """
        return self.solve(dt)

    def __call__(self, dt: float) -> SolverResult:
        """
        Shortcut for solve() - enables solver(dt) syntax.

        Args:
            dt: Timestep size (days).

        Returns:
            SolverResult with convergence info and solution.
        """
        return self.solve(dt)

    def __repr__(self) -> str:
        return (
            f"NewtonSolver(nc={self.nc}, max_iter={self.max_iter}, "
            f"tol={self.tol:.1e}, device={self.device})"
        )


# =============================================================================
# Time Step Scheduler
# =============================================================================

class TimeStepScheduler(nn.Module):
    """
    Adaptive timestep scheduler.

    Adjusts timestep based on solver convergence behavior.

    Args:
        dt_init: Initial timestep (days).
        dt_min: Minimum timestep (days).
        dt_max: Maximum timestep (days).
        grow_factor: Factor to increase timestep after convergence.
        cut_factor: Factor to reduce timestep after failure.
        target_iter: Target Newton iterations for optimal timestep.

    Example:
        >>> scheduler = TimeStepScheduler(dt_init=1.0, dt_max=30.0)
        >>> dt = scheduler.dt
        >>> if converged:
        ...     scheduler.step(n_iter=4)
        ... else:
        ...     scheduler.cut()
    """

    def __init__(
        self,
        dt_init: float = 1.0,
        dt_min: float = 1e-6,
        dt_max: float = 365.0,
        grow_factor: float = 1.5,
        cut_factor: float = 0.5,
        target_iter: int = 5
    ):
        super().__init__()

        self._dt = dt_init
        self._dt_init = dt_init
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.grow_factor = grow_factor
        self.cut_factor = cut_factor
        self.target_iter = target_iter

    @property
    def dt(self) -> float:
        """Current timestep (days)."""
        return self._dt

    def step(self, n_iter: int) -> None:
        """
        Update timestep after successful convergence.

        Args:
            n_iter: Number of Newton iterations taken.
        """
        if n_iter <= self.target_iter:
            self._dt = min(self._dt * self.grow_factor, self.dt_max)

    def cut(self) -> bool:
        """
        Reduce timestep after convergence failure.

        Returns:
            False if timestep is already at minimum.
        """
        self._dt *= self.cut_factor
        if self._dt < self.dt_min:
            self._dt = self.dt_min
            return False
        return True

    def reset(self, dt: float = None) -> None:
        """
        Reset timestep to given or initial value.

        Args:
            dt: New timestep value. If None, use initial value.
        """
        if dt is not None:
            self._dt = dt
        else:
            self._dt = self._dt_init

    def forward(self) -> float:
        """
        Forward pass returns current timestep.

        Returns:
            Current timestep (days).
        """
        return self._dt
