"""
Flow propagator module for reservoir simulation.

Provides FlowPropagator class,
performing time-stepping simulation through the reservoir model.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import logging

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field

from .constants import DEVICE, DTYPE, DT_GROW_FACTOR, DT_CUT_FACTOR, TARGET_ITER
from .model import ReservoirModel
from .solver import NewtonSolver, TimeStepScheduler, SolverResult
from .well import Well, compute_well_index

logger = logging.getLogger(__name__)


# =============================================================================
# Simulation Result
# =============================================================================

@dataclass
class SimulationResult:
    """
    Result of flow simulation.

    Contains time series of field quantities and well histories.

    Attributes:
        time: Time points (days).
        pressure: Pressure at each time (list of tensors).
        saturation: Water saturation at each time.
        well_data: Well histories by name.
        converged: Whether simulation completed successfully.
        final_time: Actual end time reached.
    """
    time: List[float] = field(default_factory=list)
    pressure: List[torch.Tensor] = field(default_factory=list)
    saturation: List[torch.Tensor] = field(default_factory=list)
    well_data: Dict[str, Dict] = field(default_factory=dict)
    converged: bool = True
    final_time: float = 0.0

    def to_dict(self) -> Dict:
        """
        Convert to dictionary.

        Returns:
            Dictionary with numpy arrays and metadata.
        """
        return {
            'time': self.time,
            'pressure': [p.cpu().numpy() for p in self.pressure],
            'saturation': [s.cpu().numpy() for s in self.saturation],
            'well_data': self.well_data,
            'converged': self.converged,
            'final_time': self.final_time
        }


# =============================================================================
# Flow Propagator
# =============================================================================

class FlowPropagator(nn.Module):
    """
    Flow propagator for time-stepping simulation.

    Performs fully-implicit time integration of oil-water flow
    using Newton-Raphson solver with adaptive timestepping.

    Supports two modes of operation:
    1. Batch mode: Run entire simulation with forward() or __call__()
    2. Step mode: Control simulation step-by-step with time_step() and step_to()

    Args:
        model: ReservoirModel instance.
        dt_init: Initial timestep (days).
        dt_min: Minimum timestep (days).
        dt_max: Maximum timestep (days).
        max_newton_iter: Maximum Newton iterations per timestep.
        newton_tol: Newton convergence tolerance.

    Attributes:
        t_current: Current simulation time (days).
        pressure_history: Dict mapping time to pressure tensor snapshots.
        saturation_history: Dict mapping time to saturation tensor snapshots.

    Example:
        Batch mode:

        >>> model = ReservoirModel(nx=30, ny=15, nz=1)
        >>> # ... configure model ...
        >>> model.initialize(po=4000.0, sw=0.2)
        >>> propagator = FlowPropagator(model)
        >>> result = propagator(t_end=365.0, verbose=True)

        Step mode:

        >>> propagator = FlowPropagator(model)
        >>> propagator.step_to(100.0)  # Run to t=100 days
        >>> propagator.add_well('INJ', well_config)  # Add well
        >>> propagator.step_to(200.0)  # Continue to t=200 days
    """

    def __init__(
        self,
        model: ReservoirModel,
        dt_init: float = 1.0,
        dt_min: float = 1e-6,
        dt_max: float = 30.0,
        max_newton_iter: int = 20,
        newton_tol: float = 1e-6
    ):
        super().__init__()

        if not model.is_initialized:
            raise ValueError(
                "ReservoirModel must be initialized before creating "
                "FlowPropagator. Call model.initialize(po=..., sw=...) first."
            )

        self.model = model
        self.dt_init = dt_init
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.max_newton_iter = max_newton_iter
        self.newton_tol = newton_tol

        # Solver and scheduler (lazy initialization)
        self._solver: Optional[NewtonSolver] = None
        self._scheduler: Optional[TimeStepScheduler] = None

        # Current simulation time
        self._t_current: float = 0.0

        # State history for step-by-step simulation
        self._pressure_history: Dict[float, torch.Tensor] = {}
        self._saturation_history: Dict[float, torch.Tensor] = {}

        # Track if initialized for step mode
        self._step_mode_initialized: bool = False

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def t_current(self) -> float:
        """Current simulation time (days)."""
        return self._t_current

    @property
    def pressure_history(self) -> Dict[float, torch.Tensor]:
        """Pressure snapshots at recorded times."""
        return self._pressure_history

    @property
    def saturation_history(self) -> Dict[float, torch.Tensor]:
        """Saturation snapshots at recorded times."""
        return self._saturation_history

    @property
    def nx(self) -> int:
        """Number of cells in x-direction."""
        return self.model.nx

    @property
    def ny(self) -> int:
        """Number of cells in y-direction."""
        return self.model.ny

    @property
    def nz(self) -> int:
        """Number of cells in z-direction."""
        return self.model.nz

    # -------------------------------------------------------------------------
    # Batch Mode Interface
    # -------------------------------------------------------------------------

    def forward(
        self,
        t_end: float,
        report_step: Optional[float] = None,
        verbose: bool = False,
        callback: Optional[Callable] = None
    ) -> SimulationResult:
        """
        Run simulation to specified end time (batch mode).

        Args:
            t_end: End time (days).
            report_step: Interval for saving results (days).
            verbose: Print progress information.
            callback: Optional callback(t, state) after each timestep.

        Returns:
            SimulationResult with time series data.

        Raises:
            ValueError: If model not initialized.
        """
        if not self.model.is_initialized:
            raise ValueError("Model not initialized")

        # Initialize solver and scheduler
        self._init_solver()

        # Setup result container
        result = SimulationResult()

        # Determine report times
        if report_step is None:
            report_step = t_end / 10.0
        report_times = self._generate_report_times(t_end, report_step)
        next_report_idx = 0

        # Record initial state
        self._record_state(result, 0.0)

        # Time loop
        current_time = 0.0
        timestep_count = 0

        while current_time < t_end - 1e-10:
            # Determine timestep
            dt = min(self._scheduler.dt, t_end - current_time)

            # Hit next report time exactly
            if next_report_idx < len(report_times):
                next_report = report_times[next_report_idx]
                if current_time + dt > next_report - 1e-10:
                    dt = next_report - current_time

            # Store previous state for potential rollback
            self.model.fluid.update_timestep()

            # Solve timestep
            solver_result = self._solver.solve(dt)

            if solver_result.converged:
                # Update time
                current_time += dt
                timestep_count += 1

                # Update model state
                self.model.fluid.set_state(solver_result.po, solver_result.sw)
                self.model.fluid.compute_properties(self.model.connlist)

                # Record well rates
                self.model.wells.record_all_states(current_time)

                # Adjust timestep
                self._scheduler.step(solver_result.iterations)

                # Check for report time
                if next_report_idx < len(report_times):
                    if abs(current_time - report_times[next_report_idx]) < 1e-10:
                        self._record_state(result, current_time)
                        next_report_idx += 1

                # Callback
                if callback is not None:
                    callback(current_time, self.model.get_state())

                # Verbose output
                if verbose:
                    self._print_progress(
                        current_time, t_end, dt,
                        solver_result.iterations, solver_result.residual_norm
                    )
            else:
                # Convergence failure - cut timestep
                self.model.fluid.restore_previous()

                if not self._scheduler.cut():
                    if verbose:
                        logger.warning(f"Simulation failed at t={current_time:.2f} days")
                    result.converged = False
                    break

                if verbose:
                    logger.warning(f"Cutting timestep to {self._scheduler.dt:.4e}")

        # Final record if not at report time
        if len(result.time) == 0 or result.time[-1] < current_time - 1e-10:
            self._record_state(result, current_time)

        # Store well histories
        result.well_data = self.model.wells.get_all_histories()
        result.final_time = current_time

        if verbose:
            logger.info(f"Simulation completed: {timestep_count} timesteps, "
                        f"final time = {current_time:.2f} days")

        return result

    def __call__(
        self,
        t_end: float,
        report_step: Optional[float] = None,
        verbose: bool = False,
        callback: Optional[Callable] = None
    ) -> SimulationResult:
        """
        Callable interface for forward pass.

        Args:
            t_end: End time (days).
            report_step: Interval for saving results (days).
            verbose: Print progress information.
            callback: Optional callback(t, state) after each timestep.

        Returns:
            SimulationResult with time series data.
        """
        return self.forward(t_end, report_step, verbose, callback)

    # -------------------------------------------------------------------------
    # Step-by-Step Mode Interface
    # -------------------------------------------------------------------------

    def _init_step_mode(self) -> None:
        """Initialize for step-by-step simulation."""
        if self._step_mode_initialized:
            return

        if not self.model.is_initialized:
            raise ValueError("Model not initialized")

        self._init_solver()
        self._t_current = 0.0

        # Record initial state
        self._record_history(0.0)
        self.model.wells.record_all_states(0.0)

        self._step_mode_initialized = True

    def _record_history(self, time: float) -> None:
        """Record current state to history dictionaries."""
        self._pressure_history[time] = self.model.fluid.oil.p.clone()
        self._saturation_history[time] = self.model.fluid.water.s.clone()

    def time_step(self, verbose: bool = False) -> bool:
        """
        Execute a single timestep.

        Args:
            verbose: Print progress information.

        Returns:
            True if timestep converged, False otherwise.

        Example:
            >>> propagator = FlowPropagator(model)
            >>> while propagator.t_current < 100.0:
            ...     converged = propagator.time_step()
            ...     if not converged:
            ...         print("Timestep failed")
        """
        self._init_step_mode()

        dt = self._scheduler.dt

        # Store previous state
        self.model.fluid.update_timestep()

        # Solve
        result = self._solver.solve(dt)

        if result.converged:
            self._t_current += dt

            # Update model state
            self.model.fluid.set_state(result.po, result.sw)
            self.model.fluid.compute_properties(self.model.connlist)

            # Record well rates
            self.model.wells.record_all_states(self._t_current)

            # Adjust timestep
            self._scheduler.step(result.iterations)

            if verbose:
                logger.info(f"t = {self._t_current:.2f}, dt = {dt:.4f}, "
                            f"iter = {result.iterations}, res = {result.residual_norm:.2e}")

            return True
        else:
            # Restore previous state
            self.model.fluid.restore_previous()
            self._scheduler.cut()

            if verbose:
                logger.warning(f"Timestep failed at t = {self._t_current:.2f}, "
                               f"cutting to dt = {self._scheduler.dt:.4e}")

            return False

    def step_to(self, t_target: float, verbose: bool = False) -> None:
        """
        Run simulation until target time.

        Args:
            t_target: Target time (days).
            verbose: Print progress information.

        Example:
            >>> propagator = FlowPropagator(model)
            >>> propagator.step_to(100.0)   # Run to t=100
            >>> propagator.step_to(200.0)   # Continue to t=200
            >>> propagator.step_to(500.0)   # Continue to t=500
        """
        self._init_step_mode()

        while self._t_current < t_target - 1e-10:
            # Adjust dt to hit target exactly
            dt_remaining = t_target - self._t_current
            if self._scheduler.dt > dt_remaining:
                self._scheduler._dt = dt_remaining

            self.time_step(verbose=verbose)

        # Record state at this checkpoint
        self._record_history(self._t_current)

    def run_to(
        self,
        t_end: float,
        report_step: Optional[float] = None,
        verbose: bool = False
    ) -> SimulationResult:
        """
        Run simulation from current time to end time.

        Similar to forward() but continues from current state instead
        of starting from t=0.

        Args:
            t_end: End time (days).
            report_step: Report interval (days).
            verbose: Print progress information.

        Returns:
            SimulationResult with time series data.

        Example:
            >>> propagator.step_to(500.0)        # First phase
            >>> propagator.add_well('INJ', cfg)  # Modify wells
            >>> result = propagator.run_to(1000.0)  # Continue with results
        """
        self._init_step_mode()

        if report_step is None:
            report_step = (t_end - self._t_current) / 10.0

        result = SimulationResult()

        # Record current state as starting point
        result.time.append(self._t_current)
        result.pressure.append(self.model.fluid.oil.p.clone())
        result.saturation.append(self.model.fluid.water.s.clone())

        # Generate report times
        report_times = []
        t = self._t_current + report_step
        while t < t_end:
            report_times.append(t)
            t += report_step
        report_times.append(t_end)
        next_report_idx = 0

        while self._t_current < t_end - 1e-10:
            # Determine timestep
            dt = min(self._scheduler.dt, t_end - self._t_current)

            # Hit report time exactly
            if next_report_idx < len(report_times):
                next_report = report_times[next_report_idx]
                if self._t_current + dt > next_report - 1e-10:
                    self._scheduler._dt = next_report - self._t_current

            converged = self.time_step(verbose=False)

            if converged:
                # Check for report time
                if next_report_idx < len(report_times):
                    if abs(self._t_current - report_times[next_report_idx]) < 1e-10:
                        result.time.append(self._t_current)
                        result.pressure.append(self.model.fluid.oil.p.clone())
                        result.saturation.append(self.model.fluid.water.s.clone())
                        self._record_history(self._t_current)
                        next_report_idx += 1

                if verbose:
                    progress = 100.0 * self._t_current / t_end
                    logger.info(f"t={self._t_current:8.2f} ({progress:5.1f}%)")
            else:
                if self._scheduler.dt < self.dt_min:
                    result.converged = False
                    break

        result.well_data = self.model.wells.get_all_histories()
        result.final_time = self._t_current

        return result

    # -------------------------------------------------------------------------
    # Dynamic Well Control
    # -------------------------------------------------------------------------

    def add_well(
        self,
        well_type: str,
        well_config: Dict,
        compute_wi: bool = True
    ) -> None:
        """
        Add a well during simulation.

        Args:
            well_type: 'PROD' for producer, 'INJ' for injector.
            well_config: Well configuration dictionary with keys:
                - name: Well name (required)
                - perforation: List of (i, j, k) tuples, 1-based (required)
                - radius: Wellbore radius in ft (default: 0.5)
                - skin: Skin factor (default: 0.0)
                - mode: 'BHP' or 'RATE' (default: 'BHP')
                - target: Control target value (required)
                - phase: Phase for rate control (default: 'LIQ')
                - bhp_limit: BHP limit for rate control (optional)
            compute_wi: Whether to compute well index automatically.

        Example:
            >>> propagator.add_well('INJ', {
            ...     'name': 'I2',
            ...     'perforation': [(12, 10, 1)],
            ...     'radius': 0.5,
            ...     'mode': 'bhp',
            ...     'target': 6500.0
            ... })
        """
        self._init_step_mode()

        well = Well(
            name=well_config['name'],
            well_type=well_type.upper(),
            device=self.model.device,
            dtype=self.model.dtype
        )

        # Add perforations
        for perf in well_config['perforation']:
            i, j, k = perf[0] - 1, perf[1] - 1, perf[2] - 1  # Convert to 0-based
            cell_idx = self.model.grid.ijk_to_global(i, j, k)

            if compute_wi:
                rw = well_config.get('radius', 0.5)
                skin = well_config.get('skin', 0.0)

                dx = float(self.model.grid.dx[i])
                dy = float(self.model.grid.dy[j])
                dz = float(self.model.grid.dz[k])
                kx = float(self.model.rock.perm_x[cell_idx])
                ky = float(self.model.rock.perm_y[cell_idx])

                wi = compute_well_index(kx, ky, dx, dy, dz, rw, skin)
            else:
                wi = well_config.get('wi', 1.0)

            depth = self.model.grid.depth[cell_idx].item()
            well.add_perforation(cell_idx=cell_idx, wi=wi, depth=depth)

        # Set control
        mode = well_config.get('mode', 'bhp').upper()
        target = well_config['target']

        if mode == 'BHP':
            well.set_control('BHP', target=target)
        elif mode == 'SHUT':
            well.set_control('RATE', target=0.0)
        else:
            phase = well_config.get('phase', 'LIQ')
            bhp_limit = well_config.get('bhp_limit', None)
            well.set_control('RATE', target=target, phase=phase, bhp_limit=bhp_limit)

        # Add to model
        self.model.wells.add_well(well)

        # Update solver's well reference
        self._solver.wells = self.model.wells

    def shut_well(self, well_name: str) -> None:
        """
        Shut a well (set rate to zero).

        Args:
            well_name: Name of well to shut.

        Example:
            >>> propagator.shut_well('I1')
        """
        self._init_step_mode()

        well = self.model.wells.get_well(well_name)
        well.set_control('RATE', target=0.0)

    def modify_well_control(
        self,
        well_name: str,
        control_type: str,
        target: float,
        phase: str = 'LIQ',
        bhp_limit: Optional[float] = None
    ) -> None:
        """
        Modify well control during simulation.

        Args:
            well_name: Name of well to modify.
            control_type: 'BHP' or 'RATE'.
            target: New control target.
            phase: Phase for rate control.
            bhp_limit: BHP limit for rate control.

        Example:
            >>> # Switch from BHP to rate control
            >>> propagator.modify_well_control('P1', 'RATE', target=-500.0, phase='LIQ')
        """
        self._init_step_mode()

        well = self.model.wells.get_well(well_name)
        well.set_control(control_type, target=target, phase=phase, bhp_limit=bhp_limit)

    def get_well_data(self, well_name: str) -> Dict:
        """
        Get well history data.

        Args:
            well_name: Name of well.

        Returns:
            Dictionary with keys: time, qo, qw, bhp, wct.
        """
        return self.model.wells.get_well(well_name).history.to_dict()

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def reset(self) -> None:
        """Reset propagator and model to initial state."""
        self.model.reset()

        if self._scheduler is not None:
            self._scheduler.reset(self.dt_init)

        self._t_current = 0.0
        self._pressure_history.clear()
        self._saturation_history.clear()
        self._step_mode_initialized = False

    def get_model(self) -> ReservoirModel:
        """
        Get underlying model.

        Returns:
            ReservoirModel instance.
        """
        return self.model

    def get_state(self) -> Dict[str, torch.Tensor]:
        """
        Get current model state.

        Returns:
            State dictionary from model.
        """
        return self.model.get_state()

    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------

    def _init_solver(self) -> None:
        """Initialize solver and scheduler."""
        if self._solver is None:
            self._solver = NewtonSolver(
                grid=self.model.grid,
                rock=self.model.rock,
                fluid=self.model.fluid,
                wells=self.model.wells,
                max_iter=self.max_newton_iter,
                tol=self.newton_tol,
                device=self.model.device,
                dtype=self.model.dtype
            )

        if self._scheduler is None:
            self._scheduler = TimeStepScheduler(
                dt_init=self.dt_init,
                dt_min=self.dt_min,
                dt_max=self.dt_max,
                grow_factor=DT_GROW_FACTOR,
                cut_factor=DT_CUT_FACTOR,
                target_iter=TARGET_ITER
            )

    def _generate_report_times(
        self,
        t_end: float,
        report_step: float
    ) -> List[float]:
        """Generate list of report times."""
        times = []
        t = report_step
        while t < t_end:
            times.append(t)
            t += report_step
        times.append(t_end)
        return times

    def _record_state(self, result: SimulationResult, time: float) -> None:
        """Record current state to result."""
        result.time.append(time)
        result.pressure.append(self.model.fluid.oil.p.clone())
        result.saturation.append(self.model.fluid.water.s.clone())

    def _print_progress(
        self,
        t: float,
        t_end: float,
        dt: float,
        n_iter: int,
        res_norm: float
    ) -> None:
        """Print progress information."""
        progress = 100.0 * t / t_end
        logger.info(f"t={t:8.2f} ({progress:5.1f}%) | dt={dt:8.4f} | "
                    f"iter={n_iter:2d} | res={res_norm:.2e}")
