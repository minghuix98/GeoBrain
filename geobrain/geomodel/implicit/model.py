"""
ImplicitModel: multi-series implicit geological model with fault support.

Orchestrates Cokriging interpolation across multiple geological series with
ERODE/ONLAP stacking and simple fault displacement. Fully differentiable
via PyTorch autograd.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional

from .data import (
    ImplicitModelConfig,
    SeriesDefinition,
    FaultDefinition,
    InterpolationInput,
    StackRelation,
)
from .kernel import CubicKernel, GaussianKernel, CovarianceKernel
from .interpolator import CokrigingInterpolator


class ImplicitModel(nn.Module):
    """Differentiable implicit geological model.

    Builds a 3D (or 2D) geological block model from surface contact points
    and orientation measurements using Universal Cokriging. Supports:

    - Multiple geological series with ERODE / ONLAP stacking
    - Simple fault displacement (differentiable tanh-based sign function)
    - End-to-end differentiability for gradient-based optimization

    Args:
        config: Model configuration (extent, resolution, kernel, etc.).
        series: List of geological series definitions (oldest first).
        faults: Optional list of fault definitions.

    Example:
        >>> model = ImplicitModel(config, series=[series1, series2])
        >>> result = model(soft=True)
        >>> block = result['block']  # (nx*ny[*nz],) soft lithology ids
    """

    def __init__(
        self,
        config: ImplicitModelConfig,
        series: List[SeriesDefinition],
        faults: Optional[List[FaultDefinition]] = None,
    ):
        super().__init__()
        self.config = config
        self.series = series
        self.faults = faults or []

        # Build kernel
        self.kernel = self._build_kernel(config)

        # Build interpolator
        self.interpolator = CokrigingInterpolator(
            kernel=self.kernel,
            drift_degree=config.drift_degree,
        )

        # Fault displacements as learnable parameters
        if self.faults:
            self.fault_displacements = nn.ParameterList([
                nn.Parameter(torch.tensor(f.displacement, dtype=config.torch_dtype))
                for f in self.faults
            ])
        else:
            self.fault_displacements = nn.ParameterList()

    @staticmethod
    def _build_kernel(config: ImplicitModelConfig) -> CovarianceKernel:
        """Instantiate the covariance kernel from config."""
        c_o = config.computed_c_o
        a = config.range
        if config.kernel == "cubic":
            return CubicKernel(a=a, c_o=c_o)
        elif config.kernel == "gaussian":
            return GaussianKernel(a=a, c_o=c_o)
        else:
            raise ValueError(f"Unknown kernel: {config.kernel!r}. Use 'cubic' or 'gaussian'.")

    # ------------------------------------------------------------------
    # Fault processing
    # ------------------------------------------------------------------

    def _interpolate_fault(
        self,
        fault: FaultDefinition,
        displacement_param: nn.Parameter,
        grid: torch.Tensor,
    ) -> dict:
        """Interpolate a single fault and compute displacement field.

        Returns a dict with fault scalar field, sign field, displacement vector,
        and the fault normal direction.
        """
        inp = InterpolationInput.from_series(
            SeriesDefinition(
                name=fault.name,
                surface_points=fault.surface_points,
                orientations=fault.orientations,
                relation=StackRelation.FAULT,
            )
        )
        # Move data to device/dtype
        inp = self._move_input(inp)

        # Solve and evaluate
        weights = self.interpolator.solve(inp)
        Z_fault = self.interpolator.evaluate(grid, inp, weights)

        # Iso-value: mean at surface points
        Z_at_sp = self.interpolator.evaluate(inp.sp_coords, inp, weights)
        iso = Z_at_sp.mean()

        # Smooth sign function (differentiable)
        sign = torch.tanh(50.0 * (Z_fault - iso))

        # Fault normal: average orientation direction
        fault_normal = inp.ori_gradients.mean(dim=0)
        fault_normal = fault_normal / fault_normal.norm().clamp(min=1e-12)

        return {
            'scalar_field': Z_fault,
            'sign': sign,
            'displacement': displacement_param,
            'normal': fault_normal,
            'affected': fault.affected_series_indices,
        }

    def _apply_fault_displacement(
        self,
        grid: torch.Tensor,
        fault_info: dict,
    ) -> torch.Tensor:
        """Apply fault displacement to grid coordinates.

        x' = x + sign * displacement * fault_normal
        """
        offset = (
            fault_info['sign'].unsqueeze(-1)
            * fault_info['displacement']
            * fault_info['normal'].unsqueeze(0)
        )
        return grid + offset

    # ------------------------------------------------------------------
    # Series interpolation
    # ------------------------------------------------------------------

    def _interpolate_series(
        self,
        series: SeriesDefinition,
        grid: torch.Tensor,
        soft: bool,
        temperature: float,
    ) -> dict:
        """Interpolate a single geological series on the given grid.

        Returns scalar field, iso values, and block (hard or soft).
        """
        inp = InterpolationInput.from_series(series)
        inp = self._move_input(inp)

        weights = self.interpolator.solve(inp)
        Z = self.interpolator.evaluate(grid, inp, weights)

        # Iso-values from surface points
        Z_at_sp = self.interpolator.evaluate(inp.sp_coords, inp, weights)
        iso_vals = self.interpolator.compute_iso_values(
            Z_at_sp, inp.sp_surface_id, inp.n_surfaces,
        )

        if soft:
            block = self.interpolator.scalar_field_to_block_soft(Z, iso_vals, temperature)
        else:
            block = self.interpolator.scalar_field_to_block(Z, iso_vals)

        return {
            'scalar_field': Z,
            'iso_vals': iso_vals,
            'block': block,
        }

    # ------------------------------------------------------------------
    # Multi-series stacking
    # ------------------------------------------------------------------

    @staticmethod
    def _stack_erode(
        final: torch.Tensor,
        block_new: torch.Tensor,
        soft: bool,
        temperature: float,
    ) -> torch.Tensor:
        """ERODE stacking: new series overwrites where it has lithology."""
        if soft:
            mask = torch.sigmoid(temperature * (block_new - 0.5))
            return mask * block_new + (1.0 - mask) * final
        else:
            has_lith = block_new > 0
            return torch.where(has_lith, block_new, final)

    @staticmethod
    def _stack_onlap(
        final: torch.Tensor,
        block_new: torch.Tensor,
        soft: bool,
        temperature: float,
    ) -> torch.Tensor:
        """ONLAP stacking: new series fills only unassigned regions."""
        if soft:
            mask_empty = torch.sigmoid(temperature * (0.5 - final))
            has_lith = torch.sigmoid(temperature * (block_new - 0.5))
            fill = mask_empty * has_lith
            return fill * block_new + (1.0 - fill) * final
        else:
            is_empty = final == 0
            has_lith = block_new > 0
            fill = is_empty & has_lith
            return torch.where(fill, block_new, final)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _move_input(self, inp: InterpolationInput) -> InterpolationInput:
        """Move InterpolationInput tensors to model device and dtype."""
        dev = self.config.torch_device
        dt = self.config.torch_dtype
        return InterpolationInput(
            sp_coords=inp.sp_coords.to(device=dev, dtype=dt),
            sp_surface_id=inp.sp_surface_id.to(device=dev),
            ori_coords=inp.ori_coords.to(device=dev, dtype=dt),
            ori_gradients=inp.ori_gradients.to(device=dev, dtype=dt),
            sp_nugget=inp.sp_nugget,
            ori_nugget=inp.ori_nugget,
            ndim=inp.ndim,
            n_surfaces=inp.n_surfaces,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        soft: bool = True,
        temperature: float = 50.0,
    ) -> Dict[str, object]:
        """Run the full implicit geological model.

        Steps:
            1. Interpolate fault scalar fields and compute displacements.
            2. For each series (oldest to newest):
               a. Apply relevant fault displacements to grid.
               b. Cokriging interpolation → scalar field → block.
            3. Stack series according to ERODE / ONLAP rules.

        Args:
            soft: If True, use differentiable sigmoid classification.
            temperature: Sigmoid sharpness for soft classification.

        Returns:
            Dict with keys:
                'block': (n_points,) final lithology ids (soft or hard).
                'scalar_fields': List of per-series scalar field tensors.
                'grid': (n_points, D) evaluation grid coordinates.
        """
        grid = self.config.make_grid()

        # Step 1: Process faults
        fault_infos = []
        for i, fault in enumerate(self.faults):
            info = self._interpolate_fault(fault, self.fault_displacements[i], grid)
            fault_infos.append(info)

        # Step 2 & 3: Process series and stack
        final_block = torch.zeros(grid.shape[0], dtype=grid.dtype, device=grid.device)
        scalar_fields = []
        series_offset = 0  # offset lithology ids so series don't clash

        for s_idx, series_def in enumerate(self.series):
            # Apply relevant faults to grid
            displaced_grid = grid.clone()
            for fi in fault_infos:
                affected = fi['affected']
                if affected is None or s_idx in affected:
                    displaced_grid = self._apply_fault_displacement(displaced_grid, fi)

            # Interpolate this series
            result = self._interpolate_series(series_def, displaced_grid, soft, temperature)
            scalar_fields.append(result['scalar_field'])

            # Offset block ids by previous series count
            block_new = result['block'] + series_offset

            # Stack
            relation = series_def.relation
            if s_idx == 0 or relation == StackRelation.BASEMENT:
                final_block = block_new
            elif relation == StackRelation.ERODE:
                final_block = self._stack_erode(final_block, block_new, soft, temperature)
            elif relation == StackRelation.ONLAP:
                final_block = self._stack_onlap(final_block, block_new, soft, temperature)

            # Accumulate offset (n_surfaces defines the number of iso-surfaces,
            # which gives n_surfaces + 1 distinct regions, but we use n_surfaces
            # as the offset to keep ids contiguous)
            inp = InterpolationInput.from_series(series_def)
            series_offset += inp.n_surfaces + 1

        return {
            'block': final_block,
            'scalar_fields': scalar_fields,
            'grid': grid,
        }
