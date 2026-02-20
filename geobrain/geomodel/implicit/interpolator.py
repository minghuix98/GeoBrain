"""
Cokriging interpolator for implicit geological modeling.

Implements the Universal Cokriging system with gradient constraints:
assembles the kriging matrix, solves for weights, evaluates the scalar
field, and converts it to lithology blocks (hard or differentiable).

Reference:
    Lajaunie, C., Courrioux, G., & Manuel, L. (1997). Foliation fields and
    3D cartography in geology. Mathematical Geology, 29(4), 571-584.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
"""

import torch
import torch.nn as nn

from .data import InterpolationInput
from .kernel import CovarianceKernel


class CokrigingInterpolator(nn.Module):
    """Universal Cokriging solver with gradient constraints.

    Builds and solves the linear system:

        [C_uu + eps_u*I   C_up         F_u ] [w_u]   [1]
        [C_up^T           C_pp + eps_p*I F_p] [w_p] = [0]
        [F_u^T            F_p^T          0  ] [ l ]   [0]

    Then evaluates the scalar field at query points and converts to
    lithology identifiers.

    Args:
        kernel: Covariance kernel instance (CubicKernel or GaussianKernel).
        drift_degree: Polynomial drift degree (0=constant, 1=linear).
    """

    def __init__(self, kernel: CovarianceKernel, drift_degree: int = 1):
        super().__init__()
        self.kernel = kernel
        self.drift_degree = drift_degree

    # ------------------------------------------------------------------
    # Drift basis functions
    # ------------------------------------------------------------------

    def _drift_functions(self, coords: torch.Tensor) -> torch.Tensor:
        """Evaluate polynomial drift basis at given coordinates.

        degree 0: [1]
        degree 1: [1, x, y(, z)]

        Args:
            coords: (N, D) coordinates.

        Returns:
            (N, n_drift) drift matrix.
        """
        N = coords.shape[0]
        ones = torch.ones(N, 1, dtype=coords.dtype, device=coords.device)
        if self.drift_degree == 0:
            return ones
        # degree 1: [1, x, y, ...]
        return torch.cat([ones, coords], dim=-1)

    def _drift_gradients(self, coords: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
        """Drift functions differentiated by gradient direction.

        For degree 0: all zeros (constant drift has zero gradient).
        For degree 1: drift = [1, x, y, z], gradient = [0, G_x, G_y, G_z].

        Args:
            coords: (M, D) orientation coordinates.
            gradients: (M, D) unit gradient vectors.

        Returns:
            (M, n_drift) drift-gradient matrix.
        """
        M, D = coords.shape
        if self.drift_degree == 0:
            return torch.zeros(M, 1, dtype=coords.dtype, device=coords.device)
        # degree 1: derivative of [1, x_1, ..., x_D] w.r.t. direction g
        # d(1)/dg = 0, d(x_d)/dg = g_d
        zeros = torch.zeros(M, 1, dtype=coords.dtype, device=coords.device)
        return torch.cat([zeros, gradients], dim=-1)

    @property
    def _n_drift(self) -> int:
        """Number of drift basis functions (set after build_system)."""
        return self._cached_n_drift

    # ------------------------------------------------------------------
    # System assembly
    # ------------------------------------------------------------------

    def build_system(self, inp: InterpolationInput):
        """Assemble the Cokriging matrix A and right-hand-side b.

        Args:
            inp: Packed interpolation input.

        Returns:
            A: (S, S) kriging matrix, S = M + N + n_drift.
            b: (S,) right-hand-side vector.
        """
        M = inp.ori_coords.shape[0]
        N = inp.sp_coords.shape[0]

        # Covariance blocks
        C_uu = self.kernel.cov_uu(
            inp.ori_coords, inp.ori_coords,
            inp.ori_gradients, inp.ori_gradients,
        )  # (M, M)
        C_up = self.kernel.cov_up(
            inp.ori_coords, inp.sp_coords,
            inp.ori_gradients,
        )  # (M, N)
        C_pp = self.kernel.cov_pp(inp.sp_coords, inp.sp_coords)  # (N, N)

        # Add nugget
        C_uu = C_uu + inp.ori_nugget * torch.eye(M, dtype=C_uu.dtype, device=C_uu.device)
        C_pp = C_pp + inp.sp_nugget * torch.eye(N, dtype=C_pp.dtype, device=C_pp.device)

        # Drift blocks
        F_u = self._drift_gradients(inp.ori_coords, inp.ori_gradients)  # (M, n_drift)
        F_p = self._drift_functions(inp.sp_coords)                      # (N, n_drift)
        n_drift = F_u.shape[1]
        self._cached_n_drift = n_drift

        S = M + N + n_drift

        A = torch.zeros(S, S, dtype=C_uu.dtype, device=C_uu.device)
        b = torch.zeros(S, dtype=C_uu.dtype, device=C_uu.device)

        # Fill blocks
        A[:M, :M] = C_uu
        A[:M, M:M+N] = C_up
        A[M:M+N, :M] = C_up.T
        A[M:M+N, M:M+N] = C_pp
        A[:M, M+N:] = F_u
        A[M:M+N, M+N:] = F_p
        A[M+N:, :M] = F_u.T
        A[M+N:, M:M+N] = F_p.T

        # RHS: orientation constraints = 1, rest = 0
        b[:M] = 1.0

        return A, b

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(self, inp: InterpolationInput) -> torch.Tensor:
        """Solve the Cokriging system for weights.

        Uses torch.linalg.solve with lstsq fallback for singular systems.

        Args:
            inp: Packed interpolation input.

        Returns:
            weights: (S,) weight vector [w_u, w_p, lambda].
        """
        A, b = self.build_system(inp)
        try:
            weights = torch.linalg.solve(A, b)
        except torch.linalg.LinAlgError:
            # Fallback to least-squares for near-singular systems
            result = torch.linalg.lstsq(A, b.unsqueeze(-1))
            weights = result.solution.squeeze(-1)
        return weights

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        query: torch.Tensor,
        inp: InterpolationInput,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the scalar field at query points.

        Z(x) = sum_i w_u_i * [sum_d G_d^(i) * dC/dh_d(x, ori_i)]
             + sum_j w_p_j * C(x, sp_j)
             + sum_k lambda_k * f_k(x)

        Args:
            query: (Q, D) query coordinates.
            inp: Interpolation input (for data point locations).
            weights: (S,) solution weights from solve().

        Returns:
            Z: (Q,) scalar field values.
        """
        M = inp.ori_coords.shape[0]
        N = inp.sp_coords.shape[0]

        w_u = weights[:M]
        w_p = weights[M:M+N]
        w_drift = weights[M+N:]

        # Term 1: gradient contributions
        # cov_up computes (ori, sp, gradients) → (M, N)
        # Here we need (query acts as "sp", ori stays as ori)
        # cov_up(ori, query, gradients) → (M, Q), then dot with w_u → (Q,)
        C_uq = self.kernel.cov_up(inp.ori_coords, query, inp.ori_gradients)  # (M, Q)
        term_u = (w_u.unsqueeze(-1) * C_uq).sum(dim=0)  # (Q,)

        # Term 2: point covariance contributions
        C_pq = self.kernel.cov_pp(inp.sp_coords, query)  # (N, Q)
        term_p = (w_p.unsqueeze(-1) * C_pq).sum(dim=0)  # (Q,)

        # Term 3: drift contributions
        F_q = self._drift_functions(query)  # (Q, n_drift)
        term_drift = (F_q * w_drift.unsqueeze(0)).sum(dim=-1)  # (Q,)

        return term_u + term_p + term_drift

    def interpolate(
        self,
        query: torch.Tensor,
        inp: InterpolationInput,
    ) -> torch.Tensor:
        """Solve and evaluate in one call.

        Args:
            query: (Q, D) query coordinates.
            inp: Interpolation input.

        Returns:
            Z: (Q,) scalar field values at query points.
        """
        weights = self.solve(inp)
        return self.evaluate(query, inp, weights)

    # ------------------------------------------------------------------
    # Scalar field → lithology
    # ------------------------------------------------------------------

    def compute_iso_values(
        self,
        Z_at_sp: torch.Tensor,
        surface_ids: torch.Tensor,
        n_surfaces: int,
    ) -> torch.Tensor:
        """Compute iso-surface values as the mean Z per surface.

        Args:
            Z_at_sp: (N,) scalar field values at surface points.
            surface_ids: (N,) integer surface id for each point.
            n_surfaces: Total number of surfaces.

        Returns:
            iso_vals: (n_surfaces,) iso-surface values, sorted ascending.
        """
        iso_vals = torch.zeros(n_surfaces, dtype=Z_at_sp.dtype, device=Z_at_sp.device)
        for k in range(n_surfaces):
            mask = surface_ids == k
            if mask.any():
                iso_vals[k] = Z_at_sp[mask].mean()
        # Sort iso values for consistent ordering
        iso_vals, _ = iso_vals.sort()
        return iso_vals

    def scalar_field_to_block(
        self,
        Z: torch.Tensor,
        iso_vals: torch.Tensor,
    ) -> torch.Tensor:
        """Hard classification: assign lithology id based on scalar field intervals.

        Each grid point gets the index of the interval it falls into.

        Args:
            Z: (Q,) scalar field values.
            iso_vals: (n_surfaces,) sorted iso-surface values.

        Returns:
            block: (Q,) integer lithology ids (0 to n_surfaces).
        """
        # Count how many iso-values each Z exceeds
        # Z > iso_vals[k] for each k → sum gives the block id
        block = torch.zeros_like(Z, dtype=torch.long)
        for k in range(len(iso_vals)):
            block = block + (Z > iso_vals[k]).long()
        return block

    def scalar_field_to_block_soft(
        self,
        Z: torch.Tensor,
        iso_vals: torch.Tensor,
        temperature: float = 50.0,
    ) -> torch.Tensor:
        """Differentiable soft classification using sigmoid.

        block_soft = sum_k sigmoid(T * (Z - iso_k))

        This produces a continuous approximation of the lithology id,
        enabling gradient flow through the classification step.

        Args:
            Z: (Q,) scalar field values.
            iso_vals: (n_surfaces,) sorted iso-surface values.
            temperature: Sigmoid sharpness (higher = closer to hard boundary).

        Returns:
            block_soft: (Q,) continuous lithology ids.
        """
        block_soft = torch.zeros_like(Z)
        for k in range(len(iso_vals)):
            block_soft = block_soft + torch.sigmoid(temperature * (Z - iso_vals[k]))
        return block_soft
