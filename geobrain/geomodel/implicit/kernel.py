"""
Covariance kernels and their analytical derivatives for Cokriging interpolation.

Implements the kernel functions needed for Universal Cokriging with gradient
constraints. Each kernel provides C(r), C'(r), C''(r) and the derived
covariance blocks: C_pp, C_up, C_uu.

Mathematical reference:
    Lajaunie, C., Courrioux, G., & Manuel, L. (1997). Foliation fields and
    3D cartography in geology. Mathematical Geology, 29(4), 571-584.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class CovarianceKernel(ABC, nn.Module):
    """Abstract covariance kernel with first and second order derivatives.

    All kernels are parameterized by:
        a  — range (correlation length)
        c_o — sill (variance at lag 0)

    Subclasses must implement C(r), C_prime(r), C_double_prime(r) where
    r = ||h|| / a is the normalized distance.
    """

    def __init__(self, a: float, c_o: float):
        super().__init__()
        self.a = a
        self.c_o = c_o

    # ------------------------------------------------------------------
    # Abstract: radial basis function and its derivatives w.r.t. r
    # ------------------------------------------------------------------

    @abstractmethod
    def C(self, r: torch.Tensor) -> torch.Tensor:
        """Kernel value C(r)."""

    @abstractmethod
    def C_prime(self, r: torch.Tensor) -> torch.Tensor:
        """First derivative dC/dr."""

    @abstractmethod
    def C_double_prime(self, r: torch.Tensor) -> torch.Tensor:
        """Second derivative d²C/dr²."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def compute_r(self, x1: torch.Tensor, x2: torch.Tensor):
        """Compute displacement vectors and normalized distances.

        Args:
            x1: (N, D) coordinates.
            x2: (M, D) coordinates.

        Returns:
            h: (N, M, D) displacement vectors  h = x1 - x2.
            r: (N, M) normalized distances ||h|| / a.
        """
        # h[i, j, d] = x1[i, d] - x2[j, d]
        h = x1.unsqueeze(1) - x2.unsqueeze(0)  # (N, M, D)
        r = h.norm(dim=-1) / self.a             # (N, M)
        return h, r

    # ------------------------------------------------------------------
    # Covariance blocks
    # ------------------------------------------------------------------

    def cov_pp(self, sp1: torch.Tensor, sp2: torch.Tensor) -> torch.Tensor:
        """Point-point covariance C_pp[i,j] = C(||sp1_i - sp2_j|| / a).

        Args:
            sp1: (N, D) first set of surface points.
            sp2: (M, D) second set of surface points.

        Returns:
            (N, M) covariance matrix.
        """
        _, r = self.compute_r(sp1, sp2)
        return self.C(r)

    def cov_up(
        self,
        ori: torch.Tensor,
        sp: torch.Tensor,
        gradients: torch.Tensor,
    ) -> torch.Tensor:
        """Gradient-point cross-covariance (C_up block).

        C_up[i, j] = sum_d  G_d^(i) * dC/dh_d(ori_i, sp_j)

        where dC/dh_d = C'(r) * h_d / (r * a).

        Args:
            ori: (M, D) orientation locations.
            sp:  (N, D) surface point locations.
            gradients: (M, D) unit gradient vectors at orientation points.

        Returns:
            (M, N) covariance matrix.
        """
        # h[i,j,d] = ori_i - sp_j
        h, r = self.compute_r(ori, sp)  # h: (M, N, D), r: (M, N)
        r_safe = r.clamp(min=1e-12)

        Cp = self.C_prime(r)  # (M, N)

        # dC/dh_d = C'(r) * h_d / (r * a)  →  (M, N, D)
        dC_dh = Cp.unsqueeze(-1) * h / (r_safe.unsqueeze(-1) * self.a)

        # Contract with gradient: sum over d
        # gradients: (M, D) → (M, 1, D)
        result = (gradients.unsqueeze(1) * dC_dh).sum(dim=-1)  # (M, N)
        return result

    def cov_uu(
        self,
        ori1: torch.Tensor,
        ori2: torch.Tensor,
        g1: torch.Tensor,
        g2: torch.Tensor,
    ) -> torch.Tensor:
        """Gradient-gradient covariance (C_uu block).

        C_uu[i,j] = sum_{d,e} G_d^(i) * G_e^(j) * d²C/(dh_d dh_e)

        where:
            d²C/(dh_d dh_e) = [C''(r) - C'(r)/r] * h_d*h_e / (r² * a²)
                             + C'(r) / (r * a²) * delta_{de}

        Args:
            ori1: (M1, D) first set of orientation locations.
            ori2: (M2, D) second set of orientation locations.
            g1:   (M1, D) unit gradient vectors at ori1.
            g2:   (M2, D) unit gradient vectors at ori2.

        Returns:
            (M1, M2) covariance matrix.
        """
        h, r = self.compute_r(ori1, ori2)  # h: (M1,M2,D), r: (M1,M2)
        r_safe = r.clamp(min=1e-12)

        Cp = self.C_prime(r)              # (M1, M2)
        Cpp = self.C_double_prime(r)      # (M1, M2)

        a2 = self.a ** 2
        r2 = r_safe ** 2

        # Term 1 coefficient: [C''(r) - C'(r)/r] / (r² * a²)
        coeff1 = (Cpp - Cp / r_safe) / (r2 * a2)  # (M1, M2)

        # Term 2 coefficient: C'(r) / (r * a²)
        coeff2 = Cp / (r_safe * a2)  # (M1, M2)

        # g1: (M1, D), g2: (M2, D)
        # We need: sum_{d,e} g1_d * g2_e * [coeff1 * h_d * h_e + coeff2 * delta_{de}]

        # Part A: coeff1 * (g1 . h) * (g2 . h)
        # g1_dot_h[i,j] = sum_d g1[i,d] * h[i,j,d]
        g1_dot_h = (g1.unsqueeze(1) * h).sum(dim=-1)  # (M1, M2)
        g2_dot_h = (g2.unsqueeze(0) * h).sum(dim=-1)  # (M1, M2)
        part_a = coeff1 * g1_dot_h * g2_dot_h

        # Part B: coeff2 * (g1 . g2)
        g1_dot_g2 = (g1.unsqueeze(1) * g2.unsqueeze(0)).sum(dim=-1)  # (M1, M2)
        part_b = coeff2 * g1_dot_g2

        return part_a + part_b


class CubicKernel(CovarianceKernel):
    """Cubic covariance kernel (compact support, C² smooth).

    This is the default kernel used in GemPy.

    C(r) = c_o * (1 - 7r² + 35/4 r³ - 7/2 r⁵ + 3/4 r⁷)   for r < 1
         = 0                                                  for r >= 1
    """

    def C(self, r: torch.Tensor) -> torch.Tensor:
        r2 = r * r
        r3 = r2 * r
        r5 = r2 * r3
        r7 = r2 * r5
        val = self.c_o * (1.0 - 7.0 * r2 + 8.75 * r3 - 3.5 * r5 + 0.75 * r7)
        return torch.where(r < 1.0, val, torch.zeros_like(val))

    def C_prime(self, r: torch.Tensor) -> torch.Tensor:
        """dC/dr."""
        r2 = r * r
        r4 = r2 * r2
        r6 = r2 * r4
        val = self.c_o * (-14.0 * r + 26.25 * r2 - 17.5 * r4 + 5.25 * r6)
        return torch.where(r < 1.0, val, torch.zeros_like(val))

    def C_double_prime(self, r: torch.Tensor) -> torch.Tensor:
        """d²C/dr²."""
        r3 = r * r * r
        r5 = r3 * r * r
        val = self.c_o * (-14.0 + 52.5 * r - 70.0 * r3 + 31.5 * r5)
        return torch.where(r < 1.0, val, torch.zeros_like(val))


class GaussianKernel(CovarianceKernel):
    """Gaussian (squared-exponential) covariance kernel.

    C(r) = c_o * exp(-r²)

    Infinitely differentiable but not compactly supported.
    """

    def C(self, r: torch.Tensor) -> torch.Tensor:
        return self.c_o * torch.exp(-r * r)

    def C_prime(self, r: torch.Tensor) -> torch.Tensor:
        return self.c_o * (-2.0 * r) * torch.exp(-r * r)

    def C_double_prime(self, r: torch.Tensor) -> torch.Tensor:
        r2 = r * r
        return self.c_o * (4.0 * r2 - 2.0) * torch.exp(-r2)
