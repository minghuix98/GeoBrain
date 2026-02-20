"""
Zoeppritz equations for exact P-P reflection coefficients.

Implements the full Zoeppritz solution using complex arithmetic
to handle post-critical angle reflections.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
import numpy as np
from torch import Tensor
from typing import Union

from .base import AVOModel, vectorize_angles


class Zoeppritz(AVOModel):
    """
    Full Zoeppritz solution for P-P reflection coefficients.

    Computes exact reflection coefficients by solving the Zoeppritz
    equations at a plane interface. Uses complex arithmetic to properly
    handle post-critical angles where total internal reflection occurs.

    Example:
        Basic usage:
        >>> zoep = Zoeppritz()
        >>> rc = zoep(vp1=2000, vs1=1000, rho1=2.0,
        ...           vp2=2500, vs2=1250, rho2=2.2,
        ...           theta=[0, 15, 30, 45, 60])
        >>> print(rc.shape)  # [5, ...]

        With 2D velocity models:
        >>> vp1 = torch.ones(100, 200) * 2000
        >>> rc = zoep(vp1, vs1, rho1, vp2, vs2, rho2, theta=[0, 30])
        >>> print(rc.shape)  # [2, 100, 200]

    Note:
        - Valid for all angles 0-90 degrees
        - Returns complex values beyond critical angle
        - Magnitude gives reflection amplitude
        - Phase gives wavelet phase shift
    """

    def __init__(self):
        super().__init__()
        self._name = 'zoeppritz'
        self._valid_angles = (0, 90)

    @vectorize_angles
    def forward(
        self,
        vp1: Tensor,
        vs1: Tensor,
        rho1: Tensor,
        vp2: Tensor,
        vs2: Tensor,
        rho2: Tensor,
        theta: Tensor,
    ) -> Tensor:
        """
        Compute exact P-P reflection coefficient.

        Args:
            vp1: P-wave velocity of upper layer (m/s).
            vs1: S-wave velocity of upper layer (m/s).
            rho1: Density of upper layer (kg/m³).
            vp2: P-wave velocity of lower layer (m/s).
            vs2: S-wave velocity of lower layer (m/s).
            rho2: Density of lower layer (kg/m³).
            theta: Incidence angles in radians with shape [n_angles, 1, ...].

        Returns:
            Reflection coefficients with shape [n_angles, *input_shape].
            Complex dtype if post-critical angles present.
        """
        eps = 1e-6

        # Add small value to avoid division by zero
        vp1 = vp1 + eps
        vs1 = vs1 + eps
        vp2 = vp2 + eps
        vs2 = vs2 + eps

        # Ray parameter (horizontal slowness)
        p = torch.sin(theta) / vp1

        def complex_arcsin(x: Tensor) -> Tensor:
            """Arcsin that handles |x| > 1 with complex output."""
            real_mask = torch.abs(x) <= 1.0
            x_clamped = torch.clamp(x, -1.0, 1.0)
            result = torch.arcsin(x_clamped)

            # Use complex formula for |x| > 1
            if not real_mask.all():
                result = result.to(torch.complex64)
                x_c = x.to(torch.complex64)
                i = torch.tensor(1j, dtype=torch.complex64, device=x.device)
                complex_result = -i * torch.log(
                    i * x_c + torch.sqrt(1 - x_c ** 2 + 0j)
                )
                result = torch.where(real_mask, result, complex_result)

            return result

        # Transmitted/reflected angles via Snell's law
        theta_p2 = complex_arcsin(p * vp2)
        theta_s1 = complex_arcsin(p * vs1)
        theta_s2 = complex_arcsin(p * vs2)

        # Ensure consistent complex dtype if any angle is complex
        tensors = [theta, theta_p2, theta_s1, theta_s2, p]
        if any(t.is_complex() for t in tensors):
            theta = theta.to(torch.complex64)
            theta_p2 = theta_p2.to(torch.complex64)
            theta_s1 = theta_s1.to(torch.complex64)
            theta_s2 = theta_s2.to(torch.complex64)
            p = p.to(torch.complex64)

        # Trigonometric functions
        cos1 = torch.cos(theta)
        cos2 = torch.cos(theta_p2)
        cos_s1 = torch.cos(theta_s1)
        cos_s2 = torch.cos(theta_s2)
        sin_s1 = torch.sin(theta_s1)
        sin_s2 = torch.sin(theta_s2)

        # Zoeppritz matrix coefficients
        a = rho2 * (1 - 2 * sin_s2 ** 2) - rho1 * (1 - 2 * sin_s1 ** 2)
        b = rho2 * (1 - 2 * sin_s2 ** 2) + 2 * rho1 * sin_s1 ** 2
        c = rho1 * (1 - 2 * sin_s1 ** 2) + 2 * rho2 * sin_s2 ** 2
        d = 2 * (rho2 * vs2 ** 2 - rho1 * vs1 ** 2)

        E = b * cos1 / vp1 + c * cos2 / vp2
        F = b * cos_s1 / vs1 + c * cos_s2 / vs2
        G = a - d * cos1 / vp1 * cos_s2 / vs2
        H = a - d * cos2 / vp2 * cos_s1 / vs1

        # Denominator (avoid division by zero)
        D = E * F + G * H * p ** 2
        D = torch.where(
            torch.abs(D) < eps,
            torch.tensor(eps, dtype=D.dtype, device=D.device),
            D
        )

        # P-P reflection coefficient
        numerator = (
            F * (b * cos1 / vp1 - c * cos2 / vp2) -
            H * p ** 2 * (a + d * cos1 / vp1 * cos_s2 / vs2)
        )
        Rpp = numerator / D

        # Return real if imaginary part is negligible
        if Rpp.is_complex() and torch.abs(Rpp.imag).max() < eps:
            Rpp = Rpp.real

        return Rpp