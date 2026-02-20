"""
Linearized AVO approximations: Aki-Richards and Shuey.

These approximations are computationally efficient alternatives
to the full Zoeppritz equations for small contrasts and moderate angles.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
from torch import Tensor
from typing import Tuple

from .base import AVOModel, vectorize_angles


class AkiRichards(AVOModel):
    """
    Aki-Richards linearized approximation.
    
    A four-term linearization of the Zoeppritz equations valid
    for small velocity/density contrasts and angles up to ~40°.
    
    The approximation is:

        R(θ) ≈ (1/2)(1 + tan²θ)(Δvp/vp)
             - 4k²sin²θ(Δvs/vs)
             + (1/2)(1 - 4k²sin²θ)(Δρ/ρ)

    where k = vs/vp is the velocity ratio.

    Example:
        >>> ar = AkiRichards()
        >>> rc = ar(vp1=2000, vs1=1000, rho1=2.0,
        ...         vp2=2100, vs2=1050, rho2=2.1,
        ...         theta=[0, 10, 20, 30])

    Note:
        - Valid for angles 0-40 degrees
        - Requires small contrasts (< 20%)
        - More accurate than Shuey for moderate contrasts
    """

    def __init__(self):
        super().__init__()
        self._name = 'aki_richards'
        self._valid_angles = (0, 40)

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
        Compute Aki-Richards reflection coefficient.

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
        """
        # Average properties
        vp = (vp1 + vp2) / 2
        vs = (vs1 + vs2) / 2
        rho = (rho1 + rho2) / 2

        # Contrasts (relative changes)
        dvp = vp2 - vp1
        dvs = vs2 - vs1
        drho = rho2 - rho1

        # Angle functions
        sin2 = torch.sin(theta) ** 2
        tan2 = torch.tan(theta) ** 2

        # Vs/Vp ratio squared
        k2 = (vs / vp) ** 2

        # Aki-Richards formula (three terms)
        term1 = 0.5 * (1 + tan2) * (dvp / vp)
        term2 = -4 * k2 * sin2 * (dvs / vs)
        term3 = 0.5 * (1 - 4 * k2 * sin2) * (drho / rho)

        return term1 + term2 + term3


class Shuey(AVOModel):
    """
    Shuey's three-term AVO approximation.

    A simplified form expressing reflectivity in terms of
    interpretable AVO attributes (intercept, gradient, curvature):

        R(θ) = R₀ + G·sin²θ + F·(tan²θ - sin²θ)

    where:
        R₀ = Intercept (normal incidence reflectivity)
        G  = Gradient (AVO gradient)
        F  = Curvature (far-offset term)

    Example:
        Basic usage:
        >>> shuey = Shuey()
        >>> rc = shuey(vp1=2000, vs1=1000, rho1=2.0,
        ...            vp2=2500, vs2=1250, rho2=2.2,
        ...            theta=[0, 15, 30, 45])

        Get AVO attributes directly:
        >>> R0, G, F = shuey.avo_attributes(vp1, vs1, rho1, vp2, vs2, rho2)

    Note:
        - Valid for angles 0-45 degrees
        - Most commonly used in industry
        - Direct access to AVO attributes for interpretation
    """

    def __init__(self):
        super().__init__()
        self._name = 'shuey'
        self._valid_angles = (0, 45)

    def _compute_attributes(
        self,
        vp1: Tensor,
        vs1: Tensor,
        rho1: Tensor,
        vp2: Tensor,
        vs2: Tensor,
        rho2: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute AVO attributes (R0, G, F).

        Args:
            vp1: P-wave velocity of upper layer (m/s).
            vs1: S-wave velocity of upper layer (m/s).
            rho1: Density of upper layer (kg/m³).
            vp2: P-wave velocity of lower layer (m/s).
            vs2: S-wave velocity of lower layer (m/s).
            rho2: Density of lower layer (kg/m³).

        Returns:
            Tuple of (R0, G, F) tensors:
                - R0: Intercept (normal incidence reflectivity)
                - G: Gradient
                - F: Curvature
        """
        # Average properties
        vp = (vp1 + vp2) / 2
        vs = (vs1 + vs2) / 2
        rho = (rho1 + rho2) / 2

        # Contrasts
        dvp = vp2 - vp1
        dvs = vs2 - vs1
        drho = rho2 - rho1

        # Vs/Vp ratio squared
        k2 = (vs / vp) ** 2

        # Shuey coefficients
        R0 = 0.5 * (dvp / vp + drho / rho)
        G = 0.5 * (dvp / vp) - 2 * k2 * (drho / rho + 2 * dvs / vs)
        F = 0.5 * (dvp / vp)

        return R0, G, F

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
        Compute Shuey reflection coefficient.

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
        """
        # Compute AVO attributes
        R0, G, F = self._compute_attributes(vp1, vs1, rho1, vp2, vs2, rho2)

        # Angle terms
        sin2 = torch.sin(theta) ** 2
        tan2 = torch.tan(theta) ** 2

        # Shuey equation: R(θ) = R₀ + G·sin²θ + F·(tan²θ - sin²θ)
        return R0 + G * sin2 + F * (tan2 - sin2)

    def avo_attributes(
        self,
        vp1, vs1, rho1,
        vp2, vs2, rho2,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute AVO attributes directly.

        Args:
            vp1: P-wave velocity of upper layer (m/s).
            vs1: S-wave velocity of upper layer (m/s).
            rho1: Density of upper layer (kg/m³).
            vp2: P-wave velocity of lower layer (m/s).
            vs2: S-wave velocity of lower layer (m/s).
            rho2: Density of lower layer (kg/m³).

        Returns:
            Tuple of (R0, G, F):
                - R0: Intercept (normal incidence reflectivity)
                - G: Gradient
                - F: Curvature

        Example:
            >>> R0, G, F = shuey.avo_attributes(vp1, vs1, rho1, vp2, vs2, rho2)
            >>>
            >>> # AVO classification
            >>> class_I = (R0 < 0) & (G > 0)    # Dim spot
            >>> class_II = torch.abs(R0) < 0.02  # Near-zero intercept
            >>> class_III = (R0 > 0) & (G < 0)   # Bright spot
        """
        from .base import as_tensor

        vp1 = as_tensor(vp1)
        vs1 = as_tensor(vs1)
        rho1 = as_tensor(rho1)
        vp2 = as_tensor(vp2)
        vs2 = as_tensor(vs2)
        rho2 = as_tensor(rho2)

        return self._compute_attributes(vp1, vs1, rho1, vp2, vs2, rho2)