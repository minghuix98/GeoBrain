#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Anisotropy models for rock physics.

This module provides models for computing elastic anisotropy in
rocks, including Thomsen parameters for VTI media, Backus averaging
for layered media, and Bond transformation for stiffness rotation.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
from torch import Tensor
from typing import Tuple

from ..core import AnisotropyModel, register, as_tensor, EPS, PI


@register(name="Thomsen", category="anisotropy")
class Thomsen(AnisotropyModel):
    """
    Thomsen parameters for VTI media.

    Computes weak anisotropy parameters (ε, γ, δ) and angle-dependent
    velocities from stiffness coefficients for vertically transverse
    isotropic (VTI) media.

    The Thomsen parameters characterize weak anisotropy:
        ε = (C11 - C33) / (2*C33)      (P-wave anisotropy)
        γ = (C66 - C44) / (2*C44)      (SH-wave anisotropy)
        δ = ((C13+C44)² - (C33-C44)²) / (2*C33*(C33-C44))

    References:
        Thomsen, L. (1986). Weak elastic anisotropy. Geophysics, 51(10),
        1954-1966.

    Args:
        None (stateless model)

    Example:
        >>> thomsen = Thomsen()
        >>> VP, VSV, VSH, eps, gam, delta = thomsen(
        ...     C11=100, C33=80, C13=30, C44=25, C66=30,
        ...     rho=2.5, theta=30
        ... )
        >>> print(f"ε={eps.item():.3f}, γ={gam.item():.3f}")
    """

    def __init__(self):
        super().__init__()

    def forward(
            self,
            C11: Tensor,
            C33: Tensor,
            C13: Tensor,
            C44: Tensor,
            C66: Tensor,
            rho: Tensor,
            theta: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Compute Thomsen parameters and angle-dependent velocities.

        Args:
            C11: Stiffness coefficient C11 (GPa)
            C33: Stiffness coefficient C33 (GPa)
            C13: Stiffness coefficient C13 (GPa)
            C44: Stiffness coefficient C44 (GPa)
            C66: Stiffness coefficient C66 (GPa)
            rho: Density (g/cm³)
            theta: Propagation angle from vertical (degrees)

        Returns:
            Tuple of (VP, VSV, VSH, epsilon, gamma, delta):
                - VP: P-wave velocity (m/s)
                - VSV: SV-wave velocity (m/s)
                - VSH: SH-wave velocity (m/s)
                - epsilon: P-wave anisotropy parameter
                - gamma: SH-wave anisotropy parameter
                - delta: Near-vertical anisotropy parameter
        """
        C11, C33, C13 = as_tensor(C11), as_tensor(C33), as_tensor(C13)
        C44, C66, rho = as_tensor(C44), as_tensor(C66), as_tensor(rho)
        theta = as_tensor(theta) * PI / 180

        # Vertical velocities
        alpha = torch.sqrt(C33 / rho) * 1e3
        beta = torch.sqrt(C44 / rho) * 1e3

        # Thomsen parameters
        epsilon = 0.5 * (C11 - C33) / C33
        gamma = 0.5 * (C66 - C44) / C44
        delta = ((C13 + C44) ** 2 - (C33 - C44) ** 2) / (2 * C33 * (C33 - C44) + EPS)

        # Angle-dependent velocities (weak anisotropy approximation)
        sin_t, cos_t = torch.sin(theta), torch.cos(theta)
        VP = alpha * (1 + delta * sin_t ** 2 * cos_t ** 2 + epsilon * sin_t ** 4)
        VSV = beta * (1 + alpha ** 2 / beta ** 2 * (epsilon - delta) * sin_t ** 2 * cos_t ** 2)
        VSH = beta * (1 + gamma * sin_t ** 2)

        return VP, VSV, VSH, epsilon, gamma, delta


@register(name="Backus", category="anisotropy")
class Backus(AnisotropyModel):
    """
    Backus averaging for layered media.

    Computes effective VTI stiffness from a stack of thin isotropic
    layers using long-wavelength equivalent medium theory. The layers
    must be thin compared to the seismic wavelength.

    The effective medium is VTI (transversely isotropic with vertical
    symmetry axis) even if individual layers are isotropic.

    References:
        Backus, G.E. (1962). Long-wave elastic anisotropy produced by
        horizontal layering. Journal of Geophysical Research, 67(11),
        4427-4440.

    Args:
        None (stateless model)

    Example:
        From elastic moduli:
        >>> backus = Backus()
        >>> f = torch.tensor([0.6, 0.4])         # Volume fractions
        >>> lamda = torch.tensor([10.0, 5.0])    # Lamé parameter (GPa)
        >>> G = torch.tensor([30.0, 10.0])       # Shear modulus (GPa)
        >>> C11, C33, C13, C44, C66 = backus(f, lamda, G)

        From well logs:
        >>> C11, C33, C13, C44, C66, rho = backus.from_logs(
        ...     Vp=torch.tensor([4000, 3000]),
        ...     Vs=torch.tensor([2500, 1800]),
        ...     rho=torch.tensor([2.5, 2.3]),
        ...     thickness=torch.tensor([10, 5])
        ... )
    """

    def __init__(self):
        super().__init__()

    def forward(
            self,
            f: Tensor,
            lamda: Tensor,
            G: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Compute effective VTI stiffness from layer properties.

        Args:
            f: Volume fractions of each layer (will be normalized)
            lamda: Lamé first parameter for each layer (GPa)
            G: Shear modulus for each layer (GPa)

        Returns:
            Tuple of (C11, C33, C13, C44, C66) effective stiffness (GPa)
        """
        f, lamda, G = as_tensor(f), as_tensor(lamda), as_tensor(G)
        f = f / torch.sum(f)  # Normalize fractions
        M = lamda + 2 * G  # P-wave modulus

        # Backus averaging formulas
        C33 = 1.0 / torch.sum(f / M)
        C44 = 1.0 / torch.sum(f / G)
        C66 = torch.sum(f * G)
        C13 = C33 * torch.sum(f * lamda / M)
        C11 = torch.sum(f * 4 * G * (lamda + G) / M) + C33 * torch.sum(f * lamda / M) ** 2

        return C11, C33, C13, C44, C66

    def from_logs(
            self,
            Vp: Tensor,
            Vs: Tensor,
            rho: Tensor,
            thickness: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Compute effective VTI stiffness from well log data.

        Args:
            Vp: P-wave velocity for each layer (m/s)
            Vs: S-wave velocity for each layer (m/s)
            rho: Density for each layer (g/cm³)
            thickness: Thickness of each layer (m)

        Returns:
            Tuple of (C11, C33, C13, C44, C66, rho_eff):
                - C11-C66: Effective stiffness coefficients (GPa)
                - rho_eff: Effective density (g/cm³)
        """
        Vp, Vs, rho = as_tensor(Vp), as_tensor(Vs), as_tensor(rho)
        thickness = as_tensor(thickness)

        # Convert velocities to moduli
        G = rho * Vs ** 2
        lamda = rho * Vp ** 2 - 2 * G
        f = thickness / torch.sum(thickness)

        C11, C33, C13, C44, C66 = self.forward(f, lamda, G)
        rho_eff = torch.sum(f * rho)

        return C11, C33, C13, C44, C66, rho_eff


@register(name="BondTransform", category="anisotropy", aliases=["Bond"])
class BondTransform(AnisotropyModel):
    """
    Bond transformation for stiffness matrix rotation.

    Rotates a 6x6 Voigt stiffness matrix around a specified axis
    using the Bond transformation matrix. This is useful for
    converting between different anisotropy orientations (e.g.,
    VTI to TTI, rotation around any principal axis).

    References:
        Bond, W.L. (1943). The mathematics of the physical properties of
        crystals. Bell System Technical Journal, 22(1), 1-72.
        Auld, B.A. (1973). Acoustic Fields and Waves in Solids. Wiley.

    Args:
        None (stateless model)

    Example:
        >>> bond = BondTransform()
        >>> C_vti = torch.eye(6) * 10  # Example VTI stiffness
        >>> C_rotated = bond(C_vti, theta=45, axis=1)  # Rotate 45° around x-axis
    """

    def __init__(self):
        super().__init__()

    def forward(self, C: Tensor, theta: Tensor, axis: int = 3) -> Tensor:
        """
        Rotate stiffness matrix by angle theta around specified axis.

        Args:
            C: 6x6 Voigt stiffness matrix (GPa)
            theta: Rotation angle (degrees)
            axis: Rotation axis (1 for x-axis, 2 for y-axis, 3 for z-axis)

        Returns:
            Rotated 6x6 stiffness matrix (GPa)

        Raises:
            ValueError: If axis is not 1, 2, or 3
        """
        C = as_tensor(C)
        theta_rad = as_tensor(theta) * PI / 180
        cos_t = torch.cos(theta_rad).item()
        sin_t = torch.sin(theta_rad).item()

        # Build rotation matrix elements based on axis
        if axis == 3:  # Rotation around z-axis
            b11, b22, b33 = cos_t, cos_t, 1.0
            b12, b21 = sin_t, -sin_t
            b13, b31, b23, b32 = 0.0, 0.0, 0.0, 0.0
        elif axis == 1:  # Rotation around x-axis
            b11 = 1.0
            b22, b33 = cos_t, cos_t
            b23, b32 = -sin_t, sin_t
            b12, b21, b13, b31 = 0.0, 0.0, 0.0, 0.0
        elif axis == 2:  # Rotation around y-axis
            b11, b33 = cos_t, cos_t
            b22 = 1.0
            b13, b31 = sin_t, -sin_t
            b12, b21, b23, b32 = 0.0, 0.0, 0.0, 0.0
        else:
            raise ValueError(f"axis must be 1, 2, or 3, got {axis}")

        # Build Bond transformation matrix blocks
        M1 = torch.tensor([[b11 ** 2, b12 ** 2, b13 ** 2],
                           [b21 ** 2, b22 ** 2, b23 ** 2],
                           [b31 ** 2, b32 ** 2, b33 ** 2]], dtype=C.dtype)
        M2 = torch.tensor([[b12 * b13, b13 * b11, b11 * b12],
                           [b22 * b23, b23 * b21, b21 * b22],
                           [b32 * b33, b33 * b31, b31 * b32]], dtype=C.dtype)
        M3 = torch.tensor([[b21 * b31, b22 * b32, b23 * b33],
                           [b31 * b11, b32 * b12, b33 * b13],
                           [b11 * b21, b12 * b22, b13 * b23]], dtype=C.dtype)
        M4 = torch.tensor([[b22 * b33 + b23 * b32, b21 * b33 + b23 * b31, b22 * b31 + b21 * b32],
                           [b12 * b33 + b13 * b32, b11 * b33 + b13 * b31, b11 * b32 + b12 * b31],
                           [b22 * b13 + b12 * b23, b11 * b23 + b13 * b21, b22 * b11 + b12 * b21]], dtype=C.dtype)

        # Assemble 6x6 Bond matrix
        M = torch.zeros(6, 6, dtype=C.dtype)
        M[:3, :3] = M1
        M[:3, 3:] = M2
        M[3:, :3] = M3
        M[3:, 3:] = M4

        return M @ C @ M.T


# =============================================================================
# Additional Anisotropy Models
# =============================================================================

@register(name="VelocityAzimuthHTI", category="anisotropy")
class VelocityAzimuthHTI(AnisotropyModel):
    """
    Azimuth-dependent velocities for HTI medium.

    Computes P, SH, and SV phase velocities as a function of
    azimuth angle for a horizontally transverse isotropic medium.

    References:
        Ruger, A. (1997). P-wave reflection coefficients for transversely
        isotropic models with vertical and horizontal axis of symmetry.
        Geophysics, 62(3), 713-722.

    Args:
        None (stateless model)
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "HTI azimuth-dependent velocities")

    def forward(
            self,
            C: Tensor,
            rho: Tensor,
            azimuth: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute azimuth-dependent velocities.

        Args:
            C: 6x6 stiffness matrix of HTI medium (GPa)
            rho: Density (g/cm3)
            azimuth: Azimuth angle (degrees)

        Returns:
            Tuple of (VP, VSH, VSV) in km/s
        """
        C = as_tensor(C)
        rho = as_tensor(rho)
        azimuth_rad = as_tensor(azimuth) * PI / 180.0

        C11 = C[0, 0]
        C12 = C[0, 1]
        C33 = C[2, 2]
        C44 = C[3, 3]
        C55 = C[4, 4]

        sin_a = torch.sin(azimuth_rad)
        cos_a = torch.cos(azimuth_rad)
        sin2 = sin_a ** 2
        cos2 = cos_a ** 2

        R = torch.sqrt(
            ((C33 - C55) * sin2 - (C11 - C55) * cos2) ** 2 +
            4.0 * (C12 + C55) ** 2 * sin2 * cos2
        )

        Vp2 = 0.5 / rho * ((C33 + C55) * sin2 + (C11 + C55) * cos2 + R)
        Vsh2 = 0.5 / rho * ((C33 + C55) * sin2 + (C11 + C55) * cos2 - R)
        Vsv2 = 1.0 / rho * (C55 + (C44 - C55) * sin2)

        VP = torch.sqrt(torch.clamp(Vp2, min=EPS))
        VSH = torch.sqrt(torch.clamp(Vsh2, min=EPS))
        VSV = torch.sqrt(torch.clamp(Vsv2, min=EPS))

        return VP, VSH, VSV


@register(name="VelocityAzimuthVTI", category="anisotropy")
class VelocityAzimuthVTI(AnisotropyModel):
    """
    Azimuth-dependent velocities for VTI medium.

    Computes P, SH, and SV phase velocities as a function of
    propagation angle for a vertically transverse isotropic medium.

    References:
        Thomsen, L. (1986). Weak elastic anisotropy. Geophysics, 51(10),
        1954-1966.

    Args:
        None (stateless model)
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "VTI azimuth-dependent velocities")

    def forward(
            self,
            C: Tensor,
            rho: Tensor,
            theta: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute angle-dependent velocities.

        Args:
            C: 6x6 stiffness matrix of VTI medium (GPa)
            rho: Density (g/cm3)
            theta: Propagation angle from vertical (degrees)

        Returns:
            Tuple of (VP, VSH, VSV) in km/s
        """
        C = as_tensor(C)
        rho = as_tensor(rho)
        theta_rad = as_tensor(theta) * PI / 180.0

        C11 = C[0, 0]
        C13 = C[0, 2]
        C33 = C[2, 2]
        C44 = C[3, 3]
        C66 = C[5, 5]

        sin_a = torch.sin(theta_rad)
        cos_a = torch.cos(theta_rad)
        sin2 = sin_a ** 2
        cos2 = cos_a ** 2

        R = torch.sqrt(
            ((C11 - C44) * sin2 - (C33 - C44) * cos2) ** 2 +
            (C13 + C44) ** 2 * torch.sin(2.0 * theta_rad) ** 2
        )

        Vp2 = 0.5 / rho * (C11 * sin2 + C33 * cos2 + C44 + R)
        Vsv2 = 0.5 / rho * (C11 * sin2 + C33 * cos2 + C44 - R)
        Vsh2 = 1.0 / rho * (C66 * sin2 + C44 * cos2)

        VP = torch.sqrt(torch.clamp(Vp2, min=EPS))
        VSH = torch.sqrt(torch.clamp(Vsh2, min=EPS))
        VSV = torch.sqrt(torch.clamp(Vsv2, min=EPS))

        return VP, VSH, VSV


@register(name="ThomsenTsvankin", category="anisotropy")
class ThomsenTsvankin(AnisotropyModel):
    """
    Thomsen-Tsvankin orthorhombic anisotropy parameters.

    Computes 7 anisotropy parameters for an orthorhombic medium
    from the 9 independent stiffness coefficients.

    References:
        Tsvankin, I. (1997). Anisotropic parameters and P-wave velocity for
        orthorhombic media. Geophysics, 62(4), 1292-1309.

    Args:
        None (stateless model)
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "Thomsen-Tsvankin orthorhombic parameters")

    def forward(
            self,
            C11: Tensor,
            C22: Tensor,
            C33: Tensor,
            C12: Tensor,
            C13: Tensor,
            C23: Tensor,
            C44: Tensor,
            C55: Tensor,
            C66: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Compute Thomsen-Tsvankin parameters.

        Args:
            C11-C66: Independent stiffness coefficients (GPa)

        Returns:
            Tuple of (eps1, delta1, gamma1, eps2, delta2, gamma2, delta3)
        """
        C11, C22, C33 = as_tensor(C11), as_tensor(C22), as_tensor(C33)
        C12, C13, C23 = as_tensor(C12), as_tensor(C13), as_tensor(C23)
        C44, C55, C66 = as_tensor(C44), as_tensor(C55), as_tensor(C66)

        # Symmetry plane with normal in x1 direction
        eps1 = (C22 - C33) / (2.0 * C33 + EPS)
        delta1 = ((C23 + C44) ** 2 - (C33 - C44) ** 2) / (2.0 * C33 * (C33 - C44) + EPS)
        gamma1 = (C66 - C55) / (2.0 * C55 + EPS)

        # Symmetry plane with normal in x2 direction
        eps2 = (C11 - C33) / (2.0 * C33 + EPS)
        delta2 = ((C13 + C55) ** 2 - (C33 - C55) ** 2) / (2.0 * C33 * (C33 - C55) + EPS)
        gamma2 = (C66 - C44) / (2.0 * C44 + EPS)

        # Horizontal plane with normal in x3 direction
        delta3 = ((C12 + C66) ** 2 - (C11 - C66) ** 2) / (2.0 * C11 * (C11 - C66) + EPS)

        return eps1, delta1, gamma1, eps2, delta2, gamma2, delta3