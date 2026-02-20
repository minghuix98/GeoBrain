#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Effective medium models for rock physics.

This module implements theoretical bounds and effective medium theories
for computing elastic properties of composite materials.

Bounds:
    - Voigt (upper bound): Iso-strain average
    - Reuss (lower bound): Iso-stress average
    - Voigt-Reuss-Hill: Arithmetic mean of bounds
    - Hashin-Shtrikman: Tightest possible bounds for isotropic composites

Effective Medium Theories:
    - DEM (Differential Effective Medium)
    - Self-Consistent Approximation
    - Critical Porosity Model

Crack/Inclusion Models:
    - Hudson (aligned penny-shaped cracks)
    - Hudson Random (randomly oriented cracks)
    - Kuster-Toksöz (spherical inclusions)
    - Eshelby-Cheng (cracked isotropic rock)

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
from torch import Tensor
from typing import Tuple, Optional

from ..core import EffectiveModel, register, as_tensor, ensure_same_device, EPS, PI
from ..utils import write_vti_matrix, write_hti_matrix, write_ortho_matrix


# =============================================================================
# Bounds
# =============================================================================

@register(name="Voigt", category="effective")
class Voigt(EffectiveModel):
    """
    Voigt upper bound (iso-strain average).

    The Voigt bound assumes uniform strain throughout the composite,
    giving the arithmetic (volume-weighted) average:

        M_Voigt = Σ(f_i * M_i)

    This is an upper bound for the effective modulus of any
    isotropic composite.

    References:
        Voigt, W. (1928). Lehrbuch der Kristallphysik. Teubner, Leipzig.

    Args:
        None (stateless model)

    Example:
        >>> voigt = Voigt()
        >>> M = torch.tensor([36.6, 21.0])  # Quartz, Clay bulk moduli (GPa)
        >>> f = torch.tensor([0.7, 0.3])    # Volume fractions
        >>> K_voigt = voigt(M, f)
        >>> print(f"K_Voigt = {K_voigt.item():.1f} GPa")
        K_Voigt = 31.9 GPa
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "Voigt upper bound (iso-strain)")
        self.set_metadata("reference", "Voigt (1928)")

    def forward(self, M: Tensor, f: Tensor) -> Tensor:
        """
        Compute Voigt average.

        Args:
            M: Moduli of components (GPa)
            f: Volume fractions (should sum to 1)

        Returns:
            Effective modulus (GPa)
        """
        M, f = as_tensor(M), as_tensor(f)
        M, f = ensure_same_device(M, f)
        return torch.sum(f * M)


@register(name="Reuss", category="effective")
class Reuss(EffectiveModel):
    """
    Reuss lower bound (iso-stress average).

    The Reuss bound assumes uniform stress throughout the composite,
    giving the harmonic (volume-weighted) average:

        M_Reuss = 1 / Σ(f_i / M_i)

    This is a lower bound for the effective modulus of any
    isotropic composite.

    References:
        Reuss, A. (1929). Berechnung der Fliessgrenze von Mischkristallen.
        Zeitschrift fur Angewandte Mathematik und Mechanik, 9(1), 49-58.

    Args:
        None (stateless model)

    Example:
        >>> reuss = Reuss()
        >>> M = torch.tensor([36.6, 21.0])  # Quartz, Clay bulk moduli (GPa)
        >>> f = torch.tensor([0.7, 0.3])    # Volume fractions
        >>> K_reuss = reuss(M, f)
        >>> print(f"K_Reuss = {K_reuss.item():.1f} GPa")
        K_Reuss = 30.2 GPa
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "Reuss lower bound (iso-stress)")
        self.set_metadata("reference", "Reuss (1929)")

    def forward(self, M: Tensor, f: Tensor) -> Tensor:
        """
        Compute Reuss average.

        Args:
            M: Moduli of components (GPa)
            f: Volume fractions (should sum to 1)

        Returns:
            Effective modulus (GPa)
        """
        M, f = as_tensor(M), as_tensor(f)
        M, f = ensure_same_device(M, f)
        return 1.0 / torch.sum(f / (M + EPS))


@register(name="VRH", category="effective", aliases=["VoigtReussHill", "Hill"])
class VRH(EffectiveModel):
    """
    Voigt-Reuss-Hill average.

    The VRH average is the arithmetic mean of Voigt and Reuss bounds:

        M_VRH = 0.5 * (M_Voigt + M_Reuss)

    This provides a reasonable estimate for isotropic polycrystalline
    aggregates and is widely used in rock physics.

    References:
        Hill, R. (1952). The elastic behaviour of a crystalline aggregate.
        Proceedings of the Physical Society A, 65(5), 349-354.

    Args:
        None (stateless model)

    Example:
        >>> vrh = VRH()
        >>> M = torch.tensor([36.6, 21.0])  # Moduli (GPa)
        >>> f = torch.tensor([0.7, 0.3])    # Volume fractions
        >>> M_v, M_r, M_vrh = vrh(M, f)
        >>> print(f"VRH average: {M_vrh.item():.1f} GPa")
        VRH average: 31.0 GPa
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "Voigt-Reuss-Hill average")
        self.set_metadata("reference", "Hill (1952)")
        self._voigt = Voigt()
        self._reuss = Reuss()

    def forward(self, M: Tensor, f: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute VRH average.

        Args:
            M: Moduli of components (GPa)
            f: Volume fractions

        Returns:
            Tuple of (M_voigt, M_reuss, M_vrh) in GPa
        """
        M_voigt = self._voigt(M, f)
        M_reuss = self._reuss(M, f)
        M_vrh = 0.5 * (M_voigt + M_reuss)
        return M_voigt, M_reuss, M_vrh


@register(name="HashinShtrikman", category="effective", aliases=["HS"])
class HashinShtrikman(EffectiveModel):
    """
    Hashin-Shtrikman bounds for two-phase composites.

    The HS bounds are the tightest possible bounds for the effective
    moduli of an isotropic composite of two isotropic phases:

        K_upper: Stiff phase as matrix (upper bound)
        K_lower: Soft phase as matrix (lower bound)

    References:
        Hashin, Z. & Shtrikman, S. (1963). A variational approach to the
        theory of the elastic behaviour of multiphase materials. Journal of
        the Mechanics and Physics of Solids, 11(2), 127-140.

    Args:
        None (stateless model)

    Example:
        >>> hs = HashinShtrikman()
        >>> K_up, K_lo, G_up, G_lo = hs(
        ...     K1=36.6, G1=45.0,  # Quartz (stiff phase)
        ...     K2=21.0, G2=7.0,   # Clay (soft phase)
        ...     f=0.7              # Quartz volume fraction
        ... )
        >>> print(f"K bounds: [{K_lo.item():.1f}, {K_up.item():.1f}] GPa")
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "Hashin-Shtrikman bounds")
        self.set_metadata("reference", "Hashin & Shtrikman (1963)")

    def forward(
            self,
            K1: Tensor,
            G1: Tensor,
            K2: Tensor,
            G2: Tensor,
            f: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Compute Hashin-Shtrikman bounds.

        Args:
            K1: Bulk modulus of phase 1 (stiff phase, GPa)
            G1: Shear modulus of phase 1 (GPa)
            K2: Bulk modulus of phase 2 (soft phase, GPa)
            G2: Shear modulus of phase 2 (GPa)
            f: Volume fraction of phase 1

        Returns:
            Tuple of (K_upper, K_lower, G_upper, G_lower) in GPa
        """
        K1, G1 = as_tensor(K1), as_tensor(G1)
        K2, G2 = as_tensor(K2), as_tensor(G2)
        f = as_tensor(f)
        K1, G1, K2, G2, f = ensure_same_device(K1, G1, K2, G2, f)

        # Upper bounds (stiff phase as matrix)
        zeta_K1 = K1 + 4.0 * G1 / 3.0
        K_upper = K1 + (1.0 - f) / (1.0 / (K2 - K1 + EPS) + f / (zeta_K1 + EPS))

        zeta_G1 = (K1 + 2.0 * G1) / (5.0 * G1 * (K1 + 4.0 * G1 / 3.0) + EPS)
        G_upper = G1 + (1.0 - f) / (1.0 / (G2 - G1 + EPS) + 2.0 * f * zeta_G1)

        # Lower bounds (soft phase as matrix)
        zeta_K2 = K2 + 4.0 * G2 / 3.0
        K_lower = K2 + f / (1.0 / (K1 - K2 + EPS) + (1.0 - f) / (zeta_K2 + EPS))

        zeta_G2 = (K2 + 2.0 * G2) / (5.0 * G2 * (K2 + 4.0 * G2 / 3.0) + EPS)
        G_lower = G2 + f / (1.0 / (G1 - G2 + EPS) + 2.0 * (1.0 - f) * zeta_G2)

        return K_upper, K_lower, G_upper, G_lower


# =============================================================================
# Effective Medium Theories
# =============================================================================

@register(name="DEM", category="effective")
class DEM(EffectiveModel):
    """
    Differential Effective Medium model.

    DEM builds the composite incrementally by adding infinitesimal
    amounts of inclusion to the host material. The effective moduli
    are computed by numerical integration of coupled ODEs.

    References:
        Berryman, J.G. (1992). Single-scattering approximations for
        coefficients in Biot's equations of poroelasticity. Journal of the
        Acoustical Society of America, 91(2), 551-571.

        Norris, A.N. (1985). A differential scheme for the effective moduli
        of composites. Mechanics of Materials, 4(1), 1-16.

    Args:
        n_steps: Number of integration steps. Default: 100

    Example:
        >>> dem = DEM(n_steps=200)
        >>> K_eff, G_eff = dem(
        ...     K_host=76.8, G_host=32.0,  # Calcite matrix (GPa)
        ...     K_inc=0.0, G_inc=0.0,       # Dry pores
        ...     phi=0.20                    # 20% porosity
        ... )
    """

    def __init__(self, n_steps: int = 100):
        super().__init__()
        self.n_steps = n_steps
        self.set_metadata("description", "Differential Effective Medium")
        self.set_metadata("reference", "Norris (1985)")

    def forward(
            self,
            K_host: Tensor,
            G_host: Tensor,
            K_inc: Tensor,
            G_inc: Tensor,
            phi: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute effective moduli using DEM.

        Args:
            K_host: Host material bulk modulus (GPa)
            G_host: Host material shear modulus (GPa)
            K_inc: Inclusion bulk modulus (GPa, 0 for pores)
            G_inc: Inclusion shear modulus (GPa, 0 for pores)
            phi: Volume fraction of inclusions (porosity)

        Returns:
            Tuple of (K_effective, G_effective) in GPa
        """
        K = as_tensor(K_host).clone()
        G = as_tensor(G_host).clone()
        K_i = as_tensor(K_inc)
        G_i = as_tensor(G_inc)
        phi = as_tensor(phi)

        dt = phi / self.n_steps

        for i in range(self.n_steps):
            t = i * dt

            # P and Q factors for spherical inclusions
            zeta = G / 6.0 * (9.0 * K + 8.0 * G) / (K + 2.0 * G + EPS)
            P = (K + 4.0 * G / 3.0) / (K_i + 4.0 * G / 3.0 + EPS)
            Q = (G + zeta) / (G_i + zeta + EPS)

            # Integration step
            scale = dt / (1.0 - t + EPS)
            K = K + scale * (K_i - K) * P
            G = G + scale * (G_i - G) * Q

        return K, G

    def _forward_dry(
            self,
            K_host: Tensor,
            G_host: Tensor,
            phi: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Optimized DEM for dry pores (K_inc=0, G_inc=0).

        When inclusions are empty pores, the ODE simplifies because
        K_i=0 and G_i=0, avoiding unnecessary additions with zero.

        Args:
            K_host: Host material bulk modulus (GPa)
            G_host: Host material shear modulus (GPa)
            phi: Volume fraction of pores (porosity)

        Returns:
            Tuple of (K_effective, G_effective) in GPa
        """
        K = as_tensor(K_host).clone()
        G = as_tensor(G_host).clone()
        phi = as_tensor(phi)

        dt = phi / self.n_steps

        for i in range(self.n_steps):
            t = i * dt
            scale = dt / (1.0 - t + EPS)

            # Simplified P and Q for K_i=0, G_i=0
            P = (K + 4.0 * G / 3.0) / (4.0 * G / 3.0 + EPS)
            zeta = G / 6.0 * (9.0 * K + 8.0 * G) / (K + 2.0 * G + EPS)
            Q = (G + zeta) / (zeta + EPS)

            # Simplified: (K_i - K)*P = -K*P, (G_i - G)*Q = -G*Q
            K = K - scale * K * P
            G = G - scale * G * Q

        return K, G

    def compute_dry_rock(self, K0, G0, phi, **params):
        # For dry rock: host = mineral, inclusions = pores (K=0, G=0)
        return self._forward_dry(K0, G0, phi)


@register(name="SelfConsistent", category="effective", aliases=["SC"])
class SelfConsistent(EffectiveModel):
    """
    Self-Consistent approximation for two-phase composites.

    The SC method treats each phase as an inclusion embedded in the
    effective medium itself, solved iteratively until convergence.

    References:
        Berryman, J.G. (1980). Long-wavelength propagation in composite
        elastic media I. Spherical inclusions. Journal of the Acoustical
        Society of America, 68(6), 1809-1819.

    Args:
        max_iter: Maximum iterations. Default: 50
        tol: Convergence tolerance. Default: 1e-6

    Example:
        >>> sc = SelfConsistent()
        >>> K_eff, G_eff = sc(
        ...     K1=36.6, G1=45.0,  # Phase 1 (GPa)
        ...     K2=21.0, G2=7.0,   # Phase 2 (GPa)
        ...     f1=0.7             # Phase 1 volume fraction
        ... )
    """

    def __init__(self, max_iter: int = 50, tol: float = 1e-6):
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol
        self.set_metadata("description", "Self-Consistent approximation")
        self.set_metadata("reference", "Berryman (1980)")

    def forward(
            self,
            K1: Tensor,
            G1: Tensor,
            K2: Tensor,
            G2: Tensor,
            f1: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute effective moduli using SC approximation.

        Args:
            K1: Bulk modulus of phase 1 (GPa)
            G1: Shear modulus of phase 1 (GPa)
            K2: Bulk modulus of phase 2 (GPa)
            G2: Shear modulus of phase 2 (GPa)
            f1: Volume fraction of phase 1

        Returns:
            Tuple of (K_effective, G_effective) in GPa
        """
        K1, G1 = as_tensor(K1), as_tensor(G1)
        K2, G2 = as_tensor(K2), as_tensor(G2)
        f1 = as_tensor(f1)
        f2 = 1.0 - f1

        # Initial guess: Voigt average
        K = f1 * K1 + f2 * K2
        G = f1 * G1 + f2 * G2

        for _ in range(self.max_iter):
            K_old, G_old = K.clone(), G.clone()

            # Update K
            zeta_K = K + 4.0 * G / 3.0
            K = zeta_K * (f1 * K1 / (K1 + zeta_K - K + EPS) +
                          f2 * K2 / (K2 + zeta_K - K + EPS))

            # Update G
            zeta_G = G / 6.0 * (9.0 * K + 8.0 * G) / (K + 2.0 * G + EPS)
            numer = f1 * G1 / (G1 + zeta_G + EPS) + f2 * G2 / (G2 + zeta_G + EPS)
            G = (zeta_G + G) * numer / (1.0 + numer + EPS)

            # Check convergence
            if (torch.abs(K - K_old) < self.tol).all() and \
                    (torch.abs(G - G_old) < self.tol).all():
                break

        return K, G


@register(name="CriticalPorosity", category="effective")
class CriticalPorosity(EffectiveModel):
    """
    Critical porosity model.

    A simple linear model where moduli decrease to zero at the
    critical porosity (suspension limit):

        K_dry = K_0 * (1 - φ/φ_c)
        G_dry = G_0 * (1 - φ/φ_c)

    References:
        Nur, A., Mavko, G., Dvorkin, J., & Galmudi, D. (1998). Critical
        porosity: A key to relating physical properties to porosity in rocks.
        The Leading Edge, 17(3), 357-362.

    Args:
        phi_c: Critical porosity. Default: 0.40

    Example:
        >>> cp = CriticalPorosity(phi_c=0.40)
        >>> K_dry, G_dry = cp(K0=36.6, G0=45.0, phi=0.25)
        >>> print(f"K_dry = {K_dry.item():.1f} GPa")
    """

    def __init__(self, phi_c: float = 0.40):
        super().__init__()
        self.phi_c = phi_c
        self.set_metadata("description", "Critical porosity model")
        self.set_metadata("reference", "Nur et al. (1998)")

    def forward(
            self,
            K0: Tensor,
            G0: Tensor,
            phi: Tensor,
            phi_c: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute dry rock moduli.

        Args:
            K0: Mineral bulk modulus (GPa)
            G0: Mineral shear modulus (GPa)
            phi: Porosity (fraction)
            phi_c: Critical porosity (optional override)

        Returns:
            Tuple of (K_dry, G_dry) in GPa
        """
        K0, G0, phi = as_tensor(K0), as_tensor(G0), as_tensor(phi)
        phi_c = as_tensor(phi_c if phi_c is not None else self.phi_c)

        factor = torch.clamp(1.0 - phi / phi_c, min=0.0)
        return K0 * factor, G0 * factor

    def compute_dry_rock(self, K0, G0, phi, **params):
        return self.forward(K0, G0, phi, phi_c=params.get('phi_c'))


# =============================================================================
# Crack and Inclusion Models
# =============================================================================

@register(name="Hudson", category="effective", tags=["crack"])
class Hudson(EffectiveModel):
    """
    Hudson's crack model for aligned penny-shaped cracks.

    Computes effective stiffness of an isotropic rock containing
    a single set of aligned penny-shaped cracks, producing VTI
    or HTI symmetry depending on the crack orientation.

    The model uses first or second-order perturbation theory and
    is valid for low crack densities (ε < 0.1).

    References:
        Hudson, J.A. (1980). Overall properties of a cracked solid.
        Mathematical Proceedings of the Cambridge Philosophical Society,
        88(2), 371-384.

        Hudson, J.A. (1981). Wave speeds and attenuation of elastic waves in
        material containing cracks. Geophysical Journal of the Royal
        Astronomical Society, 64(1), 133-150.

    Args:
        order: Perturbation order (1 or 2). Default: 1
        axis: Symmetry axis (3 for VTI, 1 for HTI). Default: 3

    Example:
        >>> hudson = Hudson(order=1, axis=3)
        >>> C_eff = hudson(
        ...     K=36.6, G=45.0,       # Background moduli (GPa)
        ...     Ki=0.0, Gi=0.0,       # Dry cracks
        ...     alpha=0.01,           # Crack aspect ratio
        ...     crd=0.05              # Crack density
        ... )
        >>> print(f"C_eff shape: {C_eff.shape}")  # [6, 6]
    """

    def __init__(self, order: int = 1, axis: int = 3):
        super().__init__()
        self.order = order
        self.axis = axis
        self.set_metadata("description", "Hudson crack model")
        self.set_metadata("reference", "Hudson (1981)")

    def forward(
            self,
            K: Tensor,
            G: Tensor,
            Ki: Tensor,
            Gi: Tensor,
            alpha: Tensor,
            crd: Tensor
    ) -> Tensor:
        """
        Compute effective stiffness matrix for cracked rock.

        Args:
            K: Background bulk modulus (GPa)
            G: Background shear modulus (GPa)
            Ki: Crack fill bulk modulus (GPa, 0 for dry)
            Gi: Crack fill shear modulus (GPa, 0 for dry)
            alpha: Crack aspect ratio (typically 0.001-0.1)
            crd: Crack density ε = N*a³/V (typically < 0.1)

        Returns:
            6x6 Voigt stiffness matrix (GPa)
        """
        K, G = as_tensor(K), as_tensor(G)
        Ki, Gi = as_tensor(Ki), as_tensor(Gi)
        alpha, crd = as_tensor(alpha), as_tensor(crd)

        lamda = K - 2.0 * G / 3.0

        # Crack compliance parameters
        kappa = (Ki + 4.0 * Gi / 3.0) * (lamda + 2.0 * G) / \
                (PI * alpha * G * (lamda + G) + EPS)
        M = 4.0 * Gi * (lamda + G) / (PI * alpha * G * (3.0 * lamda + 4.0 * G) + EPS)

        # U parameters
        U1 = 16.0 * (lamda + 2.0 * G) / (3.0 * (3.0 * lamda + 4.0 * G) * (1.0 + M) + EPS)
        U3 = 4.0 * (lamda + 2.0 * G) / (3.0 * (lamda + G) * (1.0 + kappa) + EPS)

        # First-order corrections
        C11_1 = -lamda ** 2 * crd * U3 / G
        C13_1 = -lamda * (lamda + 2.0 * G) * crd * U3 / G
        C33_1 = -(lamda + 2.0 * G) ** 2 * crd * U3 / G
        C44_1 = -G * crd * U1

        # Second-order corrections (if requested)
        if self.order == 2:
            q = 15.0 * lamda ** 2 / G ** 2 + 28.0 * lamda / G + 28.0
            C11_1 = C11_1 + q / 15.0 * lamda ** 2 / (lamda + 2.0 * G) * (crd * U3) ** 2
            C13_1 = C13_1 + q / 15.0 * lamda * (crd * U3) ** 2
            C33_1 = C33_1 + q / 15.0 * (lamda + 2.0 * G) * (crd * U3) ** 2
            C44_1 = C44_1 + 2.0 / 15.0 * G * (3.0 * lamda + 8.0 * G) / \
                    (lamda + 2.0 * G) * (crd * U1) ** 2

        # Effective stiffness
        C11 = lamda + 2.0 * G + C11_1
        C13 = lamda + C13_1
        C33 = lamda + 2.0 * G + C33_1
        C44 = G + C44_1
        C66 = G

        if self.axis == 3:
            return write_vti_matrix(C11, C33, C13, C44, C66)
        else:
            return write_hti_matrix(C33, C11, C13, C66, C44)


@register(name="HudsonRandom", category="effective", tags=["crack"])
class HudsonRandom(EffectiveModel):
    """
    Hudson's model for randomly oriented cracks.

    When cracks are randomly oriented, the effective medium remains
    isotropic. This model computes the effective bulk and shear
    moduli for such a medium.

    References:
        Hudson, J.A. (1990). Overall elastic properties of isotropic
        materials with arbitrary distribution of circular cracks. Geophysical
        Journal International, 102(2), 465-469.

    Args:
        None (stateless model)

    Example:
        >>> hr = HudsonRandom()
        >>> K_eff, G_eff = hr(
        ...     K=36.6, G=45.0,     # Background moduli (GPa)
        ...     Ki=0.0, Gi=0.0,     # Dry cracks
        ...     alpha=0.01,         # Crack aspect ratio
        ...     crd=0.05            # Crack density
        ... )
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "Hudson random crack model")
        self.set_metadata("reference", "Hudson (1981)")

    def forward(
            self,
            K: Tensor,
            G: Tensor,
            Ki: Tensor,
            Gi: Tensor,
            alpha: Tensor,
            crd: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute effective moduli for randomly oriented cracks.

        Args:
            K: Background bulk modulus (GPa)
            G: Background shear modulus (GPa)
            Ki: Crack fill bulk modulus (GPa, 0 for dry)
            Gi: Crack fill shear modulus (GPa, 0 for dry)
            alpha: Crack aspect ratio
            crd: Crack density

        Returns:
            Tuple of (K_effective, G_effective) in GPa
        """
        K, G = as_tensor(K), as_tensor(G)
        Ki, Gi = as_tensor(Ki), as_tensor(Gi)
        alpha, crd = as_tensor(alpha), as_tensor(crd)

        lamda = K - 2.0 * G / 3.0

        # Crack compliance parameters
        kappa = (Ki + 4.0 * Gi / 3.0) * (lamda + 2.0 * G) / \
                (PI * alpha * G * (lamda + G) + EPS)
        M = 4.0 * Gi * (lamda + G) / (PI * alpha * G * (3.0 * lamda + 4.0 * G) + EPS)

        # U parameters
        U1 = 16.0 * (lamda + 2.0 * G) / (3.0 * (3.0 * lamda + 4.0 * G) * (1.0 + M) + EPS)
        U3 = 4.0 * (lamda + 2.0 * G) / (3.0 * (lamda + G) * (1.0 + kappa) + EPS)

        # First-order corrections for random orientation
        G_1 = -2.0 * G * crd * (3.0 * U1 + 2.0 * U3)
        lamda_1 = 1.0 / 3.0 * (-(3.0 * lamda + 2.0 * G) ** 2 * crd * U3 / (3.0 * G) - 2.0 * G_1)

        # Effective moduli
        G_eff = G + G_1
        K_eff = (lamda + lamda_1) + 2.0 * G_eff / 3.0

        return K_eff, G_eff


@register(name="KusterToksoz", category="effective", aliases=["KT"], tags=["inclusion"])
class KusterToksoz(EffectiveModel):
    """
    Kuster-Toksöz model for spherical inclusions.

    Computes effective moduli of a rock with spherical inclusions
    (pores or other phases). Valid for low inclusion concentrations
    (typically φ < 0.1-0.2).

    References:
        Kuster, G.T. & Toksoz, M.N. (1974). Velocity and attenuation of
        seismic waves in two-phase media: Part I. Theoretical formulations.
        Geophysics, 39(5), 587-606.

    Args:
        None (stateless model)

    Example:
        >>> kt = KusterToksoz()
        >>> K_eff, G_eff = kt(
        ...     Km=36.6, Gm=45.0,   # Matrix moduli (GPa)
        ...     Ki=2.25, Gi=0.0,    # Inclusion moduli (water-filled)
        ...     phi=0.15            # Inclusion volume fraction
        ... )
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "Kuster-Toksöz inclusion model")
        self.set_metadata("reference", "Kuster & Toksöz (1974)")

    def forward(
            self,
            Km: Tensor,
            Gm: Tensor,
            Ki: Tensor,
            Gi: Tensor,
            phi: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute effective moduli for spherical inclusions.

        Args:
            Km: Matrix bulk modulus (GPa)
            Gm: Matrix shear modulus (GPa)
            Ki: Inclusion bulk modulus (GPa)
            Gi: Inclusion shear modulus (GPa)
            phi: Inclusion volume fraction (porosity)

        Returns:
            Tuple of (K_effective, G_effective) in GPa
        """
        Km, Gm = as_tensor(Km), as_tensor(Gm)
        Ki, Gi = as_tensor(Ki), as_tensor(Gi)
        phi = as_tensor(phi)

        # P and Q for spheres
        P = (Km + 4.0 * Gm / 3.0) / (Ki + 4.0 * Gm / 3.0 + EPS)
        zeta = Gm / 6.0 * (9.0 * Km + 8.0 * Gm) / (Km + 2.0 * Gm + EPS)
        Q = (Gm + zeta) / (Gi + zeta + EPS)

        # Effective moduli
        K_eff = Km + phi * (Ki - Km) * P / \
                (1.0 + (1.0 - phi) * (Ki - Km) * P / (Km + 4.0 * Gm / 3.0) + EPS)

        G_eff = Gm + phi * (Gi - Gm) * Q / \
                (1.0 + (1.0 - phi) * (Gi - Gm) * Q / (Gm + EPS) + EPS)

        return K_eff, G_eff


@register(name="EshelbyCheng", category="effective", tags=["crack", "anisotropic"])
class EshelbyCheng(EffectiveModel):
    """
    Eshelby-Cheng model for cracked isotropic rock.

    Computes VTI effective stiffness for aligned oblate spheroidal
    cracks using Eshelby's inclusion theory.

    References:
        Eshelby, J.D. (1957). The determination of the elastic field of an
        ellipsoidal inclusion. Proceedings of the Royal Society A,
        241(1226), 376-396.

        Cheng, C.H. (1993). Crack models for a transversely isotropic
        medium. Journal of Geophysical Research, 98(B1), 675-684.

    Args:
        None (stateless model)

    Example:
        >>> ec = EshelbyCheng()
        >>> C_eff = ec(
        ...     K=36.6, G=45.0,   # Background moduli (GPa)
        ...     phi=0.02,         # Crack porosity
        ...     alpha=0.01,       # Crack aspect ratio
        ...     Kf=0.0            # Fluid modulus (0 for dry)
        ... )
        >>> print(f"C_eff shape: {C_eff.shape}")  # [6, 6]
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "Eshelby-Cheng crack model")
        self.set_metadata("reference", "Cheng (1993)")

    def forward(
            self,
            K: Tensor,
            G: Tensor,
            phi: Tensor,
            alpha: Tensor,
            Kf: Tensor = 0.0
    ) -> Tensor:
        """
        Compute VTI stiffness for cracked rock.

        Args:
            K: Matrix bulk modulus (GPa)
            G: Matrix shear modulus (GPa)
            phi: Crack porosity (fraction)
            alpha: Crack aspect ratio
            Kf: Fluid bulk modulus (GPa, 0 for dry cracks)

        Returns:
            6x6 VTI stiffness matrix (GPa)
        """
        K, G = as_tensor(K), as_tensor(G)
        phi, alpha = as_tensor(phi), as_tensor(alpha)
        Kf = as_tensor(Kf)

        lamda = K - 2.0 * G / 3.0
        sigma = (3.0 * K - 2.0 * G) / (6.0 * K + 2.0 * G + EPS)

        # Eshelby tensor components
        R = (1.0 - 2.0 * sigma) / (8.0 * PI * (1.0 - sigma) + EPS)
        Q = 3.0 * R / (1.0 - 2.0 * sigma + EPS)

        # Geometric integrals for oblate spheroid
        Sa = torch.sqrt(1.0 - alpha ** 2 + EPS)
        Ia = 2.0 * PI * alpha * (torch.acos(alpha) - alpha * Sa) / (Sa ** 3 + EPS)
        Ic = 4.0 * PI - 2.0 * Ia
        Iac = (Ic - Ia) / (3.0 * Sa ** 2 + EPS)
        Iaa = PI - 3.0 * Iac / 4.0
        Iab = Iaa / 3.0

        # S tensor components
        S11 = Q * Iaa + R * Ia
        S33 = Q * (4.0 * PI / 3.0 - 2.0 * Iac * alpha ** 2) + Ic * R
        S12 = Q * Iab - R * Ia
        S13 = Q * Iac * alpha ** 2 - R * Ia
        S31 = Q * Iac - R * Ic
        S1212 = Q * Iab + R * Ia
        S1313 = Q * (1.0 + alpha ** 2) * Iac / 2.0 + R * (Ia + Ic) / 2.0

        # Fluid effect
        C = Kf / (3.0 * (K - Kf) + EPS)
        D = S33 * S11 + S33 * S12 - 2.0 * S31 * S13 - \
            (S11 + S12 + S33 - 1.0 - 3.0 * C) - \
            C * (S11 + S12 + 2.0 * (S33 - S13 - S31))
        E = S33 * S11 - S31 * S13 - (S33 + S11 - 2.0 * C - 1.0) + \
            C * (S31 + S13 - S11 - S33)

        # Stiffness perturbations
        C11_1 = lamda * (S31 - S33 + 1.0) + 2.0 * G * E / (D * (S12 - S11 + 1.0) + EPS)
        C33_1 = ((lamda + 2.0 * G) * (-S12 - S11 + 1.0) + 2.0 * lamda * S13 + 4.0 * G * C) / (D + EPS)
        C13_1 = ((lamda + 2.0 * G) * (S13 + S31) - 4.0 * G * C +
                 lamda * (S13 - S12 - S11 - S33 + 2.0)) / (2.0 * D + EPS)
        C44_1 = G / (1.0 - 2.0 * S1313 + EPS)
        C66_1 = G / (1.0 - 2.0 * S1212 + EPS)

        # Effective stiffness
        C11 = lamda + 2.0 * G - phi * C11_1
        C33 = lamda + 2.0 * G - phi * C33_1
        C13 = lamda - phi * C13_1
        C44 = G - phi * (G - C44_1)
        C66 = G - phi * (G - C66_1)

        return write_vti_matrix(C11, C33, C13, C44, C66)


# =============================================================================
# Additional Effective Medium Models
# =============================================================================

@register(name="SwissCheese", category="effective", tags=["dilute"])
class SwissCheese(EffectiveModel):
    """
    Swiss cheese model for dilute spherical pores.

    Assumes non-interacting spherical cavities in an unbounded
    homogeneous solid (dilute distribution):

        Kdry = Ks / (1 + (1 + 3Ks/(4Gs)) * phi)
        Gdry = Gs / (1 + (15Ks + 20Gs) / (9Ks + 8Gs) * phi)

    References:
        Mackenzie, J.K. (1950). The elastic constants of a solid containing
        spherical holes. Proceedings of the Physical Society B, 63(1), 2-11.

    Args:
        None (stateless model)
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "Swiss cheese (dilute pores) model")

    def forward(
            self, Ks: Tensor, Gs: Tensor, phi: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute effective moduli for dilute spherical pores.

        Args:
            Ks: Matrix bulk modulus (GPa)
            Gs: Matrix shear modulus (GPa)
            phi: Porosity (fraction)

        Returns:
            Tuple of (Kdry, Gdry) in GPa
        """
        Ks, Gs, phi = as_tensor(Ks), as_tensor(Gs), as_tensor(phi)
        Kdry = Ks / (1.0 + (1.0 + 3.0 * Ks / (4.0 * Gs + EPS)) * phi)
        Gdry = Gs / (1.0 + (15.0 * Ks + 20.0 * Gs) / (9.0 * Ks + 8.0 * Gs + EPS) * phi)
        return Kdry, Gdry


@register(name="DiluteCrack", category="effective", tags=["crack"])
class DiluteCrack(EffectiveModel):
    """
    Non-interacting randomly oriented crack model.

    Computes effective moduli for dilute, randomly oriented cracks
    using first-order perturbation theory.

    References:
        Walsh, J.B. (1965). The effect of cracks on the compressibility of
        rock. Journal of Geophysical Research, 70(2), 381-389.

    Args:
        None (stateless model)
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "Dilute crack model")

    def forward(
            self, Ks: Tensor, Gs: Tensor, cd: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute effective moduli for dilute cracks.

        Args:
            Ks: Uncracked bulk modulus (GPa)
            Gs: Uncracked shear modulus (GPa)
            cd: Crack density

        Returns:
            Tuple of (K_eff, G_eff) in GPa
        """
        Ks, Gs, cd = as_tensor(Ks), as_tensor(Gs), as_tensor(cd)
        nu = (3.0 * Ks - 2.0 * Gs) / (2.0 * (3.0 * Ks + Gs) + EPS)
        K_eff = Ks / (1.0 + 16.0 / 9.0 * (1.0 - nu ** 2) / (1.0 - 2.0 * nu + EPS) * cd)
        G_eff = Gs / (1.0 + 32.0 * (1.0 - nu) * (5.0 - nu) / (45.0 * (2.0 - nu) + EPS) * cd)
        return K_eff, G_eff


@register(name="OConnellBudiansky", category="effective", tags=["crack"])
class OConnellBudiansky(EffectiveModel):
    """
    O'Connell-Budiansky model for dry penny-shaped cracks.

    Closed-form solution for randomly oriented dry cracks in the
    limiting case of zero aspect ratio.

    References:
        O'Connell, R.J. & Budiansky, B. (1974). Seismic velocities in dry
        and saturated cracked solids. Journal of Geophysical Research,
        79(35), 5412-5426.

    Args:
        None (stateless model)
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "O'Connell-Budiansky dry cracks")
        self.set_metadata("reference", "O'Connell & Budiansky (1974)")

    def forward(
            self, K0: Tensor, G0: Tensor, crd: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute effective moduli for dry penny-shaped cracks.

        Args:
            K0: Background bulk modulus (GPa)
            G0: Background shear modulus (GPa)
            crd: Crack density

        Returns:
            Tuple of (K_dry, G_dry) in GPa
        """
        K0, G0, crd = as_tensor(K0), as_tensor(G0), as_tensor(crd)
        nu0 = (3.0 * K0 - 2.0 * G0) / (6.0 * K0 + 2.0 * G0 + EPS)
        nu_eff = nu0 * (1.0 - 16.0 * crd / 9.0)
        K_dry = K0 * (1.0 - 16.0 * (1.0 - nu_eff ** 2) * crd / (9.0 * (1.0 - 2.0 * nu_eff) + EPS))
        G_dry = G0 * (1.0 - 32.0 * (1.0 - nu_eff) * (5.0 - nu_eff) * crd / (45.0 * (2.0 - nu_eff) + EPS))
        return K_dry, G_dry


@register(name="OConnellBudianskyFl", category="effective", tags=["crack"])
class OConnellBudianskyFl(EffectiveModel):
    """
    O'Connell-Budiansky model for fluid-saturated cracks.

    Iterative self-consistent solution for randomly oriented cracks
    saturated with a soft fluid.

    References:
        O'Connell, R.J. & Budiansky, B. (1974). Seismic velocities in dry
        and saturated cracked solids. Journal of Geophysical Research,
        79(35), 5412-5426.

    Args:
        max_iter: Maximum iterations. Default: 100
        tol: Convergence tolerance. Default: 1e-6
    """

    def __init__(self, max_iter: int = 100, tol: float = 1e-6):
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol
        self.set_metadata("description", "O'Connell-Budiansky saturated cracks")

    def forward(
            self,
            K0: Tensor,
            G0: Tensor,
            Kfl: Tensor,
            crd: Tensor,
            alpha: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute effective moduli for fluid-saturated cracks.

        Args:
            K0: Background bulk modulus (GPa)
            G0: Background shear modulus (GPa)
            Kfl: Fluid bulk modulus (GPa)
            crd: Crack density
            alpha: Crack aspect ratio

        Returns:
            Tuple of (K_sat, G_sat) in GPa
        """
        K0, G0 = as_tensor(K0), as_tensor(G0)
        Kfl, crd, alpha = as_tensor(Kfl), as_tensor(crd), as_tensor(alpha)

        nu0 = (3.0 * K0 - 2.0 * G0) / (6.0 * K0 + 2.0 * G0 + EPS)
        w = Kfl / (alpha * K0 + EPS)

        # Iterative solution for nu_eff and D
        nu_eff = as_tensor(0.2)
        D = as_tensor(0.9)

        for _ in range(self.max_iter):
            nu_old, D_old = nu_eff.clone(), D.clone()

            # Update nu_eff from eq. 23
            numer = 45.0 / 16.0 * (nu0 - nu_eff) / (1.0 - nu_eff ** 2 + EPS) * (2.0 - nu_eff)
            denom_term = D * (1.0 + 3.0 * nu0) * (2.0 - nu_eff) - 2.0 * (1.0 - 2.0 * nu0)
            residual1 = numer / (denom_term + EPS) - crd

            # Update D from eq. 25
            a_coeff = crd
            b_coeff = -(crd + 9.0 / 16.0 * (1.0 - 2.0 * nu_eff) / (1.0 - nu0 ** 2 + EPS) + 3.0 * w / (4.0 * PI))
            c_coeff = 9.0 / 16.0 * (1.0 - 2.0 * nu_eff) / (1.0 - nu0 ** 2 + EPS)

            # Solve quadratic for D
            disc = b_coeff ** 2 - 4.0 * a_coeff * c_coeff
            disc = torch.clamp(disc, min=0.0)
            D = (-b_coeff - torch.sqrt(disc)) / (2.0 * a_coeff + EPS)
            D = torch.clamp(D, min=0.01, max=1.0)

            # Adjust nu_eff using Newton-like step
            nu_eff = nu_eff - 0.1 * residual1
            nu_eff = torch.clamp(nu_eff, min=-0.5, max=0.499)

            if (torch.abs(nu_eff - nu_old) < self.tol).all() and \
                    (torch.abs(D - D_old) < self.tol).all():
                break

        K_sat = K0 * (1.0 - 16.0 * (1.0 - nu_eff ** 2) * crd * D / (9.0 * (1.0 - 2.0 * nu_eff) + EPS))
        G_sat = G0 * (1.0 - 32.0 / 45.0 * (1.0 - nu_eff) * (D + 3.0 / (2.0 - nu_eff + EPS)) * crd)
        return K_sat, G_sat


@register(name="SCDilute", category="effective", tags=["inclusion"])
class SCDilute(EffectiveModel):
    """
    Dilute spherical micro-inclusions in two-phase composite.

    Random distribution of dilute spherical elastic inclusions
    with prescribed macro stress or macro strain.

    References:
        Berryman, J.G. (1980). Long-wavelength propagation in composite
        elastic media I. Spherical inclusions. Journal of the Acoustical
        Society of America, 68(6), 1809-1819.

    Args:
        mode: 'stress' or 'strain'. Default: 'stress'
    """

    def __init__(self, mode: str = 'stress'):
        super().__init__()
        self.mode = mode
        self.set_metadata("description", "Dilute spherical inclusions (SC)")

    def forward(
            self,
            Km: Tensor,
            Gm: Tensor,
            Ki: Tensor,
            Gi: Tensor,
            f: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute effective moduli for dilute spherical inclusions.

        Args:
            Km: Matrix bulk modulus (GPa)
            Gm: Matrix shear modulus (GPa)
            Ki: Inclusion bulk modulus (GPa)
            Gi: Inclusion shear modulus (GPa)
            f: Volume fraction of inclusions

        Returns:
            Tuple of (K, G): normalized effective moduli
        """
        Km, Gm = as_tensor(Km), as_tensor(Gm)
        Ki, Gi, f = as_tensor(Ki), as_tensor(Gi), as_tensor(f)

        nu = 0.5 * (3.0 * Km - 2.0 * Gm) / (3.0 * Km + Gm + EPS)
        S1 = 1.0 / 3.0 * (1.0 + nu) / (1.0 - nu + EPS)
        S2 = 2.0 / 15.0 * (4.0 - 5.0 * nu) / (1.0 - nu + EPS)

        if self.mode == 'stress':
            K = (1.0 + f * (Km / (Km - Ki + EPS) - S1) ** (-1)) ** (-1)
            G = (1.0 + f * (Gm / (Gm - Gi + EPS) - S2) ** (-1)) ** (-1)
        else:  # strain
            K = 1.0 - f * (Km / (Km - Ki + EPS) - S1) ** (-1)
            G = 1.0 - f * (Gm / (Gm - Gi + EPS) - S2) ** (-1)
        return K, G


@register(name="SCFlex", category="effective", tags=["inclusion"])
class SCFlex(EffectiveModel):
    """
    Iterative self-consistent model for two-phase composite.

    Not limited to pore inclusions; handles arbitrary two-phase
    composites with spherical inclusions.

    References:
        Berryman, J.G. (1980). Long-wavelength propagation in composite
        elastic media I. Spherical inclusions. Journal of the Acoustical
        Society of America, 68(6), 1809-1819.

    Args:
        max_iter: Maximum iterations. Default: 50
    """

    def __init__(self, max_iter: int = 50):
        super().__init__()
        self.max_iter = max_iter
        self.set_metadata("description", "Flexible self-consistent (two-phase)")

    def forward(
            self,
            f: Tensor,
            Km: Tensor,
            Ki: Tensor,
            Gm: Tensor,
            Gi: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute effective moduli iteratively.

        Args:
            f: Volume fraction of inclusion phase
            Km: Matrix bulk modulus (GPa)
            Ki: Inclusion bulk modulus (GPa)
            Gm: Matrix shear modulus (GPa)
            Gi: Inclusion shear modulus (GPa)

        Returns:
            Tuple of (K_eff, G_eff) in GPa
        """
        f = as_tensor(f)
        Km, Ki = as_tensor(Km), as_tensor(Ki)
        Gm, Gi = as_tensor(Gm), as_tensor(Gi)

        K_eff = Km.clone()
        G_eff = Km.clone()

        for _ in range(self.max_iter):
            nu = 0.5 * (3.0 * K_eff - 2.0 * G_eff) / (3.0 * K_eff + G_eff + EPS)
            S1 = 1.0 / 3.0 * (1.0 + nu) / (1.0 - nu + EPS)
            S2 = 2.0 / 15.0 * (4.0 - 5.0 * nu) / (1.0 - nu + EPS)

            K_eff = (1.0 - f * K_eff * (Km - Ki) / (Km * (K_eff - Ki) + EPS)
                     * (K_eff / (K_eff - Ki + EPS) - S1) ** (-1)) * Km
            G_eff = (1.0 - f * G_eff * (Gm - Gi) / (Gm * (G_eff - Gi) + EPS)
                     * (G_eff / (G_eff - Gi + EPS) - S2) ** (-1)) * Gm

        return K_eff, G_eff


@register(name="MTAverage", category="effective", aliases=["MoriTanaka"])
class MTAverage(EffectiveModel):
    """
    Modified Mori-Tanaka scheme for two-phase composite.

    Computes effective moduli of a two-phase composite without
    explicit matrix using the modified Mori-Tanaka scheme
    (Iwakuma 2003).

    References:
        Mori, T. & Tanaka, K. (1973). Average stress in matrix and average
        elastic energy of materials with misfitting inclusions. Acta
        Metallurgica, 21(5), 571-574.

    Args:
        None (stateless model)
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "Modified Mori-Tanaka average")
        self.set_metadata("reference", "Iwakuma (2003)")

    def forward(
            self,
            f: Tensor,
            Kmat: Tensor,
            Gmat: Tensor,
            K1: Tensor,
            G1: Tensor,
            K2: Tensor,
            G2: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute effective moduli.

        Args:
            f: Volume fraction of phase 1
            Kmat: Virtual matrix bulk modulus (GPa)
            Gmat: Virtual matrix shear modulus (GPa)
            K1: Bulk modulus of phase 1 (GPa)
            G1: Shear modulus of phase 1 (GPa)
            K2: Bulk modulus of phase 2 (GPa)
            G2: Shear modulus of phase 2 (GPa)

        Returns:
            Tuple of (K_ave, G_ave) in GPa
        """
        f = as_tensor(f)
        Kmat, Gmat = as_tensor(Kmat), as_tensor(Gmat)
        K1, G1 = as_tensor(K1), as_tensor(G1)
        K2, G2 = as_tensor(K2), as_tensor(G2)

        nu = (3.0 * Kmat - 2.0 * Gmat) / (2.0 * (3.0 * Kmat + Gmat) + EPS)
        alpha_mt = (1.0 + nu) / (3.0 * (1.0 - nu) + EPS)
        beta = 2.0 * (4.0 - 5.0 * nu) / (15.0 * (1.0 - nu) + EPS)

        dK1 = Kmat - (Kmat - K1) * alpha_mt
        dK2 = Kmat - (Kmat - K2) * alpha_mt
        dG1 = Gmat - (Gmat - G1) * beta
        dG2 = Gmat - (Gmat - G2) * beta

        K_ave = (f * Kmat * K1 / (dK1 + EPS) + (1.0 - f) * Kmat * K2 / (dK2 + EPS)) / \
                (f * Kmat / (dK1 + EPS) + (1.0 - f) * Kmat / (dK2 + EPS) + EPS)
        G_ave = (f * Gmat * G1 / (dG1 + EPS) + (1.0 - f) * Gmat * G2 / (dG2 + EPS)) / \
                (f * Gmat / (dG1 + EPS) + (1.0 - f) * Gmat / (dG2 + EPS) + EPS)

        return K_ave, G_ave


@register(name="PQ", category="effective", tags=["geometry"])
class PQ(EffectiveModel):
    """
    Berryman geometric strain concentration factors P and Q.

    Computes P and Q factors for prolate and oblate spheroids
    used in DEM, KT, and self-consistent methods.

    References:
        Berryman, J.G. (1980). Long-wavelength propagation in composite
        elastic media. Journal of the Acoustical Society of America, 68(6),
        1809-1819.

    Args:
        None (stateless model)
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "Berryman P,Q factors")
        self.set_metadata("reference", "Berryman (1980)")

    def forward(
            self,
            Km: Tensor,
            Gm: Tensor,
            Ki: Tensor,
            Gi: Tensor,
            alpha: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute strain concentration factors.

        Args:
            Km: Matrix bulk modulus (GPa)
            Gm: Matrix shear modulus (GPa)
            Ki: Inclusion bulk modulus (GPa)
            Gi: Inclusion shear modulus (GPa)
            alpha: Aspect ratio (<1 oblate, >1 prolate, 1 sphere)

        Returns:
            Tuple of (P, Q): strain concentration factors
        """
        Km, Gm = as_tensor(Km), as_tensor(Gm)
        Ki, Gi = as_tensor(Ki), as_tensor(Gi)
        alpha = as_tensor(alpha)

        # Spherical case
        is_sphere = torch.abs(alpha - 1.0) < 1e-4

        # For non-sphere: compute theta and f
        alpha_c = torch.where(is_sphere, as_tensor(0.999), alpha)

        # Oblate spheroid (alpha < 1)
        theta_oblate = alpha_c / (1.0 - alpha_c ** 2 + EPS) ** 1.5 * (
                torch.acos(torch.clamp(alpha_c, max=1.0 - 1e-7)) -
                alpha_c * torch.sqrt(torch.clamp(1.0 - alpha_c ** 2, min=EPS))
        )
        # Prolate spheroid (alpha > 1)
        theta_prolate = alpha_c / (alpha_c ** 2 - 1.0 + EPS) ** 1.5 * (
                alpha_c * torch.sqrt(alpha_c ** 2 - 1.0 + EPS) -
                torch.acosh(torch.clamp(alpha_c, min=1.0 + 1e-7))
        )

        theta = torch.where(alpha_c < 1.0, theta_oblate, theta_prolate)

        f_val = alpha_c ** 2 * (3.0 * theta - 2.0) / (1.0 - alpha_c ** 2 + EPS)
        A = Gi / (Gm + EPS) - 1.0
        B = (Ki / (Km + EPS) - Gi / (Gm + EPS)) / 3.0
        R = Gm / (Km + 4.0 / 3.0 * Gm + EPS)

        F1 = 1.0 + A * (1.5 * (f_val + theta) - R * (1.5 * f_val + 2.5 * theta - 4.0 / 3.0))
        F2 = 1.0 + A * (1.0 + 1.5 * (f_val + theta) - R * (1.5 * f_val + 2.5 * theta)) + \
             B * (3.0 - 4.0 * R) + A * (A + 3.0 * B) * (1.5 - 2.0 * R) * \
             (f_val + theta - R * (f_val - theta + 2.0 * theta ** 2))
        F3 = 1.0 + A * (1.0 - f_val - 1.5 * theta + R * (f_val + theta))
        F4 = 1.0 + A / 4.0 * (f_val + 3.0 * theta - R * (f_val - theta))
        F5 = A * (-f_val + R * (f_val + theta - 4.0 / 3.0)) + B * theta * (3.0 - 4.0 * R)
        F6 = 1.0 + A * (1.0 + f_val - R * (f_val + theta)) + B * (1.0 - theta) * (3.0 - 4.0 * R)
        F7 = 2.0 + A / 4.0 * (3.0 * f_val + 9.0 * theta - R * (3.0 * f_val + 5.0 * theta)) + \
             B * theta * (3.0 - 4.0 * R)
        F8 = A * (1.0 - 2.0 * R + f_val / 2.0 * (R - 1.0) + theta / 2.0 * (5.0 * R - 3.0)) + \
             B * (1.0 - theta) * (3.0 - 4.0 * R)
        F9 = A * ((R - 1.0) * f_val - R * theta) + B * theta * (3.0 - 4.0 * R)

        Tiijj = 3.0 * F1 / (F2 + EPS)
        Tijij = Tiijj / 3.0 + 2.0 / (F3 + EPS) + 1.0 / (F4 + EPS) + \
                (F4 * F5 + F6 * F7 - F8 * F9) / (F2 * F4 + EPS)
        P_gen = Tiijj / 3.0
        Q_gen = (Tijij - P_gen) / 5.0

        # Spherical case
        P_sphere = (Km + 4.0 * Gm / 3.0) / (Ki + 4.0 * Gm / 3.0 + EPS)
        zeta = Gm / 6.0 * (9.0 * Km + 8.0 * Gm) / (Km + 2.0 * Gm + EPS)
        Q_sphere = (Gm + zeta) / (Gi + zeta + EPS)

        P = torch.where(is_sphere, P_sphere, P_gen)
        Q = torch.where(is_sphere, Q_sphere, Q_gen)
        return P, Q


@register(name="HudsonOrtho", category="effective", tags=["crack"])
class HudsonOrtho(EffectiveModel):
    """
    Hudson's crack model for three orthogonal crack sets.

    Computes orthorhombic effective stiffness for a medium with
    three crack sets with normals aligned along 1, 2, and 3 axes.

    References:
        Hudson, J.A. (1981). Wave speeds and attenuation of elastic waves in
        material containing cracks. Geophysical Journal of the Royal
        Astronomical Society, 64(1), 133-150.

    Args:
        None (stateless model)
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "Hudson orthogonal cracks")
        self.set_metadata("reference", "Hudson (1981)")

    def forward(
            self,
            K: Tensor,
            G: Tensor,
            Ki: Tensor,
            Gi: Tensor,
            alpha: Tensor,
            crd: Tensor
    ) -> Tensor:
        """
        Compute orthorhombic stiffness for three orthogonal crack sets.

        Args:
            K: Background bulk modulus (GPa)
            G: Background shear modulus (GPa)
            Ki: Crack fill bulk modulus (GPa)
            Gi: Crack fill shear modulus (GPa)
            alpha: Aspect ratios [alpha1, alpha2, alpha3]
            crd: Crack densities [crd1, crd2, crd3]

        Returns:
            6x6 orthorhombic stiffness matrix (GPa)
        """
        K, G = as_tensor(K), as_tensor(G)
        Ki, Gi = as_tensor(Ki), as_tensor(Gi)
        alpha, crd = as_tensor(alpha), as_tensor(crd)

        lamda = K - 2.0 * G / 3.0

        kapa = (Ki + 4.0 * Gi / 3.0) * (lamda + 2.0 * G) / (PI * alpha * G * (lamda + G) + EPS)
        M = 4.0 * Gi * (lamda + G) / (PI * alpha * G * (3.0 * lamda + 4.0 * G) + EPS)
        U1 = 16.0 * (lamda + 2.0 * G) / (3.0 * (3.0 * lamda + 4.0 * G) * (1.0 + M) + EPS)
        U3 = 4.0 * (lamda + 2.0 * G) / (3.0 * (lamda + G) * (1.0 + kapa) + EPS)

        # First order corrections per crack set
        C11_1 = -lamda ** 2 * crd * U3 / G
        C13_1 = -lamda * (lamda + 2.0 * G) * crd * U3 / G
        C33_1 = -(lamda + 2.0 * G) ** 2 * crd * U3 / G
        C44_1 = -G * crd * U1
        C12_1 = C11_1

        # Combine three sets (normals along 1, 2, 3)
        C11_o = lamda + 2.0 * G + C33_1[0] + C11_1[1] + C11_1[2]
        C12_o = lamda + C13_1[0] + C13_1[1] + C12_1[2]
        C13_o = lamda + C13_1[0] + C12_1[1] + C13_1[2]
        C22_o = lamda + 2.0 * G + C11_1[0] + C33_1[1] + C11_1[2]
        C23_o = lamda + C12_1[0] + C13_1[1] + C13_1[2]
        C33_o = lamda + 2.0 * G + C11_1[0] + C11_1[1] + C33_1[2]
        C44_o = G + C44_1[1] + C44_1[2]
        C55_o = G + C44_1[0] + C44_1[2]
        C66_o = G + C44_1[0] + C44_1[1]

        return write_ortho_matrix(C11_o, C22_o, C33_o, C12_o, C13_o, C23_o, C44_o, C55_o, C66_o)


@register(name="HudsonCone", category="effective", tags=["crack"])
class HudsonCone(EffectiveModel):
    """
    Hudson's crack model for conically distributed crack normals.

    Crack normals randomly distributed at a fixed angle from the
    TI symmetry axis, forming a cone. Produces VTI symmetry.

    References:
        Hudson, J.A. (1981). Wave speeds and attenuation of elastic waves in
        material containing cracks. Geophysical Journal of the Royal
        Astronomical Society, 64(1), 133-150.

    Args:
        None (stateless model)
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "Hudson cone-distributed cracks")
        self.set_metadata("reference", "Hudson (1981)")

    def forward(
            self,
            K: Tensor,
            G: Tensor,
            Ki: Tensor,
            Gi: Tensor,
            alpha: Tensor,
            crd: Tensor,
            theta: Tensor
    ) -> Tensor:
        """
        Compute VTI stiffness for conically distributed cracks.

        Args:
            K: Background bulk modulus (GPa)
            G: Background shear modulus (GPa)
            Ki: Crack fill bulk modulus (GPa)
            Gi: Crack fill shear modulus (GPa)
            alpha: Crack aspect ratio
            crd: Total crack density
            theta: Fixed angle between crack normal and symmetry axis (degrees)

        Returns:
            6x6 VTI stiffness matrix (GPa)
        """
        K, G = as_tensor(K), as_tensor(G)
        Ki, Gi = as_tensor(Ki), as_tensor(Gi)
        alpha, crd = as_tensor(alpha), as_tensor(crd)
        theta = as_tensor(theta) * PI / 180.0

        lamda = K - 2.0 * G / 3.0

        kapa = (Ki + 4.0 * Gi / 3.0) * (lamda + 2.0 * G) / (PI * alpha * G * (lamda + G) + EPS)
        M = 4.0 * Gi * (lamda + G) / (PI * alpha * G * (3.0 * lamda + 4.0 * G) + EPS)
        U1 = 16.0 * (lamda + 2.0 * G) / (3.0 * (3.0 * lamda + 4.0 * G) * (1.0 + M) + EPS)
        U3 = 4.0 * (lamda + 2.0 * G) / (3.0 * (lamda + G) * (1.0 + kapa) + EPS)

        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        sin2 = sin_t ** 2
        cos2 = cos_t ** 2
        sin4 = sin_t ** 4

        C11_1 = -crd / (2.0 * G) * (
                U3 * (2.0 * lamda ** 2 + 4.0 * lamda * G * sin2 + 3.0 * G ** 2 * sin4) +
                U1 * G ** 2 * sin2 * (4.0 - 3.0 * sin2))
        C33_1 = -crd / G * (
                U3 * (lamda + 2.0 * G * cos2) ** 2 +
                U1 * G ** 2 * 4.0 * cos2 * sin2)
        C12_1 = -crd / (2.0 * G) * (
                U3 * (2.0 * lamda ** 2 + 4.0 * lamda * G * sin2 + G ** 2 * sin4) -
                U1 * G ** 2 * sin4)
        C13_1 = -crd / G * (
                U3 * (lamda + G * sin2) * (lamda + 2.0 * G * cos2) -
                U1 * G ** 2 * 2.0 * sin2 * cos2)
        C44_1 = -crd / 2.0 * G * (
                U3 * 4.0 * sin2 * cos2 +
                U1 * (sin2 + 2.0 * cos2 - 4.0 * sin2 * cos2))
        C66_1 = -crd / 2.0 * G * (
                U3 * 4.0 * sin4 +
                U1 * sin2 * (2.0 - sin2))

        C11_c = lamda + 2.0 * G + C11_1
        C33_c = lamda + 2.0 * G + C33_1
        C13_c = lamda + C13_1
        C44_c = G + C44_1
        C66_c = G + C66_1

        return write_vti_matrix(C11_c, C33_c, C13_c, C44_c, C66_c)