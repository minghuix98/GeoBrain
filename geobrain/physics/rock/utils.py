#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functions for rock physics calculations.

This module provides common utility functions used across rock physics
models, including:
    - Velocity-moduli conversions
    - Elastic parameter computations (Poisson's ratio, Young's modulus, etc.)
    - Stiffness matrix construction (isotropic, VTI, HTI)
    - Coordination number estimation

All functions support PyTorch tensors for autodiff compatibility.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
from torch import Tensor
from typing import Tuple, Union, List

# =============================================================================
# Type Aliases and Constants
# =============================================================================

TensorLike = Union[float, int, List[float], Tensor]
PI = 3.141592653589793
EPS = 1e-10


def ensure_same_device(*tensors: Tensor):
    """
    Move all tensors to the same device, preferring CUDA over CPU.

    When mixing scalar parameters (CPU) with data tensors (CUDA),
    this ensures all operands are on the same device.
    """
    device = None
    for t in tensors:
        if isinstance(t, Tensor) and t.device.type != 'cpu':
            device = t.device
            break
    if device is None:
        return tensors
    return tuple(t.to(device) if isinstance(t, Tensor) else t for t in tensors)


def as_tensor(x: TensorLike, dtype: torch.dtype = torch.float32) -> Tensor:
    """
    Convert input to PyTorch tensor.

    Args:
        x: Input value (scalar, list, or tensor)
        dtype: Target data type. Default: torch.float32

    Returns:
        PyTorch tensor of at least 1 dimension
    """
    if isinstance(x, Tensor):
        return x.to(dtype)
    return torch.atleast_1d(torch.as_tensor(x, dtype=dtype))


# =============================================================================
# Velocity-Moduli Conversions
# =============================================================================

def v_from_moduli(
        K: TensorLike,
        G: TensorLike,
        rho: TensorLike
) -> Tuple[Tensor, Tensor]:
    """
    Compute seismic velocities from elastic moduli and density.

    Uses the standard isotropic relations:

        Vp = sqrt((K + 4G/3) / ρ)
        Vs = sqrt(G / ρ)

    Args:
        K: Bulk modulus (GPa)
        G: Shear modulus (GPa)
        rho: Density (g/cm³)

    Returns:
        Tuple of (Vp, Vs):
            - Vp: P-wave velocity (m/s)
            - Vs: S-wave velocity (m/s)

    Example:
        >>> K, G, rho = 36.6, 45.0, 2.65  # Quartz
        >>> Vp, Vs = v_from_moduli(K, G, rho)
        >>> print(f"Vp = {Vp.item():.0f} m/s, Vs = {Vs.item():.0f} m/s")
        Vp = 6038 m/s, Vs = 4121 m/s

    Note:
        Input moduli in GPa, density in g/cc gives output in m/s.
    """
    K, G, rho = as_tensor(K), as_tensor(G), as_tensor(rho)
    K, G, rho = ensure_same_device(K, G, rho)

    # M = K + 4G/3 (P-wave modulus)
    M = K + 4.0 / 3.0 * G

    # Convert GPa·cc/g to (km/s)² then to m/s
    Vp = torch.sqrt(M / (rho + EPS)) * 1000.0
    Vs = torch.sqrt(G / (rho + EPS)) * 1000.0

    return Vp, Vs


def moduli_from_v(
        Vp: TensorLike,
        Vs: TensorLike,
        rho: TensorLike
) -> Tuple[Tensor, Tensor]:
    """
    Compute elastic moduli from seismic velocities and density.

    Inverse of v_from_moduli:

        G = ρ * Vs²
        K = ρ * Vp² - 4G/3

    Args:
        Vp: P-wave velocity (m/s)
        Vs: S-wave velocity (m/s)
        rho: Density (g/cm³)

    Returns:
        Tuple of (K, G):
            - K: Bulk modulus (GPa)
            - G: Shear modulus (GPa)

    Example:
        >>> Vp, Vs, rho = 6038, 4121, 2.65
        >>> K, G = moduli_from_v(Vp, Vs, rho)
        >>> print(f"K = {K.item():.1f} GPa, G = {G.item():.1f} GPa")
        K = 36.6 GPa, G = 45.0 GPa
    """
    Vp, Vs, rho = as_tensor(Vp), as_tensor(Vs), as_tensor(rho)
    Vp, Vs, rho = ensure_same_device(Vp, Vs, rho)

    # Convert m/s to km/s for calculation
    vp_km = Vp / 1000.0
    vs_km = Vs / 1000.0

    # G = ρ * Vs² (in GPa when Vs in km/s and ρ in g/cc)
    G = rho * vs_km ** 2

    # K = ρ * Vp² - 4G/3
    K = rho * vp_km ** 2 - 4.0 * G / 3.0

    return K, G


# =============================================================================
# Elastic Parameter Calculations
# =============================================================================

def poisson(K: TensorLike, G: TensorLike) -> Tensor:
    """
    Compute Poisson's ratio from bulk and shear moduli.

    Formula:
        ν = (3K - 2G) / (6K + 2G)

    Args:
        K: Bulk modulus (any consistent units)
        G: Shear modulus (same units as K)

    Returns:
        Poisson's ratio (dimensionless, typically 0 to 0.5)

    Example:
        >>> nu = poisson(K=36.6, G=45.0)
        >>> print(f"Poisson's ratio = {nu.item():.3f}")
        Poisson's ratio = 0.074

    Note:
        For fluids (G=0), returns 0.5. Valid range: -1 < ν < 0.5.
    """
    K, G = as_tensor(K), as_tensor(G)
    K, G = ensure_same_device(K, G)
    return (3.0 * K - 2.0 * G) / (6.0 * K + 2.0 * G + EPS)


def lame_lambda(K: TensorLike, G: TensorLike) -> Tensor:
    """
    Compute Lamé's first parameter (λ) from bulk and shear moduli.

    Formula:
        λ = K - 2G/3

    Args:
        K: Bulk modulus
        G: Shear modulus

    Returns:
        Lamé's first parameter (same units as input)

    Example:
        >>> lamda = lame_lambda(K=36.6, G=45.0)
        >>> print(f"Lambda = {lamda.item():.1f} GPa")
        Lambda = 6.6 GPa
    """
    K, G = as_tensor(K), as_tensor(G)
    K, G = ensure_same_device(K, G)
    return K - 2.0 * G / 3.0


def youngs_modulus(K: TensorLike, G: TensorLike) -> Tensor:
    """
    Compute Young's modulus from bulk and shear moduli.

    Formula:
        E = 9KG / (3K + G)

    Args:
        K: Bulk modulus
        G: Shear modulus

    Returns:
        Young's modulus (same units as input)

    Example:
        >>> E = youngs_modulus(K=36.6, G=45.0)
        >>> print(f"Young's modulus = {E.item():.1f} GPa")
        Young's modulus = 95.3 GPa
    """
    K, G = as_tensor(K), as_tensor(G)
    K, G = ensure_same_device(K, G)
    return 9.0 * K * G / (3.0 * K + G + EPS)


def bulk_modulus(E: TensorLike, nu: TensorLike) -> Tensor:
    """
    Compute bulk modulus from Young's modulus and Poisson's ratio.

    Formula:
        K = E / (3(1 - 2ν))

    Args:
        E: Young's modulus
        nu: Poisson's ratio

    Returns:
        Bulk modulus (same units as E)
    """
    E, nu = as_tensor(E), as_tensor(nu)
    E, nu = ensure_same_device(E, nu)
    return E / (3.0 * (1.0 - 2.0 * nu) + EPS)


def shear_modulus(E: TensorLike, nu: TensorLike) -> Tensor:
    """
    Compute shear modulus from Young's modulus and Poisson's ratio.

    Formula:
        G = E / (2(1 + ν))

    Args:
        E: Young's modulus
        nu: Poisson's ratio

    Returns:
        Shear modulus (same units as E)
    """
    E, nu = as_tensor(E), as_tensor(nu)
    E, nu = ensure_same_device(E, nu)
    return E / (2.0 * (1.0 + nu) + EPS)


# =============================================================================
# Input Validation
# =============================================================================

def validate_porosity(
        phi: TensorLike,
        name: str = "phi",
        clamp: bool = False
) -> Tensor:
    """
    Validate that porosity is in the physical range [0, 1].

    Args:
        phi: Porosity value(s)
        name: Parameter name for error messages. Default: "phi"
        clamp: If True, clamp values to [0, 1] instead of raising.
               Default: False

    Returns:
        Validated porosity tensor

    Raises:
        ValueError: If any value is outside [0, 1] and clamp is False

    Example:
        >>> phi = validate_porosity(torch.tensor([0.2, 0.3]))  # OK
        >>> phi = validate_porosity(torch.tensor([1.5]), clamp=True)  # Clamped to 1.0
    """
    phi = as_tensor(phi)
    if clamp:
        return torch.clamp(phi, min=0.0, max=1.0)
    if (phi < 0.0).any() or (phi > 1.0).any():
        raise ValueError(
            f"{name} must be in [0, 1], got values in "
            f"[{phi.min().item():.4f}, {phi.max().item():.4f}]"
        )
    return phi


def validate_positive(x: TensorLike, name: str) -> Tensor:
    """
    Validate that a parameter is strictly positive.

    Args:
        x: Parameter value(s)
        name: Parameter name for error messages

    Returns:
        Validated tensor

    Raises:
        ValueError: If any value is <= 0

    Example:
        >>> K = validate_positive(torch.tensor([36.6]), name="K0")
    """
    x = as_tensor(x)
    if (x <= 0.0).any():
        raise ValueError(
            f"{name} must be positive, got min value {x.min().item():.6f}"
        )
    return x


# =============================================================================
# Stiffness Matrix Construction
# =============================================================================

def write_iso_matrix(K: TensorLike, G: TensorLike) -> Tensor:
    """
    Construct 6x6 stiffness matrix for isotropic medium.

    The stiffness matrix C_ij in Voigt notation:

        | C11  C12  C12   0    0    0  |
        | C12  C11  C12   0    0    0  |
        | C12  C12  C11   0    0    0  |
        |  0    0    0   C44   0    0  |
        |  0    0    0    0   C44   0  |
        |  0    0    0    0    0   C44 |

    where C11 = λ + 2μ = K + 4G/3, C12 = λ = K - 2G/3, C44 = μ = G.

    Args:
        K: Bulk modulus (GPa)
        G: Shear modulus (GPa)

    Returns:
        6x6 stiffness tensor (GPa)

    Example:
        >>> C = write_iso_matrix(K=36.6, G=45.0)
        >>> print(f"C11 = {C[0,0]:.1f}, C44 = {C[3,3]:.1f}")
        C11 = 96.6, C44 = 45.0
    """
    K, G = as_tensor(K).squeeze(), as_tensor(G).squeeze()
    K, G = ensure_same_device(K, G)

    lamda = K - 2.0 * G / 3.0
    C11 = lamda + 2.0 * G

    C = torch.zeros(6, 6, dtype=K.dtype, device=K.device)

    # Diagonal elements
    C[0, 0] = C[1, 1] = C[2, 2] = C11
    C[3, 3] = C[4, 4] = C[5, 5] = G

    # Off-diagonal elements
    C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = lamda

    return C


def write_vti_matrix(
        C11: TensorLike,
        C33: TensorLike,
        C13: TensorLike,
        C44: TensorLike,
        C66: TensorLike
) -> Tensor:
    """
    Construct 6x6 stiffness matrix for VTI medium.

    VTI (Vertical Transverse Isotropy) has a vertical axis of symmetry,
    typical for layered media.

    The stiffness matrix:

        | C11  C12  C13   0    0    0  |
        | C12  C11  C13   0    0    0  |
        | C13  C13  C33   0    0    0  |
        |  0    0    0   C44   0    0  |
        |  0    0    0    0   C44   0  |
        |  0    0    0    0    0   C66 |

    where C12 = C11 - 2*C66.

    Args:
        C11: Horizontal P-wave modulus (GPa)
        C33: Vertical P-wave modulus (GPa)
        C13: Coupling modulus (GPa)
        C44: Vertical shear modulus (GPa)
        C66: Horizontal shear modulus (GPa)

    Returns:
        6x6 stiffness tensor (GPa)

    Example:
        >>> C = write_vti_matrix(C11=100, C33=80, C13=30, C44=25, C66=35)
        >>> print(f"C12 = {C[0,1]:.0f}")  # C12 = C11 - 2*C66 = 30
        C12 = 30
    """
    C11 = as_tensor(C11).squeeze()
    C33 = as_tensor(C33).squeeze()
    C13 = as_tensor(C13).squeeze()
    C44 = as_tensor(C44).squeeze()
    C66 = as_tensor(C66).squeeze()
    C11, C33, C13, C44, C66 = ensure_same_device(C11, C33, C13, C44, C66)

    C12 = C11 - 2.0 * C66

    C = torch.zeros(6, 6, dtype=C11.dtype, device=C11.device)

    C[0, 0] = C[1, 1] = C11
    C[2, 2] = C33
    C[0, 1] = C[1, 0] = C12
    C[0, 2] = C[2, 0] = C[1, 2] = C[2, 1] = C13
    C[3, 3] = C[4, 4] = C44
    C[5, 5] = C66

    return C


def write_hti_matrix(
        C11: TensorLike,
        C33: TensorLike,
        C13: TensorLike,
        C44: TensorLike,
        C55: TensorLike
) -> Tensor:
    """
    Construct 6x6 stiffness matrix for HTI medium.

    HTI (Horizontal Transverse Isotropy) has a horizontal axis of symmetry,
    typical for vertical fractures. Here we assume symmetry axis along x1.

    The stiffness matrix:

        | C11  C13  C13   0    0    0  |
        | C13  C33  C23   0    0    0  |
        | C13  C23  C33   0    0    0  |
        |  0    0    0   C44   0    0  |
        |  0    0    0    0   C55   0  |
        |  0    0    0    0    0   C55 |

    where C23 = C33 - 2*C44.

    Args:
        C11: Modulus along symmetry axis (GPa)
        C33: Modulus perpendicular to symmetry axis (GPa)
        C13: Coupling modulus (GPa)
        C44: Shear modulus in symmetry plane (GPa)
        C55: Shear modulus perpendicular to symmetry axis (GPa)

    Returns:
        6x6 stiffness tensor (GPa)
    """
    C11 = as_tensor(C11).squeeze()
    C33 = as_tensor(C33).squeeze()
    C13 = as_tensor(C13).squeeze()
    C44 = as_tensor(C44).squeeze()
    C55 = as_tensor(C55).squeeze()
    C11, C33, C13, C44, C55 = ensure_same_device(C11, C33, C13, C44, C55)

    C23 = C33 - 2.0 * C44

    C = torch.zeros(6, 6, dtype=C11.dtype, device=C11.device)

    C[0, 0] = C11
    C[1, 1] = C[2, 2] = C33
    C[0, 1] = C[1, 0] = C[0, 2] = C[2, 0] = C13
    C[1, 2] = C[2, 1] = C23
    C[3, 3] = C44
    C[4, 4] = C[5, 5] = C55

    return C


def write_ortho_matrix(
        C11: TensorLike,
        C22: TensorLike,
        C33: TensorLike,
        C12: TensorLike,
        C13: TensorLike,
        C23: TensorLike,
        C44: TensorLike,
        C55: TensorLike,
        C66: TensorLike
) -> Tensor:
    """
    Construct 6x6 stiffness matrix for orthorhombic medium.

    Args:
        C11-C66: Independent stiffness coefficients (GPa)

    Returns:
        6x6 stiffness tensor (GPa)
    """
    C11 = as_tensor(C11).squeeze()
    C22 = as_tensor(C22).squeeze()
    C33 = as_tensor(C33).squeeze()
    C12 = as_tensor(C12).squeeze()
    C13 = as_tensor(C13).squeeze()
    C23 = as_tensor(C23).squeeze()
    C44 = as_tensor(C44).squeeze()
    C55 = as_tensor(C55).squeeze()
    C66 = as_tensor(C66).squeeze()
    C11, C22, C33, C12, C13, C23, C44, C55, C66 = ensure_same_device(
        C11, C22, C33, C12, C13, C23, C44, C55, C66)

    C = torch.zeros(6, 6, dtype=C11.dtype, device=C11.device)
    C[0, 0] = C11
    C[1, 1] = C22
    C[2, 2] = C33
    C[0, 1] = C[1, 0] = C12
    C[0, 2] = C[2, 0] = C13
    C[1, 2] = C[2, 1] = C23
    C[3, 3] = C44
    C[4, 4] = C55
    C[5, 5] = C66
    return C


# =============================================================================
# Granular Media Utilities
# =============================================================================

def coordination_number(phi: TensorLike) -> Tensor:
    """
    Estimate coordination number from porosity using Murphy's relation.

    The coordination number (Cn) is the average number of contacts
    per grain in a granular medium.

    Murphy's empirical relation:
        Cn = 20 - 34φ + 14φ²

    Args:
        phi: Porosity (fraction, 0-1)

    Returns:
        Coordination number (clamped to range [4, 12])

    Example:
        >>> Cn = coordination_number(phi=0.36)
        >>> print(f"Cn = {Cn.item():.1f}")
        Cn = 9.6

    Note:
        For random sphere packs, Cn ≈ 6-9. Clamped to physical range [4, 12].
    """
    phi = as_tensor(phi)
    Cn = 20.0 - 34.0 * phi + 14.0 * phi ** 2
    return torch.clamp(Cn, min=4.0, max=12.0)


def critical_porosity_default(lithology: str = 'sandstone') -> float:
    """
    Get default critical porosity for common lithologies.

    Critical porosity (φc) is the porosity above which the rock
    becomes a suspension and cannot support shear stress.

    Args:
        lithology: Rock type ('sandstone', 'limestone', 'shale', 'chalk')

    Returns:
        Critical porosity value

    Example:
        >>> phi_c = critical_porosity_default('sandstone')
        >>> print(f"Critical porosity = {phi_c}")
        Critical porosity = 0.4
    """
    defaults = {
        'sandstone': 0.40,
        'sand': 0.40,
        'limestone': 0.35,
        'carbonate': 0.35,
        'shale': 0.30,
        'chalk': 0.65,
        'clay': 0.55,
    }
    return defaults.get(lithology.lower(), 0.40)


# =============================================================================
# QI (Quantitative Interpretation) Utilities
# =============================================================================

def matrix_modulus(
        vsh: TensorLike,
        phi_c: TensorLike,
        phi_0: TensorLike,
        M_sh: TensorLike,
        M_g: TensorLike,
        M_c: TensorLike
) -> Tensor:
    """
    Compute matrix modulus during cementation using VRH Hill average.

    As cement fills pore space, the solid matrix composition changes.
    This function computes the effective modulus of the solid frame
    (everything except pore space) as a VRH average of shale, grain,
    and cement components.

    Args:
        vsh: Bulk volume fraction of shale
        phi_c: Critical porosity
        phi_0: Current porosity (array or scalar)
        M_sh: Modulus of shale (GPa)
        M_g: Modulus of grain material (GPa)
        M_c: Modulus of cement (GPa)

    Returns:
        Matrix modulus (GPa)
    """
    vsh = as_tensor(vsh)
    phi_c = as_tensor(phi_c)
    phi_0 = as_tensor(phi_0)
    M_sh, M_g, M_c = as_tensor(M_sh), as_tensor(M_g), as_tensor(M_c)
    vsh, phi_c, phi_0, M_sh, M_g, M_c = ensure_same_device(
        vsh, phi_c, phi_0, M_sh, M_g, M_c)

    fc = phi_c - phi_0  # cement fraction
    denom = 1.0 - phi_0 + EPS
    vsh_s = vsh / denom          # shale volume fraction in matrix
    fgs = (1.0 - phi_c - vsh) / denom  # grain fraction in matrix
    fcs = fc / denom             # cement fraction in matrix

    # Voigt (upper) bound
    M_voigt = vsh_s * M_sh + fgs * M_g + fcs * M_c
    # Reuss (lower) bound
    M_reuss = 1.0 / (vsh_s / (M_sh + EPS) + fgs / (M_g + EPS)
                      + fcs / (M_c + EPS) + EPS)
    # Hill average
    M_mat = 0.5 * (M_voigt + M_reuss)
    return M_mat


def den_matrix(
        vsh: TensorLike,
        phi_c: TensorLike,
        phi_0: TensorLike,
        den_sh: TensorLike,
        den_g: TensorLike,
        den_c: TensorLike
) -> Tensor:
    """
    Compute matrix density during cementation using Voigt (linear) average.

    Same volume fraction logic as :func:`matrix_modulus`, but computes
    a linear (Voigt) average of component densities.

    Args:
        vsh: Bulk volume fraction of shale
        phi_c: Critical porosity
        phi_0: Current porosity (array or scalar)
        den_sh: Density of shale (g/cm³)
        den_g: Density of grain (g/cm³)
        den_c: Density of cement (g/cm³)

    Returns:
        Matrix density (g/cm³)
    """
    vsh = as_tensor(vsh)
    phi_c = as_tensor(phi_c)
    phi_0 = as_tensor(phi_0)
    den_sh, den_g, den_c = as_tensor(den_sh), as_tensor(den_g), as_tensor(den_c)
    vsh, phi_c, phi_0, den_sh, den_g, den_c = ensure_same_device(
        vsh, phi_c, phi_0, den_sh, den_g, den_c)

    fc = phi_c - phi_0
    denom = 1.0 - phi_0 + EPS
    vsh_s = vsh / denom
    fgs = (1.0 - phi_c - vsh) / denom
    fcs = fc / denom

    den_mat = vsh_s * den_sh + fgs * den_g + fcs * den_c
    return den_mat


def normalize_kde(density: TensorLike) -> Tensor:
    """
    Normalize KDE density values to [0, 1] range.

    Maps density values by dividing by the maximum value so that
    the peak density maps to 1.0.

    Args:
        density: KDE density values

    Returns:
        Normalized density in [0, 1]
    """
    density = as_tensor(density)
    den_max = density.max()
    return density / (den_max + EPS)