"""
Elastic wave equation FDTD kernels.

Implements finite-difference time-domain kernels for elastic
wave propagation using the stress-velocity formulation.
Supports 4th, 6th, 8th order staggered grid schemes.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from typing import Dict, List, Tuple

from .acoustic_2d import pad_model


# =============================================================================
# Finite Difference Coefficients
# =============================================================================

@torch.jit.script
def fd_coefficients(order: int, grid_scheme: str) -> Tensor:
    """
    Calculate finite difference coefficients.
    
    Computes optimal FD coefficients for the specified order
    and grid scheme using Taylor series expansion.

    Args:
        order: FD order (half-width of stencil). E.g., 2 for 4th order.
        grid_scheme: Grid type.
            - 'r': Regular grid
            - 's': Staggered grid

    Returns:
        FD coefficients tensor with shape (order,).

    Raises:
        ValueError: If grid_scheme is not 'r' or 's'.

    Example:
        >>> c = fd_coefficients(order=2, grid_scheme='s')
        >>> print(c)  # [1.125, -0.0416667]
    """
    C = torch.zeros(order)

    if grid_scheme == 'r':
        B = torch.cat((torch.tensor([1 / 2]), torch.zeros(order - 1)))
        A = torch.empty((order, order), dtype=torch.float32)
        for i in range(order):
            for j in range(order):
                A[i, j] = (j + 1) ** (2 * (i + 1) - 1)
    elif grid_scheme == 's':
        B = torch.cat((torch.tensor([1.0]), torch.zeros(order - 1)))
        A = torch.empty((order, order), dtype=torch.float32)
        for i in range(order):
            for j in range(order):
                A[i, j] = (2 * (j + 1) - 1) ** (2 * (i + 1) - 1)
    else:
        raise ValueError("Grid scheme must be 'r' or 's'")

    C = torch.linalg.inv(A) @ B
    return C


# =============================================================================
# 4th Order FD Operators
# =============================================================================

@torch.jit.script
def Dxf_4(a: Tensor, fdc: Tensor, ii: Tensor, jj: Tensor) -> Tensor:
    """Forward difference in x-direction (4th order)."""
    return (
        fdc[0] * (a[:, ii, :][:, :, jj + 1] - a[:, ii, :][:, :, jj]) +
        fdc[1] * (a[:, ii, :][:, :, jj + 2] - a[:, ii, :][:, :, jj - 1])
    )


@torch.jit.script
def Dzf_4(a: Tensor, fdc: Tensor, ii: Tensor, jj: Tensor) -> Tensor:
    """Forward difference in z-direction (4th order)."""
    return (
        fdc[0] * (a[:, ii + 1, :][:, :, jj] - a[:, ii, :][:, :, jj]) +
        fdc[1] * (a[:, ii + 2, :][:, :, jj] - a[:, ii - 1, :][:, :, jj])
    )


@torch.jit.script
def Dxb_4(a: Tensor, fdc: Tensor, ii: Tensor, jj: Tensor) -> Tensor:
    """Backward difference in x-direction (4th order)."""
    return (
        fdc[0] * (a[:, ii, :][:, :, jj] - a[:, ii, :][:, :, jj - 1]) +
        fdc[1] * (a[:, ii, :][:, :, jj + 1] - a[:, ii, :][:, :, jj - 2])
    )


@torch.jit.script
def Dzb_4(a: Tensor, fdc: Tensor, ii: Tensor, jj: Tensor) -> Tensor:
    """Backward difference in z-direction (4th order)."""
    return (
        fdc[0] * (a[:, ii, :][:, :, jj] - a[:, ii - 1, :][:, :, jj]) +
        fdc[1] * (a[:, ii + 1, :][:, :, jj] - a[:, ii - 2, :][:, :, jj])
    )


# =============================================================================
# 6th Order FD Operators
# =============================================================================

@torch.jit.script
def Dxf_6(a: Tensor, fdc: Tensor, ii: Tensor, jj: Tensor) -> Tensor:
    """Forward difference in x-direction (6th order)."""
    return (
        fdc[0] * (a[:, ii, :][:, :, jj + 1] - a[:, ii, :][:, :, jj]) +
        fdc[1] * (a[:, ii, :][:, :, jj + 2] - a[:, ii, :][:, :, jj - 1]) +
        fdc[2] * (a[:, ii, :][:, :, jj + 3] - a[:, ii, :][:, :, jj - 2])
    )


@torch.jit.script
def Dzf_6(a: Tensor, fdc: Tensor, ii: Tensor, jj: Tensor) -> Tensor:
    """Forward difference in z-direction (6th order)."""
    return (
        fdc[0] * (a[:, ii + 1, :][:, :, jj] - a[:, ii, :][:, :, jj]) +
        fdc[1] * (a[:, ii + 2, :][:, :, jj] - a[:, ii - 1, :][:, :, jj]) +
        fdc[2] * (a[:, ii + 3, :][:, :, jj] - a[:, ii - 2, :][:, :, jj])
    )


@torch.jit.script
def Dxb_6(a: Tensor, fdc: Tensor, ii: Tensor, jj: Tensor) -> Tensor:
    """Backward difference in x-direction (6th order)."""
    return (
        fdc[0] * (a[:, ii, :][:, :, jj] - a[:, ii, :][:, :, jj - 1]) +
        fdc[1] * (a[:, ii, :][:, :, jj + 1] - a[:, ii, :][:, :, jj - 2]) +
        fdc[2] * (a[:, ii, :][:, :, jj + 2] - a[:, ii, :][:, :, jj - 3])
    )


@torch.jit.script
def Dzb_6(a: Tensor, fdc: Tensor, ii: Tensor, jj: Tensor) -> Tensor:
    """Backward difference in z-direction (6th order)."""
    return (
        fdc[0] * (a[:, ii, :][:, :, jj] - a[:, ii - 1, :][:, :, jj]) +
        fdc[1] * (a[:, ii + 1, :][:, :, jj] - a[:, ii - 2, :][:, :, jj]) +
        fdc[2] * (a[:, ii + 2, :][:, :, jj] - a[:, ii - 3, :][:, :, jj])
    )


# =============================================================================
# 8th Order FD Operators
# =============================================================================

@torch.jit.script
def Dxf_8(a: Tensor, fdc: Tensor, ii: Tensor, jj: Tensor) -> Tensor:
    """Forward difference in x-direction (8th order)."""
    return (
        fdc[0] * (a[:, ii, :][:, :, jj + 1] - a[:, ii, :][:, :, jj]) +
        fdc[1] * (a[:, ii, :][:, :, jj + 2] - a[:, ii, :][:, :, jj - 1]) +
        fdc[2] * (a[:, ii, :][:, :, jj + 3] - a[:, ii, :][:, :, jj - 2]) +
        fdc[3] * (a[:, ii, :][:, :, jj + 4] - a[:, ii, :][:, :, jj - 3])
    )


@torch.jit.script
def Dzf_8(a: Tensor, fdc: Tensor, ii: Tensor, jj: Tensor) -> Tensor:
    """Forward difference in z-direction (8th order)."""
    return (
        fdc[0] * (a[:, ii + 1, :][:, :, jj] - a[:, ii, :][:, :, jj]) +
        fdc[1] * (a[:, ii + 2, :][:, :, jj] - a[:, ii - 1, :][:, :, jj]) +
        fdc[2] * (a[:, ii + 3, :][:, :, jj] - a[:, ii - 2, :][:, :, jj]) +
        fdc[3] * (a[:, ii + 4, :][:, :, jj] - a[:, ii - 3, :][:, :, jj])
    )


@torch.jit.script
def Dxb_8(a: Tensor, fdc: Tensor, ii: Tensor, jj: Tensor) -> Tensor:
    """Backward difference in x-direction (8th order)."""
    return (
        fdc[0] * (a[:, ii, :][:, :, jj] - a[:, ii, :][:, :, jj - 1]) +
        fdc[1] * (a[:, ii, :][:, :, jj + 1] - a[:, ii, :][:, :, jj - 2]) +
        fdc[2] * (a[:, ii, :][:, :, jj + 2] - a[:, ii, :][:, :, jj - 3]) +
        fdc[3] * (a[:, ii, :][:, :, jj + 3] - a[:, ii, :][:, :, jj - 4])
    )


@torch.jit.script
def Dzb_8(a: Tensor, fdc: Tensor, ii: Tensor, jj: Tensor) -> Tensor:
    """Backward difference in z-direction (8th order)."""
    return (
        fdc[0] * (a[:, ii, :][:, :, jj] - a[:, ii - 1, :][:, :, jj]) +
        fdc[1] * (a[:, ii + 1, :][:, :, jj] - a[:, ii - 2, :][:, :, jj]) +
        fdc[2] * (a[:, ii + 2, :][:, :, jj] - a[:, ii - 3, :][:, :, jj]) +
        fdc[3] * (a[:, ii + 3, :][:, :, jj] - a[:, ii - 4, :][:, :, jj])
    )


# =============================================================================
# Model Utilities
# =============================================================================

@torch.jit.script
def pad_model_elastic(
    v: Tensor,
    nabc: int,
    fs_offset: int,
    free_surface: bool,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """
    Pad model for elastic wave simulation.

    Extends the model by replicating edge values into the
    absorbing boundary region, with special handling for
    free surface boundary.

    Args:
        v: Input model with shape (nz, nx).
        nabc: Number of absorbing boundary layers.
        fs_offset: Free surface offset (fd_order // 2).
        free_surface: Whether using free surface at top.
        device: Computation device. Default: 'cpu'.

    Returns:
        Padded model with shape (nz_pml, nx_pml).

    Example:
        >>> vp = torch.ones(100, 200) * 3000.0
        >>> vp_padded = pad_model_elastic(
        ...     vp, nabc=20, fs_offset=2, free_surface=True
        ... )
    """
    nz, nx = v.shape
    nx_pml = nx + 2 * nabc

    if free_surface:
        nz_pml = nz + nabc + fs_offset
    else:
        nz_pml = nz + 2 * nabc + fs_offset

    cc = torch.zeros((nz_pml, nx_pml), device=device)

    z_start = fs_offset if free_surface else nabc + fs_offset

    # Copy original model to center
    cc[z_start:z_start + nz, nabc:nabc + nx] = v

    # Extend boundaries
    cc[:z_start, nabc:nabc + nx] = cc[z_start, nabc:nabc + nx].expand(z_start, -1)
    cc[z_start + nz:, nabc:nabc + nx] = cc[z_start + nz - 1, nabc:nabc + nx].expand(
        nz_pml - z_start - nz, -1
    )
    cc[:, :nabc] = cc[:, [nabc]].expand(-1, nabc)
    cc[:, nabc + nx:] = cc[:, [nabc + nx - 1]].expand(-1, nabc)

    return cc


def thomsen_to_elastic(
    vp: Tensor,
    vs: Tensor,
    rho: Tensor,
    eps: Tensor,
    delta: Tensor,
    gamma: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Convert Thomsen parameters to elastic moduli.

    Converts VTI anisotropy parameters (Thomsen, 1986) to
    elastic stiffness tensor components.

    Args:
        vp: Vertical P-wave velocity (m/s).
        vs: Vertical S-wave velocity (m/s).
        rho: Density (kg/m³).
        eps: Thomsen epsilon (P-wave anisotropy).
        delta: Thomsen delta (near-vertical anisotropy).
        gamma: Thomsen gamma (S-wave anisotropy).

    Returns:
        Tuple of elastic moduli (C11, C13, C33, C44, C66) in Pa.

    Example:
        >>> C11, C13, C33, C44, C66 = thomsen_to_elastic(
        ...     vp, vs, rho, eps=0.1, delta=0.05, gamma=0.08
        ... )
    """
    C33 = vp ** 2 * rho
    C44 = vs ** 2 * rho
    C11 = C33 * (1 + 2 * eps)
    C66 = C44 * (1 + 2 * gamma)
    C13 = torch.sqrt(2 * C33 * (C33 - C44) * delta + (C33 - C44) ** 2) - C44
    return C11, C13, C33, C44, C66


def lame_from_velocities(
    vp: Tensor,
    vs: Tensor,
    rho: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Calculate Lamé parameters from velocities.

    Computes elastic moduli from P-wave velocity, S-wave velocity,
    and density using standard relationships.

    Args:
        vp: P-wave velocity (m/s).
        vs: S-wave velocity (m/s).
        rho: Density (kg/m³).

    Returns:
        Tuple of (mu, lamu, lam, b):
            - mu: Shear modulus (Pa)
            - lamu: P-wave modulus λ + 2μ (Pa)
            - lam: First Lamé parameter λ (Pa)
            - b: Buoyancy 1/ρ (m³/kg)

    Example:
        >>> mu, lamu, lam, b = lame_from_velocities(vp=3000, vs=1500, rho=2200)
    """
    mu = vs ** 2 * rho
    lamu = vp ** 2 * rho
    lam = lamu - 2 * mu
    b = 1.0 / rho
    return mu, lamu, lam, b


def staggered_grid_parameters(
    mu: Tensor,
    b: Tensor,
    C44: Tensor,
    C55: Tensor,
    C66: Tensor,
    nx: int,
    nz: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Calculate staggered grid parameters by averaging.

    Computes averaged elastic parameters at staggered grid
    locations for the velocity-stress formulation.

    Args:
        mu: Shear modulus (Pa) with shape (nz, nx).
        b: Buoyancy 1/ρ (m³/kg) with shape (nz, nx).
        C44: Elastic modulus C44 (Pa) with shape (nz, nx).
        C55: Elastic modulus C55 (Pa) with shape (nz, nx).
        C66: Elastic modulus C66 (Pa) with shape (nz, nx).
        nx: Number of grid points in x-direction.
        nz: Number of grid points in z-direction.

    Returns:
        Tuple of staggered parameters (bx, bz, muxz, C44_stag, C55_stag, C66_stag).
    """
    # Average buoyancy at staggered locations
    bx = 0.5 * (b[:, 0:nx - 1] + b[:, 1:nx])
    bz = 0.5 * (b[0:nz - 1, :] + b[1:nz, :])

    # Average moduli at corner locations (5-point average)
    muxz = 0.2 * (
        mu[1:nz - 1, 1:nx - 1] + mu[2:nz, 1:nx - 1] +
        mu[1:nz - 1, 2:nx] + mu[2:nz, 1:nx - 1] + mu[2:nz, 2:nx]
    )

    C44_stag = 0.2 * (
        C44[1:nz - 1, 1:nx - 1] + C44[2:nz, 1:nx - 1] +
        C44[1:nz - 1, 2:nx] + C44[2:nz, 1:nx - 1] + C44[2:nz, 2:nx]
    )

    C55_stag = 0.2 * (
        C55[1:nz - 1, 1:nx - 1] + C55[2:nz, 1:nx - 1] +
        C55[1:nz - 1, 2:nx] + C55[2:nz, 1:nx - 1] + C55[2:nz, 2:nx]
    )

    C66_stag = 0.2 * (
        C66[1:nz - 1, 1:nx - 1] + C66[2:nz, 1:nx - 1] +
        C66[1:nz - 1, 2:nx] + C66[2:nz, 1:nx - 1] + C66[2:nz, 2:nx]
    )

    return bx, bz, muxz, C44_stag, C55_stag, C66_stag


@torch.jit.script
def elastic_step_forward(
    nx: int,
    nz: int,
    dx: float,
    dz: float,
    dt: float,
    nabc: int,
    free_surface: bool,
    src_x: Tensor,
    src_z: Tensor,
    src_n: int,
    src_v: Tensor,
    rcv_x: Tensor,
    rcv_z: Tensor,
    rcv_n: int,
    kappa_s: Tensor,
    kappa_sxz: Tensor,
    kappa_vx: Tensor,
    kappa_vz: Tensor,
    alpha_lamu: Tensor,
    alpha_lam: Tensor,
    alpha_mu: Tensor,
    alpha_bx: Tensor,
    alpha_bz: Tensor,
    c1: float,
    c2: float,
    sigma_xx: Tensor,
    sigma_zz: Tensor,
    sigma_xz: Tensor,
    vx: Tensor,
    vz: Tensor,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Time-stepping segment for 2D elastic wave equation.

    Uses 4th-order staggered grid finite difference scheme
    with velocity-stress formulation.

    Args:
        nx: Number of grid points in x-direction.
        nz: Number of grid points in z-direction.
        dx: Grid spacing in x-direction (m).
        dz: Grid spacing in z-direction (m).
        dt: Time step (s).
        nabc: Number of absorbing boundary layers.
        free_surface: Whether to use free surface at top.
        src_x: Source x-positions (grid indices).
        src_z: Source z-positions (grid indices).
        src_n: Number of sources.
        src_v: Source wavelet values for this segment.
        rcv_x: Receiver x-positions (grid indices).
        rcv_z: Receiver z-positions (grid indices).
        rcv_n: Number of receivers.
        kappa_s: Damping for normal stresses (nz_pml, nx_pml).
        kappa_sxz: Damping for shear stress (nz_pml-1, nx_pml-1).
        kappa_vx: Damping for vx (nz_pml, nx_pml).
        kappa_vz: Damping for vz (nz_pml, nx_pml).
        alpha_lamu: P-wave modulus coefficient (nz_pml, nx_pml).
        alpha_lam: Lambda coefficient (nz_pml, nx_pml).
        alpha_mu: Shear modulus coefficient (nz_pml-1, nx_pml-1).
        alpha_bx: Buoyancy coefficient for vx (nz_pml, nx_pml).
        alpha_bz: Buoyancy coefficient for vz (nz_pml, nx_pml).
        c1: FD stencil coefficient (9/8).
        c2: FD stencil coefficient (-1/24).
        sigma_xx: Normal stress xx field (src_n, nz_pml, nx_pml).
        sigma_zz: Normal stress zz field (src_n, nz_pml, nx_pml).
        sigma_xz: Shear stress field (src_n, nz_pml-1, nx_pml-1).
        vx: Horizontal velocity field (src_n, nz_pml, nx_pml-1).
        vz: Vertical velocity field (src_n, nz_pml-1, nx_pml).
        device: Computation device.
        dtype: Data type.

    Returns:
        Tuple of 11 tensors:
            - sigma_xx, sigma_zz, sigma_xz, vx, vz: Updated fields
            - rcv_p, rcv_vx, rcv_vz: Recorded seismograms
            - fwd_p, fwd_vx, fwd_vz: Accumulated wavefield energy
    """
    sigma_xx = sigma_xx.clone()
    sigma_zz = sigma_zz.clone()
    sigma_xz = sigma_xz.clone()
    vx = vx.clone()
    vz = vz.clone()

    nt = src_v.shape[-1]
    fs = nabc if free_surface else 1
    nx_pml = nx + 2 * nabc
    nz_pml = nz + 2 * nabc

    # Initialize recorded values
    rcv_p = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_vx = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_vz = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)

    # Initialize forward wavefield accumulators
    fwd_p = torch.zeros((nz, nx), dtype=dtype, device=device)
    fwd_vx = torch.zeros((nz, nx), dtype=dtype, device=device)
    fwd_vz = torch.zeros((nz, nx), dtype=dtype, device=device)

    # Normal stress update range (same as acoustic pressure)
    szs = fs + 1
    sze = nz_pml - 2
    sxs = 2
    sxe = nx_pml - 2

    # Shear stress update range
    # Need vx[xz_zs-1] valid (>= 0) and vz[xz_zs-1] valid (>= 0)
    xz_zs = fs
    if xz_zs < 1:
        xz_zs = 1
    xz_ze = nz_pml - 3
    xz_xs = 1
    xz_xe = nx_pml - 3

    # Velocity update ranges (adjusted for half-grid sigma_xz access)
    # vx needs sigma_xz z-stencil: z-2 >= 0 requires vx_zs >= 2
    vx_zs = fs
    if vx_zs < 2:
        vx_zs = 2
    vx_ze = nz_pml - 2  # sigma_xz z+1 must be <= nz_pml-2
    vx_xs = 1
    vx_xe = nx_pml - 2

    vz_zs = fs
    vz_ze = nz_pml - 2
    vz_xs = 2   # sigma_xz x-2 must be >= 0
    vz_xe = nx_pml - 2  # sigma_xz x+1 must be <= nx_pml-2

    for it in range(nt):
        # ====== 1. Update normal stresses (sigma_xx, sigma_zz) ======
        # FD(dvx/dx) at integer grid: backward difference
        dvx_dx = (
            c1 * (vx[:, szs:sze, sxs:sxe] -
                  vx[:, szs:sze, sxs - 1:sxe - 1]) +
            c2 * (vx[:, szs:sze, sxs + 1:sxe + 1] -
                  vx[:, szs:sze, sxs - 2:sxe - 2])
        )
        # FD(dvz/dz) at integer grid: backward difference
        dvz_dz = (
            c1 * (vz[:, szs:sze, sxs:sxe] -
                  vz[:, szs - 1:sze - 1, sxs:sxe]) +
            c2 * (vz[:, szs + 1:sze + 1, sxs:sxe] -
                  vz[:, szs - 2:sze - 2, sxs:sxe])
        )

        sigma_xx[:, szs:sze, sxs:sxe] = (
            (1.0 - kappa_s[szs:sze, sxs:sxe]) *
            sigma_xx[:, szs:sze, sxs:sxe] +
            alpha_lamu[szs:sze, sxs:sxe] * dvx_dx +
            alpha_lam[szs:sze, sxs:sxe] * dvz_dz
        )

        sigma_zz[:, szs:sze, sxs:sxe] = (
            (1.0 - kappa_s[szs:sze, sxs:sxe]) *
            sigma_zz[:, szs:sze, sxs:sxe] +
            alpha_lam[szs:sze, sxs:sxe] * dvx_dx +
            alpha_lamu[szs:sze, sxs:sxe] * dvz_dz
        )

        # ====== 2. Update shear stress (sigma_xz) ======
        # FD(dvx/dz) at half-grid: forward difference in z
        dvx_dz = (
            c1 * (vx[:, xz_zs + 1:xz_ze + 1, xz_xs:xz_xe] -
                  vx[:, xz_zs:xz_ze, xz_xs:xz_xe]) +
            c2 * (vx[:, xz_zs + 2:xz_ze + 2, xz_xs:xz_xe] -
                  vx[:, xz_zs - 1:xz_ze - 1, xz_xs:xz_xe])
        )
        # FD(dvz/dx) at half-grid: forward difference in x
        dvz_dx = (
            c1 * (vz[:, xz_zs:xz_ze, xz_xs + 1:xz_xe + 1] -
                  vz[:, xz_zs:xz_ze, xz_xs:xz_xe]) +
            c2 * (vz[:, xz_zs:xz_ze, xz_xs + 2:xz_xe + 2] -
                  vz[:, xz_zs:xz_ze, xz_xs - 1:xz_xe - 1])
        )

        sigma_xz[:, xz_zs:xz_ze, xz_xs:xz_xe] = (
            (1.0 - kappa_sxz[xz_zs:xz_ze, xz_xs:xz_xe]) *
            sigma_xz[:, xz_zs:xz_ze, xz_xs:xz_xe] +
            alpha_mu[xz_zs:xz_ze, xz_xs:xz_xe] * (dvx_dz + dvz_dx)
        )

        # ====== 3. Source injection (isotropic explosion) ======
        if src_z.dim() == 1:
            src_update = dt * (src_v[it] if len(src_v.shape) == 1 else src_v[:, it])
            sigma_xx[torch.arange(src_n), src_z, src_x] = (
                sigma_xx[torch.arange(src_n), src_z, src_x] + src_update
            )
            sigma_zz[torch.arange(src_n), src_z, src_x] = (
                sigma_zz[torch.arange(src_n), src_z, src_x] + src_update
            )
        else:
            for i in range(src_n):
                src_update = dt * (src_v[i, it] if len(src_v.shape) == 2 else src_v[i, :, it])
                sigma_xx[i, src_z[i], src_x[i]] = sigma_xx[i, src_z[i], src_x[i]] + src_update
                sigma_zz[i, src_z[i], src_x[i]] = sigma_zz[i, src_z[i], src_x[i]] + src_update

        # ====== 4. Free surface BC for stresses ======
        if free_surface:
            sigma_zz[:, fs, :] = 0
            sigma_zz[:, :fs, :] = 0
            if fs >= 1:
                sigma_xz[:, :fs, :] = 0

        # ====== 5. Update vx ======
        # FD(dsigma_xx/dx) forward difference
        dsxx_dx = (
            c1 * (sigma_xx[:, vx_zs:vx_ze, vx_xs + 1:vx_xe + 1] -
                  sigma_xx[:, vx_zs:vx_ze, vx_xs:vx_xe]) +
            c2 * (sigma_xx[:, vx_zs:vx_ze, vx_xs + 2:vx_xe + 2] -
                  sigma_xx[:, vx_zs:vx_ze, vx_xs - 1:vx_xe - 1])
        )
        # FD(dsigma_xz/dz) backward difference
        dsxz_dz = (
            c1 * (sigma_xz[:, vx_zs:vx_ze, vx_xs:vx_xe] -
                  sigma_xz[:, vx_zs - 1:vx_ze - 1, vx_xs:vx_xe]) +
            c2 * (sigma_xz[:, vx_zs + 1:vx_ze + 1, vx_xs:vx_xe] -
                  sigma_xz[:, vx_zs - 2:vx_ze - 2, vx_xs:vx_xe])
        )

        vx[:, vx_zs:vx_ze, vx_xs:vx_xe] = (
            (1.0 - kappa_vx[vx_zs:vx_ze, vx_xs:vx_xe]) *
            vx[:, vx_zs:vx_ze, vx_xs:vx_xe] +
            alpha_bx[vx_zs:vx_ze, vx_xs:vx_xe] * (dsxx_dx + dsxz_dz)
        )

        # ====== 6. Update vz ======
        # FD(dsigma_xz/dx) backward difference
        dsxz_dx = (
            c1 * (sigma_xz[:, vz_zs:vz_ze, vz_xs:vz_xe] -
                  sigma_xz[:, vz_zs:vz_ze, vz_xs - 1:vz_xe - 1]) +
            c2 * (sigma_xz[:, vz_zs:vz_ze, vz_xs + 1:vz_xe + 1] -
                  sigma_xz[:, vz_zs:vz_ze, vz_xs - 2:vz_xe - 2])
        )
        # FD(dsigma_zz/dz) forward difference
        dszz_dz = (
            c1 * (sigma_zz[:, vz_zs + 1:vz_ze + 1, vz_xs:vz_xe] -
                  sigma_zz[:, vz_zs:vz_ze, vz_xs:vz_xe]) +
            c2 * (sigma_zz[:, vz_zs + 2:vz_ze + 2, vz_xs:vz_xe] -
                  sigma_zz[:, vz_zs - 1:vz_ze - 1, vz_xs:vz_xe])
        )

        vz[:, vz_zs:vz_ze, vz_xs:vz_xe] = (
            (1.0 - kappa_vz[vz_zs:vz_ze, vz_xs:vz_xe]) *
            vz[:, vz_zs:vz_ze, vz_xs:vz_xe] +
            alpha_bz[vz_zs:vz_ze, vz_xs:vz_xe] * (dsxz_dx + dszz_dz)
        )

        # ====== 7. Free surface BC for vz (mirror) ======
        if free_surface:
            vz[:, fs - 1, :] = vz[:, fs, :]

        # ====== 8. Record seismograms ======
        rcv_p[:, it, :] = -(sigma_xx[:, rcv_z, rcv_x] + sigma_zz[:, rcv_z, rcv_x]) / 2.0
        rcv_vx[:, it, :] = vx[:, rcv_z, rcv_x]
        rcv_vz[:, it, :] = vz[:, rcv_z, rcv_x]

        # ====== 9. Accumulate forward wavefield energy ======
        p_field = -(sigma_xx + sigma_zz) / 2.0
        fwd_p += torch.sum(
            p_field * p_field, dim=0
        )[nabc:nabc + nz, nabc:nabc + nx].detach()
        fwd_vx += torch.sum(
            vx * vx, dim=0
        )[nabc:nabc + nz, nabc:nabc + nx].detach()
        fwd_vz += torch.sum(
            vz * vz, dim=0
        )[nabc:nabc + nz, nabc:nabc + nx].detach()

    return (
        sigma_xx, sigma_zz, sigma_xz, vx, vz,
        rcv_p, rcv_vx, rcv_vz,
        fwd_p, fwd_vx, fwd_vz,
    )


def elastic_forward_kernel(
    nx: int,
    nz: int,
    dx: float,
    dz: float,
    nt: int,
    dt: float,
    nabc: int,
    free_surface: bool,
    src_x: Tensor,
    src_z: Tensor,
    src_n: int,
    src_v: Tensor,
    rcv_x: Tensor,
    rcv_z: Tensor,
    rcv_n: int,
    damp: Tensor,
    vp: Tensor,
    vs: Tensor,
    rho: Tensor,
    checkpoint_segments: int = 1,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
    snapshot_every: int = 0,
) -> Dict[str, Tensor]:
    """
    Forward simulation kernel for 2D elastic wave equation.

    Main entry point for elastic wave propagation. Handles model
    padding, coefficient computation, and time stepping with
    optional gradient checkpointing.

    Args:
        nx: Number of grid points in x-direction.
        nz: Number of grid points in z-direction.
        dx: Grid spacing in x-direction (m).
        dz: Grid spacing in z-direction (m).
        nt: Number of time steps.
        dt: Time step (s).
        nabc: Number of absorbing boundary layers.
        free_surface: Whether to use free surface at top.
        src_x: Source x-positions (grid indices).
        src_z: Source z-positions (grid indices).
        src_n: Number of sources.
        src_v: Source wavelets with shape (src_n, nt) or (nt,).
        rcv_x: Receiver x-positions (grid indices).
        rcv_z: Receiver z-positions (grid indices).
        rcv_n: Number of receivers.
        damp: Damping coefficients from boundary module.
        vp: P-wave velocity model (m/s) with shape (nz, nx).
        vs: S-wave velocity model (m/s) with shape (nz, nx).
        rho: Density model (kg/m³) with shape (nz, nx).
        checkpoint_segments: Number of segments for gradient checkpointing.
        device: Computation device. Default: 'cpu'.
        dtype: Data type. Default: torch.float32.
        snapshot_every: Save pressure wavefield snapshot every N steps.
            0 disables snapshots. When > 0, checkpointing is adjusted
            to capture snapshots at segment boundaries. Default: 0.

    Returns:
        Dictionary containing:
            - 'p': Pressure seismograms (src_n, nt, rcv_n)
            - 'vx': Horizontal velocity seismograms
            - 'vz': Vertical velocity seismograms
            - 'forward_wavefield_p': Pressure wavefield energy (nz, nx)
            - 'forward_wavefield_vx': Horizontal velocity energy
            - 'forward_wavefield_vz': Vertical velocity energy
            - 'snapshots': Wavefield snapshots (n_snaps, nz, nx),
              only present when snapshot_every > 0.
    """
    # Pad model parameters
    vp_pad = pad_model(vp, nabc, nz, nx, device=device)
    vs_pad = pad_model(vs, nabc, nz, nx, device=device)
    rho_pad = pad_model(rho, nabc, nz, nx, device=device)

    nx_pml = nx + 2 * nabc
    nz_pml = nz + 2 * nabc
    fs = nabc if free_surface else 1

    # Adjust source and receiver positions for padding
    src_x = src_x + nabc
    src_z = src_z + nabc
    rcv_x = rcv_x + nabc
    rcv_z = rcv_z + nabc

    # Compute Lame parameters on padded grid
    mu = vs_pad ** 2 * rho_pad
    lamu = vp_pad ** 2 * rho_pad
    lam = lamu - 2 * mu
    b = 1.0 / rho_pad

    # Staggered averages
    # mu_xz: harmonic mean at (i+1/2, j+1/2), shape (nz_pml-1, nx_pml-1)
    mu_xz = torch.zeros(nz_pml - 1, nx_pml - 1, dtype=dtype, device=device)
    mu00 = mu[:nz_pml - 1, :nx_pml - 1]
    mu10 = mu[1:nz_pml, :nx_pml - 1]
    mu01 = mu[:nz_pml - 1, 1:nx_pml]
    mu11 = mu[1:nz_pml, 1:nx_pml]
    # Harmonic mean; handle zero mu (fluid regions)
    mu_sum = mu00 + mu10 + mu01 + mu11
    nonzero = (mu00 > 0) & (mu10 > 0) & (mu01 > 0) & (mu11 > 0)
    inv_sum = torch.zeros_like(mu_sum)
    inv_sum[nonzero] = (
        1.0 / mu00[nonzero] + 1.0 / mu10[nonzero] +
        1.0 / mu01[nonzero] + 1.0 / mu11[nonzero]
    )
    mu_xz[nonzero] = 4.0 / inv_sum[nonzero]

    # bx: arithmetic mean at vx position (i, j+1/2)
    bx = torch.zeros(nz_pml, nx_pml, dtype=dtype, device=device)
    bx[:, :nx_pml - 1] = 0.5 * (b[:, :nx_pml - 1] + b[:, 1:nx_pml])

    # bz: arithmetic mean at vz position (i+1/2, j)
    bz = torch.zeros(nz_pml, nx_pml, dtype=dtype, device=device)
    bz[:nz_pml - 1, :] = 0.5 * (b[:nz_pml - 1, :] + b[1:nz_pml, :])

    # 4th-order FD coefficients
    c1 = 9.0 / 8.0
    c2 = -1.0 / 24.0

    # Precompute alpha coefficients (material * dt / dz)
    alpha_lamu = lamu * dt / dz
    alpha_lam = lam * dt / dz
    alpha_mu = mu_xz * dt / dz
    alpha_bx = bx * dt / dz
    alpha_bz = bz * dt / dz

    # Precompute kappa (damping) coefficients
    kappa_s = damp * dt

    # kappa_sxz: average damp at shear stress position (i+1/2, j+1/2)
    kappa_sxz = torch.zeros(nz_pml - 1, nx_pml - 1, dtype=dtype, device=device)
    kappa_sxz = 0.25 * (
        damp[:nz_pml - 1, :nx_pml - 1] + damp[1:nz_pml, :nx_pml - 1] +
        damp[:nz_pml - 1, 1:nx_pml] + damp[1:nz_pml, 1:nx_pml]
    ) * dt

    # kappa_vx: average damp at vx position (same as acoustic kappa2)
    kappa_vx = torch.zeros_like(damp, device=device)
    kappa_vx[:, 1:nx_pml - 2] = 0.5 * (
        damp[:, 1:nx_pml - 2] + damp[:, 2:nx_pml - 1]
    ) * dt

    # kappa_vz: average damp at vz position (same as acoustic kappa3)
    kappa_vz = torch.zeros_like(damp, device=device)
    kappa_vz[fs:nz_pml - 2, :] = 0.5 * (
        damp[fs:nz_pml - 2, :] + damp[fs + 1:nz_pml - 1, :]
    ) * dt

    # Initialize fields
    sigma_xx = torch.zeros((src_n, nz_pml, nx_pml), dtype=dtype, device=device)
    sigma_zz = torch.zeros((src_n, nz_pml, nx_pml), dtype=dtype, device=device)
    sigma_xz = torch.zeros((src_n, nz_pml - 1, nx_pml - 1), dtype=dtype, device=device)
    vx_field = torch.zeros((src_n, nz_pml, nx_pml - 1), dtype=dtype, device=device)
    vz_field = torch.zeros((src_n, nz_pml - 1, nx_pml), dtype=dtype, device=device)

    # Initialize recorded waveforms
    rcv_p = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_vx = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_vz = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)

    # Initialize wavefield accumulators
    fwd_p = torch.zeros((nz, nx), dtype=dtype, device=device)
    fwd_vx = torch.zeros((nz, nx), dtype=dtype, device=device)
    fwd_vz = torch.zeros((nz, nx), dtype=dtype, device=device)

    # Build chunk list
    if snapshot_every > 0:
        # Fixed-size chunks for snapshot collection
        chunk_list = []
        for t0 in range(0, nt, snapshot_every):
            chunk_list.append(src_v[..., t0:min(t0 + snapshot_every, nt)])
        snapshots = []
    else:
        chunk_list = list(torch.chunk(src_v, checkpoint_segments, dim=-1))
        snapshots = None

    # Time stepping with checkpointing
    k = 0
    for i, chunk in enumerate(chunk_list):
        if snapshot_every > 0:
            # Direct call (no checkpointing) for snapshot mode
            result = elastic_step_forward(
                nx, nz, dx, dz, dt,
                nabc, free_surface,
                src_x, src_z, src_n, chunk,
                rcv_x, rcv_z, rcv_n,
                kappa_s, kappa_sxz, kappa_vx, kappa_vz,
                alpha_lamu, alpha_lam, alpha_mu, alpha_bx, alpha_bz,
                c1, c2,
                sigma_xx, sigma_zz, sigma_xz, vx_field, vz_field,
                device, dtype,
            )
        elif checkpoint_segments > 1:
            result = checkpoint(
                elastic_step_forward,
                nx, nz, dx, dz, dt,
                nabc, free_surface,
                src_x, src_z, src_n, chunk,
                rcv_x, rcv_z, rcv_n,
                kappa_s, kappa_sxz, kappa_vx, kappa_vz,
                alpha_lamu, alpha_lam, alpha_mu, alpha_bx, alpha_bz,
                c1, c2,
                sigma_xx, sigma_zz, sigma_xz, vx_field, vz_field,
                device, dtype,
                use_reentrant=True
            )
        else:
            result = elastic_step_forward(
                nx, nz, dx, dz, dt,
                nabc, free_surface,
                src_x, src_z, src_n, chunk,
                rcv_x, rcv_z, rcv_n,
                kappa_s, kappa_sxz, kappa_vx, kappa_vz,
                alpha_lamu, alpha_lam, alpha_mu, alpha_bx, alpha_bz,
                c1, c2,
                sigma_xx, sigma_zz, sigma_xz, vx_field, vz_field,
                device, dtype,
            )

        sigma_xx = result[0]
        sigma_zz = result[1]
        sigma_xz = result[2]
        vx_field = result[3]
        vz_field = result[4]
        rcv_p_temp = result[5]
        rcv_vx_temp = result[6]
        rcv_vz_temp = result[7]
        fwd_p_temp = result[8]
        fwd_vx_temp = result[9]
        fwd_vz_temp = result[10]

        # Save recorded waveforms
        chunk_len = chunk.shape[-1]
        rcv_p[:, k:k + chunk_len] = rcv_p_temp
        rcv_vx[:, k:k + chunk_len] = rcv_vx_temp
        rcv_vz[:, k:k + chunk_len] = rcv_vz_temp

        # Accumulate wavefield
        fwd_p += fwd_p_temp.detach()
        fwd_vx += fwd_vx_temp.detach()
        fwd_vz += fwd_vz_temp.detach()

        # Capture snapshot (first shot pressure field)
        if snapshot_every > 0:
            p_snap = -(sigma_xx[0] + sigma_zz[0]) / 2.0
            snapshots.append(
                p_snap[nabc:nabc + nz, nabc:nabc + nx].detach().cpu().clone()
            )

        k += chunk_len

    out = {
        'p': rcv_p,
        'vx': rcv_vx,
        'vz': rcv_vz,
        'forward_wavefield_p': fwd_p,
        'forward_wavefield_vx': fwd_vx,
        'forward_wavefield_vz': fwd_vz,
    }
    if snapshot_every > 0:
        out['snapshots'] = torch.stack(snapshots, dim=0)
    return out