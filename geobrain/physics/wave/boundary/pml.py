"""
Perfectly Matched Layer (PML) boundary condition.

PML is the most effective absorbing boundary condition for wave
simulation, providing minimal artificial reflections.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import numpy as np
from math import log
from typing import Tuple


def bc_pml(
    nx: int,
    nz: int,
    dx: float,
    dz: float,
    pml: int,
    vmax: float,
    free_surface: bool = True,
) -> np.ndarray:
    """
    Compute PML damping coefficients.

    Creates a 2D array of damping values that increase quadratically
    from the domain interior toward the boundaries.

    The damping profile follows:

        σ(x) = κ × (x/L)²

    where κ is computed to achieve target reflection coefficient R = 10⁻⁶.

    Args:
        nx: Number of grid points in x-direction.
        nz: Number of grid points in z-direction.
        dx: Grid spacing in x-direction (m).
        dz: Grid spacing in z-direction (m).
        pml: Number of PML layers.
        vmax: Maximum velocity in model (m/s), used to scale damping.
        free_surface: If True, no PML at top boundary. Default: True.

    Returns:
        2D damping array with shape (nz_pml, nx_pml).

    Example:
        >>> damp = bc_pml(
        ...     nx=200, nz=100, dx=10, dz=10,
        ...     pml=20, vmax=3000, free_surface=True
        ... )
        >>> print(damp.shape)  # (120, 240)

    Note:
        The reflection coefficient R is set to 10⁻⁶ for minimal
        artificial reflections.
    """
    # Compute padded dimensions
    if free_surface:
        nx_pml = nx + 2 * pml
        nz_pml = nz + pml
    else:
        nx_pml = nx + 2 * pml
        nz_pml = nz + 2 * pml

    # Initialize damping array
    damp_global = np.zeros((nx_pml, nz_pml))
    damp = np.zeros(pml)

    # PML parameters
    a = (pml - 1) * dx
    R = 1e-6  # Target reflection coefficient
    kappa = -3.0 * vmax * log(R) / (2.0 * a)

    # Compute damping profile (quadratic)
    for ix in range(pml):
        xa = ix * dx / a
        damp[ix] = kappa * xa * xa

    # Apply to left and right boundaries
    for ix in range(pml):
        for iz in range(nz_pml):
            damp_global[pml - ix - 1, iz] = damp[ix]  # Left
            damp_global[nx_pml - pml + ix, iz] = damp[ix]  # Right

    # Apply to top and bottom boundaries
    for iz in range(pml):
        start_x = pml - iz
        end_x = nx_pml - (pml - iz)

        if not free_surface:
            damp_global[start_x:end_x, pml - iz - 1] = damp[iz]  # Top

        damp_global[start_x:end_x, nz_pml - pml + iz] = damp[iz]  # Bottom

    return damp_global.T


def bc_pml_3d(
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    dy: float,
    dz: float,
    pml: int,
    vmax: float,
    free_surface: bool = True,
) -> np.ndarray:
    """
    Compute 3D PML damping coefficients.

    Creates a 3D array of damping values that increase quadratically
    from the domain interior toward the boundaries on all six faces.

    The damping profile follows:

        σ(x) = κ × (x/L)²

    where κ is computed to achieve target reflection coefficient R = 10⁻⁶.

    Args:
        nx: Number of grid points in x-direction.
        ny: Number of grid points in y-direction.
        nz: Number of grid points in z-direction.
        dx: Grid spacing in x-direction (m).
        dy: Grid spacing in y-direction (m).
        dz: Grid spacing in z-direction (m).
        pml: Number of PML layers.
        vmax: Maximum velocity in model (m/s), used to scale damping.
        free_surface: If True, no PML at top boundary. Default: True.

    Returns:
        3D damping array with shape (nz_pml, ny_pml, nx_pml).

    Example:
        >>> damp = bc_pml_3d(
        ...     nx=64, ny=64, nz=64, dx=10, dy=10, dz=10,
        ...     pml=10, vmax=3000, free_surface=False
        ... )
        >>> print(damp.shape)  # (84, 84, 84)
    """
    # Compute padded dimensions
    nx_pml = nx + 2 * pml
    ny_pml = ny + 2 * pml
    if free_surface:
        nz_pml = nz + pml
    else:
        nz_pml = nz + 2 * pml

    # Initialize damping array (nz_pml, ny_pml, nx_pml)
    damp_global = np.zeros((nz_pml, ny_pml, nx_pml))

    # PML parameters
    a = (pml - 1) * dx
    R = 1e-6  # Target reflection coefficient
    kappa = -3.0 * vmax * log(R) / (2.0 * a)

    # Compute damping profile (quadratic)
    damp = np.zeros(pml)
    for i in range(pml):
        xa = i * dx / a
        damp[i] = kappa * xa * xa

    # Apply to x-faces (left and right)
    for ix in range(pml):
        damp_global[:, :, pml - ix - 1] = np.maximum(
            damp_global[:, :, pml - ix - 1], damp[ix]
        )
        damp_global[:, :, nx_pml - pml + ix] = np.maximum(
            damp_global[:, :, nx_pml - pml + ix], damp[ix]
        )

    # Apply to y-faces (front and back)
    for iy in range(pml):
        damp_global[:, pml - iy - 1, :] = np.maximum(
            damp_global[:, pml - iy - 1, :], damp[iy]
        )
        damp_global[:, ny_pml - pml + iy, :] = np.maximum(
            damp_global[:, ny_pml - pml + iy, :], damp[iy]
        )

    # Apply to z-faces (top and bottom)
    for iz in range(pml):
        if not free_surface:
            damp_global[pml - iz - 1, :, :] = np.maximum(
                damp_global[pml - iz - 1, :, :], damp[iz]
            )
        damp_global[nz_pml - pml + iz, :, :] = np.maximum(
            damp_global[nz_pml - pml + iz, :, :], damp[iz]
        )

    return damp_global


def bc_pml_xz(
    nx: int,
    nz: int,
    dx: float,
    dz: float,
    pml: int,
    vmax: float,
    free_surface: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute split PML damping coefficients for elastic waves.

    Returns separate damping arrays for x and z directions,
    required for the split-field PML formulation in elastic
    wave simulation.

    Args:
        nx: Number of grid points in x-direction.
        nz: Number of grid points in z-direction.
        dx: Grid spacing in x-direction (m).
        dz: Grid spacing in z-direction (m).
        pml: Number of PML layers.
        vmax: Maximum velocity in model (m/s).
        free_surface: If True, no PML at top. Default: True.

    Returns:
        Tuple of (BCx, BCz) damping arrays, each with shape
        (nz_pml, nx_pml).

    Example:
        >>> bcx, bcz = bc_pml_xz(
        ...     nx=200, nz=100, dx=10, dz=10,
        ...     pml=20, vmax=3000
        ... )
        >>> print(bcx.shape, bcz.shape)
    """
    # Compute padded dimensions
    if free_surface:
        nx_pml = nx + 2 * pml
        nz_pml = nz + pml
    else:
        nx_pml = nx + 2 * pml
        nz_pml = nz + 2 * pml

    # PML coefficient
    R = 1e-6
    ppml = -np.log(R) * 3 * vmax / (2 * pml ** 3)

    # Initialize arrays
    BCx = np.zeros((nz_pml, nx_pml))
    BCz = np.zeros((nz_pml, nx_pml))

    # Apply damping
    for k in range(1, pml + 1):
        coef = (pml - k + 1) ** 2 * ppml

        # X-direction (left and right)
        BCx[:, k - 1] = coef / dx
        BCx[:, nx_pml - k] = coef / dx

        # Z-direction (top and bottom)
        if not free_surface:
            BCz[k - 1, :] = coef / dz
        BCz[nz_pml - k, :] = coef / dz

    return BCx, BCz