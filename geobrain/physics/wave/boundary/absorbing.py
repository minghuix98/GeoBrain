"""
Absorbing boundary conditions: Gerjan and SinCos.

Alternative absorbing boundary conditions for wave simulation.
Generally less effective than PML but computationally simpler.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import numpy as np


def bc_gerjan(
    nx: int,
    nz: int,
    dx: float,
    dz: float,
    pml: int,
    alpha: float = 0.0053,
    free_surface: bool = True,
) -> np.ndarray:
    """
    Compute Gerjan exponential damping coefficients.
    
    Implements the sponge layer approach with exponential decay:

        G = exp(-(α × (N - i))²)

    Args:
        nx: Number of grid points in x-direction.
        nz: Number of grid points in z-direction.
        dx: Grid spacing in x-direction (m).
        dz: Grid spacing in z-direction (m).
        pml: Number of boundary layers.
        alpha: Attenuation factor. Default: 0.0053.
        free_surface: If True, no damping at top. Default: True.

    Returns:
        2D damping weight array with shape (nz_pml, nx_pml).
        Values range from 0 (full damping) to 1 (no damping).

    Example:
        >>> damp = bc_gerjan(nx=200, nz=100, dx=10, dz=10, pml=20)
        >>> # Apply: wavefield *= damp

    Note:
        Unlike PML, Gerjan returns multiplicative weights.
        Apply as: wavefield = wavefield * damp
    """
    # Compute damping weights
    wt = np.exp(-(alpha * (pml - np.arange(1, pml + 1))) ** 2)

    # Compute padded dimensions
    if free_surface:
        nx_pml = nx + 2 * pml
        nz_pml = nz + pml
    else:
        nx_pml = nx + 2 * pml
        nz_pml = nz + 2 * pml

    # Initialize with ones (no damping in interior)
    damp = np.ones((nz_pml, nx_pml))

    # Apply damping at boundaries
    if free_surface:
        for k in range(1, len(wt) + 1):
            # Left boundary
            damp[:nz_pml - k + 1, k - 1] = wt[k - 1]
            # Bottom boundary
            damp[nz_pml - k, k - 1:nx_pml - k + 1] = wt[k - 1]
            # Right boundary
            damp[:nz_pml - k + 1, nx_pml - k] = wt[k - 1]
    else:
        for k in range(1, len(wt) + 1):
            # Left boundary
            damp[k - 1:nz_pml - k + 1, k - 1] = wt[k - 1]
            # Right boundary
            damp[k:nz_pml - k + 1, nx_pml - k] = wt[k - 1]
            # Bottom boundary
            damp[nz_pml - k, k - 1:nx_pml - k + 1] = wt[k - 1]
            # Top boundary
            damp[k - 1, k - 1:nx_pml - k + 1] = wt[k - 1]

    return damp


def bc_sincos(
    nx: int,
    nz: int,
    dx: float,
    dz: float,
    pml: int,
    free_surface: bool = False,
) -> np.ndarray:
    """
    Compute sinusoidal damping coefficients.

    Uses a sin² taper for smooth transition at boundaries:

        w = sin²(π × i / (2N))

    Args:
        nx: Number of grid points in x-direction.
        nz: Number of grid points in z-direction.
        dx: Grid spacing in x-direction (m).
        dz: Grid spacing in z-direction (m).
        pml: Number of boundary layers.
        free_surface: If True, no damping at top. Default: False.

    Returns:
        2D damping weight array with shape (nz_pml, nx_pml).
        Values range from 0 (full damping) to 1 (no damping).

    Example:
        >>> damp = bc_sincos(nx=200, nz=100, dx=10, dz=10, pml=20)
        >>> # Apply: wavefield *= damp
    """
    # Compute padded dimensions
    if free_surface:
        nx_pml = nx + 2 * pml
        nz_pml = nz + pml
    else:
        nx_pml = nx + 2 * pml
        nz_pml = nz + 2 * pml

    # Initialize with ones
    damp = np.ones((nz_pml, nx_pml))

    # Apply sin² taper at boundaries
    for i in range(pml):
        sin_val = np.sin(np.pi / 2 * i / pml) ** 2

        # Top (if not free surface)
        if not free_surface:
            damp[i, :] *= sin_val

        # Bottom
        damp[-i - 1, :] *= sin_val

        # Left
        damp[:, i] *= sin_val

        # Right
        damp[:, -i - 1] *= sin_val

    return damp