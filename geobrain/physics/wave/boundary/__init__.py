"""
Absorbing boundary conditions for wave simulation.

Provides various methods to suppress artificial reflections
at computational domain boundaries.

Available methods:
    - PML: Perfectly Matched Layer, most effective, recommended
    - Gerjan: Exponential sponge layer
    - SinCos: Trigonometric taper

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import numpy as np
from typing import Union, Tuple

from .pml import bc_pml, bc_pml_3d, bc_pml_xz
from .absorbing import bc_gerjan, bc_sincos


def create_boundary(
    bc_type: str,
    nx: int,
    nz: int,
    dx: float,
    dz: float,
    n_layers: int,
    vmax: float = None,
    alpha: float = 0.0053,
    free_surface: bool = False,
    ny: int = None,
    dy: float = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Factory function to create boundary condition arrays.

    Creates the appropriate damping/weight arrays based on the
    specified boundary type.

    Args:
        bc_type: Boundary type. Options:
            - 'pml': Perfectly Matched Layer
            - 'pml_xz': Split PML for elastic (returns tuple)
            - 'gerjan': Exponential damping
            - 'sincos': Trigonometric taper
        nx: Number of grid points in x-direction.
        nz: Number of grid points in z-direction.
        dx: Grid spacing in x-direction (m).
        dz: Grid spacing in z-direction (m).
        n_layers: Number of boundary layers.
        vmax: Maximum velocity (m/s). Required for PML types.
        alpha: Attenuation factor for Gerjan. Default: 0.0053.
        free_surface: If True, no absorbing at top. Default: False.
        ny: Number of grid points in y-direction (3-D only).
        dy: Grid spacing in y-direction (m) (3-D only).

    Returns:
        For 2-D 'pml', 'gerjan', 'sincos': Single 2D array (nz_pml, nx_pml).
        For 2-D 'pml_xz': Tuple of (BCx, BCz) arrays.
        For 3-D 'pml': Single 3D array (nz_pml, ny_pml, nx_pml).

    Raises:
        ValueError: If bc_type is unknown or vmax not provided for PML.

    Example:
        PML boundary:
        >>> damp = create_boundary(
        ...     'pml', nx=200, nz=100,
        ...     dx=10, dz=10, n_layers=20, vmax=3000
        ... )

        Split PML for elastic:
        >>> bcx, bcz = create_boundary(
        ...     'pml_xz', nx=200, nz=100,
        ...     dx=10, dz=10, n_layers=20, vmax=3000
        ... )
    """
    bc_type = bc_type.lower()

    # 3-D dispatch
    if ny is not None:
        if bc_type == 'pml':
            if vmax is None:
                raise ValueError("vmax is required for PML boundary")
            if dy is None:
                raise ValueError("dy is required for 3-D PML boundary")
            return bc_pml_3d(nx, ny, nz, dx, dy, dz, n_layers, vmax, free_surface)
        else:
            raise NotImplementedError(
                f"3-D boundary type '{bc_type}' is not yet supported. "
                f"Only 'pml' is available for 3-D."
            )

    if bc_type == 'pml':
        if vmax is None:
            raise ValueError("vmax is required for PML boundary")
        return bc_pml(nx, nz, dx, dz, n_layers, vmax, free_surface)

    elif bc_type == 'pml_xz':
        if vmax is None:
            raise ValueError("vmax is required for PML boundary")
        return bc_pml_xz(nx, nz, dx, dz, n_layers, vmax, free_surface)

    elif bc_type == 'gerjan':
        return bc_gerjan(nx, nz, dx, dz, n_layers, alpha, free_surface)

    elif bc_type == 'sincos':
        return bc_sincos(nx, nz, dx, dz, n_layers, free_surface)

    else:
        raise ValueError(
            f"Unknown boundary type: '{bc_type}'. "
            f"Choose from: 'pml', 'pml_xz', 'gerjan', 'sincos'"
        )


__all__ = [
    # PML
    'bc_pml',
    'bc_pml_3d',
    'bc_pml_xz',
    # Other absorbing
    'bc_gerjan',
    'bc_sincos',
    # Factory
    'create_boundary',
]