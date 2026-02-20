"""
2D/3D property field visualization.

Functions for displaying velocity models, porosity fields,
pressure maps, and other spatial properties.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Union

from ._utils import to_numpy, add_colorbar, ensure_ax


def plot_field(
    data,
    dx: float = 1.0,
    dz: float = 1.0,
    ax: Optional[plt.Axes] = None,
    cmap: str = 'viridis',
    colorbar: bool = True,
    label: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a 2D property field.

    Args:
        data: 2D array (nz, nx) of property values.
        dx: Horizontal grid spacing (meters). Default: 1.0.
        dz: Vertical grid spacing (meters). Default: 1.0.
        ax: Existing axes. If None, creates new figure.
        cmap: Colormap name. Default: 'viridis'.
        colorbar: Whether to add colorbar. Default: True.
        label: Colorbar label (e.g. 'Vp (m/s)'). Default: None.
        figsize: Figure size if creating new figure.
        **kwargs: Passed to ax.imshow().

    Returns:
        (fig, ax) tuple.

    Example:
        >>> fig, ax = plot_field(vp, dx=12.5, dz=12.5,
        ...                     cmap='jet', label='Vp (m/s)')
    """
    data = to_numpy(data)
    nz, nx = data.shape

    fig, ax = ensure_ax(ax, figsize)

    extent = [0, (nx - 1) * dx, (nz - 1) * dz, 0]

    defaults = dict(
        aspect='auto',
        cmap=cmap,
        extent=extent,
        interpolation='bilinear',
    )
    defaults.update(kwargs)

    im = ax.imshow(data, **defaults)
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Depth (m)')

    if colorbar:
        add_colorbar(im, ax, label=label)

    return fig, ax


def plot_slices(
    data,
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
    ix: Optional[int] = None,
    iy: Optional[int] = None,
    iz: Optional[int] = None,
    cmap: str = 'viridis',
    colorbar: bool = True,
    label: Optional[str] = None,
    figsize: Tuple[float, float] = (16, 5),
    **kwargs,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot three orthogonal slices of a 3D volume.

    Displays XY (depth slice), XZ (inline), and YZ (crossline)
    views side by side.

    Args:
        data: 3D array (nz, ny, nx) of property values.
        dx, dy, dz: Grid spacing in each direction. Default: 1.0.
        ix, iy, iz: Slice indices. If None, uses middle of each axis.
        cmap: Colormap name. Default: 'viridis'.
        colorbar: Whether to add colorbars. Default: True.
        label: Colorbar label. Default: None.
        figsize: Figure size.
        **kwargs: Passed to ax.imshow().

    Returns:
        (fig, axes) tuple where axes has shape (3,).

    Example:
        >>> fig, axes = plot_slices(volume, dx=10, dy=10, dz=10,
        ...                        label='Vp (m/s)')
    """
    data = to_numpy(data)
    nz, ny, nx = data.shape

    if ix is None:
        ix = nx // 2
    if iy is None:
        iy = ny // 2
    if iz is None:
        iz = nz // 2

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    vmin = kwargs.pop('vmin', data.min())
    vmax = kwargs.pop('vmax', data.max())

    # XY slice (depth slice at iz)
    slice_xy = data[iz, :, :]
    extent_xy = [0, (nx - 1) * dx, (ny - 1) * dy, 0]
    im0 = axes[0].imshow(
        slice_xy, aspect='auto', cmap=cmap,
        extent=extent_xy, vmin=vmin, vmax=vmax, **kwargs,
    )
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    axes[0].set_title(f'XY slice (z={iz * dz:.0f} m)')
    if colorbar:
        add_colorbar(im0, axes[0], label=label)

    # XZ slice (inline at iy)
    slice_xz = data[:, iy, :]
    extent_xz = [0, (nx - 1) * dx, (nz - 1) * dz, 0]
    im1 = axes[1].imshow(
        slice_xz, aspect='auto', cmap=cmap,
        extent=extent_xz, vmin=vmin, vmax=vmax, **kwargs,
    )
    axes[1].set_xlabel('X (m)')
    axes[1].set_ylabel('Depth (m)')
    axes[1].set_title(f'XZ slice (y={iy * dy:.0f} m)')
    if colorbar:
        add_colorbar(im1, axes[1], label=label)

    # YZ slice (crossline at ix)
    slice_yz = data[:, :, ix]
    extent_yz = [0, (ny - 1) * dy, (nz - 1) * dz, 0]
    im2 = axes[2].imshow(
        slice_yz, aspect='auto', cmap=cmap,
        extent=extent_yz, vmin=vmin, vmax=vmax, **kwargs,
    )
    axes[2].set_xlabel('Y (m)')
    axes[2].set_ylabel('Depth (m)')
    axes[2].set_title(f'YZ slice (x={ix * dx:.0f} m)')
    if colorbar:
        add_colorbar(im2, axes[2], label=label)

    fig.tight_layout()
    return fig, axes


def plot_comparison(
    fields: List,
    titles: Optional[List[str]] = None,
    dx: float = 1.0,
    dz: float = 1.0,
    cmap: str = 'viridis',
    share_clim: bool = True,
    colorbar: bool = True,
    label: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot multiple 2D fields side by side for comparison.

    Typical use: [true, predicted, difference].

    Args:
        fields: List of 2D arrays (nz, nx).
        titles: List of subplot titles. Default: None.
        dx, dz: Grid spacing. Default: 1.0.
        cmap: Colormap name. Can be a single string (applied to all)
            or a list of strings (one per field). Default: 'viridis'.
        share_clim: Use the same color limits across all panels.
            Default: True.
        colorbar: Whether to add colorbars. Default: True.
        label: Colorbar label. Default: None.
        figsize: Figure size. Default: auto-computed.
        **kwargs: Passed to ax.imshow().

    Returns:
        (fig, axes) tuple.

    Example:
        >>> fig, axes = plot_comparison(
        ...     [vp_true, vp_pred, vp_true - vp_pred],
        ...     titles=['True', 'Predicted', 'Difference'],
        ...     cmap=['jet', 'jet', 'seismic'],
        ...     label='Vp (m/s)',
        ... )
    """
    n = len(fields)
    arrays = [to_numpy(f) for f in fields]

    if titles is None:
        titles = [None] * n

    if isinstance(cmap, str):
        cmaps = [cmap] * n
    else:
        cmaps = list(cmap)

    if figsize is None:
        figsize = (5 * n, 5)

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = np.array([axes])

    if share_clim:
        global_vmin = min(a.min() for a in arrays)
        global_vmax = max(a.max() for a in arrays)
    else:
        global_vmin, global_vmax = None, None

    for i, (arr, title, cm) in enumerate(zip(arrays, titles, cmaps)):
        nz, nx = arr.shape
        extent = [0, (nx - 1) * dx, (nz - 1) * dz, 0]

        im_kwargs = dict(
            aspect='auto', cmap=cm, extent=extent,
            interpolation='bilinear',
        )
        if share_clim:
            im_kwargs['vmin'] = global_vmin
            im_kwargs['vmax'] = global_vmax
        im_kwargs.update(kwargs)

        im = axes[i].imshow(arr, **im_kwargs)
        axes[i].set_xlabel('Distance (m)')
        axes[i].set_ylabel('Depth (m)')
        if title:
            axes[i].set_title(title)
        if colorbar:
            add_colorbar(im, axes[i], label=label)

    fig.tight_layout()
    return fig, axes
