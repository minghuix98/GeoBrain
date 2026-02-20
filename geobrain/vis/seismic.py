"""
Seismic data visualization.

Functions for displaying seismic sections, wiggle traces,
and shot gathers.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

from ._utils import to_numpy, symmetric_clim, add_colorbar, ensure_ax


def plot_section(
    data,
    dt: float = 1.0,
    dx: float = 1.0,
    ax: Optional[plt.Axes] = None,
    cmap: str = 'seismic',
    clip: float = 99,
    colorbar: bool = True,
    label: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a seismic section as an image.

    Args:
        data: 2D array (nt, nx) of seismic amplitudes.
        dt: Time sample interval (seconds). Default: 1.0.
        dx: Trace spacing (meters). Default: 1.0.
        ax: Existing axes. If None, creates new figure.
        cmap: Colormap name. Default: 'seismic'.
        clip: Percentile for symmetric amplitude clipping.
            Default: 99.
        colorbar: Whether to add colorbar. Default: True.
        label: Colorbar label. Default: None.
        figsize: Figure size if creating new figure.
        **kwargs: Passed to ax.imshow().

    Returns:
        (fig, ax) tuple.

    Example:
        >>> fig, ax = plot_section(shot_data, dt=0.001, dx=12.5)
        >>> ax.set_title('Shot Gather #1')
        >>> fig.savefig('section.png')
    """
    data = to_numpy(data)
    nt, nx = data.shape

    fig, ax = ensure_ax(ax, figsize)

    vmin, vmax = symmetric_clim(data, clip)
    extent = [0, (nx - 1) * dx, (nt - 1) * dt, 0]

    defaults = dict(
        aspect='auto',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        interpolation='bilinear',
    )
    defaults.update(kwargs)

    im = ax.imshow(data, **defaults)
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Time (s)')

    if colorbar:
        add_colorbar(im, ax, label=label)

    return fig, ax


def plot_wiggle(
    data,
    dt: float = 1.0,
    ax: Optional[plt.Axes] = None,
    trace_step: int = 1,
    fill: bool = True,
    gain: float = 0.8,
    color: str = 'k',
    linewidth: float = 0.5,
    figsize: Tuple[float, float] = (10, 6),
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot seismic traces as wiggle display.

    Args:
        data: 2D array (nt, nx) of seismic amplitudes.
        dt: Time sample interval (seconds). Default: 1.0.
        ax: Existing axes. If None, creates new figure.
        trace_step: Plot every N-th trace. Default: 1.
        fill: Fill positive amplitudes. Default: True.
        gain: Amplitude scaling relative to trace spacing.
            Default: 0.8.
        color: Line color. Default: 'k'.
        linewidth: Line width. Default: 0.5.
        figsize: Figure size if creating new figure.
        **kwargs: Passed to ax.plot().

    Returns:
        (fig, ax) tuple.

    Example:
        >>> fig, ax = plot_wiggle(shot_data, dt=0.001, trace_step=5)
    """
    data = to_numpy(data)
    nt, nx = data.shape

    fig, ax = ensure_ax(ax, figsize)

    t = np.arange(nt) * dt
    trace_indices = range(0, nx, trace_step)

    for i in trace_indices:
        trace = data[:, i]
        amax = np.abs(trace).max()
        if amax == 0:
            continue
        trace_norm = trace / amax * trace_step * gain
        x = trace_norm + i

        ax.plot(x, t, color=color, linewidth=linewidth, **kwargs)
        if fill:
            ax.fill_betweenx(
                t, i, x,
                where=(trace_norm > 0),
                color=color, alpha=0.5,
            )

    ax.set_xlim(-trace_step, nx)
    ax.set_ylim(t[-1], t[0])
    ax.set_xlabel('Trace')
    ax.set_ylabel('Time (s)')

    return fig, ax


def plot_gather(
    data,
    dt: float = 1.0,
    dx: float = 1.0,
    clip: float = 99,
    trace_step: int = 10,
    figsize: Tuple[float, float] = (14, 8),
    **kwargs,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot a shot gather with image (left) and wiggle (right).

    Args:
        data: 2D array (nt, nx) of seismic amplitudes.
        dt: Time sample interval (seconds). Default: 1.0.
        dx: Trace spacing (meters). Default: 1.0.
        clip: Percentile for amplitude clipping. Default: 99.
        trace_step: Wiggle trace decimation. Default: 10.
        figsize: Figure size.
        **kwargs: Passed to imshow and plot.

    Returns:
        (fig, axes) tuple where axes has shape (2,).

    Example:
        >>> fig, axes = plot_gather(shot_data, dt=0.001, dx=12.5)
        >>> axes[0].set_title('Image')
        >>> axes[1].set_title('Wiggle')
    """
    data = to_numpy(data)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    plot_section(data, dt=dt, dx=dx, ax=axes[0], clip=clip)
    axes[0].set_title('Image')

    plot_wiggle(data, dt=dt, ax=axes[1], trace_step=trace_step)
    axes[1].set_title('Wiggle')

    fig.tight_layout()
    return fig, axes
