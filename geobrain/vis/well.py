"""
Well log visualization.

Multi-track display for well log curves.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict, Union

from ._utils import to_numpy


# Default colors for well log tracks
_TRACK_COLORS = [
    '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
]


def plot_logs(
    curves,
    depth=None,
    names: Optional[List[str]] = None,
    units: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    depth_label: str = 'Depth (m)',
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    **kwargs,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot well log curves as multi-track display.

    Each curve gets its own track (subplot) sharing a common depth
    axis on the left.

    Args:
        curves: Well log data. Accepts:
            - dict: {name: 1D_array}
            - LASData: from geobrain.io.read_las()
            - 2D array: (n_samples, n_curves), requires names
        depth: Depth array. If None:
            - Inferred from LASData.index
            - Or uses np.arange(n_samples)
        names: Curve names (required if curves is ndarray).
        units: Curve units for x-axis labels.
        colors: Line colors per track. Cycles through defaults.
        depth_label: Y-axis label. Default: 'Depth (m)'.
        figsize: Figure size. Default: auto-computed.
        title: Figure super-title.
        **kwargs: Passed to ax.plot().

    Returns:
        (fig, axes) tuple where axes has shape (n_curves,).

    Example:
        >>> from geobrain.io import read_las
        >>> las = read_las('well.las')
        >>> fig, axes = plot_logs(las, names=['GR', 'DT', 'RHOB'])

        With dict:
        >>> fig, axes = plot_logs(
        ...     {'GR': gr_data, 'DT': dt_data},
        ...     depth=depth,
        ...     units=['API', 'us/ft'],
        ... )
    """
    # ── Parse input ─────────────────────────────────────────────
    if hasattr(curves, 'curve_names') and hasattr(curves, 'data'):
        # LASData object
        las = curves
        if names is None:
            # Skip depth index (first curve)
            names = las.curve_names[1:]
        if depth is None:
            depth = las.index
        if units is None:
            unit_map = dict(zip(las.curve_names, las.curve_units))
            units = [unit_map.get(n, '') for n in names]
        curve_dict = {n: las[n] for n in names}

    elif isinstance(curves, dict):
        curve_dict = {k: to_numpy(v) for k, v in curves.items()}
        if names is None:
            names = list(curve_dict.keys())
        else:
            curve_dict = {n: curve_dict[n] for n in names}
        if depth is None:
            depth = np.arange(len(next(iter(curve_dict.values()))))

    else:
        # ndarray (n_samples, n_curves)
        arr = to_numpy(curves)
        if names is None:
            names = [f'Curve_{i}' for i in range(arr.shape[1])]
        curve_dict = {
            names[i]: arr[:, i] for i in range(arr.shape[1])
        }
        if depth is None:
            depth = np.arange(arr.shape[0])

    depth = to_numpy(depth)
    n_tracks = len(names)

    if units is None:
        units = [''] * n_tracks
    if colors is None:
        colors = [_TRACK_COLORS[i % len(_TRACK_COLORS)]
                  for i in range(n_tracks)]

    if figsize is None:
        figsize = (2.5 * n_tracks, 8)

    # ── Create figure ───────────────────────────────────────────
    fig, axes = plt.subplots(
        1, n_tracks,
        figsize=figsize,
        sharey=True,
    )
    if n_tracks == 1:
        axes = np.array([axes])

    for i, (name, unit, color) in enumerate(zip(names, units, colors)):
        ax = axes[i]
        values = to_numpy(curve_dict[name])

        ax.plot(values, depth, color=color, linewidth=1.0, **kwargs)
        ax.set_xlim(np.nanmin(values), np.nanmax(values))

        xlabel = f'{name} ({unit})' if unit else name
        ax.set_xlabel(xlabel)
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.set_ylabel(depth_label)
            ax.invert_yaxis()

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    fig.tight_layout()
    return fig, axes
