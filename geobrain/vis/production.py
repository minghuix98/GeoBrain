"""
Production and reservoir simulation visualization.

Functions for displaying well production data, water cut,
and reservoir state maps from flow simulation results.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict, Union

from ._utils import to_numpy, ensure_ax, add_colorbar


def plot_production(
    time,
    rates: Dict[str, np.ndarray],
    ax: Optional[plt.Axes] = None,
    labels: Optional[Dict[str, str]] = None,
    colors: Optional[Dict[str, str]] = None,
    figsize: Tuple[float, float] = (10, 5),
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot production rate curves.

    Args:
        time: Time array (days).
        rates: Dict of {key: rate_array}. Common keys: 'qo', 'qw', 'qg'.
        ax: Existing axes. If None, creates new figure.
        labels: Custom labels for each key. Defaults to standard names.
        colors: Custom colors for each key.
        figsize: Figure size if creating new figure.
        **kwargs: Passed to ax.plot().

    Returns:
        (fig, ax) tuple.

    Example:
        >>> fig, ax = plot_production(
        ...     time, {'qo': oil_rate, 'qw': water_rate},
        ...     labels={'qo': 'Oil', 'qw': 'Water'},
        ... )
    """
    time = to_numpy(time)

    default_labels = {
        'qo': 'Oil Rate', 'qw': 'Water Rate', 'qg': 'Gas Rate',
        'ql': 'Liquid Rate', 'qi': 'Injection Rate',
    }
    default_colors = {
        'qo': '#2ca02c', 'qw': '#1f77b4', 'qg': '#d62728',
        'ql': 'black', 'qi': '#1f77b4',
    }

    if labels is None:
        labels = default_labels
    if colors is None:
        colors = default_colors

    fig, ax = ensure_ax(ax, figsize)

    for key, rate in rates.items():
        rate = to_numpy(rate)
        lbl = labels.get(key, key)
        clr = colors.get(key, None)
        ax.plot(time, rate, color=clr, label=lbl, **kwargs)

    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Rate (STB/day)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_watercut(
    time,
    qo,
    qw,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (10, 5),
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot water cut (fractional flow) curve.

    Args:
        time: Time array (days).
        qo: Oil production rate array.
        qw: Water production rate array.
        ax: Existing axes. If None, creates new figure.
        figsize: Figure size if creating new figure.
        **kwargs: Passed to ax.plot().

    Returns:
        (fig, ax) tuple.

    Example:
        >>> fig, ax = plot_watercut(time, qo, qw)
    """
    time = to_numpy(time)
    qo = to_numpy(qo)
    qw = to_numpy(qw)

    total = np.abs(qo) + np.abs(qw)
    wcut = np.where(total > 0, np.abs(qw) / total, 0.0)

    fig, ax = ensure_ax(ax, figsize)

    ax.plot(time, wcut, color='#1f77b4', **kwargs)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Water Cut')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_well_summary(
    time,
    well_data: Dict[str, np.ndarray],
    well_name: str = '',
    figsize: Tuple[float, float] = (14, 8),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot a 4-panel well summary: oil rate, water rate, water cut, BHP.

    Args:
        time: Time array (days).
        well_data: Dict with keys 'qo', 'qw', and optionally 'bhp'.
        well_name: Well name for the title.
        figsize: Figure size.

    Returns:
        (fig, axes) tuple where axes has shape (2, 2).

    Example:
        >>> fig, axes = plot_well_summary(
        ...     time, {'qo': qo, 'qw': qw, 'bhp': bhp},
        ...     well_name='P1',
        ... )
    """
    time = to_numpy(time)

    has_bhp = 'bhp' in well_data
    ncols = 2
    nrows = 2 if has_bhp else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1:
        axes = axes.reshape(1, -1)

    qo = to_numpy(well_data['qo'])
    qw = to_numpy(well_data['qw'])

    # Oil rate
    axes[0, 0].plot(time, qo, color='#2ca02c')
    axes[0, 0].set_ylabel('Oil Rate (STB/day)')
    axes[0, 0].set_title('Oil Production')
    axes[0, 0].grid(True, alpha=0.3)

    # Water rate
    axes[0, 1].plot(time, np.abs(qw), color='#1f77b4')
    axes[0, 1].set_ylabel('Water Rate (STB/day)')
    axes[0, 1].set_title('Water Production')
    axes[0, 1].grid(True, alpha=0.3)

    if has_bhp:
        bhp = to_numpy(well_data['bhp'])

        # Water cut
        total = np.abs(qo) + np.abs(qw)
        wcut = np.where(total > 0, np.abs(qw) / total, 0.0)
        axes[1, 0].plot(time, wcut, color='#d62728')
        axes[1, 0].set_ylabel('Water Cut')
        axes[1, 0].set_ylim(-0.05, 1.05)
        axes[1, 0].set_title('Water Cut')
        axes[1, 0].grid(True, alpha=0.3)

        # BHP
        axes[1, 1].plot(time, bhp, color='#9467bd')
        axes[1, 1].set_ylabel('BHP (psi)')
        axes[1, 1].set_title('Bottom-hole Pressure')
        axes[1, 1].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel('Time (days)')

    if well_name:
        fig.suptitle(f'Well {well_name}', fontsize=14)

    fig.tight_layout()
    return fig, axes
