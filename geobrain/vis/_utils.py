"""
Internal visualization utilities.

Not part of the public API.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Any


def to_numpy(x) -> np.ndarray:
    """Convert torch.Tensor or array-like to numpy ndarray."""
    if hasattr(x, 'detach'):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def symmetric_clim(
    data: np.ndarray,
    percentile: float = 99,
) -> Tuple[float, float]:
    """
    Compute symmetric color limits for diverging colormaps.

    Returns (-vmax, vmax) where vmax is the given percentile
    of abs(data).
    """
    vmax = np.percentile(np.abs(data[np.isfinite(data)]), percentile)
    return -vmax, vmax


def add_colorbar(
    im: Any,
    ax: plt.Axes,
    label: Optional[str] = None,
    **kwargs,
) -> None:
    """Add a colorbar with consistent sizing."""
    defaults = dict(fraction=0.046, pad=0.04)
    defaults.update(kwargs)
    cb = plt.colorbar(im, ax=ax, **defaults)
    if label:
        cb.set_label(label)


def ensure_ax(
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8, 6),
) -> Tuple[plt.Figure, plt.Axes]:
    """Return (fig, ax), creating new figure if ax is None."""
    if ax is not None:
        return ax.figure, ax
    return plt.subplots(figsize=figsize)
