"""
Geophysics-specific colormaps.

Registers a small set of domain-specific colormaps with matplotlib.
Call register_colormaps() once at import time; after that use them
by name in any matplotlib call, e.g. cmap='geo_velocity'.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.pyplot as plt

# ── Velocity colormap (blue → cyan → green → yellow → red) ──────────
_velocity_colors = [
    (0.0, 0.0, 0.5),   # dark blue
    (0.0, 0.0, 1.0),   # blue
    (0.0, 0.7, 1.0),   # cyan
    (0.0, 0.8, 0.0),   # green
    (1.0, 1.0, 0.0),   # yellow
    (1.0, 0.5, 0.0),   # orange
    (0.8, 0.0, 0.0),   # red
]

# ── Impedance colormap (warm tones) ─────────────────────────────────
_impedance_colors = [
    (0.1, 0.1, 0.4),
    (0.2, 0.3, 0.7),
    (0.3, 0.6, 0.5),
    (0.9, 0.9, 0.3),
    (0.9, 0.5, 0.1),
    (0.7, 0.1, 0.1),
]

# ── Facies discrete colormap ────────────────────────────────────────
_facies_colors = [
    '#F5DEB3',  # sand (wheat)
    '#4682B4',  # shale (steel blue)
    '#228B22',  # carbonate (forest green)
    '#DAA520',  # silt (goldenrod)
    '#8B4513',  # clay (saddle brown)
    '#708090',  # coal (slate gray)
    '#B22222',  # salt (firebrick)
    '#9370DB',  # anhydrite (medium purple)
]

# ── Porosity colormap (white → blue) ────────────────────────────────
_porosity_colors = [
    (0.95, 0.95, 0.95),
    (0.7, 0.85, 0.95),
    (0.3, 0.6, 0.9),
    (0.1, 0.3, 0.7),
    (0.05, 0.1, 0.4),
]

# ── Registry ────────────────────────────────────────────────────────
GEO_COLORMAPS = {
    'geo_velocity': LinearSegmentedColormap.from_list(
        'geo_velocity', _velocity_colors, N=256,
    ),
    'geo_impedance': LinearSegmentedColormap.from_list(
        'geo_impedance', _impedance_colors, N=256,
    ),
    'geo_porosity': LinearSegmentedColormap.from_list(
        'geo_porosity', _porosity_colors, N=256,
    ),
    'geo_facies': ListedColormap(_facies_colors, name='geo_facies'),
}

_registered = False


def register_colormaps() -> None:
    """Register geophysics colormaps with matplotlib (idempotent)."""
    global _registered
    if _registered:
        return
    for name, cmap in GEO_COLORMAPS.items():
        try:
            plt.colormaps.register(cmap, name=name)
        except (ValueError, AttributeError):
            pass
    _registered = True


def get_colormap(name: str):
    """
    Get a colormap by name.

    Checks geophysics colormaps first, then falls back to matplotlib.
    """
    if name in GEO_COLORMAPS:
        return GEO_COLORMAPS[name]
    return plt.get_cmap(name)
