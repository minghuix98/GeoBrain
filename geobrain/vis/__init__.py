"""
GeoBrain Visualization Module.

Function-based visualization for geophysical data.
All plot functions return (fig, ax) or (fig, axes) for
further customization with matplotlib.

Submodules:
    - seismic: Seismic sections, wiggle traces, shot gathers
    - field: 2D/3D property fields (velocity, porosity, etc.)
    - well: Multi-track well log display
    - production: Production rates, water cut, well summary
    - colormaps: Geophysics-specific colormaps

Example:
    >>> from geobrain.vis import plot_section, plot_field, plot_logs
    >>>
    >>> fig, ax = plot_section(seismic_data, dt=0.001, dx=12.5)
    >>> fig, ax = plot_field(vp_model, dx=10, dz=10, label='Vp (m/s)')
    >>> fig, axes = plot_logs(las_data, names=['GR', 'DT', 'RHOB'])

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

from .seismic import plot_section, plot_wiggle, plot_gather
from .field import plot_field, plot_slices, plot_comparison
from .well import plot_logs
from .production import plot_production, plot_watercut, plot_well_summary
from .colormaps import register_colormaps, get_colormap, GEO_COLORMAPS

# Register geophysics colormaps on import
register_colormaps()

__all__ = [
    # Seismic
    'plot_section',
    'plot_wiggle',
    'plot_gather',
    # Field
    'plot_field',
    'plot_slices',
    'plot_comparison',
    # Well
    'plot_logs',
    # Production / flow simulation
    'plot_production',
    'plot_watercut',
    'plot_well_summary',
    # Colormaps
    'register_colormaps',
    'get_colormap',
    'GEO_COLORMAPS',
]
