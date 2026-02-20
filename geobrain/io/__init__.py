"""
I/O utilities for geophysical data formats.

Supported formats:
    - LAS: Log ASCII Standard (well log data) via lasio
    - SEG-Y: SEG-Y rev1/rev2 (seismic data) via segyio
    - Eclipse: Reservoir simulation files via resdata
      (EGRID, INIT, UNRST, SMSPEC/UNSMRY)

Example:
    >>> from geobrain.io import read_las, read_segy, read_segy_volume
    >>>
    >>> # Well log
    >>> las = read_las('well.las')
    >>> gr = las['GR']
    >>>
    >>> # Seismic traces
    >>> segy = read_segy('shots.sgy', ignore_geometry=True)
    >>> traces = segy.traces
    >>>
    >>> # Velocity model from SEG-Y
    >>> vp, info = read_segy_volume('marmousi_vp.sgy')
    >>>
    >>> # Eclipse simulation case
    >>> from geobrain.io import read_egrid, read_restart, read_summary
    >>> grid = read_egrid('MODEL.EGRID')
    >>> rst = read_restart('MODEL.UNRST', keywords=['PRESSURE', 'SWAT'])
    >>> smry = read_summary('MODEL')

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

from .las import read_las, write_las, LASData
from .segy import read_segy, write_segy, read_segy_volume, SEGYData
from .resgrid import (
    read_egrid, read_init, read_restart, read_summary, read_case,
    EclGrid, EclInit, EclRestart, EclSummary,
)

__all__ = [
    # LAS
    'read_las',
    'write_las',
    'LASData',
    # SEG-Y
    'read_segy',
    'write_segy',
    'read_segy_volume',
    'SEGYData',
    # Eclipse
    'read_egrid',
    'read_init',
    'read_restart',
    'read_summary',
    'read_case',
    'EclGrid',
    'EclInit',
    'EclRestart',
    'EclSummary',
]
