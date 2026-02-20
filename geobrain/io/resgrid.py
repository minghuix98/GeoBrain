"""
Eclipse/OPM reservoir simulation file I/O.

Thin wrapper around resdata for reading Eclipse binary formats:
EGRID (grid), INIT (static properties), UNRST (restart/dynamic),
UNSMRY (summary/production data).

Reference:
    Equinor/resdata: https://github.com/equinor/resdata

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

try:
    from resdata.grid import Grid as _RdGrid
    from resdata.resfile import ResdataFile as _RdFile
    from resdata.summary import Summary as _RdSummary
    _HAS_RESDATA = True
except ImportError:
    _HAS_RESDATA = False


def _require_resdata():
    if not _HAS_RESDATA:
        raise ImportError(
            "resdata is required for Eclipse file I/O: pip install resdata"
        )


# ════════════════════════════════════════════════════════════════
# Data Containers
# ════════════════════════════════════════════════════════════════

@dataclass
class EclGrid:
    """
    Container for Eclipse grid data.

    Attributes:
        nx, ny, nz: Grid dimensions.
        n_active: Number of active cells.
        actnum: Active cell mask, shape (nz*ny*nx,).
        coord: Pillar coordinates (COORD keyword).
        zcorn: Corner depths (ZCORN keyword).
        volumes: Cell volumes, shape (n_active,) or (n_global,).
        xyz: Cell centers, shape (n_active, 3).
        unit_system: Unit system string ('METRIC', 'FIELD', etc.).
    """
    nx: int
    ny: int
    nz: int
    n_active: int
    actnum: np.ndarray
    coord: np.ndarray
    zcorn: np.ndarray
    volumes: np.ndarray
    xyz: np.ndarray
    unit_system: str = 'METRIC'

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.nz, self.ny, self.nx)

    @property
    def n_global(self) -> int:
        return self.nx * self.ny * self.nz

    def __repr__(self) -> str:
        return (
            f"EclGrid(nx={self.nx}, ny={self.ny}, nz={self.nz}, "
            f"active={self.n_active}/{self.n_global}, "
            f"unit={self.unit_system})"
        )


@dataclass
class EclInit:
    """
    Container for Eclipse INIT (static properties) data.

    Attributes:
        properties: Dict of {keyword: array}. Common keywords:
            PORO, PERMX, PERMY, PERMZ, NTG, SATNUM, etc.
        keywords: List of all available keywords.
    """
    properties: Dict[str, np.ndarray]
    keywords: List[str]

    def __getitem__(self, key: str) -> np.ndarray:
        return self.properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.properties

    def __repr__(self) -> str:
        return f"EclInit(keywords={self.keywords})"


@dataclass
class EclRestart:
    """
    Container for Eclipse restart (dynamic) data.

    Attributes:
        report_steps: List of available report step numbers.
        data: Dict of {step: {keyword: array}}.
            Common keywords: PRESSURE, SWAT, SGAS, RS, etc.
        keywords: List of unique keywords across all steps.
    """
    report_steps: List[int]
    data: Dict[int, Dict[str, np.ndarray]]
    keywords: List[str]

    def get(self, keyword: str, step: int) -> np.ndarray:
        """Get a keyword at a specific report step."""
        return self.data[step][keyword]

    def get_all(self, keyword: str) -> np.ndarray:
        """
        Get a keyword across all report steps.

        Returns:
            2D array of shape (n_steps, n_cells).
        """
        return np.stack(
            [self.data[s][keyword] for s in self.report_steps]
        )

    def __repr__(self) -> str:
        return (
            f"EclRestart(steps={self.report_steps}, "
            f"keywords={self.keywords})"
        )


@dataclass
class EclSummary:
    """
    Container for Eclipse summary (production/injection) data.

    Attributes:
        keys: List of summary vector keys (e.g. 'FOPR', 'WBHP:P1').
        dates: Array of datetime64.
        days: Array of simulation days.
        data: Dict of {key: 1D_array}.
        start_date: Simulation start date.
    """
    keys: List[str]
    dates: np.ndarray
    days: np.ndarray
    data: Dict[str, np.ndarray]
    start_date: Any = None

    def __getitem__(self, key: str) -> np.ndarray:
        return self.data[key]

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Return all vectors as {key: array} dict."""
        return dict(self.data)

    def __repr__(self) -> str:
        n = len(self.dates)
        return (
            f"EclSummary(n_timesteps={n}, n_keys={len(self.keys)}, "
            f"start={self.start_date})"
        )


# ════════════════════════════════════════════════════════════════
# Read Functions
# ════════════════════════════════════════════════════════════════

def read_egrid(path: str) -> EclGrid:
    """
    Read an Eclipse EGRID (or GRID) file.

    Args:
        path: Path to .EGRID or .GRID file.

    Returns:
        EclGrid with grid geometry and topology.

    Example:
        >>> grid = read_egrid('MODEL.EGRID')
        >>> print(grid.nx, grid.ny, grid.nz)
        >>> print(grid.volumes.shape)
    """
    _require_resdata()

    g = _RdGrid(path)

    nx, ny, nz = g.nx, g.ny, g.nz

    # ACTNUM
    actnum_kw = g.export_actnum()
    actnum = actnum_kw.numpy_copy()

    # COORD / ZCORN
    coord = np.array(g.export_coord())
    zcorn = np.array(g.export_zcorn())

    # Cell volumes
    vol_kw = g.create_volume_keyword()
    volumes = vol_kw.numpy_copy()

    # Cell centers (active cells only)
    n_active = g.get_num_active()
    xyz = np.empty((n_active, 3), dtype=np.float64)
    for ai in range(n_active):
        gi = g.get_global_index(active_index=ai)
        xyz[ai] = g.get_xyz(global_index=gi)

    unit_system = str(g.unit_system) if g.unit_system else 'METRIC'

    return EclGrid(
        nx=nx, ny=ny, nz=nz,
        n_active=n_active,
        actnum=actnum,
        coord=coord,
        zcorn=zcorn,
        volumes=volumes,
        xyz=xyz,
        unit_system=unit_system,
    )


def read_init(
    path: str,
    keywords: Optional[List[str]] = None,
) -> EclInit:
    """
    Read an Eclipse INIT file (static cell properties).

    Args:
        path: Path to .INIT file.
        keywords: List of keywords to extract. If None, reads
            all available keywords.

    Returns:
        EclInit with property arrays.

    Example:
        >>> init = read_init('MODEL.INIT')
        >>> poro = init['PORO']
        >>> permx = init['PERMX']
    """
    _require_resdata()

    f = _RdFile(path)
    all_kws = sorted(set(f.keys()))

    if keywords is None:
        keywords = all_kws

    properties = {}
    found = []
    for kw in keywords:
        if f.has_kw(kw):
            kw_obj = f.iget_named_kw(kw, 0)
            properties[kw] = kw_obj.numpy_copy()
            found.append(kw)

    return EclInit(properties=properties, keywords=found)


def read_restart(
    path: str,
    keywords: Optional[List[str]] = None,
    steps: Optional[List[int]] = None,
) -> EclRestart:
    """
    Read an Eclipse UNRST (unified restart) file.

    Args:
        path: Path to .UNRST file.
        keywords: Keywords to extract (e.g. ['PRESSURE', 'SWAT']).
            If None, extracts all dynamic keywords.
        steps: Report steps to read. If None, reads all.

    Returns:
        EclRestart with dynamic data at each report step.

    Example:
        >>> rst = read_restart('MODEL.UNRST',
        ...                    keywords=['PRESSURE', 'SWAT'])
        >>> p = rst.get('PRESSURE', step=1)
        >>> p_all = rst.get_all('PRESSURE')  # (n_steps, n_cells)
    """
    _require_resdata()

    f = _RdFile(path)
    all_steps = list(f.report_steps)

    if steps is not None:
        all_steps = [s for s in all_steps if s in steps]

    # Discover keywords from first step
    all_kws = sorted(set(f.keys()))
    # Filter out internal keywords
    _skip = {
        'INTEHEAD', 'DOUBHEAD', 'LOGIHEAD', 'STARTSOL', 'ENDSOL',
        'SEQNUM', 'IGRP', 'SGRP', 'XGRP', 'ZGRP',
    }

    if keywords is None:
        keywords = [k for k in all_kws if k not in _skip]

    data = {}
    found_kws = set()

    for step in all_steps:
        rv = f.restart_view(report_step=step)
        step_data = {}
        for kw in keywords:
            try:
                kw_obj = rv.iget_named_kw(kw, 0)
                step_data[kw] = kw_obj.numpy_copy()
                found_kws.add(kw)
            except (KeyError, IndexError):
                pass
        data[step] = step_data

    return EclRestart(
        report_steps=all_steps,
        data=data,
        keywords=sorted(found_kws),
    )


def read_summary(
    prefix: str,
    keys: Optional[List[str]] = None,
) -> EclSummary:
    """
    Read Eclipse summary files (.SMSPEC + .UNSMRY).

    Args:
        prefix: Case prefix (without extension), e.g. 'MODEL'
            or full path to .SMSPEC/.UNSMRY.
        keys: Summary vector keys to extract (e.g. ['FOPR', 'FWCT']).
            If None, reads all.

    Returns:
        EclSummary with production/injection time series.

    Example:
        >>> smry = read_summary('MODEL')
        >>> fopr = smry['FOPR']
        >>> dates = smry.dates
        >>> print(smry.keys)
    """
    _require_resdata()

    s = _RdSummary(prefix)

    all_keys = sorted(s.keys())
    if keys is not None:
        all_keys = [k for k in keys if s.has_key(k)]

    dates = s.numpy_dates
    days = np.array(s.days)

    data = {}
    for key in all_keys:
        data[key] = s.numpy_vector(key)

    return EclSummary(
        keys=all_keys,
        dates=dates,
        days=days,
        data=data,
        start_date=s.start_date,
    )


def read_case(
    prefix: str,
    restart_keywords: Optional[List[str]] = None,
    init_keywords: Optional[List[str]] = None,
    summary_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Read a complete Eclipse simulation case.

    Convenience function that reads all available files for a case
    (EGRID, INIT, UNRST, SMSPEC+UNSMRY).

    Args:
        prefix: Case prefix (e.g. 'MODEL' or '/path/to/MODEL').
        restart_keywords: Keywords for restart file.
        init_keywords: Keywords for INIT file.
        summary_keys: Keys for summary file.

    Returns:
        Dict with keys 'grid', 'init', 'restart', 'summary'.
        Missing files are set to None.

    Example:
        >>> case = read_case('MODEL')
        >>> grid = case['grid']
        >>> poro = case['init']['PORO']
        >>> pressure = case['restart'].get('PRESSURE', step=1)
        >>> fopr = case['summary']['FOPR']
    """
    import os

    # Strip extension if provided
    base = prefix
    for ext in ['.EGRID', '.GRID', '.INIT', '.UNRST', '.SMSPEC',
                '.UNSMRY', '.DATA']:
        if base.upper().endswith(ext):
            base = base[:-len(ext)]
            break

    result = {
        'grid': None,
        'init': None,
        'restart': None,
        'summary': None,
    }

    # Grid
    for ext in ['.EGRID', '.GRID']:
        path = base + ext
        if os.path.exists(path):
            result['grid'] = read_egrid(path)
            break

    # INIT
    path = base + '.INIT'
    if os.path.exists(path):
        result['init'] = read_init(path, keywords=init_keywords)

    # Restart
    path = base + '.UNRST'
    if os.path.exists(path):
        result['restart'] = read_restart(
            path, keywords=restart_keywords,
        )

    # Summary
    smspec = base + '.SMSPEC'
    if os.path.exists(smspec):
        result['summary'] = read_summary(base, keys=summary_keys)

    return result
