"""
LAS file I/O for well log data.

Thin wrapper around lasio for reading/writing Log ASCII Standard files.
Returns numpy arrays for easy integration with the rest of GeoBrain.

Reference:
    Canadian Well Logging Society. LAS file format specification.
    https://www.cwls.org/products/

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import numpy as np
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field

try:
    import lasio
except ImportError:
    lasio = None


def _require_lasio():
    if lasio is None:
        raise ImportError(
            "lasio is required for LAS file I/O: pip install lasio"
        )


@dataclass
class LASData:
    """
    Container for LAS file data.

    Attributes:
        data: 2D array of shape (n_samples, n_curves).
        curve_names: List of curve mnemonics (e.g. ['DEPTH', 'GR', 'DT']).
        curve_units: List of curve units (e.g. ['m', 'API', 'us/ft']).
        index: 1D array of depth/time values (first curve).
        well_info: Dict of well header fields {mnemonic: value}.
        params: Dict of parameter section fields {mnemonic: value}.
    """
    data: np.ndarray
    curve_names: List[str]
    curve_units: List[str]
    index: np.ndarray
    well_info: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return self.data.shape[0]

    @property
    def n_curves(self) -> int:
        return self.data.shape[1]

    def __getitem__(self, name: str) -> np.ndarray:
        """Get curve data by mnemonic name."""
        if name not in self.curve_names:
            raise KeyError(
                f"Curve '{name}' not found. "
                f"Available: {self.curve_names}"
            )
        idx = self.curve_names.index(name)
        return self.data[:, idx]

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Return curves as {name: array} dict."""
        return {
            name: self.data[:, i]
            for i, name in enumerate(self.curve_names)
        }

    def __repr__(self) -> str:
        well = self.well_info.get('WELL', 'unknown')
        return (
            f"LASData(well='{well}', n_samples={self.n_samples}, "
            f"curves={self.curve_names})"
        )


def read_las(
    path: str,
    curves: Optional[List[str]] = None,
    null_policy: str = 'none',
) -> LASData:
    """
    Read a LAS file.

    Args:
        path: Path to the LAS file.
        curves: List of curve mnemonics to extract. If None, reads all.
        null_policy: How to handle null values.
            'none': keep as-is (NaN). 'strict': raise on nulls.

    Returns:
        LASData with well log data and metadata.

    Example:
        >>> las = read_las('well.las')
        >>> print(las.curve_names)
        ['DEPTH', 'GR', 'DT', 'RHOB']
        >>> gr = las['GR']
        >>> depth = las.index
    """
    _require_lasio()

    las = lasio.read(path, null_policy=null_policy)

    # Extract well info
    well_info = {}
    for item in las.well:
        well_info[item.mnemonic] = item.value

    # Extract params
    params = {}
    for item in las.params:
        params[item.mnemonic] = item.value

    # All curve names and units
    all_names = [c.mnemonic for c in las.curves]
    all_units = [c.unit for c in las.curves]

    if curves is not None:
        # Filter to requested curves
        indices = []
        for name in curves:
            if name not in all_names:
                raise KeyError(
                    f"Curve '{name}' not found in LAS file. "
                    f"Available: {all_names}"
                )
            indices.append(all_names.index(name))

        data = las.data[:, indices]
        curve_names = [all_names[i] for i in indices]
        curve_units = [all_units[i] for i in indices]
    else:
        data = las.data
        curve_names = all_names
        curve_units = all_units

    index = las.index

    return LASData(
        data=data,
        curve_names=curve_names,
        curve_units=curve_units,
        index=index,
        well_info=well_info,
        params=params,
    )


def write_las(
    path: str,
    data: Union[np.ndarray, Dict[str, np.ndarray]],
    curve_names: Optional[List[str]] = None,
    curve_units: Optional[List[str]] = None,
    index: Optional[np.ndarray] = None,
    index_name: str = 'DEPTH',
    index_unit: str = 'm',
    well_info: Optional[Dict[str, Any]] = None,
    version: float = 2.0,
) -> None:
    """
    Write data to a LAS file.

    Args:
        path: Output file path.
        data: Either a 2D array (n_samples, n_curves) or a dict
            {curve_name: 1D_array}. If dict, curve_names/index are
            inferred from keys (first key used as index).
        curve_names: Curve mnemonics (required if data is ndarray).
        curve_units: Curve units. Defaults to empty strings.
        index: Depth/time index array. If None, uses first curve
            or np.arange.
        index_name: Mnemonic for the index curve. Default: 'DEPTH'.
        index_unit: Unit for the index curve. Default: 'm'.
        well_info: Dict of well header fields to set.
        version: LAS version (1.2 or 2.0). Default: 2.0.

    Example:
        >>> write_las('out.las', data, curve_names=['GR', 'DT'],
        ...           index=depth, well_info={'WELL': 'Test-1'})

        With dict input:
        >>> write_las('out.las', {'DEPTH': depth, 'GR': gr, 'DT': dt})
    """
    _require_lasio()

    las = lasio.LASFile()

    # Handle dict input
    if isinstance(data, dict):
        keys = list(data.keys())
        if index is None:
            index_name = keys[0]
            index = data[keys[0]]
            keys = keys[1:]

        curve_names = curve_names or keys
        curve_units = curve_units or [''] * len(curve_names)
        arrays = [data[k] for k in curve_names]
    else:
        if curve_names is None:
            curve_names = [f'CURVE_{i}' for i in range(data.shape[1])]
        curve_units = curve_units or [''] * len(curve_names)

        if index is None:
            index = np.arange(data.shape[0], dtype=np.float64)

        arrays = [data[:, i] for i in range(data.shape[1])]

    # Set index curve
    las.append_curve(index_name, index, unit=index_unit)

    # Add data curves
    for name, unit, arr in zip(curve_names, curve_units, arrays):
        las.append_curve(name, arr, unit=unit)

    # Set well info
    if well_info:
        for key, val in well_info.items():
            las.well[key] = lasio.HeaderItem(
                mnemonic=key, value=val
            )

    las.write(path, version=version)
