"""
SEG-Y file I/O for seismic data.

Thin wrapper around segyio for reading/writing SEG-Y files.
Returns numpy arrays for easy integration with the rest of GeoBrain.

Reference:
    SEG Technical Standards Committee (2017).
    SEG-Y revision 2 Data Exchange Format.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass, field

try:
    import segyio
except ImportError:
    segyio = None


def _require_segyio():
    if segyio is None:
        raise ImportError(
            "segyio is required for SEG-Y file I/O: pip install segyio"
        )


@dataclass
class SEGYData:
    """
    Container for SEG-Y file data.

    Attributes:
        traces: 2D array of shape (n_traces, n_samples).
        sample_interval: Sample interval in microseconds.
        samples: 1D array of sample positions (time/depth).
        n_traces: Number of traces.
        n_samples: Number of samples per trace.
        ilines: Inline numbers (None if unstructured).
        xlines: Crossline numbers (None if unstructured).
        text_header: Textual file header string.
    """
    traces: np.ndarray
    sample_interval: float
    samples: np.ndarray
    n_traces: int
    n_samples: int
    ilines: Optional[np.ndarray] = None
    xlines: Optional[np.ndarray] = None
    text_header: str = ''

    @property
    def is_structured(self) -> bool:
        return self.ilines is not None and self.xlines is not None

    def to_volume(self) -> np.ndarray:
        """
        Reshape traces to 3D volume (n_ilines, n_xlines, n_samples).

        Only works for structured (inline/crossline) data.

        Returns:
            3D numpy array.

        Raises:
            ValueError: If data is unstructured.
        """
        if not self.is_structured:
            raise ValueError(
                "Cannot reshape to volume: data is unstructured. "
                "Use read_segy(path, ignore_geometry=False) on a "
                "structured SEG-Y file."
            )
        ni = len(self.ilines)
        nx = len(self.xlines)
        return self.traces.reshape(ni, nx, self.n_samples)

    def __repr__(self) -> str:
        if self.is_structured:
            return (
                f"SEGYData(ilines={len(self.ilines)}, "
                f"xlines={len(self.xlines)}, "
                f"n_samples={self.n_samples}, "
                f"dt={self.sample_interval}us)"
            )
        return (
            f"SEGYData(n_traces={self.n_traces}, "
            f"n_samples={self.n_samples}, "
            f"dt={self.sample_interval}us)"
        )


def read_segy(
    path: str,
    ignore_geometry: bool = False,
) -> SEGYData:
    """
    Read a SEG-Y file.

    Args:
        path: Path to the SEG-Y file.
        ignore_geometry: If True, skip inline/crossline geometry
            inference. Use this for unstructured or shot-gather data.

    Returns:
        SEGYData with traces and metadata.

    Example:
        Unstructured (shot gathers, generic):
        >>> segy = read_segy('shots.sgy', ignore_geometry=True)
        >>> print(segy.traces.shape)  # (n_traces, n_samples)

        Structured (3D cube):
        >>> segy = read_segy('cube.sgy')
        >>> volume = segy.to_volume()  # (n_il, n_xl, n_samples)
    """
    _require_segyio()

    with segyio.open(path, 'r', ignore_geometry=ignore_geometry,
                     strict=False) as f:
        traces = segyio.tools.collect(f.trace[:])
        dt = segyio.tools.dt(f)
        samples = f.samples.copy()
        n_traces = f.tracecount
        n_samples = len(f.samples)

        ilines = None
        xlines = None
        if not ignore_geometry and not f.unstructured:
            ilines = f.ilines.copy()
            xlines = f.xlines.copy()

        text_header = ''
        try:
            raw = f.text[0]
            text_header = raw.decode('utf-8', errors='replace') \
                if isinstance(raw, bytes) else str(raw)
        except Exception:
            pass

    return SEGYData(
        traces=traces,
        sample_interval=dt,
        samples=samples,
        n_traces=n_traces,
        n_samples=n_samples,
        ilines=ilines,
        xlines=xlines,
        text_header=text_header,
    )


def read_segy_volume(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Read a SEG-Y file as a 2D model (nz, nx) from trace data.

    Convenience function for loading velocity/property models stored
    as SEG-Y. Treats each trace as a vertical column and stacks them
    horizontally.

    Args:
        path: Path to the SEG-Y file.

    Returns:
        Tuple of (volume, info):
            - volume: 2D array (n_samples, n_traces) = (nz, nx).
            - info: Dict with 'n_traces', 'n_samples', 'dt'.

    Example:
        >>> vp, info = read_segy_volume('marmousi_vp.sgy')
        >>> print(vp.shape)  # (nz, nx)
    """
    segy = read_segy(path, ignore_geometry=True)

    # Transpose: (n_traces, n_samples) -> (n_samples, n_traces) = (nz, nx)
    volume = segy.traces.T

    info = {
        'n_traces': segy.n_traces,
        'n_samples': segy.n_samples,
        'dt': segy.sample_interval,
        'samples': segy.samples,
    }

    return volume, info


def write_segy(
    path: str,
    data: np.ndarray,
    dt: float = 1000.0,
    format: int = 5,
    ilines: Optional[np.ndarray] = None,
    xlines: Optional[np.ndarray] = None,
    text_header: Optional[str] = None,
) -> None:
    """
    Write data to a SEG-Y file.

    Args:
        path: Output file path.
        data: 2D array (n_traces, n_samples) or 3D array
            (n_ilines, n_xlines, n_samples).
        dt: Sample interval in microseconds. Default: 1000.
        format: SEG-Y sample format. Default: 5 (IEEE float32).
            1=IBM float, 5=IEEE float, 2=int32, 3=int16.
        ilines: Inline numbers for 3D data. If None and data is 3D,
            uses np.arange(n_ilines).
        xlines: Crossline numbers for 3D data. If None and data is 3D,
            uses np.arange(n_xlines).
        text_header: Optional textual header string.

    Example:
        Unstructured:
        >>> write_segy('out.sgy', traces)  # (n_traces, n_samples)

        Structured 3D:
        >>> write_segy('cube.sgy', volume)  # (ni, nx, ns)
    """
    _require_segyio()

    data = np.asarray(data, dtype=np.float32)

    spec = segyio.spec()
    spec.format = format

    if data.ndim == 3:
        ni, nx, ns = data.shape
        spec.ilines = ilines if ilines is not None else np.arange(ni)
        spec.xlines = xlines if xlines is not None else np.arange(nx)
        spec.samples = np.arange(ns) * dt
        spec.sorting = 2  # inline sorting

        with segyio.create(path, spec) as f:
            trace_idx = 0
            for i in range(ni):
                for x in range(nx):
                    f.trace[trace_idx] = data[i, x, :]
                    f.header[trace_idx].update({
                        segyio.TraceField.INLINE_3D: int(spec.ilines[i]),
                        segyio.TraceField.CROSSLINE_3D: int(spec.xlines[x]),
                    })
                    trace_idx += 1

            f.bin.update({
                segyio.BinField.Interval: int(dt),
            })

            if text_header:
                f.text[0] = _format_text_header(text_header)

    elif data.ndim == 2:
        n_traces, ns = data.shape
        spec.tracecount = n_traces
        spec.samples = np.arange(ns) * dt

        with segyio.create(path, spec) as f:
            for i in range(n_traces):
                f.trace[i] = data[i, :]

            f.bin.update({
                segyio.BinField.Interval: int(dt),
            })

            if text_header:
                f.text[0] = _format_text_header(text_header)
    else:
        raise ValueError(
            f"Data must be 2D (n_traces, n_samples) or "
            f"3D (n_il, n_xl, n_samples), got {data.ndim}D"
        )


def _format_text_header(text: str) -> str:
    """Format a text string as a SEG-Y textual header."""
    lines = text.split('\n')
    header_dict = {}
    for i, line in enumerate(lines[:40], start=1):
        header_dict[i] = line
    return segyio.tools.create_text_header(header_dict)
