"""
Wave simulation result container.

Provides a structured dataclass for wave propagation outputs,
replacing raw dictionaries with typed, documented fields.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
from torch import Tensor
from dataclasses import dataclass, fields
from typing import Optional


@dataclass
class WaveResult:
    """
    Structured result from wave propagation.

    Holds seismograms recorded at receiver locations and accumulated
    forward wavefields (energy maps) used for imaging / gradient
    computation.

    Supports dict-like access for backward compatibility:
        ``result['p']`` is equivalent to ``result.p``.

    Args:
        p: Pressure seismograms, shape ``(n_shots, nt, n_receivers)``.
        vx: Horizontal (x) velocity seismograms.
        vz: Vertical (z) velocity seismograms.
        vy: Transverse (y) velocity seismograms (3-D only).
        forward_wavefield_p: Accumulated pressure energy, shape
            ``(nz, nx)`` (2-D) or ``(nz, ny, nx)`` (3-D).
        forward_wavefield_vx: Accumulated vx energy.
        forward_wavefield_vz: Accumulated vz energy.
        forward_wavefield_vy: Accumulated vy energy (3-D only).

    Example:
        >>> result = propagator.forward()
        >>> result.p.shape       # (n_shots, nt, n_receivers)
        >>> result['p'].shape    # dict-style also works
    """

    # Seismograms (n_shots, nt, n_receivers)
    p: Tensor
    vx: Tensor
    vz: Tensor

    # Forward wavefields (nz, nx) or (nz, ny, nx)
    forward_wavefield_p: Tensor
    forward_wavefield_vx: Tensor
    forward_wavefield_vz: Tensor

    # 3-D only (None for 2-D)
    vy: Optional[Tensor] = None
    forward_wavefield_vy: Optional[Tensor] = None

    # Wavefield snapshots (n_snaps, nz, nx), None when not requested
    snapshots: Optional[Tensor] = None

    # ---- dict-like access for backward compatibility ----

    def __getitem__(self, key: str) -> Tensor:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) and getattr(self, key) is not None

    def keys(self):
        """Return field names that are not None."""
        return [f.name for f in fields(self) if getattr(self, f.name) is not None]

    def values(self):
        """Return non-None field values."""
        return [getattr(self, k) for k in self.keys()]

    def items(self):
        """Return (name, value) pairs for non-None fields."""
        return [(k, getattr(self, k)) for k in self.keys()]

    @classmethod
    def from_dict(cls, d: dict) -> "WaveResult":
        """Create a WaveResult from a kernel output dictionary."""
        return cls(
            p=d['p'],
            vx=d['vx'],
            vz=d['vz'],
            forward_wavefield_p=d['forward_wavefield_p'],
            forward_wavefield_vx=d['forward_wavefield_vx'],
            forward_wavefield_vz=d['forward_wavefield_vz'],
            vy=d.get('vy'),
            forward_wavefield_vy=d.get('forward_wavefield_vy'),
            snapshots=d.get('snapshots'),
        )

    @property
    def ndim(self) -> int:
        """Return 3 if vy is present, else 2."""
        return 3 if self.vy is not None else 2

    @property
    def n_shots(self) -> int:
        """Number of shots."""
        return self.p.shape[0]

    @property
    def nt(self) -> int:
        """Number of time steps."""
        return self.p.shape[1]

    @property
    def n_receivers(self) -> int:
        """Number of receivers."""
        return self.p.shape[2]
