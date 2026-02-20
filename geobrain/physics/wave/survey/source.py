"""
Seismic source definition.

Manages source positions, wavelets, and moment tensors for
seismic wave simulation.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import numpy as np
from typing import List, Optional, Union


class Source:
    """
    Seismic source manager.

    Handles source geometry, wavelets, and moment tensors for
    single or multiple shot simulations.  Supports both 2-D
    (x, z) and 3-D (x, y, z) acquisition geometries.

    Args:
        nt: Number of time samples.
        dt: Time sampling interval (s).
        f0: Dominant frequency (Hz), used for wavelet generation.

    Attributes:
        num: Number of sources.
        loc_x: List of source x-positions (grid indices).
        loc_z: List of source z-positions (grid indices).
        loc_y: List of source y-positions (grid indices, 3-D only).
        wavelet: List of source wavelets.
        moment_tensor: List of 3x3 moment tensors.
        t: Time array with shape (nt,).

    Example:
        Single source:
        >>> source = Source(nt=1000, dt=0.001, f0=15.0)
        >>> source.add_source(x=100, z=5, wavelet=ricker)
        >>> print(source)

        Multiple sources:
        >>> source.add_sources(
        ...     x=np.arange(10, 190, 20),
        ...     z=np.ones(9, dtype=int) * 5,
        ...     wavelet=ricker
        ... )
    """

    def __init__(self, nt: int, dt: float, f0: float) -> None:
        self.nt = nt
        self.dt = dt
        self.f0 = f0
        self.t = np.arange(nt) * dt

        # Source storage
        self.loc_x: List[int] = []
        self.loc_z: List[int] = []
        self.loc_y: List[int] = []
        self.type: List[str] = []
        self.wavelet: List[np.ndarray] = []
        self.moment_tensor: List[np.ndarray] = []
        self.num: int = 0

    def __repr__(self) -> str:
        """String representation."""
        if self.num == 0:
            return "Source: empty"

        src_x = np.array(self.loc_x)
        src_z = np.array(self.loc_z)

        info = (
            f"Source:\n"
            f"  Number: {self.num}\n"
            f"  Time: {self.nt} samples @ {self.dt * 1000:.2f} ms\n"
            f"  X range: {src_x.min()} - {src_x.max()} (grid)\n"
        )
        if self.loc_y:
            src_y = np.array(self.loc_y)
            info += f"  Y range: {src_y.min()} - {src_y.max()} (grid)\n"
        info += (
            f"  Z range: {src_z.min()} - {src_z.max()} (grid)\n"
            f"  Types: {list(set(self.type))}"
        )
        return info

    def add_source(
        self,
        x: int,
        z: int,
        wavelet: np.ndarray,
        y: Optional[int] = None,
        source_type: str = 'mt',
        moment_tensor: Optional[np.ndarray] = None,
    ) -> None:
        """
        Add a single source.

        Args:
            x: Source x-position (grid index).
            z: Source z-position (grid index).
            wavelet: Source wavelet array with shape (nt,).
            y: Source y-position (grid index, 3-D only).
            source_type: Source type identifier. Default: 'mt'.
            moment_tensor: 3x3 moment tensor. Default: identity matrix
                (isotropic explosion).

        Raises:
            ValueError: If wavelet length doesn't match nt or moment
                tensor shape is not (3, 3).

        Example:
            >>> source.add_source(x=100, z=5, wavelet=ricker)
        """
        if wavelet.shape[0] != self.nt:
            raise ValueError(
                f"Wavelet length {wavelet.shape[0]} != nt {self.nt}"
            )

        if moment_tensor is None:
            moment_tensor = np.eye(3)

        if moment_tensor.shape != (3, 3):
            raise ValueError(
                f"Moment tensor must be 3x3, got {moment_tensor.shape}"
            )

        self.loc_x.append(x)
        self.loc_z.append(z)
        if y is not None:
            self.loc_y.append(y)
        self.type.append(source_type)
        self.wavelet.append(wavelet.copy())
        self.moment_tensor.append(moment_tensor.copy())
        self.num += 1

    def add_sources(
        self,
        x: np.ndarray,
        z: np.ndarray,
        wavelet: np.ndarray,
        y: Optional[np.ndarray] = None,
        source_type: str = 'mt',
        moment_tensor: Optional[np.ndarray] = None,
    ) -> None:
        """
        Add multiple sources with the same wavelet.

        Args:
            x: Array of source x-positions (grid indices).
            z: Array of source z-positions (grid indices).
            wavelet: Source wavelet with shape (nt,), shared by all sources.
            y: Array of source y-positions (grid indices, 3-D only).
            source_type: Source type identifier. Default: 'mt'.
            moment_tensor: 3x3 moment tensor, shared by all sources.
                Default: identity matrix.

        Raises:
            ValueError: If x and z have different shapes or wavelet
                length doesn't match nt.

        Example:
            >>> source.add_sources(
            ...     x=np.arange(10, 190, 20),
            ...     z=np.ones(9, dtype=int) * 5,
            ...     wavelet=ricker
            ... )
        """
        if x.shape != z.shape:
            raise ValueError(
                f"x and z must have same shape: {x.shape} != {z.shape}"
            )
        if y is not None and y.shape != x.shape:
            raise ValueError(
                f"y must have same shape as x: {y.shape} != {x.shape}"
            )

        if wavelet.shape[0] != self.nt:
            raise ValueError(
                f"Wavelet length {wavelet.shape[0]} != nt {self.nt}"
            )

        if moment_tensor is None:
            moment_tensor = np.eye(3)

        n_sources = len(x.flatten())

        self.loc_x.extend(x.flatten().tolist())
        self.loc_z.extend(z.flatten().tolist())
        if y is not None:
            self.loc_y.extend(y.flatten().tolist())
        self.type.extend([source_type] * n_sources)
        self.wavelet.extend([wavelet.copy() for _ in range(n_sources)])
        self.moment_tensor.extend([moment_tensor.copy() for _ in range(n_sources)])
        self.num += n_sources

    def get_loc(self) -> np.ndarray:
        """
        Get source locations.

        Returns:
            Array with shape ``(num, 2)`` for 2-D (columns: x, z)
            or ``(num, 3)`` for 3-D (columns: x, y, z).
        """
        src_x = np.array(self.loc_x).reshape(-1, 1)
        src_z = np.array(self.loc_z).reshape(-1, 1)
        if self.loc_y:
            src_y = np.array(self.loc_y).reshape(-1, 1)
            return np.hstack((src_x, src_y, src_z))
        return np.hstack((src_x, src_z))

    def get_wavelet(self) -> np.ndarray:
        """
        Get source wavelets.

        Returns:
            Array with shape (num, nt).
        """
        return np.array(self.wavelet)

    def get_moment_tensor(self) -> np.ndarray:
        """
        Get moment tensors.

        Returns:
            Array with shape (num, 3, 3).
        """
        return np.array(self.moment_tensor)

    def get_type(self, unique: bool = False) -> np.ndarray:
        """
        Get source types.

        Args:
            unique: If True, return unique types only. Default: False.

        Returns:
            Array of source type strings.
        """
        types = np.array(self.type)
        if unique:
            return np.unique(types)
        return types