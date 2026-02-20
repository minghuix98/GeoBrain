"""
Seismic receiver definition.

Manages receiver positions and types for seismic data recording.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import numpy as np
from typing import List, Optional


class Receiver:
    """
    Seismic receiver manager.

    Handles receiver geometry for seismic data acquisition.
    Supports both 2-D (x, z) and 3-D (x, y, z) geometries.
    Receivers are assumed to be the same for all shots unless
    receiver masks are used in Survey.

    Args:
        nt: Number of time samples to record.
        dt: Time sampling interval (s).

    Attributes:
        num: Number of receivers.
        loc_x: List of receiver x-positions (grid indices).
        loc_z: List of receiver z-positions (grid indices).
        loc_y: List of receiver y-positions (grid indices, 3-D only).
        type: List of receiver types ('pr', 'vx', 'vy', 'vz').

    Example:
        >>> receiver = Receiver(nt=1000, dt=0.001)
        >>> receiver.add_receivers(
        ...     x=np.arange(10, 190, 2),
        ...     z=np.ones(90, dtype=int) * 5,
        ...     rcv_type='pr'
        ... )
        >>> print(receiver)
    """

    def __init__(self, nt: int, dt: float) -> None:
        self.nt = nt
        self.dt = dt

        # Receiver storage
        self.loc_x: List[int] = []
        self.loc_z: List[int] = []
        self.loc_y: List[int] = []
        self.type: List[str] = []
        self.num: int = 0

    def __repr__(self) -> str:
        """String representation."""
        if self.num == 0:
            return "Receiver: empty"

        rcv_x = np.array(self.loc_x)
        rcv_z = np.array(self.loc_z)

        info = (
            f"Receiver:\n"
            f"  Number: {self.num}\n"
            f"  Recording: {self.nt} samples @ {self.dt * 1000:.2f} ms\n"
            f"  X range: {rcv_x.min()} - {rcv_x.max()} (grid)\n"
        )
        if self.loc_y:
            rcv_y = np.array(self.loc_y)
            info += f"  Y range: {rcv_y.min()} - {rcv_y.max()} (grid)\n"
        info += (
            f"  Z range: {rcv_z.min()} - {rcv_z.max()} (grid)\n"
            f"  Types: {list(set(self.type))}"
        )
        return info

    def add_receiver(
        self, x: int, z: int, rcv_type: str, y: Optional[int] = None,
    ) -> None:
        """
        Add a single receiver.

        Args:
            x: Receiver x-position (grid index).
            z: Receiver z-position (grid index).
            rcv_type: Receiver type.
                Options:
                    - 'pr': Pressure (hydrophone)
                    - 'vx': Horizontal particle velocity (x)
                    - 'vy': Horizontal particle velocity (y, 3-D only)
                    - 'vz': Vertical particle velocity
            y: Receiver y-position (grid index, 3-D only).

        Raises:
            ValueError: If rcv_type is not recognized.

        Example:
            >>> receiver.add_receiver(x=50, z=5, rcv_type='pr')
        """
        valid_types = ('pr', 'vx', 'vy', 'vz')
        if rcv_type.lower() not in valid_types:
            raise ValueError(
                f"Receiver type must be one of {valid_types}, got '{rcv_type}'"
            )

        self.loc_x.append(x)
        self.loc_z.append(z)
        if y is not None:
            self.loc_y.append(y)
        self.type.append(rcv_type.lower())
        self.num += 1

    def add_receivers(
        self,
        x: np.ndarray,
        z: np.ndarray,
        rcv_type: str,
        y: Optional[np.ndarray] = None,
    ) -> None:
        """
        Add multiple receivers with the same type.

        Args:
            x: Array of receiver x-positions (grid indices).
            z: Array of receiver z-positions (grid indices).
            rcv_type: Receiver type ('pr', 'vx', 'vy', 'vz').
            y: Array of receiver y-positions (grid indices, 3-D only).

        Raises:
            ValueError: If x and z have different shapes or rcv_type
                is not recognized.

        Example:
            >>> receiver.add_receivers(
            ...     x=np.arange(10, 190, 2),
            ...     z=np.ones(90, dtype=int) * 5,
            ...     rcv_type='pr'
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

        valid_types = ('pr', 'vx', 'vy', 'vz')
        if rcv_type.lower() not in valid_types:
            raise ValueError(
                f"Receiver type must be one of {valid_types}, got '{rcv_type}'"
            )

        n_receivers = len(x.flatten())

        self.loc_x.extend(x.flatten().tolist())
        self.loc_z.extend(z.flatten().tolist())
        if y is not None:
            self.loc_y.extend(y.flatten().tolist())
        self.type.extend([rcv_type.lower()] * n_receivers)
        self.num += n_receivers

    def get_loc(self) -> np.ndarray:
        """
        Get receiver locations.

        Returns:
            Array with shape ``(num, 2)`` for 2-D (columns: x, z)
            or ``(num, 3)`` for 3-D (columns: x, y, z).
        """
        rcv_x = np.array(self.loc_x).reshape(-1, 1)
        rcv_z = np.array(self.loc_z).reshape(-1, 1)
        if self.loc_y:
            rcv_y = np.array(self.loc_y).reshape(-1, 1)
            return np.hstack((rcv_x, rcv_y, rcv_z))
        return np.hstack((rcv_x, rcv_z))

    def get_type(self, unique: bool = False) -> np.ndarray:
        """
        Get receiver types.

        Args:
            unique: If True, return unique types only. Default: False.

        Returns:
            Array of receiver type strings.
        """
        types = np.array(self.type)
        if unique:
            return np.unique(types)
        return types