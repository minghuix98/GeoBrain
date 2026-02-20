"""
Seismic survey definition.

Combines source and receiver geometry into a complete acquisition setup.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import numpy as np
from typing import Optional

from .source import Source
from .receiver import Receiver


class Survey:
    """
    Seismic survey manager.
    
    Combines source and receiver geometry for acquisition simulation.
    Optionally supports receiver masks for shot-dependent receiver
    selection (e.g., marine streamer acquisition).
    
    Args:
        source: Source object with shot positions and wavelets.
        receiver: Receiver object with receiver positions.
        receiver_masks: Optional boolean mask array with shape
            (n_shots, n_receivers) indicating active receivers per shot.
            Default: None (all receivers active for all shots).

    Attributes:
        n_shots: Number of shots (same as source.num).
        n_receivers: Number of receivers (same as receiver.num).
        nt: Number of time samples.
        dt: Time sampling interval (s).
        f0: Dominant frequency (Hz).

    Example:
        Basic usage:
        >>> source = Source(nt=1000, dt=0.001, f0=15.0)
        >>> source.add_sources(src_x, src_z, wavelet)
        >>>
        >>> receiver = Receiver(nt=1000, dt=0.001)
        >>> receiver.add_receivers(rcv_x, rcv_z, 'pr')
        >>>
        >>> survey = Survey(source, receiver)
        >>> print(survey)

        With receiver masks (marine streamer):
        >>> masks = np.ones((n_shots, n_receivers), dtype=bool)
        >>> # Mask out receivers behind the source
        >>> for i in range(n_shots):
        ...     masks[i, :src_idx[i]] = False
        >>> survey = Survey(source, receiver, receiver_masks=masks)
    """

    def __init__(
        self,
        source: Source,
        receiver: Receiver,
        receiver_masks: Optional[np.ndarray] = None,
    ) -> None:
        self.source = source
        self.receiver = receiver
        self.receiver_masks = None

        if receiver_masks is not None:
            self.set_receiver_masks(receiver_masks)

    def __repr__(self) -> str:
        """String representation."""
        info = (
            f"Survey:\n"
            f"  Shots: {self.n_shots}\n"
            f"  Receivers: {self.n_receivers}\n"
            f"  Time: {self.nt} samples @ {self.dt * 1000:.2f} ms\n"
            f"  Receiver masks: {'Yes' if self.receiver_masks is not None else 'No'}"
        )
        return info

    def set_receiver_masks(self, receiver_masks: np.ndarray) -> None:
        """
        Set receiver masks for shot-dependent receiver selection.

        Args:
            receiver_masks: Boolean array with shape (n_shots, n_receivers).
                True indicates active receiver.

        Raises:
            ValueError: If mask shape doesn't match survey geometry.

        Example:
            >>> masks = np.ones((n_shots, n_receivers), dtype=bool)
            >>> masks[:, :10] = False  # Disable first 10 receivers
            >>> survey.set_receiver_masks(masks)
        """
        expected_shape = (self.n_shots, self.n_receivers)

        if receiver_masks.shape != expected_shape:
            raise ValueError(
                f"Receiver mask shape {receiver_masks.shape} doesn't match "
                f"survey geometry {expected_shape}"
            )

        self.receiver_masks = receiver_masks.astype(bool)

    @property
    def n_shots(self) -> int:
        """Number of shots."""
        return self.source.num

    @property
    def n_receivers(self) -> int:
        """Number of receivers."""
        return self.receiver.num

    @property
    def nt(self) -> int:
        """Number of time samples."""
        return self.source.nt

    @property
    def dt(self) -> float:
        """Time sampling interval (s)."""
        return self.source.dt

    @property
    def f0(self) -> float:
        """Dominant frequency (Hz)."""
        return self.source.f0