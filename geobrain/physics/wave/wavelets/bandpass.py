"""
Bandpass wavelets: Ormsby and Klauder.

These wavelets provide more control over frequency content compared
to simple wavelets like Ricker.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
import numpy as np
from torch import Tensor
from typing import Tuple, Optional

from .base import WaveletGenerator


class OrmsbyWavelet(WaveletGenerator):
    """
    Ormsby wavelet with trapezoidal frequency spectrum.
    
    Defined by four corner frequencies that control the
    frequency response:

        f1 < f2: Low-frequency taper (ramp up)
        f2 to f3: Passband (flat response)
        f3 < f4: High-frequency taper (ramp down)

    Args:
        f1: Low cut frequency (Hz).
        f2: Low pass frequency (Hz).
        f3: High pass frequency (Hz).
        f4: High cut frequency (Hz).

    Example:
        >>> ormsby = OrmsbyWavelet(f1=5, f2=10, f3=40, f4=60)
        >>> wavelet, t = ormsby(f0=0, dt=0.001)  # f0 is ignored

    Note:
        The f0 parameter in forward() is ignored; frequency content
        is controlled entirely by (f1, f2, f3, f4).
    """

    def __init__(
        self,
        f1: float,
        f2: float,
        f3: float,
        f4: float,
    ):
        super().__init__()

        if not (f1 < f2 < f3 < f4):
            raise ValueError(
                f"Frequencies must satisfy f1 < f2 < f3 < f4, "
                f"got ({f1}, {f2}, {f3}, {f4})"
            )

        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.f4 = f4
        self._name = 'ormsby'

    def forward(
        self,
        f0: float,
        dt: float,
        device: Optional[str] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Generate Ormsby wavelet.

        Args:
            f0: Ignored (frequency controlled by corner frequencies).
            dt: Time sampling interval (s).
            device: Computation device. Default: 'cpu'.

        Returns:
            Tuple of (wavelet, time_axis), each with shape (nw,).
        """
        if isinstance(dt, Tensor):
            dt = dt.item()

        device = device or 'cpu'

        # Length based on lowest frequency
        nw = int(2.0 / self.f1 / dt)
        nw = 2 * (nw // 2) + 1
        nc = nw // 2

        t = torch.arange(nw, device=device, dtype=torch.float32)
        t = (t - nc) * dt

        # Avoid division by zero at t=0
        t_safe = torch.where(
            torch.abs(t) < 1e-10,
            torch.tensor(1e-10, device=device),
            t
        )

        # Ormsby formula using sinc-squared functions
        def sinc_sq(f: float) -> Tensor:
            arg = np.pi * f * t_safe
            return (torch.sin(arg) / arg) ** 2 * f ** 2

        wavelet = (
            (sinc_sq(self.f4) - sinc_sq(self.f3)) / (self.f4 - self.f3) -
            (sinc_sq(self.f2) - sinc_sq(self.f1)) / (self.f2 - self.f1)
        )

        # Handle t=0 case
        t_zero = torch.abs(t) < 1e-10
        center_value = (self.f4 + self.f3) / 2 - (self.f2 + self.f1) / 2
        wavelet = torch.where(
            t_zero,
            torch.tensor(center_value, device=device),
            wavelet
        )

        # Normalize to unit amplitude
        wavelet = wavelet / torch.max(torch.abs(wavelet))

        return wavelet, t

    def __repr__(self) -> str:
        return (
            f"OrmsbyWavelet(f1={self.f1}, f2={self.f2}, "
            f"f3={self.f3}, f4={self.f4})"
        )


class KlauderWavelet(WaveletGenerator):
    """
    Klauder wavelet (autocorrelation of linear frequency sweep).

    Used in vibroseis processing where the source is a controlled
    frequency sweep (chirp). The Klauder wavelet represents the
    theoretical autocorrelation of the sweep.

    Args:
        f_low: Start frequency of sweep (Hz).
        f_high: End frequency of sweep (Hz).
        T: Sweep duration (s).

    Example:
        >>> klauder = KlauderWavelet(f_low=10, f_high=80, T=8.0)
        >>> wavelet, t = klauder(f0=0, dt=0.002)

    Note:
        The wavelet length is determined by the sweep duration T.
        The f0 parameter in forward() is ignored.
    """

    def __init__(
        self,
        f_low: float,
        f_high: float,
        T: float,
    ):
        super().__init__()

        if f_low >= f_high:
            raise ValueError(
                f"f_low must be less than f_high: {f_low} >= {f_high}"
            )
        if T <= 0:
            raise ValueError(f"Sweep duration must be positive: {T}")

        self.f_low = f_low
        self.f_high = f_high
        self.T = T
        self._name = 'klauder'

    def forward(
        self,
        f0: float,
        dt: float,
        device: Optional[str] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Generate Klauder wavelet.

        Args:
            f0: Ignored (frequency controlled by sweep parameters).
            dt: Time sampling interval (s).
            device: Computation device. Default: 'cpu'.

        Returns:
            Tuple of (wavelet, time_axis), each with shape (nw,).
        """
        if isinstance(dt, Tensor):
            dt = dt.item()

        device = device or 'cpu'

        bandwidth = self.f_high - self.f_low
        f_center = (self.f_low + self.f_high) / 2

        # Length based on sweep duration
        nw = int(self.T / dt)
        nw = 2 * (nw // 2) + 1
        nc = nw // 2

        t = torch.arange(nw, device=device, dtype=torch.float32)
        t = (t - nc) * dt

        # Avoid division by zero
        t_safe = torch.where(
            torch.abs(t) < 1e-10,
            torch.tensor(1e-10, device=device),
            t
        )

        # Klauder formula: sinc(πBt) × cos(2πf_c t)
        arg1 = np.pi * bandwidth * t_safe
        arg2 = 2 * np.pi * f_center * t

        wavelet = torch.sin(arg1) / arg1 * torch.cos(arg2)

        # Handle t=0
        t_zero = torch.abs(t) < 1e-10
        wavelet = torch.where(
            t_zero,
            torch.tensor(1.0, device=device),
            wavelet
        )

        return wavelet, t

    def __repr__(self) -> str:
        return (
            f"KlauderWavelet(f_low={self.f_low}, f_high={self.f_high}, "
            f"T={self.T})"
        )