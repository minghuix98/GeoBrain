"""
Ricker (Mexican hat) wavelet generator.

The Ricker wavelet is the most commonly used source wavelet in
seismic modeling due to its zero DC component and smooth spectrum.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
import numpy as np
from torch import Tensor
from typing import Tuple, Optional

from .base import WaveletGenerator


class RickerWavelet(WaveletGenerator):
    """
    Ricker (Mexican hat) wavelet generator.
    
    Computes the Ricker wavelet defined as:

        w(t) = (1 - 2π²f₀²t²) × exp(-π²f₀²t²)

    The Ricker wavelet is the negative normalized second derivative
    of a Gaussian function. It has zero mean (no DC component) and
    a well-defined peak frequency.

    Args:
        length_factor: Controls wavelet length as multiples of
            dominant period. Higher values produce longer wavelets.
            Default: 2.2.

    Example:
        Basic usage:
        >>> ricker = RickerWavelet()
        >>> wavelet, t = ricker(f0=25.0, dt=0.001)
        >>> print(f"Length: {len(wavelet)}, Peak at t=0")

        With custom length:
        >>> ricker = RickerWavelet(length_factor=3.0)
        >>> wavelet, t = ricker(f0=15.0, dt=0.002)

    Note:
        The wavelet is centered at t=0 with symmetric time axis.
    """

    def __init__(self, length_factor: float = 2.2):
        super().__init__()
        self.length_factor = length_factor
        self._name = 'ricker'

    def forward(
        self,
        f0: float,
        dt: float,
        device: Optional[str] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Generate Ricker wavelet.

        Args:
            f0: Peak/dominant frequency (Hz).
            dt: Time sampling interval (s).
            device: Computation device. Default: 'cpu'.

        Returns:
            Tuple of (wavelet, time_axis):
                - wavelet: Tensor with shape (nw,)
                - time_axis: Tensor with shape (nw,) centered at t=0
        """
        # Handle tensor inputs
        if isinstance(f0, Tensor):
            f0 = f0.item()
        if isinstance(dt, Tensor):
            dt = dt.item()

        device = device or 'cpu'

        # Compute length (ensure odd for symmetry)
        nw = self.length_factor / f0 / dt
        nw = 2 * int(nw / 2) + 1
        nc = nw // 2

        # Time axis centered at 0
        t = torch.arange(nw, device=device, dtype=torch.float32)
        t = (t - nc) * dt

        # Ricker formula: w(t) = (1 - 2π²f₀²t²) × exp(-π²f₀²t²)
        arg = np.pi * f0 * t
        wavelet = (1.0 - 2.0 * arg ** 2) * torch.exp(-arg ** 2)

        return wavelet, t

    def get_length(self, f0: float, dt: float) -> int:
        """
        Get wavelet length in samples.

        Args:
            f0: Peak frequency (Hz).
            dt: Time sampling (s).

        Returns:
            Number of samples in wavelet.
        """
        nw = self.length_factor / f0 / dt
        return 2 * int(nw / 2) + 1


class GaussianWavelet(WaveletGenerator):
    """
    Gaussian wavelet generator.

    Computes a simple Gaussian pulse:

        w(t) = exp(-4f₀²t²)

    The Gaussian wavelet is smooth without negative side lobes,
    but has a broader frequency spectrum than the Ricker wavelet.

    Args:
        length_factor: Controls wavelet length. Default: 2.0.

    Example:
        >>> gaussian = GaussianWavelet()
        >>> wavelet, t = gaussian(f0=25.0, dt=0.001)
    """

    def __init__(self, length_factor: float = 2.0):
        super().__init__()
        self.length_factor = length_factor
        self._name = 'gaussian'

    def forward(
        self,
        f0: float,
        dt: float,
        device: Optional[str] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Generate Gaussian wavelet.

        Args:
            f0: Peak/dominant frequency (Hz).
            dt: Time sampling interval (s).
            device: Computation device. Default: 'cpu'.

        Returns:
            Tuple of (wavelet, time_axis), each with shape (nw,).
        """
        if isinstance(f0, Tensor):
            f0 = f0.item()
        if isinstance(dt, Tensor):
            dt = dt.item()

        device = device or 'cpu'

        nw = self.length_factor / f0 / dt
        nw = 2 * int(nw / 2) + 1
        nc = nw // 2

        t = torch.arange(nw, device=device, dtype=torch.float32)
        t = (t - nc) * dt

        # Gaussian formula: w(t) = exp(-4f₀²t²)
        wavelet = torch.exp(-4.0 * f0 ** 2 * t ** 2)

        return wavelet, t