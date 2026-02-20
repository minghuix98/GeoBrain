"""
Seismic wavelet generators.

Provides various source wavelet implementations commonly used
in seismic modeling and processing.

Available wavelets:
    - RickerWavelet: Mexican hat, most common
    - GaussianWavelet: Simple pulse without side lobes
    - OrmsbyWavelet: Bandpass with trapezoidal spectrum
    - KlauderWavelet: Vibroseis sweep autocorrelation

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
from torch import Tensor
from typing import Tuple, Optional, Union

from .base import WaveletGenerator
from .ricker import RickerWavelet, GaussianWavelet
from .bandpass import OrmsbyWavelet, KlauderWavelet


def create_wavelet(
    wavelet_type: str,
    f0: float,
    dt: float,
    nt: Optional[int] = None,
    device: str = None,
    **kwargs,
) -> Tuple[Tensor, Tensor]:
    """
    Factory function to create wavelets.

    Convenience function that creates the appropriate wavelet
    generator and returns the wavelet and time axis.

    Args:
        wavelet_type: Type of wavelet.
            Options:
                - 'ricker' or 'mexican_hat'
                - 'gaussian'
                - 'ormsby' (requires f1, f2, f3, f4 kwargs)
                - 'klauder' (requires f_low, f_high, T kwargs)
        f0: Peak/dominant frequency (Hz).
        dt: Time sampling interval (s).
        nt: Optional fixed output length. If provided, wavelet is
            padded or truncated to this length. Default: None.
        device: Computation device. Default: 'cpu'.
        **kwargs: Additional parameters for specific wavelets:
            - Ormsby: f1, f2, f3, f4 (corner frequencies in Hz)
            - Klauder: f_low, f_high (Hz), T (sweep duration in s)

    Returns:
        Tuple of (wavelet, time_axis), each with shape (nw,).

    Raises:
        ValueError: If wavelet_type is unknown.

    Example:
        Ricker wavelet:
        >>> w, t = create_wavelet('ricker', f0=25.0, dt=0.001)

        Ormsby wavelet:
        >>> w, t = create_wavelet(
        ...     'ormsby', f0=0, dt=0.001,
        ...     f1=5, f2=10, f3=40, f4=60
        ... )

        Klauder wavelet:
        >>> w, t = create_wavelet(
        ...     'klauder', f0=0, dt=0.002,
        ...     f_low=10, f_high=80, T=8.0
        ... )
    """
    wavelet_type = wavelet_type.lower()

    # Create generator
    if wavelet_type in ('ricker', 'mexican_hat'):
        generator = RickerWavelet()
    elif wavelet_type == 'gaussian':
        generator = GaussianWavelet()
    elif wavelet_type == 'ormsby':
        f1 = kwargs.pop('f1', f0 * 0.2)
        f2 = kwargs.pop('f2', f0 * 0.5)
        f3 = kwargs.pop('f3', f0 * 1.5)
        f4 = kwargs.pop('f4', f0 * 2.0)
        generator = OrmsbyWavelet(f1, f2, f3, f4)
    elif wavelet_type == 'klauder':
        f_low = kwargs.pop('f_low', 10.0)
        f_high = kwargs.pop('f_high', 80.0)
        T = kwargs.pop('T', 8.0)
        generator = KlauderWavelet(f_low, f_high, T)
    else:
        raise ValueError(
            f"Unknown wavelet type: '{wavelet_type}'. "
            f"Choose from: 'ricker', 'gaussian', 'ormsby', 'klauder'"
        )

    # Generate wavelet
    wavelet, t = generator(f0, dt, device=device)

    # Adjust length if requested
    if nt is not None:
        current_len = len(wavelet)
        if current_len < nt:
            # Pad with zeros
            pad = torch.zeros(nt - current_len, device=wavelet.device)
            wavelet = torch.cat([wavelet, pad])
            t_ext = t[-1] + dt * torch.arange(1, len(pad) + 1, device=t.device)
            t = torch.cat([t, t_ext])
        elif current_len > nt:
            # Truncate
            wavelet = wavelet[:nt]
            t = t[:nt]

    return wavelet, t


__all__ = [
    # Base class
    'WaveletGenerator',
    # Implementations
    'RickerWavelet',
    'GaussianWavelet',
    'OrmsbyWavelet',
    'KlauderWavelet',
    # Factory function
    'create_wavelet',
]