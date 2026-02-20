"""
Geostatistical simulation methods.

Implemented:
    - FFTMASimulator: FFT-based Moving Average (unconditional, fast, GPU)
    - SGSSimulator: Sequential Gaussian Simulation (conditional)

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

from .fft_ma import FFTMovingAverage, FFTMASimulator
from .sgs import SGSSimulator
from .variogram import VariogramModel, VariogramStructure

__all__ = [
    "FFTMovingAverage",
    "FFTMASimulator",
    "SGSSimulator",
    "VariogramModel",
    "VariogramStructure",
]
