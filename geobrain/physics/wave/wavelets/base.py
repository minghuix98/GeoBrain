"""
Base class for wavelet generators.

Provides the abstract interface that all wavelet implementations
must follow for consistent API across different wavelet types.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
import torch.nn as nn
from torch import Tensor
from abc import abstractmethod
from typing import Tuple, Optional


class WaveletGenerator(nn.Module):
    """
    Abstract base class for wavelet generators.
    
    All wavelet implementations should inherit from this class
    and implement the forward method.
    
    Attributes:
        _name: Wavelet type identifier string.
    
    Example:
        Subclass implementation:
        >>> class MyWavelet(WaveletGenerator):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self._name = 'my_wavelet'
        ...
        ...     def forward(self, f0, dt, device=None):
        ...         # Generate wavelet
        ...         return wavelet, time_axis
    """
    
    def __init__(self):
        super().__init__()
        self._name = 'base'
    
    @abstractmethod
    def forward(
        self,
        f0: float,
        dt: float,
        device: Optional[str] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Generate wavelet.
        
        Args:
            f0: Dominant/peak frequency (Hz).
            dt: Time sampling interval (s).
            device: Computation device. Default: 'cpu'.

        Returns:
            Tuple of (wavelet, time_axis):
                - wavelet: Tensor with shape (nw,) containing amplitude values
                - time_axis: Tensor with shape (nw,) containing time values
        """
        raise NotImplementedError

    def __call__(
        self,
        f0: float,
        dt: float,
        device: Optional[str] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Generate wavelet (callable interface)."""
        return self.forward(f0, dt, device)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @property
    def name(self) -> str:
        """Wavelet type name."""
        return self._name