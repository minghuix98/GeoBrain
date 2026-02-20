"""
Seismic survey and acquisition geometry.

Provides classes for defining and managing seismic acquisition:
    - Source: Source positions and wavelets
    - Receiver: Receiver positions and types
    - Survey: Combined acquisition geometry

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

from .source import Source
from .receiver import Receiver
from .survey import Survey

__all__ = [
    'Source',
    'Receiver',
    'Survey',
]
