"""
Decision support module for reservoir management under uncertainty.

Provides:
    - ValueOfInformation: Value of Perfect Information (VOPI) analysis.
    - ClosedLoopManager: Closed-loop reservoir management (observe → update → decide).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

from .voi import ValueOfInformation, VOIResult
from .closed_loop import ClosedLoopManager, ClosedLoopStep

__all__ = [
    'ValueOfInformation',
    'VOIResult',
    'ClosedLoopManager',
    'ClosedLoopStep',
]
