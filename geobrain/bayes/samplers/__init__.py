"""
Sampling algorithms for Bayesian inference.

Implemented:
    - SVGD: Stein Variational Gradient Descent
    - HMC: Hamiltonian Monte Carlo
    - NUTS: No-U-Turn Sampler
    - LDS: Langevin Dynamics Sampler (ULA/MALA)

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

from .svgd import SVGD
from .hmc import HMC
from .nuts import NUTS
from .lds import LDS

__all__ = ['SVGD', 'HMC', 'NUTS', 'LDS']
