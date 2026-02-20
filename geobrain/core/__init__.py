"""
GeoBrain core infrastructure.

Shared abstractions and utilities used across multiple submodules.

Contents:
    - InverseProblem: Bridges optim and bayes via a unified problem definition
    - registry: Factory pattern infrastructure (BaseRegistry, RegistryHub)
"""

from .inverse import InverseProblem
from .registry import BaseRegistry, registries, RegistryHub

__all__ = [
    'InverseProblem',
    'BaseRegistry',
    'registries',
    'RegistryHub',
]
