"""
Registry infrastructure for GeoBrain.

Provides the base registry class for factory pattern and a unified
hub for accessing all registered components.

Example:
    >>> from geobrain.core.registry import BaseRegistry, registries
    >>>
    >>> registries.simulators.list()
    >>> registries.rock_models.create('Gassmann')
"""

from .base import BaseRegistry
from .hub import registries, RegistryHub

__all__ = [
    'BaseRegistry',
    'registries',
    'RegistryHub',
]
