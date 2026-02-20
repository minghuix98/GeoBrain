"""
Simulator registry system for factory pattern.

Provides a centralized registry for simulator classes with metadata
support for discovery and documentation.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

from typing import Type, Optional, List

from geobrain.core.registry import BaseRegistry
from .base import BaseSimulator


class SimulatorRegistry(BaseRegistry):
    """
    Registry for simulator classes.

    Maintains a mapping of method names to simulator classes along with
    metadata for documentation and discovery purposes.

    Inherits core registry functionality (create, get, list, get_info,
    alias resolution) from BaseRegistry.

    Example:
        >>> from geobrain.geomodel.registry import SimulatorRegistry
        >>>
        >>> # Create simulator
        >>> simulator = SimulatorRegistry.create('fft_ma')
        >>>
        >>> # List available methods
        >>> print(SimulatorRegistry.list_methods())
        ['fft_ma', 'sgs', 'gan', 'vae', 'diffusion']
        >>>
        >>> # Get method info
        >>> info = SimulatorRegistry.get_info('fft_ma')
        >>> print(info['description'])
    """

    # Isolated storage for this registry
    _registry = {}
    _metadata = {}
    _aliases = {}

    @classmethod
    def register(
        cls,
        name: str,
        *,
        category: str = "other",
        implemented: bool = False,
        description: str = "",
        aliases: Optional[List[str]] = None,
    ):
        """
        Decorator to register a simulator class.

        Args:
            name: Unique identifier for the method.
            category: Category ('geostat' or 'generative').
            implemented: Whether the method is fully implemented.
            description: Brief description of the method.
            aliases: Alternative names for the method.

        Returns:
            Decorator function.

        Example:
            >>> @SimulatorRegistry.register(
            ...     'my_method',
            ...     category='geostat',
            ...     implemented=True,
            ...     description='My custom simulation method'
            ... )
            ... class MySimulator(GeostatSimulator):
            ...     def _simulate(self, config):
            ...         pass
        """
        return super().register(
            name,
            category=category,
            aliases=aliases,
            implemented=implemented,
            description=description,
        )

    @classmethod
    def register_class(
        cls,
        name: str,
        simulator_cls: Type[BaseSimulator],
        category: str = "other",
        implemented: bool = False,
        description: str = ""
    ) -> None:
        """
        Register a simulator class directly (non-decorator form).

        Args:
            name: Unique identifier for the method.
            simulator_cls: Simulator class to register.
            category: Category ('geostat' or 'generative').
            implemented: Whether the method is fully implemented.
            description: Brief description of the method.
        """
        cls._registry[name] = simulator_cls
        cls._metadata[name] = {
            'category': category,
            'class_name': simulator_cls.__name__,
            'implemented': implemented,
            'description': description,
        }

    @classmethod
    def list_methods(cls, category: Optional[str] = None) -> List[str]:
        """
        List registered methods.

        Args:
            category: Filter by category (None for all).

        Returns:
            List of method names.

        Example:
            >>> SimulatorRegistry.list_methods()
            ['fft_ma', 'sgs', 'gan', 'vae', 'diffusion']
            >>> SimulatorRegistry.list_methods(category='geostat')
            ['fft_ma', 'sgs']
        """
        return cls.list(category)

    @classmethod
    def list_implemented(cls) -> List[str]:
        """
        List fully implemented methods.

        Returns:
            List of implemented method names.
        """
        return [
            name for name, meta in cls._metadata.items()
            if meta.get('implemented')
        ]

    @classmethod
    def is_implemented(cls, name: str) -> bool:
        """
        Check if a method is fully implemented.

        Args:
            name: Method name.

        Returns:
            True if implemented, False otherwise.
        """
        try:
            key = cls._resolve(name)
            return cls._metadata[key].get('implemented', False)
        except KeyError:
            return False


class _RegistryProxy:
    """
    Proxy class for backward compatibility with instance-based usage.

    Allows `registry.create(...)` syntax while delegating to class methods.
    """

    def __getattr__(self, name):
        return getattr(SimulatorRegistry, name)

    def __contains__(self, name: str) -> bool:
        return name in SimulatorRegistry._registry

    def __len__(self) -> int:
        return len(SimulatorRegistry._registry)


# Global registry instance for backward compatibility
registry = _RegistryProxy()


def register_model(
    name: str,
    category: str = "other",
    implemented: bool = False,
    description: str = ""
):
    """
    Convenience decorator for registering simulators.

    Args:
        name: Unique identifier for the method.
        category: Category ('geostat' or 'generative').
        implemented: Whether fully implemented.
        description: Brief description.

    Returns:
        Decorator function.

    Example:
        >>> from geobrain.geomodel import register_model, GeostatSimulator
        >>>
        >>> @register_model(
        ...     'kriging',
        ...     category='geostat',
        ...     implemented=True,
        ...     description='Ordinary Kriging interpolation'
        ... )
        ... class KrigingSimulator(GeostatSimulator):
        ...     def _simulate(self, config):
        ...         # Implementation
        ...         pass
    """
    return SimulatorRegistry.register(
        name,
        category=category,
        implemented=implemented,
        description=description
    )
