"""
Unified registry access for GeoBrain.

Provides a single entry point for all registries in the framework,
making it easy to discover and use registered components.

Available Registries:
    - simulators: Geostatistical and generative model simulators
    - rock_models: Rock physics models (effective, fluid, granular, etc.)

Example:
    >>> from geobrain import registries
    >>>
    >>> # List all simulator methods
    >>> registries.simulators.list()
    ['fft_ma', 'sgs', 'gan', 'vae', 'diffusion']
    >>>
    >>> # Create a simulator
    >>> simulator = registries.simulators.create('fft_ma')
    >>>
    >>> # List rock physics models by category
    >>> registries.rock_models.list(category='fluid')
    ['Gassmann', 'Wood', 'Brie', ...]
    >>>
    >>> # Create a rock physics model
    >>> gassmann = registries.rock_models.create('Gassmann')
    >>>
    >>> # Get all available registries
    >>> registries.list_all()
    ['simulators', 'rock_models']

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

from typing import Dict, List, Any, Optional


class _RegistryNamespace:
    """
    Namespace wrapper for a registry, providing a clean interface.

    Wraps a registry class to provide consistent access methods
    regardless of the underlying registry implementation.
    """

    def __init__(self, registry_cls, name: str, description: str = ""):
        self._registry = registry_cls
        self._name = name
        self._description = description

    @property
    def name(self) -> str:
        """Registry name."""
        return self._name

    @property
    def description(self) -> str:
        """Registry description."""
        return self._description

    def list(self, category: Optional[str] = None) -> List[str]:
        """
        List registered items.

        Args:
            category: Filter by category (None for all).

        Returns:
            List of registered names.
        """
        return self._registry.list(category)

    def create(self, name: str, **kwargs):
        """
        Create an instance by name.

        Args:
            name: Registered name or alias.
            **kwargs: Arguments passed to constructor.

        Returns:
            New instance.
        """
        return self._registry.create(name, **kwargs)

    def get(self, name: str):
        """
        Get a registered class by name.

        Args:
            name: Registered name or alias.

        Returns:
            Registered class.
        """
        return self._registry.get(name)

    def get_info(self, name: str) -> Dict[str, Any]:
        """
        Get metadata for a registered item.

        Args:
            name: Registered name or alias.

        Returns:
            Metadata dictionary.
        """
        return self._registry.get_info(name)

    def __contains__(self, name: str) -> bool:
        """Check if name is registered."""
        return name in self._registry

    def __len__(self) -> int:
        """Number of registered items."""
        return len(self._registry)

    def __repr__(self) -> str:
        return f"Registry('{self._name}', {len(self)} items)"


class _SimulatorRegistry(_RegistryNamespace):
    """Extended namespace for simulator registry with additional methods."""

    def list_implemented(self) -> List[str]:
        """List fully implemented simulators."""
        return self._registry.list_implemented()

    def is_implemented(self, name: str) -> bool:
        """Check if a simulator is fully implemented."""
        return self._registry.is_implemented(name)

    def list_geostat(self) -> List[str]:
        """List geostatistical simulators."""
        return self.list(category='geostat')

    def list_generative(self) -> List[str]:
        """List generative model simulators."""
        return self.list(category='generative')


class _RockModelRegistry(_RegistryNamespace):
    """Extended namespace for rock model registry with additional methods."""

    def list_categories(self) -> List[str]:
        """List all categories."""
        return self._registry.list_categories()

    def list_effective(self) -> List[str]:
        """List effective medium models."""
        return self.list(category='effective')

    def list_fluid(self) -> List[str]:
        """List fluid substitution models."""
        return self.list(category='fluid')

    def list_granular(self) -> List[str]:
        """List granular media models."""
        return self.list(category='granular')

    def list_empirical(self) -> List[str]:
        """List empirical models."""
        return self.list(category='empirical')


class RegistryHub:
    """
    Central hub for accessing all GeoBrain registries.

    Provides a unified interface to discover and access components
    registered across different modules.

    Example:
        >>> from geobrain import registries
        >>>
        >>> # Access simulator registry
        >>> registries.simulators.list()
        >>> registries.simulators.create('fft_ma')
        >>>
        >>> # Access rock model registry
        >>> registries.rock_models.list(category='fluid')
        >>> registries.rock_models.create('Gassmann')
        >>>
        >>> # List all registries
        >>> registries.list_all()
        ['simulators', 'rock_models']
    """

    def __init__(self):
        self._registries: Dict[str, _RegistryNamespace] = {}
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization to avoid circular imports."""
        if self._initialized:
            return

        # Import registries here to avoid circular imports
        from geobrain.geomodel.registry import SimulatorRegistry
        from geobrain.physics.rock.core.registry import ModelRegistry

        self._registries['simulators'] = _SimulatorRegistry(
            SimulatorRegistry,
            name='simulators',
            description='Geostatistical and generative model simulators'
        )

        self._registries['rock_models'] = _RockModelRegistry(
            ModelRegistry,
            name='rock_models',
            description='Rock physics models'
        )

        self._initialized = True

    @property
    def simulators(self) -> _SimulatorRegistry:
        """
        Registry for geological model simulators.

        Categories:
            - geostat: Geostatistical methods (FFT-MA, SGS)
            - generative: Deep learning methods (GAN, VAE, Diffusion)

        Example:
            >>> registries.simulators.list()
            >>> registries.simulators.list_implemented()
            >>> registries.simulators.create('fft_ma')
        """
        self._ensure_initialized()
        return self._registries['simulators']

    @property
    def rock_models(self) -> _RockModelRegistry:
        """
        Registry for rock physics models.

        Categories:
            - effective: Effective medium theories (VRH, HS, DEM)
            - granular: Granular media models (Hertz-Mindlin, Soft/Stiff Sand)
            - fluid: Fluid substitution (Gassmann, Wood, Brie)
            - empirical: Empirical relations (Han, Gardner, Castagna)
            - anisotropy: Anisotropy models (Thomsen, Backus)
            - avo: AVO models

        Example:
            >>> registries.rock_models.list_categories()
            >>> registries.rock_models.list(category='fluid')
            >>> registries.rock_models.create('Gassmann')
        """
        self._ensure_initialized()
        return self._registries['rock_models']

    def list_all(self) -> List[str]:
        """
        List all available registry names.

        Returns:
            List of registry names.
        """
        self._ensure_initialized()
        return list(self._registries.keys())

    def get_registry(self, name: str) -> _RegistryNamespace:
        """
        Get a registry by name.

        Args:
            name: Registry name ('simulators', 'rock_models').

        Returns:
            Registry namespace.

        Raises:
            KeyError: If registry not found.
        """
        self._ensure_initialized()
        if name not in self._registries:
            raise KeyError(
                f"Unknown registry: '{name}'. "
                f"Available: {self.list_all()}"
            )
        return self._registries[name]

    def summary(self) -> str:
        """
        Get a summary of all registries.

        Returns:
            Formatted summary string.
        """
        self._ensure_initialized()
        lines = ["GeoBrain Registries", "=" * 40]

        for name, reg in self._registries.items():
            lines.append(f"\n{name} ({len(reg)} items)")
            lines.append(f"  {reg._description}")

            # Show categories if available
            if hasattr(reg, 'list_categories'):
                cats = reg.list_categories()
                lines.append(f"  Categories: {', '.join(cats)}")

        return "\n".join(lines)

    def __getitem__(self, name: str) -> _RegistryNamespace:
        """Allow dict-like access: registries['simulators']"""
        return self.get_registry(name)

    def __repr__(self) -> str:
        self._ensure_initialized()
        return f"RegistryHub({self.list_all()})"


# Global singleton instance
registries = RegistryHub()


# Convenience exports
__all__ = [
    'registries',
    'RegistryHub',
]
