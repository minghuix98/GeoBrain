"""
Base registry infrastructure for factory pattern.

Provides BaseRegistry, a generic class-level registry that can be
subclassed to create domain-specific registries with isolated storage.

Subclasses MUST declare their own class attributes to get isolated
registries::

    class MyRegistry(BaseRegistry):
        _registry = {}
        _metadata = {}
        _aliases = {}

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

from typing import Dict, Type, Optional, List, Any


class BaseRegistry:
    """
    Generic registry with class-level storage.

    Provides name-to-class mapping, alias resolution, category-based
    filtering, and metadata storage. All methods are classmethods so
    the registry can be used without instantiation.

    Subclasses must redeclare ``_registry``, ``_metadata``, and
    ``_aliases`` as empty dicts to get isolated storage.
    """

    _registry: Dict[str, Type] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}
    _aliases: Dict[str, str] = {}

    @classmethod
    def register(
        cls,
        name: str = None,
        *,
        category: str = "other",
        aliases: Optional[List[str]] = None,
        **metadata,
    ):
        """
        Decorator to register a class.

        Args:
            name: Registry key. If None, uses the class name.
            category: Category string for organization.
            aliases: Alternative names that resolve to this entry.
            **metadata: Additional metadata stored alongside the entry.

        Returns:
            Decorator function.
        """
        def decorator(registered_cls: Type) -> Type:
            key = name or registered_cls.__name__
            cls._registry[key] = registered_cls
            cls._metadata[key] = {
                'category': category,
                'class_name': registered_cls.__name__,
                **metadata,
            }
            if aliases:
                for alias in aliases:
                    cls._aliases[alias] = key
            return registered_cls
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs):
        """
        Create an instance by name or alias.

        Args:
            name: Registered name or alias.
            **kwargs: Arguments passed to the constructor.

        Returns:
            New instance of the registered class.

        Raises:
            KeyError: If name is not found.
        """
        key = cls._resolve(name)
        return cls._registry[key](**kwargs)

    @classmethod
    def get(cls, name: str) -> Type:
        """
        Get a registered class by name or alias.

        Args:
            name: Registered name or alias.

        Returns:
            The registered class.

        Raises:
            KeyError: If name is not found.
        """
        key = cls._resolve(name)
        return cls._registry[key]

    @classmethod
    def list(cls, category: Optional[str] = None) -> List[str]:
        """
        List registered names, optionally filtered by category.

        Args:
            category: If provided, only return entries in this category.

        Returns:
            List of registered names.
        """
        if category is None:
            return list(cls._registry.keys())
        return [
            n for n, m in cls._metadata.items()
            if m.get('category') == category
        ]

    @classmethod
    def get_info(cls, name: str) -> Dict[str, Any]:
        """
        Get metadata for a registered entry.

        Args:
            name: Registered name or alias.

        Returns:
            Copy of the metadata dictionary.

        Raises:
            KeyError: If name is not found.
        """
        key = cls._resolve(name)
        return cls._metadata[key].copy()

    @classmethod
    def _resolve(cls, name: str) -> str:
        """Resolve a name or alias to the canonical registry key."""
        if name in cls._registry:
            return name
        if name in cls._aliases:
            return cls._aliases[name]
        available = list(cls._registry.keys())
        raise KeyError(f"Unknown: '{name}'. Available: {available}")

    @classmethod
    def __contains__(cls, name: str) -> bool:
        """Check if a name or alias is registered."""
        return name in cls._registry or name in cls._aliases

    @classmethod
    def __len__(cls) -> int:
        """Number of registered entries (excluding aliases)."""
        return len(cls._registry)
