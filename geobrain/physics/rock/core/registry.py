"""
Model registry for rock physics models.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import warnings
from typing import Dict, Any, List, Type, Optional

from geobrain.core.registry import BaseRegistry
from .base import BaseModel
from .categories import (
    EffectiveModel, FluidModel, GranularModel,
    EmpiricalModel, AnisotropyModel, AVOModel,
    PermeabilityModel,
)


class ModelRegistry(BaseRegistry):
    """
    Registry for rock physics models.

    Inherits core registry functionality (create, get, list, get_info,
    alias resolution) from BaseRegistry. Adds category inference from
    base class and backward-compatible method names.

    Provides:
        - Automatic model registration via @register decorator
        - Model lookup by name or alias
        - Category-based model organization
        - Factory method for model instantiation
    """

    # Isolated storage for this registry
    _registry: Dict[str, Type[BaseModel]] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}
    _aliases: Dict[str, str] = {}

    @classmethod
    def register(
            cls,
            name: str = None,
            category: str = None,
            aliases: Optional[List[str]] = None,
            tags: Optional[List[str]] = None,
            **metadata
    ):
        """
        Decorator for registering models in the registry.

        Args:
            name: Model name. If None, uses class name.
            category: Model category. If None, inferred from base class.
            aliases: Alternative names for the model.
            tags: Descriptive tags for filtering.
            **metadata: Additional metadata to store.
        """
        def decorator(model_class: Type[BaseModel]) -> Type[BaseModel]:
            model_name = name or model_class.__name__

            if model_name in cls._registry:
                warnings.warn(f"Model '{model_name}' already registered, overwriting.")

            cat = category or cls._infer_category(model_class)

            # Use BaseRegistry's register decorator for core logic
            register_fn = super(ModelRegistry, cls).register(
                model_name,
                category=cat,
                aliases=aliases,
                tags=tags or [],
                **metadata,
            )
            return register_fn(model_class)

        return decorator

    @classmethod
    def _infer_category(cls, model_class: Type[BaseModel]) -> str:
        """Infer category from base class."""
        if issubclass(model_class, EffectiveModel):
            return "effective"
        if issubclass(model_class, FluidModel):
            return "fluid"
        if issubclass(model_class, GranularModel):
            return "granular"
        if issubclass(model_class, EmpiricalModel):
            return "empirical"
        if issubclass(model_class, AnisotropyModel):
            return "anisotropy"
        if issubclass(model_class, AVOModel):
            return "avo"
        if issubclass(model_class, PermeabilityModel):
            return "permeability"
        return "other"

    # ------------------------------------------------------------------
    # Backward-compatible aliases
    # ------------------------------------------------------------------

    @classmethod
    def get_model(cls, name: str) -> Type[BaseModel]:
        """Retrieve a model class by name or alias."""
        return cls.get(name)

    @classmethod
    def create_model(cls, name: str, **kwargs) -> BaseModel:
        """Create a model instance by name."""
        return cls.create(name, **kwargs)

    @classmethod
    def list_models(cls, category: str = None) -> List[str]:
        """List registered model names."""
        return cls.list(category)

    @classmethod
    def list_categories(cls) -> List[str]:
        """List all categories that have registered models."""
        cats = set()
        for meta in cls._metadata.values():
            cats.add(meta.get('category', 'other'))
        return sorted(cats)

    @classmethod
    def get_model_metadata(cls, name: str) -> Dict[str, Any]:
        """Get metadata for a model."""
        return cls.get_info(name)


# Convenience alias for the decorator
register = ModelRegistry.register
