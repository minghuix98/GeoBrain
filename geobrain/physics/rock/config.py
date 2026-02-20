#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration and presets for rock physics workflows.

This module provides:
    - RockPhysicsConfig: Configuration dataclass for workflows
    - PRESETS: Pre-defined configurations for common rock types
    - Helper functions for preset management

Presets encode domain knowledge about typical rock compositions
and modeling approaches for different lithologies.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import copy

from .core import defaults


@dataclass
class RockPhysicsConfig:
    """
    Configuration for rock physics workflow.

    Encapsulates all settings needed to define a rock physics model:
    mineral composition, fluid composition, model selection, and
    model parameters.

    Attributes:
        minerals: Volume fractions of minerals.
                 Example: {'quartz': 0.7, 'clay': 0.3}
        fluids: Volume fractions of fluids.
               Example: {'water': 0.8, 'gas': 0.2}
        models: Model selection for each workflow step:
            - 'mineral_mixer': Mineral mixing method (default: 'VRH')
            - 'dry_rock': Dry rock model (default: 'SoftSand')
            - 'fluid_sub': Fluid substitution model (default: 'Gassmann')
            - 'fluid_mix': Fluid mixing method (default: 'Wood')
        parameters: Model parameters:
            - 'phi_c': Critical porosity (default: 0.36)
            - 'Cn': Coordination number (default: 8.5)
            - 'P': Effective pressure in MPa (default: 20.0)

    Example:
        Create custom configuration:
        >>> config = RockPhysicsConfig(
        ...     minerals={'quartz': 0.8, 'clay': 0.2},
        ...     fluids={'water': 0.7, 'gas': 0.3},
        ...     models={'dry_rock': 'StiffSand', 'fluid_mix': 'Brie'},
        ...     parameters={'phi_c': 0.35, 'P': 30.0}
        ... )

        Modify existing configuration:
        >>> new_config = config.with_overrides(P=40.0, phi_c=0.32)
    """

    minerals: Dict[str, float]
    fluids: Dict[str, float]
    models: Dict[str, str] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize defaults and normalize fractions after creation."""
        self._set_defaults()
        self._normalize()

    def _set_defaults(self) -> None:
        """Set default model choices and parameters."""
        # Default model selections
        model_defaults = {
            'mineral_mixer': 'VRH',
            'dry_rock': 'SoftSand',
            'fluid_sub': 'Gassmann',
            'fluid_mix': 'Wood',
        }
        for key, value in model_defaults.items():
            self.models.setdefault(key, value)

        # Default parameters (from centralized defaults)
        param_defaults = {
            'phi_c': defaults.PHI_C,
            'Cn': defaults.CN,
            'P': defaults.P,
            'f': defaults.F,
            'e': defaults.BRIE_E,
        }
        for key, value in param_defaults.items():
            self.parameters.setdefault(key, value)

    def _normalize(self) -> None:
        """Normalize mineral and fluid fractions to sum to 1.0."""
        for d in [self.minerals, self.fluids]:
            total = sum(d.values())
            if total > 0:
                for key in d:
                    d[key] /= total

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of config

        Example:
            >>> d = config.to_dict()
            >>> import json
            >>> json.dumps(d)  # Serializable
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RockPhysicsConfig':
        """
        Create configuration from dictionary.

        Args:
            d: Dictionary with 'minerals', 'fluids', optionally
               'models' and 'parameters'

        Returns:
            RockPhysicsConfig instance

        Example:
            >>> config = RockPhysicsConfig.from_dict({
            ...     'minerals': {'quartz': 1.0},
            ...     'fluids': {'water': 1.0}
            ... })
        """
        return cls(
            minerals=d.get('minerals', {}),
            fluids=d.get('fluids', {}),
            models=d.get('models', {}),
            parameters=d.get('parameters', {}),
        )

    def copy(self) -> 'RockPhysicsConfig':
        """
        Create a deep copy of this configuration.

        Returns:
            New RockPhysicsConfig instance with copied data
        """
        return RockPhysicsConfig(
            minerals=copy.deepcopy(self.minerals),
            fluids=copy.deepcopy(self.fluids),
            models=copy.deepcopy(self.models),
            parameters=copy.deepcopy(self.parameters),
        )

    def with_overrides(self, **kwargs) -> 'RockPhysicsConfig':
        """
        Create new configuration with overridden values.

        Supports overriding:
            - 'minerals': Replace mineral composition
            - 'fluids': Replace fluid composition
            - 'models': Update model selections
            - 'parameters': Update parameters
            - Individual parameters (e.g., phi_c=0.35, P=30.0)

        Args:
            **kwargs: Values to override

        Returns:
            New RockPhysicsConfig with overrides applied

        Example:
            >>> base = get_preset('shaly_sand')
            >>> modified = base.with_overrides(
            ...     minerals={'quartz': 0.6, 'clay': 0.4},
            ...     P=35.0,
            ...     phi_c=0.33
            ... )
        """
        new = self.copy()

        for key, value in kwargs.items():
            if key == 'minerals':
                new.minerals = copy.deepcopy(value)
            elif key == 'fluids':
                new.fluids = copy.deepcopy(value)
            elif key == 'models':
                new.models.update(value)
            elif key == 'parameters':
                new.parameters.update(value)
            else:
                # Assume it's a parameter
                new.parameters[key] = value

        new._normalize()
        return new

    def __repr__(self) -> str:
        minerals_str = ', '.join(f"{k}:{v:.0%}" for k, v in self.minerals.items())
        fluids_str = ', '.join(f"{k}:{v:.0%}" for k, v in self.fluids.items())
        return (
            f"RockPhysicsConfig(\n"
            f"  minerals=[{minerals_str}],\n"
            f"  fluids=[{fluids_str}],\n"
            f"  dry_rock={self.models.get('dry_rock')},\n"
            f"  phi_c={self.parameters.get('phi_c')}\n"
            f")"
        )


# =============================================================================
# Presets
# =============================================================================

PRESETS: Dict[str, RockPhysicsConfig] = {
    # Clean sand (unconsolidated to consolidated)
    'clean_sand': RockPhysicsConfig(
        minerals={'quartz': 1.0},
        fluids={'water': 1.0},
        models={'dry_rock': 'SoftSand'},
        parameters={'phi_c': 0.40},
    ),

    # Shaly sand (common reservoir rock)
    'shaly_sand': RockPhysicsConfig(
        minerals={'quartz': 0.7, 'clay': 0.3},
        fluids={'water': 1.0},
        models={'dry_rock': 'SoftSand'},
        parameters={'phi_c': 0.35},
    ),

    # Gas sand (gas-bearing sandstone)
    'gas_sand': RockPhysicsConfig(
        minerals={'quartz': 0.8, 'clay': 0.2},
        fluids={'gas': 0.7, 'water': 0.3},
        models={'dry_rock': 'SoftSand', 'fluid_mix': 'Brie'},
        parameters={'phi_c': 0.36, 'e': 3.0},
    ),

    # Tight sand (cemented, low porosity)
    'tight_sand': RockPhysicsConfig(
        minerals={'quartz': 0.85, 'clay': 0.15},
        fluids={'gas': 0.8, 'water': 0.2},
        models={'dry_rock': 'StiffSand', 'fluid_mix': 'Brie'},
        parameters={'phi_c': 0.30},
    ),

    # Limestone (carbonate)
    'limestone': RockPhysicsConfig(
        minerals={'calcite': 0.95, 'clay': 0.05},
        fluids={'water': 1.0},
        models={'dry_rock': 'DEM'},
        parameters={'phi_c': 0.35},
    ),

    # Dolomite
    'dolomite': RockPhysicsConfig(
        minerals={'dolomite': 0.9, 'calcite': 0.1},
        fluids={'water': 1.0},
        models={'dry_rock': 'DEM'},
        parameters={'phi_c': 0.35},
    ),

    # Shale (source/seal rock)
    'shale': RockPhysicsConfig(
        minerals={'clay': 0.6, 'quartz': 0.3, 'calcite': 0.1},
        fluids={'water': 1.0},
        models={'dry_rock': 'SoftSand'},
        parameters={'phi_c': 0.30},
    ),

    # Gas shale (unconventional reservoir)
    'gas_shale': RockPhysicsConfig(
        minerals={'clay': 0.5, 'quartz': 0.35, 'kerogen': 0.15},
        fluids={'gas': 0.9, 'water': 0.1},
        models={'dry_rock': 'SoftSand', 'fluid_mix': 'Brie'},
        parameters={'phi_c': 0.25},
    ),

    # Chalk (high porosity carbonate)
    'chalk': RockPhysicsConfig(
        minerals={'calcite': 1.0},
        fluids={'water': 1.0},
        models={'dry_rock': 'CriticalPorosity'},
        parameters={'phi_c': 0.65},
    ),

    # Cemented sandstone
    'cemented_sand': RockPhysicsConfig(
        minerals={'quartz': 0.9, 'clay': 0.1},
        fluids={'water': 1.0},
        models={'dry_rock': 'ContactCement'},
        parameters={'phi_c': 0.36, 'Cn': 9.0},
    ),
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_preset(name: str) -> RockPhysicsConfig:
    """
    Retrieve a preset configuration by name.

    Returns a copy of the preset, so modifications won't affect
    the original preset.

    Args:
        name: Preset name (case-insensitive)

    Returns:
        Copy of the preset configuration

    Raises:
        ValueError: If preset not found

    Example:
        >>> config = get_preset('shaly_sand')
        >>> print(config.minerals)
        {'quartz': 0.7, 'clay': 0.3}

        >>> # Modify without affecting original preset
        >>> config.parameters['P'] = 40.0
    """
    key = name.lower().replace(' ', '_').replace('-', '_')

    if key not in PRESETS:
        available = ', '.join(sorted(PRESETS.keys()))
        raise ValueError(
            f"Unknown preset: '{name}'. "
            f"Available presets: {available}"
        )

    return PRESETS[key].copy()


def list_presets() -> List[str]:
    """
    List all available preset names.

    Returns:
        Sorted list of preset names

    Example:
        >>> presets = list_presets()
        >>> print(presets)
        ['cemented_sand', 'chalk', 'clean_sand', 'dolomite', ...]
    """
    return sorted(PRESETS.keys())


def describe_preset(name: str) -> str:
    """
    Get a human-readable description of a preset.

    Args:
        name: Preset name

    Returns:
        Formatted string describing the preset

    Example:
        >>> print(describe_preset('shaly_sand'))
        Preset: shaly_sand
        Minerals: quartz (70%), clay (30%)
        Fluids: water (100%)
        Dry rock model: SoftSand
        Critical porosity: 0.35
    """
    config = get_preset(name)

    minerals = ', '.join(f"{k} ({v:.0%})" for k, v in config.minerals.items())
    fluids = ', '.join(f"{k} ({v:.0%})" for k, v in config.fluids.items())

    return (
        f"Preset: {name}\n"
        f"Minerals: {minerals}\n"
        f"Fluids: {fluids}\n"
        f"Dry rock model: {config.models.get('dry_rock')}\n"
        f"Critical porosity: {config.parameters.get('phi_c')}"
    )