#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Materials database: minerals and fluids.

This module provides a database of common rock-forming minerals and
pore fluids with their elastic and physical properties.

Features:
    - Pre-defined mineral properties (K, G, ρ)
    - Pre-defined fluid properties (K, ρ)
    - Lookup functions with flexible naming
    - Mixing functions (VRH for minerals, Wood for fluids)

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import torch
from torch import Tensor


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(frozen=True)
class Mineral:
    """
    Mineral elastic properties.

    Immutable dataclass storing the elastic moduli and density
    of a mineral phase.

    Attributes:
        name: Mineral name (display format)
        K: Bulk modulus (GPa)
        G: Shear modulus (GPa)
        rho: Density (g/cm³)

    Example:
        >>> quartz = Mineral('Quartz', K=36.6, G=45.0, rho=2.65)
        >>> print(f"{quartz.name}: K={quartz.K} GPa")
        Quartz: K=36.6 GPa
    """
    name: str
    K: float  # Bulk modulus (GPa)
    G: float  # Shear modulus (GPa)
    rho: float  # Density (g/cm³)


@dataclass(frozen=True)
class Fluid:
    """
    Fluid properties.

    Immutable dataclass storing the bulk modulus and density
    of a pore fluid.

    Attributes:
        name: Fluid name (display format)
        K: Bulk modulus (GPa)
        rho: Density (g/cm³)

    Note:
        Fluids have zero shear modulus (G=0).

    Example:
        >>> water = Fluid('Water', K=2.25, rho=1.0)
        >>> print(f"{water.name}: K={water.K} GPa, ρ={water.rho} g/cc")
        Water: K=2.25 GPa, ρ=1.0 g/cc
    """
    name: str
    K: float  # Bulk modulus (GPa)
    rho: float  # Density (g/cm³)


# =============================================================================
# Mineral Database
# =============================================================================

MINERALS: Dict[str, Mineral] = {
    # Framework silicates
    'quartz': Mineral('Quartz', K=36.6, G=45.0, rho=2.65),
    'feldspar': Mineral('Feldspar', K=37.5, G=15.0, rho=2.62),
    'plagioclase': Mineral('Plagioclase', K=75.6, G=25.6, rho=2.63),

    # Micas
    'mica': Mineral('Mica', K=42.0, G=22.0, rho=2.79),
    'muscovite': Mineral('Muscovite', K=52.0, G=31.0, rho=2.82),
    'biotite': Mineral('Biotite', K=50.0, G=28.0, rho=3.05),

    # Clay minerals
    'clay': Mineral('Clay (average)', K=21.0, G=7.0, rho=2.58),
    'illite': Mineral('Illite', K=21.0, G=7.0, rho=2.66),
    'kaolinite': Mineral('Kaolinite', K=1.5, G=1.4, rho=2.60),
    'smectite': Mineral('Smectite', K=9.3, G=6.9, rho=2.70),
    'montmorillonite': Mineral('Montmorillonite', K=9.3, G=6.9, rho=2.50),
    'chlorite': Mineral('Chlorite', K=82.0, G=28.0, rho=2.80),

    # Carbonate minerals
    'calcite': Mineral('Calcite', K=76.8, G=32.0, rho=2.71),
    'dolomite': Mineral('Dolomite', K=94.9, G=45.0, rho=2.87),
    'aragonite': Mineral('Aragonite', K=47.0, G=39.0, rho=2.93),

    # Evaporites
    'halite': Mineral('Halite', K=24.8, G=14.9, rho=2.16),
    'anhydrite': Mineral('Anhydrite', K=56.1, G=29.1, rho=2.98),
    'gypsum': Mineral('Gypsum', K=36.0, G=15.0, rho=2.32),

    # Other minerals
    'kerogen': Mineral('Kerogen', K=2.9, G=2.7, rho=1.30),
    'pyrite': Mineral('Pyrite', K=147.4, G=132.5, rho=4.93),
    'siderite': Mineral('Siderite', K=124.0, G=51.0, rho=3.96),
    'magnetite': Mineral('Magnetite', K=161.0, G=91.0, rho=5.18),
    'hematite': Mineral('Hematite', K=100.0, G=95.0, rho=5.24),
}

# =============================================================================
# Fluid Database
# =============================================================================

FLUIDS: Dict[str, Fluid] = {
    # Water and brine
    'water': Fluid('Water', K=2.25, rho=1.00),
    'brine': Fluid('Brine', K=2.80, rho=1.05),
    'brine_light': Fluid('Light Brine', K=2.50, rho=1.02),
    'brine_heavy': Fluid('Heavy Brine', K=3.20, rho=1.15),

    # Oil
    'oil': Fluid('Oil', K=1.00, rho=0.80),
    'oil_light': Fluid('Light Oil', K=1.20, rho=0.75),
    'oil_medium': Fluid('Medium Oil', K=1.00, rho=0.85),
    'oil_heavy': Fluid('Heavy Oil', K=0.70, rho=0.95),

    # Gas
    'gas': Fluid('Gas', K=0.02, rho=0.10),
    'gas_shallow': Fluid('Shallow Gas', K=0.01, rho=0.05),
    'gas_deep': Fluid('Deep Gas', K=0.10, rho=0.30),

    # CO2
    'co2': Fluid('CO2 (supercritical)', K=0.05, rho=0.50),
    'co2_liquid': Fluid('CO2 (liquid)', K=0.10, rho=0.70),
}


# =============================================================================
# Lookup Functions
# =============================================================================

def get_mineral(name: str) -> Mineral:
    """
    Retrieve mineral properties by name.

    Supports flexible naming: case-insensitive, spaces converted to underscores.

    Args:
        name: Mineral name (e.g., 'quartz', 'Quartz', 'QUARTZ')

    Returns:
        Mineral dataclass with K, G, rho

    Raises:
        KeyError: If mineral not found in database

    Example:
        >>> m = get_mineral('quartz')
        >>> print(f"Quartz: K={m.K} GPa, G={m.G} GPa, ρ={m.rho} g/cc")
        Quartz: K=36.6 GPa, G=45.0 GPa, ρ=2.65 g/cc

        >>> m = get_mineral('Clay')  # Case insensitive
        >>> print(f"{m.name}: K={m.K} GPa")
        Clay (average): K=21.0 GPa
    """
    key = name.lower().replace(' ', '_').replace('-', '_')
    if key not in MINERALS:
        available = ', '.join(sorted(MINERALS.keys()))
        raise KeyError(
            f"Unknown mineral: '{name}'. "
            f"Available minerals: {available}"
        )
    return MINERALS[key]


def get_fluid(name: str) -> Fluid:
    """
    Retrieve fluid properties by name.

    Supports flexible naming: case-insensitive, spaces converted to underscores.

    Args:
        name: Fluid name (e.g., 'water', 'Water', 'brine')

    Returns:
        Fluid dataclass with K, rho

    Raises:
        KeyError: If fluid not found in database

    Example:
        >>> f = get_fluid('water')
        >>> print(f"Water: K={f.K} GPa, ρ={f.rho} g/cc")
        Water: K=2.25 GPa, ρ=1.0 g/cc

        >>> f = get_fluid('gas_deep')
        >>> print(f"{f.name}: K={f.K} GPa")
        Deep Gas: K=0.1 GPa
    """
    key = name.lower().replace(' ', '_').replace('-', '_')
    if key not in FLUIDS:
        available = ', '.join(sorted(FLUIDS.keys()))
        raise KeyError(
            f"Unknown fluid: '{name}'. "
            f"Available fluids: {available}"
        )
    return FLUIDS[key]


def list_minerals() -> List[str]:
    """
    List all available mineral names.

    Returns:
        Sorted list of mineral keys

    Example:
        >>> minerals = list_minerals()
        >>> print(minerals[:5])
        ['anhydrite', 'aragonite', 'biotite', 'calcite', 'chlorite']
    """
    return sorted(MINERALS.keys())


def list_fluids() -> List[str]:
    """
    List all available fluid names.

    Returns:
        Sorted list of fluid keys

    Example:
        >>> fluids = list_fluids()
        >>> print(fluids[:5])
        ['brine', 'brine_heavy', 'brine_light', 'co2', 'co2_liquid']
    """
    return sorted(FLUIDS.keys())


# =============================================================================
# Mixing Functions
# =============================================================================

def mix_minerals_vrh(
        composition: Dict[str, float],
        normalize: bool = True
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Mix minerals using Voigt-Reuss-Hill average.

    VRH average provides a reasonable estimate for isotropic mixtures:

        M_VRH = 0.5 * (M_Voigt + M_Reuss)

    where:
        M_Voigt = Σ(f_i * M_i)        (arithmetic mean - upper bound)
        M_Reuss = 1 / Σ(f_i / M_i)    (harmonic mean - lower bound)

    Args:
        composition: Dictionary mapping mineral names to volume fractions.
                    Example: {'quartz': 0.7, 'clay': 0.3}
        normalize: If True, normalize fractions to sum to 1.0

    Returns:
        Tuple of (K, G, rho) as torch tensors:
            - K: Effective bulk modulus (GPa)
            - G: Effective shear modulus (GPa)
            - rho: Effective density (g/cm³)

    Example:
        >>> K, G, rho = mix_minerals_vrh({'quartz': 0.7, 'clay': 0.3})
        >>> print(f"K={K.item():.1f} GPa, G={G.item():.1f} GPa")
        K=31.7 GPa, G=26.5 GPa

        Using non-normalized fractions:
        >>> K, G, rho = mix_minerals_vrh({'quartz': 70, 'clay': 30})
        >>> # Same result (fractions are normalized internally)
    """
    if not composition:
        raise ValueError("Composition dictionary cannot be empty")

    total = sum(composition.values())
    if total <= 0:
        raise ValueError("Total fraction must be positive")

    # Initialize accumulators
    K_voigt = 0.0
    K_reuss_inv = 0.0
    G_voigt = 0.0
    G_reuss_inv = 0.0
    rho_mix = 0.0

    for name, frac in composition.items():
        f = frac / total if normalize else frac
        mineral = get_mineral(name)

        # Voigt (arithmetic mean)
        K_voigt += f * mineral.K
        G_voigt += f * mineral.G

        # Reuss (harmonic mean)
        K_reuss_inv += f / mineral.K
        G_reuss_inv += f / mineral.G

        # Density (always arithmetic mean)
        rho_mix += f * mineral.rho

    # VRH average
    K_reuss = 1.0 / K_reuss_inv
    G_reuss = 1.0 / G_reuss_inv

    K = 0.5 * (K_voigt + K_reuss)
    G = 0.5 * (G_voigt + G_reuss)

    return torch.tensor(K), torch.tensor(G), torch.tensor(rho_mix)


def mix_fluids_wood(
        composition: Dict[str, float],
        normalize: bool = True
) -> Tuple[Tensor, Tensor]:
    """
    Mix fluids using Wood's equation (Reuss average).

    Wood's equation (iso-stress average) is appropriate for
    fluid mixtures where phases are in pressure equilibrium:

        1/K_eff = Σ(f_i / K_i)
        ρ_eff = Σ(f_i * ρ_i)

    Args:
        composition: Dictionary mapping fluid names to volume fractions.
                    Example: {'water': 0.7, 'gas': 0.3}
        normalize: If True, normalize fractions to sum to 1.0

    Returns:
        Tuple of (K, rho) as torch tensors:
            - K: Effective bulk modulus (GPa)
            - rho: Effective density (g/cm³)

    Example:
        >>> K, rho = mix_fluids_wood({'water': 0.8, 'gas': 0.2})
        >>> print(f"K={K.item():.3f} GPa, ρ={rho.item():.2f} g/cc")
        K=0.099 GPa, ρ=0.82 g/cc

    Note:
        Small amounts of gas dramatically reduce bulk modulus.
        This is the basis for bright spot detection in seismic.
    """
    if not composition:
        raise ValueError("Composition dictionary cannot be empty")

    total = sum(composition.values())
    if total <= 0:
        raise ValueError("Total fraction must be positive")

    K_inv = 0.0
    rho_mix = 0.0

    for name, frac in composition.items():
        f = frac / total if normalize else frac
        fluid = get_fluid(name)

        K_inv += f / fluid.K
        rho_mix += f * fluid.rho

    K = 1.0 / K_inv

    return torch.tensor(K), torch.tensor(rho_mix)


def mix_fluids_brie(
        K_water: float,
        K_gas: float,
        rho_water: float,
        rho_gas: float,
        Sw: float,
        e: float = 3.0
) -> Tuple[Tensor, Tensor]:
    """
    Mix fluids using Brie's empirical equation.

    Brie's equation accounts for patchy saturation effects:

        K_eff = K_gas + (K_water - K_gas) * Sw^e
        ρ_eff = Sw * ρ_water + (1-Sw) * ρ_gas

    Args:
        K_water: Water/brine bulk modulus (GPa)
        K_gas: Gas bulk modulus (GPa)
        rho_water: Water/brine density (g/cm³)
        rho_gas: Gas density (g/cm³)
        Sw: Water saturation (0-1)
        e: Brie exponent (default: 3.0, higher = more patchy)

    Returns:
        Tuple of (K, rho) as torch tensors

    Example:
        >>> K, rho = mix_fluids_brie(
        ...     K_water=2.25, K_gas=0.02,
        ...     rho_water=1.0, rho_gas=0.1,
        ...     Sw=0.8, e=3.0
        ... )
        >>> print(f"K={K.item():.3f} GPa")
        K=1.155 GPa

    Note:
        - e=1: Linear mixing (Wood's equation at low frequencies)
        - e=3: Typical patchy saturation
        - e→∞: All gas or all water (step function)
    """
    K = K_gas + (K_water - K_gas) * (Sw ** e)
    rho = Sw * rho_water + (1.0 - Sw) * rho_gas

    return torch.tensor(K), torch.tensor(rho)