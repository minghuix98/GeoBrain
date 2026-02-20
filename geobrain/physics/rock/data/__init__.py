#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data module for rock physics.

This package provides mineral and fluid property databases and
mixing functions for rock physics computations.

Features:
    - Mineral elastic properties database
    - Fluid properties database
    - Flexible lookup functions
    - VRH and Wood mixing laws

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

from .materials import (
    # Data classes
    Mineral,
    Fluid,

    # Databases
    MINERALS,
    FLUIDS,

    # Lookup functions
    get_mineral,
    get_fluid,
    list_minerals,
    list_fluids,

    # Mixing functions
    mix_minerals_vrh,
    mix_fluids_wood,
    mix_fluids_brie,
)

__all__ = [
    # Data classes
    'Mineral',
    'Fluid',

    # Databases
    'MINERALS',
    'FLUIDS',

    # Lookup functions
    'get_mineral',
    'get_fluid',
    'list_minerals',
    'list_fluids',

    # Mixing functions
    'mix_minerals_vrh',
    'mix_fluids_wood',
    'mix_fluids_brie',
]