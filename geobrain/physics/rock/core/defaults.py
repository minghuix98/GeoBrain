"""
Default parameter values for rock physics models.

This module provides centralized default values to ensure consistency
between standalone model usage and workflow usage.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

# Critical porosity (typical for sandstones)
PHI_C = 0.36

# Coordination number (typical for random packing)
CN = 8.5

# Effective pressure (MPa)
P = 20.0

# Shear reduction factor (Hertz-Mindlin)
# 1.0 = rough grains, 0.0 = smooth grains
F = 1.0

# Brie exponent for patchy saturation
BRIE_E = 3.0

# Critical porosity for CriticalPorosity model
# Note: slightly higher than PHI_C for cemented sands
PHI_C_CEMENTED = 0.40
