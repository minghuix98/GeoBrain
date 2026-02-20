#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rock Physics Module
===================

Complete PyTorch-based differentiable rock physics modeling library.

This package provides a comprehensive suite of rock physics models
for computing elastic properties and seismic velocities of rocks
and sediments. All models support PyTorch autodiff for gradient-based
optimization and inversion.

Features:
    - Effective medium theories (VRH, HS bounds, DEM, SC)
    - Granular media models (Hertz-Mindlin, soft/stiff sand, contact cement)
    - Fluid substitution (Gassmann, Wood, Brie)
    - Empirical relations (Han, Castagna, Gardner, compaction trends)
    - Anisotropy models (Thomsen, Backus, Bond transform)
    - Permeability models (Kozeny-Carman, Owolabi, etc.)
    - Unified workflow with presets for common lithologies

Example:
    >>> from geobrain.physics.rock import RockPhysicsWorkflow, Gassmann
    >>> workflow = RockPhysicsWorkflow.from_preset('shaly_sand')
    >>> Vp, Vs, rho = workflow(phi=0.2)

Model Categories:
    - Effective: VRH, HS, DEM, SC, Hudson, KT, Eshelby-Cheng, Swiss Cheese
    - Granular: Hertz-Mindlin, Soft/Stiff Sand, Contact Cement, PCM, Digby, VPCM
    - Fluid: Gassmann, Wood, Brie, Batzle-Wang, Biot, Brown-Korringa
    - Empirical: Han, Castagna, Gardner, compaction trends
    - Anisotropy: Thomsen, Backus, Bond transform, Thomsen-Tsvankin
    - Permeability: Kozeny-Carman, Owolabi, Panda-Lake, Bloch, Bernabe
    - Resistivity: Archie

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "Mingliang Liu"
__email__ = "mingliangliu@sdu.edu.cn"

# Core
from .core import (
    Tensor, TensorLike, as_tensor, PI, EPS,
    BaseModel, ComponentModel, CompositeModel,
    EffectiveModel, FluidModel, GranularModel, EmpiricalModel, AnisotropyModel, AVOModel,
    PermeabilityModel,
    ModelRegistry, register,
)
from .core.registry import ModelRegistry  # Explicit export for workflow usage

# Utils
from .utils import (
    v_from_moduli, moduli_from_v, poisson, lame_lambda, youngs_modulus,
    write_iso_matrix, write_vti_matrix, write_hti_matrix, write_ortho_matrix,
    coordination_number, validate_porosity, validate_positive,
    matrix_modulus, den_matrix, normalize_kde,
)

# Data
from .data import (
    Mineral, Fluid, MINERALS, FLUIDS,
    get_mineral, get_fluid, list_minerals, list_fluids,
    mix_minerals_vrh, mix_fluids_wood,
)

# Models - Effective
from .models.effective import (
    Voigt, Reuss, VRH, HashinShtrikman, DEM, SelfConsistent, CriticalPorosity,
    Hudson, HudsonRandom, KusterToksoz, EshelbyCheng,
    SwissCheese, DiluteCrack, OConnellBudiansky, OConnellBudianskyFl,
    SCDilute, SCFlex, MTAverage, PQ, HudsonOrtho, HudsonCone,
)

# Models - Granular
from .models.granular import (
    HertzMindlin, SoftSand, StiffSand, ContactCement, ContactCementFull,
    Walton, MUHS, PCM,
    ThomasStieber, SiltyShale, ShalySand,
    Digby, ConstantCement, VPCM, Diluting,
)

# Models - Fluid
from .models.fluid import (
    Gassmann, GassmannInverse, GassmannFluidSub, Wood, Brie, BatzleWang,
    BiotHF, BiotDispersion, GeertsmaSmitHF, GeertsmaSmitLF,
    BrownKorringaDry2Sat, BrownKorringaSat2Dry, BrownKorringaSub,
    MavkoJizba, CO2Properties, LiveOil, CO2Brine,
)

# Models - Empirical
from .models.empirical import (
    Han, RaymerHuntGardner, WyllieTimeAverage, CastagnaMudrock, GreenbergCastagna, Gardner, Krief,
    Storvoll, Japsen, Hillis, Scherbaum, Ehrenberg, RammPorosity, Sclater, Hjelstuen, StPeter,
    DensityModel,
)

# Models - Anisotropy
from .models.anisotropy import (
    Thomsen, Backus, BondTransform,
    VelocityAzimuthHTI, VelocityAzimuthVTI, ThomsenTsvankin,
)

# Models - Resistivity
from .models.resistivity import ArchieResistivity

# Models - Permeability
from .models.permeability import (
    KozenyCarman, KozenyCarmanPercolation, Owolabi, PermLogs,
    PandaLake, PandaLakeCem, Revil, Fredrich, Bloch, Bernabe,
)

# Config & Workflow
from .config import RockPhysicsConfig, PRESETS, get_preset, list_presets
from .workflow import RockPhysicsWorkflow

# QI (Quantitative Interpretation)
from .qi import Screening, ConstantCementVelocity, CementEstimator, RPT

# Default parameters
from .core import defaults

__all__ = [
    # Version
    '__version__',
    '__author__',
    '__email__',
    # Core
    'Tensor', 'TensorLike', 'as_tensor', 'PI', 'EPS',
    'BaseModel', 'ComponentModel', 'CompositeModel',
    'EffectiveModel', 'FluidModel', 'GranularModel', 'EmpiricalModel', 'AnisotropyModel', 'AVOModel',
    'PermeabilityModel',
    'ModelRegistry', 'register',
    # Utils
    'v_from_moduli', 'moduli_from_v', 'poisson', 'lame_lambda', 'youngs_modulus',
    'write_iso_matrix', 'write_vti_matrix', 'write_hti_matrix', 'write_ortho_matrix',
    'coordination_number', 'validate_porosity', 'validate_positive',
    'matrix_modulus', 'den_matrix', 'normalize_kde',
    # Data
    'Mineral', 'Fluid', 'MINERALS', 'FLUIDS',
    'get_mineral', 'get_fluid', 'list_minerals', 'list_fluids',
    'mix_minerals_vrh', 'mix_fluids_wood',
    # Effective medium models
    'Voigt', 'Reuss', 'VRH', 'HashinShtrikman', 'DEM', 'SelfConsistent', 'CriticalPorosity',
    'Hudson', 'HudsonRandom', 'KusterToksoz', 'EshelbyCheng',
    'SwissCheese', 'DiluteCrack', 'OConnellBudiansky', 'OConnellBudianskyFl',
    'SCDilute', 'SCFlex', 'MTAverage', 'PQ', 'HudsonOrtho', 'HudsonCone',
    # Granular models
    'HertzMindlin', 'SoftSand', 'StiffSand', 'ContactCement', 'ContactCementFull',
    'Walton', 'MUHS', 'PCM',
    'ThomasStieber', 'SiltyShale', 'ShalySand',
    'Digby', 'ConstantCement', 'VPCM', 'Diluting',
    # Fluid models
    'Gassmann', 'GassmannInverse', 'GassmannFluidSub', 'Wood', 'Brie', 'BatzleWang',
    'BiotHF', 'BiotDispersion', 'GeertsmaSmitHF', 'GeertsmaSmitLF',
    'BrownKorringaDry2Sat', 'BrownKorringaSat2Dry', 'BrownKorringaSub',
    'MavkoJizba', 'CO2Properties', 'LiveOil', 'CO2Brine',
    # Empirical models
    'Han', 'RaymerHuntGardner', 'WyllieTimeAverage', 'CastagnaMudrock', 'GreenbergCastagna', 'Gardner', 'Krief',
    'Storvoll', 'Japsen', 'Hillis', 'Scherbaum', 'Ehrenberg', 'RammPorosity', 'Sclater', 'Hjelstuen', 'StPeter',
    'DensityModel',
    # Anisotropy models
    'Thomsen', 'Backus', 'BondTransform',
    'VelocityAzimuthHTI', 'VelocityAzimuthVTI', 'ThomsenTsvankin',
    # Resistivity models
    'ArchieResistivity',
    # Permeability models
    'KozenyCarman', 'KozenyCarmanPercolation', 'Owolabi', 'PermLogs',
    'PandaLake', 'PandaLakeCem', 'Revil', 'Fredrich', 'Bloch', 'Bernabe',
    # QI models
    'Screening', 'ConstantCementVelocity', 'CementEstimator', 'RPT',
    # Config & Workflow
    'RockPhysicsConfig', 'PRESETS', 'get_preset', 'list_presets',
    'RockPhysicsWorkflow',
]
