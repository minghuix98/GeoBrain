#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rock physics models.

This package provides a comprehensive collection of rock physics models
for computing elastic properties of rocks and sediments.

Modules:
    - effective: Bounds and effective medium theories
    - granular: Contact theories and sand models
    - fluid: Fluid substitution and mixing
    - empirical: Velocity-porosity and compaction trends
    - anisotropy: VTI media and layer averaging
    - resistivity: Electrical property models
    - permeability: Permeability estimation models

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

from .effective import (
    Voigt, Reuss, VRH, HashinShtrikman, DEM, SelfConsistent, CriticalPorosity,
    Hudson, HudsonRandom, KusterToksoz, EshelbyCheng,
    SwissCheese, DiluteCrack, OConnellBudiansky, OConnellBudianskyFl,
    SCDilute, SCFlex, MTAverage, PQ, HudsonOrtho, HudsonCone,
)

from .granular import (
    HertzMindlin, SoftSand, StiffSand, ContactCement, ContactCementFull,
    Walton, MUHS, PCM,
    ThomasStieber, SiltyShale, ShalySand,
    Digby, ConstantCement, VPCM, Diluting,
)

from .fluid import (
    Gassmann, GassmannInverse, GassmannFluidSub, Wood, Brie, BatzleWang,
    BiotHF, BiotDispersion, GeertsmaSmitHF, GeertsmaSmitLF,
    BrownKorringaDry2Sat, BrownKorringaSat2Dry, BrownKorringaSub,
    MavkoJizba, CO2Properties, LiveOil, CO2Brine,
)

from .empirical import (
    Han, RaymerHuntGardner, WyllieTimeAverage, CastagnaMudrock, GreenbergCastagna, Gardner, Krief,
    Storvoll, Japsen, Hillis, Scherbaum, Ehrenberg, RammPorosity, Sclater, Hjelstuen, StPeter,
    DensityModel,
)

from .anisotropy import (
    Thomsen, Backus, BondTransform,
    VelocityAzimuthHTI, VelocityAzimuthVTI, ThomsenTsvankin,
)

from .resistivity import ArchieResistivity

from .permeability import (
    KozenyCarman, KozenyCarmanPercolation, Owolabi, PermLogs,
    PandaLake, PandaLakeCem, Revil, Fredrich, Bloch, Bernabe,
)

__all__ = [
    # Effective medium models
    'Voigt', 'Reuss', 'VRH', 'HashinShtrikman', 'DEM', 'SelfConsistent', 'CriticalPorosity',
    'Hudson', 'HudsonRandom', 'KusterToksoz', 'EshelbyCheng',
    'SwissCheese', 'DiluteCrack', 'OConnellBudiansky', 'OConnellBudianskyFl',
    'SCDilute', 'SCFlex', 'MTAverage', 'PQ', 'HudsonOrtho', 'HudsonCone',
    # Granular models
    'HertzMindlin', 'SoftSand', 'StiffSand', 'ContactCement', 'ContactCementFull',
    'Walton', 'MUHS', 'PCM', 'ThomasStieber', 'SiltyShale', 'ShalySand',
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
]
