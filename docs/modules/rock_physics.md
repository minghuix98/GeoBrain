# Rock Physics (`geobrain.physics.rock`)

The rock physics module transforms geological properties (porosity, mineralogy,
fluid saturation) into elastic parameters (Vp, Vs, density) for seismic
modeling. It provides 70+ differentiable PyTorch models organized by category.

## Effective Medium Models

### Voigt-Reuss-Hill (VRH) Averaging

Compute effective mineral moduli from constituent minerals:

```python
from geobrain.physics.rock import VRH
import torch

vrh = VRH()
vol_fractions = torch.tensor([0.9, 0.1])  # Quartz, clay
K_minerals = torch.tensor([36.6, 21.0])   # Bulk moduli (GPa)

K_voigt, K_reuss, K_hill = vrh(vol_fractions, K_minerals)
```

### Soft Sand Model

Granular rock model based on Hertz-Mindlin contact theory:

```python
from geobrain.physics.rock import SoftSand

soft_sand = SoftSand()
K_dry, G_dry = soft_sand(
    K_mineral=36.6,
    G_mineral=44.0,
    porosity=phi,
    critical_porosity=0.4,
    coordination_number=7,
    pressure=pressure,       # Effective pressure (MPa)
)
```

## Fluid Substitution

### Gassmann's Equation

Compute saturated rock moduli from dry-frame properties:

```python
from geobrain.physics.rock import Gassmann

gassmann = Gassmann()
K_sat, G_sat = gassmann(K_dry, G_dry, K_mineral, K_fluid, porosity)
```

## Density Model

```python
from geobrain.physics.rock import DensityModel

density = DensityModel()(porosity, rho_mineral, rho_fluid)
```

## Velocity Computation

Convert bulk and shear moduli to P- and S-wave velocities:

```python
from geobrain.physics.rock import v_from_moduli

vp, vs = v_from_moduli(K_sat, G_sat, density)
```

```{figure} ../../examples/figs/05_rock_physics_fields.png
:width: 100%
:name: fig-rock-physics-fields

Rock physics workflow: porosity to Vp, Vs, density fields.
```

## Rock Physics Workflow

Use presets for common scenarios:

```python
from geobrain.physics.rock import RockPhysicsWorkflow

workflow = RockPhysicsWorkflow.from_preset('shaly_sand')
vp, vs, rho = workflow(porosity, sw)
```

```{figure} ../../examples/figs/05_crossplot.png
:width: 100%
:name: fig-rock-physics-crossplot

Vp-porosity crossplot with rock physics model overlay.
```

## Mineral & Fluid Database

```python
from geobrain.physics.rock import MINERALS, FLUIDS, get_mineral, get_fluid

quartz = MINERALS['quartz']    # K, G, rho
brine = FLUIDS['brine']        # K, rho
```

## Available Models

| Category | Models |
|----------|--------|
| Effective medium | VRH, Hashin-Shtrikman, DEM, Self-Consistent, Critical Porosity, Hudson, Eshelby-Cheng |
| Granular media | Hertz-Mindlin, SoftSand, StiffSand, ContactCement, Walton, MUHS, PCM, Digby, Thomas-Stieber |
| Fluid substitution | Gassmann (forward & inverse), Wood, Brie, Batzle-Wang, Biot, CO<sub>2</sub> properties, LiveOil, CO<sub>2</sub>-Brine, Brown-Korringa |
| Empirical | Gardner, Han, Castagna, Krief, Raymer-Hunt, Storvoll, Japsen, Hillis, Scherbaum |
| Anisotropy | Thomsen (VTI/HTI), Backus averaging, Bond transform |
| Permeability | Kozeny-Carman, Owolabi, Panda-Lake, Revil, Fredrich, Bloch, Bernabe |
| Resistivity | Archie's law |
| Quantitative Interpretation | QI tools for crossplot analysis |

All models are differentiable PyTorch modules and can be composed into
end-to-end inversion workflows.
