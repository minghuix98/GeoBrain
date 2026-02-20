# Tutorial 05: Rock Physics Modeling

```{note}
This tutorial is available as a Python script [`examples/05_rock_physics.py`](https://github.com/GeoBrain-Project/geobrain/blob/main/examples/05_rock_physics.py) and an interactive Jupyter notebook [`examples/notebooks/05_rock_physics.ipynb`](https://github.com/GeoBrain-Project/geobrain/blob/main/examples/notebooks/05_rock_physics.ipynb).
```

This tutorial walks through a complete rock physics workflow: mineral mixing,
dry-frame modeling, fluid substitution, and velocity computation.

## What You Will Learn

- Compute effective mineral moduli with `VRH`
- Model the dry rock frame with `SoftSand`
- Perform fluid substitution using `Gassmann`
- Compute Vp and Vs from elastic moduli

## Workflow

```
Minerals (K, G, rho)
       |
       v
   VRH averaging  -->  K_m, G_m
       |
       v
   SoftSand model  -->  K_dry, G_dry
       |
       v
   Gassmann  -->  K_sat, G_sat
       |
       v
   v_from_moduli  -->  Vp, Vs
```

## Code

```python
from geobrain.physics.rock import VRH, SoftSand, Gassmann, DensityModel, v_from_moduli

vrh = VRH()
_, _, K_m = vrh(vol_fractions, K_minerals)
_, _, G_m = vrh(vol_fractions, G_minerals)

k_dry, g_dry = SoftSand()(K_m, G_m, porosity, 0.4, 7, pressure)
k_sat, g_sat = Gassmann()(k_dry, g_dry, K_m, K_fluid, porosity)
rho = DensityModel()(porosity, rho_mineral, rho_fluid)
vp, vs = v_from_moduli(k_sat, g_sat, rho)
```

## Results

```{figure} ../../examples/figs/05_rock_physics_fields.png
:width: 100%
:name: fig-05-rock-physics-fields

Rock physics workflow: porosity to Vp, Vs, and density.
```

```{figure} ../../examples/figs/05_crossplot.png
:width: 100%
:name: fig-05-crossplot

Vp-porosity crossplot with rock physics model predictions.
```

## Full Example

See [`examples/05_rock_physics.py`](https://github.com/GeoBrain-Project/geobrain/blob/main/examples/05_rock_physics.py).
