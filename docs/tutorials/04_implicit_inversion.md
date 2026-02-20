# Tutorial 04: Implicit Model Inversion

```{note}
This tutorial is available as a Python script [`examples/04_implicit_inversion.py`](https://github.com/GeoBrain-Project/geobrain/blob/main/examples/04_implicit_inversion.py) and an interactive Jupyter notebook [`examples/notebooks/04_implicit_inversion.ipynb`](https://github.com/GeoBrain-Project/geobrain/blob/main/examples/notebooks/04_implicit_inversion.ipynb).
```

This tutorial demonstrates karst cave detection by inverting post-stack
seismic data through a differentiable implicit geological model.

## What You Will Learn

- Parameterize a cave boundary using surface point z-coordinates
- Chain implicit modeling with rock physics and seismic convolution
- Run deterministic inversion with Adam and Adam+L-BFGS optimizers
- Visualize impedance models, seismic sections, and convergence curves

## Key Concepts

The full forward chain is:

```
Cave boundary (z-coords)
    -> Implicit Model (soft lithology)
    -> Rock properties (Vp, rho)
    -> Acoustic impedance & reflectivity
    -> Wavelet convolution -> Synthetic seismic
```

Because every step is differentiable, PyTorch autograd computes gradients of
the seismic misfit with respect to the cave boundary coordinates, enabling
efficient gradient-based inversion.

## Code

```python
import torch
from geobrain.geomodel.implicit import (
    ImplicitModel, ImplicitModelConfig,
    SeriesDefinition, SurfacePointData, OrientationData,
)
from geobrain.physics.wave import RickerWavelet, create_conv_matrix, normal_incidence_rc
from geobrain.optim import Inverter, bound_constraint

def forward_fn(sp_z):
    coords = torch.stack([SP_X, sp_z], dim=-1)
    sp = SurfacePointData(coords=coords, surface_id=SP_IDS)
    ori = OrientationData(coords=ORI_COORDS, gradients=ORI_GRADS)
    model = ImplicitModel(CONFIG, series=[SeriesDefinition("cave", sp, ori)])
    block = model(soft=True, temperature=50.0)['block'].reshape(NX, NZ)
    # ... rock physics & convolution ...
    return synthetic_seismic

inverter = Inverter.create(
    initial_model=sp_z_init,
    forward_fn=forward_fn,
    constraints=bound_constraint(0.10, 0.90),
)
result = inverter.run(observed_data=seis_obs, max_epochs=500, lr=1e-2)
```

## Results

```{figure} ../../examples/figs/04_seismic.png
:width: 100%
:name: fig-04-seismic

Observed and predicted seismic sections.
```

```{figure} ../../examples/figs/04_impedance.png
:width: 100%
:name: fig-04-impedance

True vs. inverted acoustic impedance.
```

```{figure} ../../examples/figs/04_cave_boundary.png
:width: 100%
:name: fig-04-cave-boundary

True vs. inverted cave boundary.
```

```{figure} ../../examples/figs/04_convergence.png
:width: 100%
:name: fig-04-convergence

Inversion convergence curve.
```

## Full Example

See [`examples/04_implicit_inversion.py`](https://github.com/GeoBrain-Project/geobrain/blob/main/examples/04_implicit_inversion.py).
