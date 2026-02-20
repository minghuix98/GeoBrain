# Tutorial 02: Advanced Geomodeling

```{note}
This tutorial is available as a Python script [`examples/02_geomodel_cosim.py`](https://github.com/GeoBrain-Project/geobrain/blob/main/examples/02_geomodel_cosim.py) and an interactive Jupyter notebook [`examples/notebooks/02_geomodel_cosim.ipynb`](https://github.com/GeoBrain-Project/geobrain/blob/main/examples/notebooks/02_geomodel_cosim.ipynb).
```

This tutorial covers advanced geostatistical techniques: multi-variable
co-simulation, Sequential Gaussian Simulation (SGS), and variogram modeling.

## What You Will Learn

- Use `CoSimConfig` to generate correlated multi-variable fields
- Run unconditional SGS with Simple Kriging and Ordinary Kriging
- Perform conditional SGS that honors well data
- Build and visualize variogram models with `VariogramModel`

## Key Concepts

**Co-simulation** produces multiple spatially correlated fields that respect
inter-variable correlations. **SGS** visits grid nodes in random order, using
kriging to estimate the local mean and variance, then drawing from the
conditional distribution. **Variograms** describe how spatial correlation
decays with distance.

## Code

```python
import torch
from geobrain.geomodel import Simulator, CoSimConfig, SimulationConfig

# Co-simulation
corr = torch.tensor([[1.0, 0.7], [0.7, 1.0]])
config = CoSimConfig(
    shape=(64, 64, 128),
    n_variables=2,
    correlation_matrix=corr,
    field_names=["phi", "vsand"],
    lh=20, lv=5,
    seed=2025,
)
sim = Simulator.create("fft_ma")
fields = sim.simulate(config)

# SGS with conditioning
sim_sgs = Simulator.create("sgs", kriging_type="ok")
sim_sgs.set_conditioning(well_values, well_locations)
fields_cond = sim_sgs.simulate(sgs_config)
```

## Results

```{figure} ../../examples/figs/02_variogram_models.png
:width: 100%
:name: fig-02-variogram-models

Composite variogram model fitting.
```

```{figure} ../../examples/figs/02_sgs_unconditional.png
:width: 100%
:name: fig-02-sgs-unconditional

SGS unconditional simulation with Simple and Ordinary Kriging.
```

```{figure} ../../examples/figs/02_sgs_conditional.png
:width: 100%
:name: fig-02-sgs-conditional

SGS conditional simulation honoring well data.
```

```{figure} ../../examples/figs/02_cosim_properties.png
:width: 100%
:name: fig-02-cosim-properties

Co-simulated porosity and sand volume fields.
```

## Full Example

See [`examples/02_geomodel_cosim.py`](https://github.com/GeoBrain-Project/geobrain/blob/main/examples/02_geomodel_cosim.py).
