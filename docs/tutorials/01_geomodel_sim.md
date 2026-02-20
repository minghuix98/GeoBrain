# Tutorial 01: Geostatistical Simulation

```{note}
This tutorial is available as a Python script [`examples/01_geomodel_sim.py`](https://github.com/GeoBrain-Project/geobrain/blob/main/examples/01_geomodel_sim.py) and an interactive Jupyter notebook [`examples/notebooks/01_geomodel_sim.ipynb`](https://github.com/GeoBrain-Project/geobrain/blob/main/examples/notebooks/01_geomodel_sim.ipynb).
```

This tutorial demonstrates how to generate 3D geological property fields
using the FFT-based Moving Average (FFT-MA) method.

## What You Will Learn

- Create simulation configurations with `SimulationConfig`
- Generate Gaussian random fields using `Simulator.create("fft_ma")`
- Transform Gaussian fields to physical property ranges (e.g., porosity)
- Generate multiple realizations for uncertainty quantification
- Compare correlation models (spherical, exponential, Gaussian)

## Key Concepts

**FFT-MA** is a spectral-domain simulation method that generates spatially
correlated random fields efficiently using Fast Fourier Transforms. It is
orders of magnitude faster than sequential methods for large grids.

## Code

```python
from geobrain.geomodel import Simulator, SimulationConfig

config = SimulationConfig(
    shape=(64, 64, 64),
    lh=20.0,
    lv=5.0,
    mean=0.0,
    std=1.0,
    seed=42,
)

sim = Simulator.create("fft_ma")
field = sim.simulate(config)
```

## Results

```{figure} ../../examples/figs/01_gaussian_field.png
:width: 100%
:name: fig-01-gaussian-field

Gaussian random field (Z-slice at z=32).
```

```{figure} ../../examples/figs/01_porosity_field.png
:width: 100%
:name: fig-01-porosity-field

Porosity field mapped to physical range [0.05, 0.40].
```

```{figure} ../../examples/figs/01_realizations.png
:width: 100%
:name: fig-01-realizations

Multiple realizations for uncertainty quantification.
```

```{figure} ../../examples/figs/01_correlation_comparison.png
:width: 100%
:name: fig-01-correlation-comparison

Correlation model comparison: spherical, exponential, Gaussian.
```

## Full Example

See [`examples/01_geomodel_sim.py`](https://github.com/GeoBrain-Project/geobrain/blob/main/examples/01_geomodel_sim.py).
