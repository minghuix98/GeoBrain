# Tutorial 03: Implicit Geological Modeling

```{note}
This tutorial is available as a Python script [`examples/03_implicit_modeling.py`](https://github.com/GeoBrain-Project/geobrain/blob/main/examples/03_implicit_modeling.py) and an interactive Jupyter notebook [`examples/notebooks/03_implicit_modeling.ipynb`](https://github.com/GeoBrain-Project/geobrain/blob/main/examples/notebooks/03_implicit_modeling.ipynb).
```

This tutorial demonstrates how to build differentiable implicit geological
models using Universal Cokriging (Lajaunie et al. 1997).

## What You Will Learn

- Define surface contact points and orientation data
- Build single-series models with 1 or 2 surfaces
- Stack multiple series with ERODE relations
- Model fault displacement using `FaultDefinition`
- Verify end-to-end differentiability with PyTorch autograd

## Key Concepts

**Implicit modeling** represents geological interfaces as iso-surfaces of a
scalar field interpolated from sparse contact points and orientation
measurements. The Cokriging formulation jointly honors both data types and
supports polynomial drift for regional trends.

**Multi-series stacking** combines independent interpolations: ERODE lets
younger series overwrite older ones, while ONLAP fills only unoccupied regions.

**Differentiability** means gradients flow from any loss function back through
the block model, scalar field, and Cokriging system to the input data —
enabling gradient-based inversion of geological parameters.

## Code

```python
import torch
from geobrain.geomodel.implicit import (
    ImplicitModel, ImplicitModelConfig,
    SeriesDefinition, SurfacePointData, OrientationData,
)

config = ImplicitModelConfig(
    extent=(0.0, 1.0, 0.0, 1.0),
    resolution=(80, 80),
    dtype="float64",
)

sp = SurfacePointData(
    coords=torch.tensor([[0.2, 0.55], [0.5, 0.50], [0.8, 0.45]],
                         dtype=torch.float64),
    surface_id=torch.tensor([0, 0, 0]),
)
ori = OrientationData(
    coords=torch.tensor([[0.35, 0.52], [0.65, 0.48]], dtype=torch.float64),
    gradients=torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float64),
)

model = ImplicitModel(config, series=[SeriesDefinition("strata", sp, ori)])
result = model(soft=True, temperature=50.0)
block = result['block']          # (6400,) soft lithology ids
sf = result['scalar_fields'][0]  # (6400,) scalar field
```

## Results

```{figure} ../../examples/figs/03_two_layer.png
:width: 100%
:name: fig-03-two-layer

Two-layer implicit model from surface points and orientations.
```

```{figure} ../../examples/figs/03_three_layer.png
:width: 100%
:name: fig-03-three-layer

Three-layer model with two surfaces.
```

```{figure} ../../examples/figs/03_multi_series.png
:width: 100%
:name: fig-03-multi-series

Multi-series stacking with erosion.
```

```{figure} ../../examples/figs/03_fault.png
:width: 100%
:name: fig-03-fault

Fault displacement modeling.
```

```{figure} ../../examples/figs/03_differentiable.png
:width: 100%
:name: fig-03-differentiable

Gradient verification: differentiable implicit modeling.
```

## Full Example

See [`examples/03_implicit_modeling.py`](https://github.com/GeoBrain-Project/geobrain/blob/main/examples/03_implicit_modeling.py).
