# Tutorial 12: Seismic Inversion (Deterministic)

```{note}
This tutorial is available as a Python script [`examples/12_seismic_inversion.py`](https://github.com/GeoBrain-Project/geobrain/blob/main/examples/12_seismic_inversion.py) and an interactive Jupyter notebook [`examples/notebooks/12_seismic_inversion.ipynb`](https://github.com/GeoBrain-Project/geobrain/blob/main/examples/notebooks/12_seismic_inversion.ipynb).
```

Recover elastic properties from seismic data using gradient-based inversion
with three parameterization strategies: Explicit, Network (Deep Image Prior),
and Latent.

## What You Will Learn

- Set up a rock-physics-based seismic forward model
- Run deterministic inversion with the `Inverter` class
- Compare three parameterization strategies:
  - **Explicit**: direct optimization of porosity values
  - **Network**: Deep Image Prior using a neural network
  - **Latent**: optimization in a learned latent space
- Evaluate inversion quality and convergence

## Key Concepts

**Parameterization** controls how the model space is represented during
inversion. Explicit parameterization optimizes physical properties directly.
Network parameterization uses a neural network as an implicit regularizer
(Deep Image Prior). Latent parameterization maps from a low-dimensional
learned space to the model domain, combining regularization with
dimensionality reduction.

Each strategy offers a different trade-off between flexibility and
regularization strength. The explicit approach is the most flexible but may
produce noisy results without additional constraints. The network approach
leverages the spectral bias of convolutional neural networks to implicitly
prefer smooth solutions. The latent approach restricts the solution to the
manifold learned by a pretrained autoencoder, providing the strongest
geological prior but the least flexibility.

## Code

```python
from geobrain.optim import Inverter, bound_constraint

inverter = Inverter(
    forward_fn=forward,
    initial_model=m0,
    constraints=bound_constraint(min_val=0.05, max_val=0.4),
)

result = inverter.run(
    observed_data=d_obs,
    max_epochs=500,
    lr=0.01,
    optimizer='adam',
)
```

## Results

The three parameterization strategies produce noticeably different porosity
estimates. The explicit approach recovers fine-scale detail but exhibits noise.
The network (DIP) approach yields a smoother result. The latent approach
constrains the solution to geologically plausible patterns.

```{figure} ../../examples/figs/12_porosity_comparison.png
:width: 100%
:name: fig-12-porosity-comparison

Porosity inversion comparison: Explicit vs. Network (DIP) vs. Latent.
```

Error maps highlight where each strategy deviates from the true model,
revealing the spatial distribution of inversion artifacts.

```{figure} ../../examples/figs/12_error_maps.png
:width: 100%
:name: fig-12-error-maps

Error maps for three parameterization strategies.
```

Convergence curves show the objective function decrease over iterations for
each approach. The latent parameterization typically converges fastest due to
its reduced dimensionality.

```{figure} ../../examples/figs/12_convergence.png
:width: 100%
:name: fig-12-convergence

Convergence curves for three inversion approaches.
```

## Full Example

See [`examples/12_seismic_inversion.py`](https://github.com/GeoBrain-Project/geobrain/blob/main/examples/12_seismic_inversion.py).
