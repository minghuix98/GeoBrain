# Tutorial 14: IBDP Bayesian Inversion with SVGD

```{note}
This tutorial is available as a Python script [`examples/14_ibdp_inversion_svgd.py`](https://github.com/GeoBrain-Project/geobrain/blob/main/examples/14_ibdp_inversion_svgd.py) and an interactive Jupyter notebook [`examples/notebooks/14_ibdp_inversion_svgd.ipynb`](https://github.com/GeoBrain-Project/geobrain/blob/main/examples/notebooks/14_ibdp_inversion_svgd.ipynb).
```

Perform Bayesian seismic inversion using Stein Variational Gradient Descent
(SVGD) in a latent space defined by a convolutional autoencoder, demonstrated
on the Illinois Basin &ndash; Decatur Project (IBDP) CO&#8322; storage site.

## What You Will Learn

- Load seismic data, prior models, and mineral properties for IBDP
- Use a pretrained convolutional autoencoder to define a latent space
- Build a posterior from `InverseProblem` and a log-prior
- Run SVGD sampling in latent space for Bayesian inference
- Decode posterior samples and visualize porosity and clay volume uncertainty

## Key Concepts

**Latent-space SVGD** performs Bayesian inference in the low-dimensional
space of a pretrained autoencoder. This reduces the dimensionality of the
inverse problem while preserving geological realism encoded by the network.

**SVGD** maintains a set of particles that are iteratively transported
toward the posterior distribution, balancing exploitation (log-posterior
gradient) and exploration (repulsive kernel force).

The IBDP dataset provides a real-world test case where porosity and clay
volume must be estimated from post-stack seismic data. The pretrained
autoencoder compresses the 2D model space into a compact latent
representation. SVGD then explores the posterior in this latent space,
producing an ensemble of geologically plausible realizations that honor both
the seismic observations and the prior geological knowledge.

## Code

```python
from geobrain.core import InverseProblem
from geobrain.bayes import SVGD

problem = InverseProblem(forward_fn=forward, observed=d_obs, noise_std=sigma)
posterior = problem.as_posterior(log_prior=prior.log_prob)

svgd = SVGD(target=posterior, lr=0.005)
result = svgd.run(n_samples=50, n_steps=1000)

# Decode latent samples to model space
models = decoder(result.samples)
```

## Results

The SVGD ensemble provides multiple posterior realizations of porosity and
clay volume, each representing a plausible subsurface model consistent with
the observed seismic data and prior information.

```{figure} ../../examples/figs/14_realizations.png
:width: 100%
:name: fig-14-realizations

SVGD posterior realizations of porosity and clay volume.
```

Summary statistics -- posterior mean and standard deviation -- quantify
the most likely model and its uncertainty. Regions with higher standard
deviation indicate areas where the data provide less constraint.

```{figure} ../../examples/figs/14_posterior_stats.png
:width: 100%
:name: fig-14-posterior-stats

Posterior statistics: mean and standard deviation.
```

Convergence diagnostics track the log-posterior and data misfit over SVGD
iterations, confirming that the particle ensemble has stabilized.

```{figure} ../../examples/figs/14_convergence.png
:width: 100%
:name: fig-14-convergence

SVGD convergence: log-posterior and data misfit.
```

## Full Example

See [`examples/14_ibdp_inversion_svgd.py`](https://github.com/GeoBrain-Project/geobrain/blob/main/examples/14_ibdp_inversion_svgd.py).
