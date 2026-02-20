# Tutorial 13: Bayesian AVO Inversion

```{note}
This tutorial is available as a Python script [`examples/13_bayes_avo_inversion.py`](https://github.com/GeoBrain-Project/geobrain/blob/main/examples/13_bayes_avo_inversion.py) and an interactive Jupyter notebook [`examples/notebooks/13_bayes_avo_inversion.ipynb`](https://github.com/GeoBrain-Project/geobrain/blob/main/examples/notebooks/13_bayes_avo_inversion.ipynb).
```

Perform Bayesian AVO inversion using all four samplers (SVGD, HMC, NUTS, LDS)
and learn the `InverseProblem` framework that bridges deterministic and
Bayesian solvers.

## What You Will Learn

- Use `InverseProblem` with `create_inverter()` for deterministic inversion
- Use `InverseProblem` with `as_posterior()` for Bayesian inference
- Set up AVO forward modeling through rock physics and Shuey reflectivity
- Run Bayesian inversion with four samplers: SVGD, HMC, NUTS, LDS
- Compare posterior mean, uncertainty, and convergence across samplers

## Key Concepts

**InverseProblem** provides a unified interface: define the forward model and
observed data once, then switch between deterministic optimization
(`create_inverter()`) and Bayesian sampling (`as_posterior()`).

**Bayesian AVO inversion** recovers a posterior distribution over elastic
properties from pre-stack seismic data, quantifying uncertainty in the
inversion result.

The tutorial proceeds in two parts. Part 1 demonstrates a deterministic
inversion using the `InverseProblem` framework to obtain a point estimate of
the elastic properties. Part 2 converts the same problem to a Bayesian
formulation and runs all four samplers, comparing the posterior distributions
and uncertainty estimates they produce.

## Code

```python
from geobrain.core import InverseProblem
from geobrain.bayes import Posterior, SVGD, HMC, NUTS, LDS

# InverseProblem framework
problem = InverseProblem(forward_fn=forward, observed=d_obs, noise_std=0.01)
inverter = problem.create_inverter(initial_model=m0)   # deterministic
posterior = problem.as_posterior(log_prior=log_prior)    # Bayesian

# Run samplers
svgd = SVGD(target=posterior, lr=1e-4)
result = svgd.run(n_samples=50, n_steps=500)
```

## Results

The deterministic inversion provides a point estimate of the elastic properties,
establishing a baseline before moving to Bayesian inference.

```{figure} ../../examples/figs/13_part1_demo.png
:width: 100%
:name: fig-13-part1-demo

Deterministic AVO inversion with InverseProblem framework.
```

The four Bayesian samplers -- SVGD, HMC, NUTS, and LDS -- each produce a
posterior distribution over elastic properties. Comparing them reveals
differences in posterior mean, credible intervals, and sampling efficiency.

```{figure} ../../examples/figs/13_bayesian_avo_comparison.png
:width: 100%
:name: fig-13-bayesian-avo-comparison

Bayesian AVO inversion: posterior comparison across four samplers.
```

## Full Example

See [`examples/13_bayes_avo_inversion.py`](https://github.com/GeoBrain-Project/geobrain/blob/main/examples/13_bayes_avo_inversion.py).
