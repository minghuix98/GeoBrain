# Bayesian Inference (`geobrain.bayes`)

The `bayes` module provides gradient-informed sampling methods for uncertainty
quantification in geophysical inverse problems.

## Overview

Rather than finding a single "best" model, Bayesian inference characterizes the
full posterior distribution of model parameters, providing uncertainty estimates
that are critical for decision-making.

GeoBrain provides four samplers, all supporting automatic differentiation
through the forward model:

| Sampler | Method | Best For |
|---------|--------|----------|
| `SVGD` | Stein Variational Gradient Descent | Multi-modal posteriors, large ensembles |
| `HMC` | Hamiltonian Monte Carlo | High-dimensional with tuned step size |
| `NUTS` | No-U-Turn Sampler | Auto-tuned HMC |
| `LDS` | Langevin Dynamics (ULA/MALA) | Simple gradient-based sampling |

## SVGD (Stein Variational Gradient Descent)

SVGD maintains an ensemble of particles that are iteratively transported to
approximate the posterior distribution:

```python
from geobrain import InverseProblem, SVGD, Posterior, Gaussian
import torch

# Define problem
problem = InverseProblem(
    forward_fn=forward,
    observed=data,
    noise_std=0.01,
)

# Define prior (Gaussian takes mean tensor and covariance matrix)
prior = Gaussian(mean=m_prior, cov=500.0**2 * torch.eye(n_params))

# Create posterior
posterior = problem.as_posterior(log_prior=prior.log_prob)

# Run SVGD
svgd = SVGD(target=posterior, lr=0.005)
result = svgd.run(
    n_samples=50,    # Number of particles
    n_steps=1000,    # Number of iterations
)

# Access results
mean_model = result.samples.mean(dim=0)   # Ensemble mean
std_model = result.samples.std(dim=0)     # Ensemble standard deviation
samples = result.samples                   # All particles [n_samples, dim]
print(result.summary())
```

```{figure} ../../examples/figs/13_bayesian_avo_comparison.png
:width: 100%
:name: fig-bayesian-avo-comparison

Bayesian AVO inversion: posterior mean and uncertainty from four samplers.
```

## Distributions

### Gaussian

Multivariate Gaussian parameterized by mean vector and covariance (or precision)
matrix:

```python
from geobrain import Gaussian
import torch

# Isotropic Gaussian
mean = torch.zeros(10)
cov = torch.eye(10)
prior = Gaussian(mean=mean, cov=cov)

# Or with precision matrix
prior = Gaussian(mean=mean, precision=torch.eye(10))

log_p = prior.log_prob(samples)   # [batch, 1]
score = prior.score(samples)      # [batch, dim]
drawn = prior.sample(100)         # [100, dim]
```

### Gaussian Mixture

```python
from geobrain import GaussianMixture
import torch

prior = GaussianMixture(
    means=[torch.tensor([2000.0]), torch.tensor([3000.0])],
    covs=[torch.tensor([[90000.0]]), torch.tensor([[160000.0]])],
    weights=torch.tensor([0.6, 0.4]),
)
```

### Custom Posterior

```python
from geobrain import Posterior

posterior = Posterior(
    log_likelihood=log_likelihood_fn,
    log_prior=log_prior_fn,
    data=d_obs,
    dim=n_params,
)
```

## Kernels

Kernels measure particle similarity for SVGD repulsion:

```python
from geobrain import RBFKernel, IMQKernel

kernel = RBFKernel()            # Radial Basis Function (default)
kernel = IMQKernel()            # Inverse Multi-Quadratic
```

## Hamiltonian Monte Carlo (HMC)

```python
from geobrain.bayes import HMC

hmc = HMC(
    target=posterior,
    step_size=1e-5,
    n_leapfrog=20,
    adapt_step_size=True,
    target_accept_rate=0.65,
)
result = hmc.run(
    n_samples=10,
    n_steps=300,
    initial_samples=init_samples,
    n_burnin=100,
)
```

## No-U-Turn Sampler (NUTS)

```python
from geobrain.bayes import NUTS

nuts = NUTS(
    target=posterior,
    target_accept_rate=0.65,
    max_tree_depth=10,
)
result = nuts.run(
    n_samples=10,
    n_steps=300,
    initial_samples=init_samples,
    n_burnin=100,
)
```

## Langevin Dynamics Sampler (LDS)

Supports both Unadjusted Langevin Algorithm (ULA) and Metropolis-Adjusted
Langevin Algorithm (MALA):

```python
from geobrain.bayes import LDS

lds = LDS(
    target=posterior,
    step_size=1e-5,
    use_mh_correction=True,      # MALA (False for ULA)
    adapt_step_size=True,
    target_accept_rate=0.574,
)
result = lds.run(
    n_samples=50,
    n_steps=500,
    initial_samples=init_samples,
    n_burnin=200,
)
```

## SamplingResult

All samplers return a `SamplingResult` dataclass:

```python
result.samples              # Final samples [n_samples, dim]
result.log_prob_history     # Mean log prob per step
result.algorithm            # Algorithm name string
result.n_steps              # Steps completed
result.total_time           # Wall time (seconds)
result.summary()            # Formatted text summary
result.to_dict()            # Serializable dictionary
result.to_numpy()           # Samples as numpy array
```

```{figure} ../../examples/figs/14_posterior_stats.png
:width: 100%
:name: fig-posterior-stats

IBDP case study: posterior mean and standard deviation of porosity.
```

## Utility Functions

```python
from geobrain.bayes import pairwise_distances, median_heuristic, mmd, energy_distance

# Compute MMD between two sample sets
distance = mmd(samples_p, samples_q)
```
