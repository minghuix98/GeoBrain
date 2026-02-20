# Architecture & Design

## Design Principles

GeoBrain is built on three foundational principles:

### End-to-End Differentiability

All forward models are implemented as PyTorch `nn.Module` subclasses, enabling
automatic differentiation through complex computational graphs. This eliminates
hand-coded Jacobians and enables gradient-based optimization for inverse
problems.

```python
# Gradients flow through the entire chain
velocity = rock_physics_model(elastic_moduli)      # Differentiable
seismic_data = wave_propagator(velocity)            # Differentiable
loss = mse_loss(seismic_data, observed_data)
loss.backward()  # Automatic gradient computation
```

### Modular Composition

Each physics module is a self-contained unit that can be used standalone or
composed with other modules. Modules communicate through standard tensor
interfaces:

```
geomodel  -->  rock_physics  -->  wave_propagation  -->  observed_data
   |               |                    |                      |
   v               v                    v                      v
porosity      velocities          seismic traces          data misfit
```

```{figure} ../examples/figs/05_rock_physics_fields.png
:name: arch-modular-composition
:alt: Modular composition: geostatistics to rock physics to elastic velocities

Modular composition: geostatistics to rock physics to elastic velocities.
```

### Bayesian-Deterministic Duality

The same forward model can be used for both deterministic inversion
(`geobrain.optim`) and Bayesian inference (`geobrain.bayes`) without
modification. The `InverseProblem` class provides a unified entry point:

```python
from geobrain import InverseProblem, SVGD

problem = InverseProblem(forward_fn=forward, observed=data, noise_std=0.01)

# Deterministic
inverter = problem.create_inverter(initial_model=m0)

# Bayesian (4 samplers: SVGD, HMC, NUTS, LDS)
posterior = problem.as_posterior(log_prior=prior)
svgd = SVGD(target=posterior)
```

## Module Organization

```
geobrain/
├── core/              # InverseProblem, registry system
├── bayes/             # Bayesian samplers and distributions
│   └── samplers/      # SVGD, HMC, NUTS, LDS (Langevin)
├── optim/             # Deterministic inversion engine
├── geomodel/          # Geological model generation
│   ├── geostats/      # FFT-MA, SGS, variograms
│   └── geogen/        # GAN, VAE, Diffusion generators
├── physics/
│   ├── rock/          # Effective medium, Gassmann, empirical
│   ├── wave/          # Acoustic & elastic propagation, AVO
│   │   ├── avo/       # Zoeppritz, Aki-Richards, Shuey
│   │   ├── boundary/  # PML, Gerjan, SinCos
│   │   ├── wavelets/  # Ricker, Gaussian, Ormsby, Klauder
│   │   ├── models/    # Acoustic & elastic velocity models
│   │   ├── propagators/ # FDTD solvers
│   │   └── survey/    # Source, Receiver, SeismicData
│   └── flow/          # Reservoir simulation
├── nn/                # Neural network layers & activations
├── data/              # Dataset loaders and transforms
├── io/                # SEG-Y, LAS, reservoir grid I/O
├── vis/               # Visualization utilities
└── decision/          # Decision-making under uncertainty
```

## Data Flow

A typical GeoBrain workflow follows this pattern:

1. **Model generation** &mdash; Create geological properties (porosity, facies)
   using geostatistics or generative models.
2. **Rock physics** &mdash; Transform geological properties into elastic
   parameters (Vp, Vs, density) via rock physics models.
3. **Forward simulation** &mdash; Simulate observed geophysical data (seismic,
   resistivity, production) from the elastic model.
4. **Inversion** &mdash; Recover subsurface properties from observed data using
   deterministic optimization or Bayesian sampling.
5. **Decision** &mdash; Use the posterior ensemble for uncertainty-aware
   decision-making.

## GPU & Memory

- All modules are GPU-compatible via PyTorch device management.
- Large-scale wave simulations use **gradient checkpointing** to trade compute
  for memory.
- 3D acoustic propagation supports domain-parallel execution.

## Extensibility

New physics modules are registered automatically via the `RegistryHub`:

```python
from geobrain.core import registries

# Register a custom forward model
@registries.physics.register("my_model")
class MyModel(nn.Module):
    def forward(self, params):
        ...
```

This allows new modules to be discovered and composed without modifying
framework code.
