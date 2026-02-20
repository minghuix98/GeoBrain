# Deterministic Inversion (`geobrain.optim`)

The `optim` module provides gradient-based optimization for solving geophysical
inverse problems.

## Inverter

The `Inverter` class is the main entry point for deterministic inversion:

```python
from geobrain import InverseProblem

# Define the inverse problem
problem = InverseProblem(
    forward_fn=forward_model,
    observed=observed_data,
    noise_std=0.01,
)

# Create an inverter
inverter = problem.create_inverter(initial_model=m0)

# Run inversion
result = inverter.run(
    problem.observed,
    max_epochs=500,
    lr=0.01,
    optimizer='adam',
)

# Access results
final_model = result.model
loss_curve = result.loss_history
print(result.summary())
```

## Parameterizations

Control how the model is represented during inversion:

### Explicit

Directly optimize physical parameters:

```python
from geobrain.optim import bound_constraint

inverter = problem.create_inverter(
    initial_model=m0,
    constraints=bound_constraint(min_val=0.05, max_val=0.4),
)
```

### Latent Space

Optimize in a reduced latent space (e.g., autoencoder bottleneck):

```python
inverter = problem.create_latent_inverter(
    decoder=decoder,
    initial_latent=z0,
)
```

### Neural Network

Use a neural network as the model generator (Deep Image Prior):

```python
inverter = problem.create_network_inverter(
    network=cnn_generator,
    fixed_input=z_fixed,
)
```

```{figure} ../../examples/figs/12_porosity_comparison.png
:width: 100%
:name: fig-porosity-comparison

Deterministic inversion: Explicit vs. Network (DIP) vs. Latent parameterizations.
```

## Loss Functions

Loss functions are plain functions, selectable by name or by reference:

```python
from geobrain.optim import mse_loss, nrmse_loss, l1_loss, huber_loss

loss = mse_loss(predictions, targets)
loss = nrmse_loss(predictions, targets)
loss = l1_loss(predictions, targets)
loss = huber_loss(predictions, targets, delta=1.0)

# Or select by name in inverter.run()
result = inverter.run(observed, loss_fn='mse')   # default
result = inverter.run(observed, loss_fn='huber')
```

## Constraints

```python
from geobrain.optim import bound_constraint, clip_constraint, sigmoid_constraint

# Hard constraint (clamp-based)
constraint = bound_constraint(min_val=1500, max_val=5000)

# Hard constraint (direct clamp, takes tensor as first arg)
clipped = clip_constraint(tensor, min_val=1500, max_val=5000)

# Soft constraint (differentiable sigmoid transform)
constraint = sigmoid_constraint(min_val=1500, max_val=5000)
```

## Regularization

Regularizers are plain functions:

```python
from geobrain.optim import l2_regularizer, l1_regularizer

reg_term = l2_regularizer(model_params)
reg_term = l1_regularizer(model_params)

# Use in inverter
inverter = Inverter.create(
    initial_model=m0,
    forward_fn=forward,
    regularizer=l2_regularizer,
)
result = inverter.run(observed, regularization_weight=0.01)
```

## Optimizers

| Optimizer | Description | Use Case |
|-----------|-------------|----------|
| Adam | Adaptive learning rate | General purpose (default) |
| L-BFGS | Quasi-Newton with Wolfe line search | Smooth objectives |
| Gauss-Newton | Second-order for least-squares | Linear/near-linear problems |

```python
# Adam (default)
result = inverter.run(observed, max_epochs=500, lr=0.01, optimizer='adam')

# L-BFGS (standalone optimizer)
from geobrain.optim import LBFGS
lbfgs = LBFGS(memory=10, max_iter=100)
result = lbfgs.minimize(fgrad_fn, x0)

# Gauss-Newton
from geobrain.optim import GaussNewton
gn = GaussNewton(forward_fn=fwd, jacobian_fn=jac,
                 obs_data=d, inv_Cd=torch.eye(N),
                 inv_Cm=torch.eye(M), x_prior=x0)
result = gn.solve(x0)
```

## InverseProblem Bridge

The `InverseProblem` class provides a unified interface that bridges
deterministic and Bayesian solvers:

```python
from geobrain import InverseProblem

problem = InverseProblem(forward_fn=forward, observed=data, noise_std=0.01)

# Deterministic
inverter = problem.create_inverter(initial_model=m0)

# Latent-space inversion
inverter = problem.create_latent_inverter(decoder=dec, initial_latent=z0)

# Network (Deep Image Prior)
inverter = problem.create_network_inverter(network=net, fixed_input=z)

# Bayesian
posterior = problem.as_posterior(log_prior=prior)
```

```{figure} ../../examples/figs/09_fwi_results.png
:width: 100%
:name: fig-fwi-results

Full Waveform Inversion result using the Inverter framework.
```
