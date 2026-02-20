"""
GeoBrain Optimization Module.

A framework for solving inverse problems using deterministic optimization
with multiple parameterization strategies.

Features:
    - Three parameterization strategies (Explicit, Latent, Network)
    - Multiple loss functions (MSE, NRMSE, L1, Huber)
    - Regularization (L2, L1)
    - Hard and soft constraints
    - Flexible optimizer support
    - L-BFGS with two-loop recursion and Wolfe line search
    - Gauss-Newton for nonlinear least squares
    - Strong Wolfe line search with cubic interpolation

Example:
    Basic usage:

    >>> from geobrain.optim import Inverter
    >>> inverter = Inverter.create(initial_model=model, forward_fn=forward)
    >>> result = inverter.run(observed_data, max_epochs=100)
    >>> print(result.summary())

    With regularization and constraints:

    >>> from geobrain.optim import Inverter, bound_constraint, l2_regularizer
    >>> inverter = Inverter.create(
    ...     initial_model=velocity,
    ...     forward_fn=forward,
    ...     regularizer=l2_regularizer,
    ...     constraints=bound_constraint(1500, 6000)
    ... )
    >>> result = inverter.run(data, regularization_weight=0.01)

    L-BFGS optimization:

    >>> from geobrain.optim import LBFGS
    >>> optimizer = LBFGS(memory=10, max_iter=100)
    >>> result = optimizer.minimize(fgrad_fn, x0)

    Gauss-Newton:

    >>> from geobrain.optim import GaussNewton
    >>> gn = GaussNewton(forward_fn=fwd, jacobian_fn=jac,
    ...     obs_data=d, inv_Cd=torch.eye(N), inv_Cm=torch.eye(M),
    ...     x_prior=x0)
    >>> result = gn.solve(x0)

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "Mingliang Liu"
__email__ = "mingliangliu@sdu.edu.cn"

# Core
from .inverter import Inverter
from .state import InversionState, InversionResult

# Parameterization strategies
from .parameterization import (
    Parameterization,
    ExplicitParameterization,
    LatentParameterization,
    NetworkParameterization,
)

# Loss functions
from .losses import (
    mse_loss,
    nrmse_loss,
    l1_loss,
    huber_loss,
    get_loss_fn,
)

# Regularizers
from .regularizers import (
    l2_regularizer,
    l1_regularizer,
    get_regularizer,
)

# Constraints
from .constraints import (
    # Hard constraints
    clip_constraint,
    positive_constraint,
    bound_constraint,
    # Soft constraints
    sigmoid_constraint,
    softplus_transform,
    exp_transform,
    tanh_transform,
    # Utilities
    compose_constraints,
)

# Line search
from .line_search import line_search_wolfe

# Optimizers
from .lbfgs import LBFGS, LBFGSResult
from .gauss_newton import GaussNewton, GaussNewtonResult


__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__email__',

    # Core
    'Inverter',
    'InversionState',
    'InversionResult',

    # Parameterization
    'Parameterization',
    'ExplicitParameterization',
    'LatentParameterization',
    'NetworkParameterization',

    # Loss functions
    'mse_loss',
    'nrmse_loss',
    'l1_loss',
    'huber_loss',
    'get_loss_fn',

    # Regularizers
    'l2_regularizer',
    'l1_regularizer',
    'get_regularizer',

    # Hard constraints
    'clip_constraint',
    'positive_constraint',
    'bound_constraint',

    # Soft constraints
    'sigmoid_constraint',
    'softplus_transform',
    'exp_transform',
    'tanh_transform',
    'compose_constraints',

    # Line search
    'line_search_wolfe',

    # Optimizers
    'LBFGS',
    'LBFGSResult',
    'GaussNewton',
    'GaussNewtonResult',
]
