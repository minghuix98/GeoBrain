"""
Shared inverse problem definition.

Bridges deterministic optimization (geobrain.optim) and
probabilistic inference (geobrain.bayes) by providing a single
problem definition that both can consume.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
from typing import Callable, Optional, Tuple


class InverseProblem:
    """
    Defines an inverse problem: given observations, infer model parameters.

    Encapsulates the forward model, observed data, and noise model into
    a single object that can drive both deterministic inversion and
    Bayesian inference.

    The bridge between the two approaches is the noise model.
    With Gaussian noise (std = sigma), the log-likelihood is::

        log p(D|theta) = -1/(2*sigma^2) * ||F(theta) - D||^2 + const

    which is proportional to the negative MSE loss. This means the same
    ``forward_fn`` and ``observed`` can be used with both ``optim`` and
    ``bayes`` without redefining the misfit.

    Args:
        forward_fn: Forward model mapping parameters to predictions.
                   Signature: (model: Tensor) -> Tensor
        observed: Observed data tensor.
        noise_std: Noise standard deviation. Controls the relative
                  weighting between data fit and prior in Bayesian mode.

    Example:
        Define a problem once, solve two ways::

            problem = InverseProblem(
                forward_fn=acoustic_forward,
                observed=seismic_data,
                noise_std=0.05,
            )

            # --- Deterministic inversion ---
            inverter = problem.create_inverter(initial_model=v0)
            result = inverter.run(problem.observed, max_epochs=100)

            # --- Bayesian inference ---
            posterior = problem.as_posterior(log_prior=my_prior)
            svgd = SVGD(target=posterior)
            samples = svgd.run(n_samples=200, n_steps=1000)
    """

    def __init__(
        self,
        forward_fn: Callable[[torch.Tensor], torch.Tensor],
        observed: torch.Tensor,
        noise_std: float = 1.0,
    ):
        self.forward_fn = forward_fn
        self.observed = (
            observed if isinstance(observed, torch.Tensor)
            else torch.as_tensor(observed, dtype=torch.float32)
        )
        self.noise_std = noise_std

    # -----------------------------------------------------------------
    # Bayesian interface
    # -----------------------------------------------------------------

    def log_likelihood(
        self,
        theta: torch.Tensor,
        data: Optional[torch.Tensor] = None,
        model_shape: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """
        Gaussian log-likelihood for a batch of parameter vectors.

        .. math::

            \\log p(D|\\theta) = -\\frac{1}{2\\sigma^2}
            \\|F(\\theta) - D\\|^2

        Args:
            theta: Parameter vectors of shape ``[batch_size, dim]``.
            data: Optional data override (uses ``self.observed`` if None).
            model_shape: If ``forward_fn`` expects shaped input, reshape
                        each vector to this shape before calling.

        Returns:
            Log-likelihoods of shape ``[batch_size]``.
        """
        obs = data if data is not None else self.observed
        obs_flat = obs.reshape(-1)

        batch_size = theta.shape[0]
        log_liks = []
        for i in range(batch_size):
            model_i = theta[i]
            if model_shape is not None:
                model_i = model_i.reshape(model_shape)
            pred = self.forward_fn(model_i)
            residual = pred.reshape(-1) - obs_flat
            ll = -0.5 * (residual / self.noise_std).pow(2).sum()
            log_liks.append(ll)
        return torch.stack(log_liks)

    def as_posterior(
        self,
        log_prior: Optional[Callable] = None,
        dim: Optional[int] = None,
        model_shape: Optional[Tuple[int, ...]] = None,
    ):
        """
        Create a :class:`~geobrain.bayes.Posterior` for Bayesian sampling.

        Args:
            log_prior: Log-prior function ``(theta) -> Tensor``.
                      If None, uses an improper uniform prior.
            dim: Parameter dimension. If None, inferred from observed data.
            model_shape: Shape to reshape each particle before calling
                        ``forward_fn``. Required when ``forward_fn``
                        expects multi-dimensional input (e.g. a 2-D
                        velocity field).

        Returns:
            Posterior distribution compatible with SVGD and other samplers.

        Example::

            def log_prior(theta):
                return -0.5 * (theta ** 2).sum(dim=-1)

            posterior = problem.as_posterior(log_prior=log_prior)
            svgd = SVGD(target=posterior)
            result = svgd.run(n_samples=200, n_steps=1000)
        """
        from geobrain.bayes import Posterior

        # Closure that captures model_shape
        def _log_likelihood(theta, data=None):
            return self.log_likelihood(
                theta, data=data, model_shape=model_shape,
            )

        return Posterior(
            log_likelihood=_log_likelihood,
            log_prior=log_prior,
            data=self.observed,
            dim=dim,
        )

    # -----------------------------------------------------------------
    # Deterministic interface
    # -----------------------------------------------------------------

    def create_inverter(self, initial_model: torch.Tensor, **kwargs):
        """
        Create an :class:`~geobrain.optim.Inverter` with explicit
        parameterization.

        Args:
            initial_model: Initial model parameters.
            **kwargs: Passed to :meth:`Inverter.create`
                     (regularizer, transform, constraints, device).

        Returns:
            Configured Inverter instance.

        Example::

            inverter = problem.create_inverter(
                initial_model=v0,
                regularizer='l2',
                constraints=bound_constraint(1500, 6000),
            )
            result = inverter.run(problem.observed, max_epochs=100)
        """
        from geobrain.optim import Inverter
        return Inverter.create(
            initial_model=initial_model,
            forward_fn=self.forward_fn,
            **kwargs,
        )

    def create_latent_inverter(
        self, decoder, initial_latent: torch.Tensor, **kwargs,
    ):
        """
        Create an Inverter with latent parameterization.

        Args:
            decoder: Pre-trained decoder network.
            initial_latent: Initial latent code.
            **kwargs: Passed to :meth:`Inverter.create_latent`.

        Returns:
            Configured Inverter instance.
        """
        from geobrain.optim import Inverter
        return Inverter.create_latent(
            decoder=decoder,
            initial_latent=initial_latent,
            forward_fn=self.forward_fn,
            **kwargs,
        )

    def create_network_inverter(
        self, network, fixed_input: torch.Tensor, **kwargs,
    ):
        """
        Create an Inverter with network parameterization (Deep Image Prior).

        Args:
            network: Neural network to optimize.
            fixed_input: Fixed input tensor.
            **kwargs: Passed to :meth:`Inverter.create_network`.

        Returns:
            Configured Inverter instance.
        """
        from geobrain.optim import Inverter
        return Inverter.create_network(
            network=network,
            fixed_input=fixed_input,
            forward_fn=self.forward_fn,
            **kwargs,
        )

    def __repr__(self) -> str:
        obs_shape = tuple(self.observed.shape)
        return (
            f"InverseProblem(observed={obs_shape}, "
            f"noise_std={self.noise_std})"
        )
