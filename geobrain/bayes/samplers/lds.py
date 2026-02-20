"""
Langevin Dynamics Sampler (LDS).

Implements ULA (Unadjusted Langevin Algorithm) and MALA
(Metropolis-Adjusted Langevin Algorithm) for sampling from
target distributions.

Reference:
    Roberts & Tweedie (1996). Exponential convergence of
    Langevin distributions and their discrete approximations.
    Bernoulli.

    Welling & Teh (2011). Bayesian Learning via Stochastic
    Gradient Langevin Dynamics. ICML.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import logging
import math
import time
from typing import Optional, List, Dict, Tuple

import torch

from ..core import Distribution, BaseSampler
from ..sampling import SamplingResult

logger = logging.getLogger(__name__)


class LDS(BaseSampler):
    """
    Langevin Dynamics Sampler.

    Uses the overdamped Langevin SDE to sample:

        dx = grad_log_p(x) * dt + sqrt(2 * dt) * dW

    where dW is a Wiener process (Brownian motion).

    Variants:
        - ULA: Unadjusted Langevin Algorithm (no MH correction).
            Asymptotically biased but fast.
        - MALA: Metropolis-Adjusted Langevin Algorithm (with MH
            correction). Asymptotically exact.

    Args:
        target: Target distribution.
        step_size: Discretization step size (epsilon).
        use_mh_correction: If True, use MALA; if False, use ULA.
            Default: False (ULA).
        adapt_step_size: Whether to adapt step size. Default: True.
        target_accept_rate: Target acceptance rate for MALA
            adaptation. Default: 0.574 (optimal for Gaussian).
        n_adapt: Number of adaptation steps. Default: 100.
        device: Computation device.

    Example:
        ULA:
        >>> lds = LDS(target=target, step_size=0.01)
        >>> result = lds.run(n_samples=100, n_steps=1000)

        MALA:
        >>> lds = LDS(target=target, step_size=0.01, use_mh_correction=True)
        >>> result = lds.run(n_samples=100, n_steps=1000)
    """

    def __init__(
        self,
        target: Optional[Distribution] = None,
        step_size: float = 0.01,
        use_mh_correction: bool = False,
        adapt_step_size: bool = True,
        target_accept_rate: float = 0.574,
        n_adapt: int = 100,
        device: Optional[torch.device] = None,
    ):
        super().__init__(target, device)
        self.step_size = step_size
        self.use_mh_correction = use_mh_correction
        self.adapt_step_size = adapt_step_size
        self.target_accept_rate = target_accept_rate
        self.n_adapt = n_adapt

    def _log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log prob for a batch of samples [n, d] -> [n, 1]."""
        return self.target.log_prob(x)

    def _score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute score for a batch of samples [n, d] -> [n, d]."""
        return self.target.score(x)

    def _ula_step(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Unadjusted Langevin Algorithm (ULA) step.

        x_{n+1} = x_n + eps * grad_log_p(x_n) + sqrt(2*eps) * z

        where z ~ N(0, I).

        Args:
            x: Current positions [n_samples, dim].

        Returns:
            New positions [n_samples, dim].
        """
        eps = self.step_size
        grad = self._score(x)
        noise = torch.randn_like(x)
        return x + eps * grad + math.sqrt(2.0 * eps) * noise

    def _mala_step(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Metropolis-Adjusted Langevin Algorithm (MALA) step.

        Proposes using ULA, then accepts/rejects per-sample with
        Metropolis-Hastings correction.

        Args:
            x: Current positions [n_samples, dim].

        Returns:
            Tuple of (new_positions, accepted_mask).
        """
        eps = self.step_size
        n_samples, dim = x.shape

        # Score at current position
        grad_x = self._score(x)

        # Proposal
        noise = torch.randn_like(x)
        x_prop = x + eps * grad_x + math.sqrt(2.0 * eps) * noise

        # Score at proposed position
        grad_prop = self._score(x_prop)

        # Log proposal density q(x_prop | x)
        # q(y|x) = N(y; x + eps*score(x), 2*eps*I)
        diff_forward = x_prop - x - eps * grad_x
        log_q_forward = -0.25 / eps * (diff_forward ** 2).sum(dim=-1)

        # Log reverse proposal density q(x | x_prop)
        diff_reverse = x - x_prop - eps * grad_prop
        log_q_reverse = -0.25 / eps * (diff_reverse ** 2).sum(dim=-1)

        # Log target density
        log_p_x = self._log_prob(x).squeeze(-1)
        log_p_prop = self._log_prob(x_prop).squeeze(-1)

        # Log acceptance ratio
        log_alpha = (log_p_prop - log_p_x) + (log_q_reverse - log_q_forward)

        # Accept/reject per sample
        log_u = torch.log(torch.rand(n_samples, device=self.device))
        accepted = log_u < log_alpha

        # Apply acceptance
        x_new = torch.where(accepted.unsqueeze(-1), x_prop, x)

        return x_new, accepted

    def step(
        self,
        samples: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Perform a single Langevin dynamics step.

        Args:
            samples: Current positions [n_samples, dim].

        Returns:
            Tuple of (updated_samples, info_dict).
        """
        if self.use_mh_correction:
            new_samples, accepted = self._mala_step(samples)
            accept_rate = accepted.float().mean().item()
        else:
            new_samples = self._ula_step(samples)
            accept_rate = 1.0  # ULA always "accepts"

        log_prob = self._log_prob(new_samples).mean().item()

        info = {
            'log_prob': log_prob,
            'accept_rate': accept_rate,
            'step_size': self.step_size,
        }

        return new_samples, info

    def _adapt_step_size_simple(
        self,
        accept_rate: float,
        iteration: int,
    ) -> None:
        """Adapt step size based on acceptance rate (MALA only)."""
        if iteration >= self.n_adapt:
            return

        adapt_rate = 1.0 / (iteration + 10.0)
        log_eps = math.log(self.step_size)
        log_eps += adapt_rate * (accept_rate - self.target_accept_rate)
        self.step_size = math.exp(log_eps)

    def run(
        self,
        n_samples: int,
        n_steps: int,
        initial_samples: Optional[torch.Tensor] = None,
        *,
        dim: Optional[int] = None,
        n_burnin: int = 0,
        verbose: bool = True,
        print_every: int = 50,
        return_trajectory: bool = False,
        trajectory_every: int = 10,
    ) -> SamplingResult:
        """
        Run Langevin dynamics sampling.

        Args:
            n_samples: Number of parallel samples.
            n_steps: Number of iterations.
            initial_samples: Starting positions [n_samples, dim].
            dim: Dimension (required if initial_samples is None).
            n_burnin: Number of burn-in steps to discard.
            verbose: Whether to print progress.
            print_every: Print frequency.
            return_trajectory: Whether to save trajectory.
            trajectory_every: Save frequency.

        Returns:
            SamplingResult containing samples and diagnostics.
        """
        if self.target is None:
            raise ValueError("Target distribution must be provided")

        # Initialize samples
        if initial_samples is not None:
            samples = initial_samples.to(self.device)
            dim = samples.shape[1]
        else:
            dim = dim or getattr(self.target, 'dim', None)
            if dim is None:
                raise ValueError("Specify dim or provide initial_samples")
            samples = torch.randn(n_samples, dim, device=self.device)

        variant = "MALA" if self.use_mh_correction else "ULA"

        # Tracking
        log_prob_history: List[float] = []
        trajectory: List[torch.Tensor] = [] if return_trajectory else None

        start_time = time.time()

        if verbose:
            logger.info(f"LDS ({variant}): {n_samples} samples, {dim} dims, "
                        f"{n_steps} steps")
            logger.info(f"  step_size={self.step_size}")
            logger.info("-" * 55)

        for step in range(n_steps):
            samples, info = self.step(samples)

            # Adaptation (MALA only)
            if (self.use_mh_correction and self.adapt_step_size
                    and step < self.n_adapt):
                self._adapt_step_size_simple(info['accept_rate'], step)

            log_prob_history.append(info['log_prob'])

            if return_trajectory and step % trajectory_every == 0:
                trajectory.append(samples.detach().cpu().clone())

            if verbose and (step % print_every == 0 or step == n_steps - 1):
                elapsed = time.time() - start_time
                msg = (
                    f"[{step+1:5d}/{n_steps}] "
                    f"log_prob: {info['log_prob']:.4e} | "
                    f"eps: {info['step_size']:.4f}"
                )
                if self.use_mh_correction:
                    msg += f" | accept: {info['accept_rate']:.3f}"
                msg += f" | time: {elapsed:.1f}s"
                logger.info(msg)

            if not math.isfinite(info['log_prob']):
                if verbose:
                    logger.warning(f"Diverged at step {step+1}")
                break

        self._samples = samples.detach()

        trajectory_tensor = None
        if return_trajectory and trajectory:
            trajectory_tensor = torch.stack(trajectory, dim=0)

        total_time = time.time() - start_time

        result = SamplingResult(
            samples=self._samples,
            log_prob_history=log_prob_history[n_burnin:],
            algorithm=f'LDS-{variant}',
            n_steps=step + 1,
            converged=False,
            total_time=total_time,
            trajectory=trajectory_tensor,
            config={
                'n_samples': n_samples,
                'dim': dim,
                'step_size': self.step_size,
                'variant': variant,
                'n_burnin': n_burnin,
            }
        )

        if verbose:
            logger.info("-" * 55)
            logger.info(f"Completed in {total_time:.2f}s")

        return result

    def __repr__(self) -> str:
        variant = "MALA" if self.use_mh_correction else "ULA"
        return (
            f"LDS(step_size={self.step_size}, variant='{variant}')"
        )
