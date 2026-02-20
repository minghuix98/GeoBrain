"""
Hamiltonian Monte Carlo (HMC) sampler.

Uses Hamiltonian dynamics to propose samples with leapfrog integration
and Metropolis-Hastings acceptance. Supports mass matrix preconditioning,
box constraints with reflection, and adaptive step size.

Reference:
    Neal (2011). MCMC using Hamiltonian dynamics.
    Handbook of Markov Chain Monte Carlo.

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


class HMC(BaseSampler):
    """
    Hamiltonian Monte Carlo sampler.

    HMC uses Hamiltonian dynamics to propose samples, achieving
    better mixing than random walk Metropolis-Hastings in high
    dimensions.

    The algorithm simulates Hamiltonian dynamics:
        dq/dt = M^{-1} p
        dp/dt = -grad U(q)

    where U(q) = -log p(q) is the potential energy and M is
    the mass matrix.

    Args:
        target: Target distribution (must implement log_prob and score).
        step_size: Leapfrog integrator step size (epsilon).
        n_leapfrog: Number of leapfrog steps per iteration.
        mass_matrix: Mass matrix M. Can be:
            - None: identity mass matrix (default)
            - 1D tensor: diagonal mass matrix
            - 2D tensor: full mass matrix
        adapt_step_size: Whether to adapt step size during warmup.
        target_accept_rate: Target acceptance rate for adaptation.
        n_adapt: Number of warmup iterations for adaptation.
        bounds: Box constraints as tensor of shape [dim, 2] where
            bounds[:, 0] are lower bounds and bounds[:, 1] are upper.
            None means unconstrained.
        device: Computation device.

    Example:
        >>> target = Gaussian(torch.zeros(10), torch.eye(10))
        >>> hmc = HMC(target=target, step_size=0.1, n_leapfrog=10)
        >>> result = hmc.run(n_samples=1000, n_steps=500)
    """

    def __init__(
        self,
        target: Optional[Distribution] = None,
        step_size: float = 0.1,
        n_leapfrog: int = 10,
        mass_matrix: Optional[torch.Tensor] = None,
        adapt_step_size: bool = True,
        target_accept_rate: float = 0.65,
        n_adapt: int = 100,
        bounds: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(target, device)
        self.step_size = step_size
        self.n_leapfrog = n_leapfrog
        self.adapt_step_size = adapt_step_size
        self.target_accept_rate = target_accept_rate
        self.n_adapt = n_adapt
        self.bounds = bounds

        # Mass matrix setup (deferred to first call if None)
        self._mass_matrix = mass_matrix
        self._inv_mass_matrix = None
        self._chol_mass_matrix = None

    def _setup_mass_matrix(self, dim: int) -> None:
        """Initialize mass matrix and its decomposition."""
        if self._mass_matrix is None:
            # Identity mass matrix
            self._inv_mass_matrix = torch.eye(dim, device=self.device)
            self._chol_mass_matrix = torch.eye(dim, device=self.device)
        elif self._mass_matrix.dim() == 1:
            # Diagonal mass matrix
            M = self._mass_matrix.to(self.device)
            self._inv_mass_matrix = torch.diag(1.0 / M)
            self._chol_mass_matrix = torch.diag(torch.sqrt(M))
        else:
            # Full mass matrix
            M = self._mass_matrix.to(self.device)
            self._inv_mass_matrix = torch.linalg.inv(M)
            self._chol_mass_matrix = torch.linalg.cholesky(M)

    def _draw_momentum(self, dim: int) -> torch.Tensor:
        """Draw momentum from N(0, M) using Cholesky decomposition."""
        z = torch.randn(dim, device=self.device)
        return self._chol_mass_matrix @ z

    def _potential_energy(self, q: torch.Tensor) -> torch.Tensor:
        """Compute potential energy U(q) = -log p(q)."""
        return -self.target.log_prob(q.unsqueeze(0)).squeeze()

    def _grad_potential(self, q: torch.Tensor) -> torch.Tensor:
        """Compute gradient of potential energy: grad U(q) = -score(q)."""
        return -self.target.score(q.unsqueeze(0)).squeeze(0)

    def _kinetic_energy(self, p: torch.Tensor) -> torch.Tensor:
        """Compute kinetic energy K(p) = 0.5 * p^T M^{-1} p."""
        return 0.5 * torch.dot(p, self._inv_mass_matrix @ p)

    def _grad_kinetic(self, p: torch.Tensor) -> torch.Tensor:
        """Compute gradient of kinetic energy: grad K(p) = M^{-1} p."""
        return self._inv_mass_matrix @ p

    def _compute_hamiltonian(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Hamiltonian H(q, p) = U(q) + K(p)."""
        return self._potential_energy(q) + self._kinetic_energy(p)

    def _reflect_constraints(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply reflection for box constraints.

        When a parameter hits a boundary, reflect it back and
        negate its momentum (Neal 2011, sec. 5.5.2).
        """
        if self.bounds is None:
            return q, p

        bounds = self.bounds.to(self.device)
        lower = bounds[:, 0]
        upper = bounds[:, 1]

        max_reflect = 100
        for _ in range(max_reflect):
            below = q < lower
            above = q > upper

            if not (below.any() or above.any()):
                break

            # Reflect from lower bound
            if below.any():
                q = torch.where(below, lower + (lower - q), q)
                p = torch.where(below, -p, p)

            # Reflect from upper bound
            if above.any():
                q = torch.where(above, upper - (q - upper), q)
                p = torch.where(above, -p, p)

        return q, p

    def _leapfrog(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Leapfrog integrator (velocity Verlet).

        Performs L leapfrog steps:
            p_{1/2} = p - (eps/2) * grad U(q)
            for i in 1..L:
                q = q + eps * M^{-1} * p
                p = p - eps * grad U(q)   (except last step)
            p = p - (eps/2) * grad U(q)

        Args:
            q: Current position.
            p: Current momentum.

        Returns:
            Tuple of (new_position, new_momentum).
        """
        eps = self.step_size
        constrained = self.bounds is not None
        q = q.clone()
        p = p.clone()

        # Half step for momentum at the beginning
        p = p - (eps / 2.0) * self._grad_potential(q)

        # Alternate full steps for position and momentum
        for i in range(self.n_leapfrog):
            # Full step for position
            q = q + eps * self._grad_kinetic(p)

            if constrained:
                q, p = self._reflect_constraints(q, p)

            # Full step for momentum (except at end of trajectory)
            if i < self.n_leapfrog - 1:
                p = p - eps * self._grad_potential(q)

        # Half step for momentum at the end
        p = p - (eps / 2.0) * self._grad_potential(q)

        # Negate momentum to make proposal symmetric
        p = -p

        return q, p

    def step(
        self,
        samples: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Perform a single HMC step for each sample (chain).

        Args:
            samples: Current positions of shape [n_chains, dim].

        Returns:
            Tuple of (updated_samples, info_dict).
            info_dict contains:
                - accept_rate: Acceptance rate across chains
                - potential_energy: Mean potential energy
                - step_size: Current step size
        """
        n_chains, dim = samples.shape

        # Ensure mass matrix is initialized
        if self._inv_mass_matrix is None:
            self._setup_mass_matrix(dim)

        n_accept = 0
        total_potential = 0.0
        new_samples = samples.clone()

        for i in range(n_chains):
            q_cur = samples[i]

            # Draw momentum
            p_cur = self._draw_momentum(dim)

            # Compute current Hamiltonian
            H_cur = self._compute_hamiltonian(q_cur, p_cur)

            # Leapfrog integration
            q_new, p_new = self._leapfrog(q_cur, p_cur)

            # Compute proposed Hamiltonian
            H_new = self._compute_hamiltonian(q_new, p_new)

            # Metropolis-Hastings acceptance
            log_accept_prob = H_cur - H_new
            if torch.log(torch.rand(1, device=self.device)) < log_accept_prob:
                new_samples[i] = q_new
                n_accept += 1
                total_potential += self._potential_energy(q_new).item()
            else:
                total_potential += self._potential_energy(q_cur).item()

        accept_rate = n_accept / n_chains
        info = {
            'accept_rate': accept_rate,
            'potential_energy': total_potential / n_chains,
            'step_size': self.step_size,
        }

        return new_samples, info

    def _adapt_step_size_simple(
        self,
        accept_rate: float,
        iteration: int,
    ) -> None:
        """
        Simple step-size adaptation based on acceptance rate.

        Adjusts step size toward target acceptance rate using
        stochastic approximation.
        """
        if iteration >= self.n_adapt:
            return

        # Robbins-Monro style update
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
        Run HMC sampling.

        Args:
            n_samples: Number of parallel chains.
            n_steps: Number of HMC iterations.
            initial_samples: Starting positions [n_samples, dim].
            dim: Dimension (required if initial_samples is None).
            n_burnin: Number of burn-in steps to discard.
            verbose: Whether to print progress.
            print_every: Print frequency.
            return_trajectory: Whether to save trajectory.
            trajectory_every: Save frequency for trajectory.

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

        # Initialize mass matrix
        self._setup_mass_matrix(dim)

        # Move bounds to device if needed
        if self.bounds is not None:
            self.bounds = self.bounds.to(self.device)

        # Tracking
        log_prob_history: List[float] = []
        trajectory: List[torch.Tensor] = [] if return_trajectory else None

        start_time = time.time()

        if verbose:
            logger.info(f"HMC: {n_samples} chains, {dim} dims, {n_steps} steps")
            logger.info(f"  step_size={self.step_size}, n_leapfrog={self.n_leapfrog}")
            logger.info("-" * 55)

        for step in range(n_steps):
            samples, info = self.step(samples)

            # Step-size adaptation during warmup
            if self.adapt_step_size and step < self.n_adapt:
                self._adapt_step_size_simple(info['accept_rate'], step)

            # Record log_prob (negative potential energy)
            log_prob_history.append(-info['potential_energy'])

            # Trajectory
            if return_trajectory and step % trajectory_every == 0:
                trajectory.append(samples.detach().cpu().clone())

            # Print
            if verbose and (step % print_every == 0 or step == n_steps - 1):
                elapsed = time.time() - start_time
                logger.info(
                    f"[{step+1:5d}/{n_steps}] "
                    f"accept: {info['accept_rate']:.3f} | "
                    f"U: {info['potential_energy']:.4e} | "
                    f"eps: {info['step_size']:.4f} | "
                    f"time: {elapsed:.1f}s"
                )

            # Divergence check
            if not math.isfinite(info['potential_energy']):
                if verbose:
                    logger.warning(f"Diverged at step {step+1}")
                break

        # Store final samples
        self._samples = samples.detach()

        # Build trajectory
        trajectory_tensor = None
        if return_trajectory and trajectory:
            trajectory_tensor = torch.stack(trajectory, dim=0)

        total_time = time.time() - start_time

        result = SamplingResult(
            samples=self._samples,
            log_prob_history=log_prob_history[n_burnin:],
            algorithm='HMC',
            n_steps=step + 1,
            converged=False,
            total_time=total_time,
            trajectory=trajectory_tensor,
            config={
                'n_samples': n_samples,
                'dim': dim,
                'step_size': self.step_size,
                'n_leapfrog': self.n_leapfrog,
                'n_burnin': n_burnin,
                'target_accept_rate': self.target_accept_rate,
            }
        )

        if verbose:
            logger.info("-" * 55)
            logger.info(f"Completed in {total_time:.2f}s")

        return result

    def __repr__(self) -> str:
        return (
            f"HMC(step_size={self.step_size}, n_leapfrog={self.n_leapfrog}, "
            f"adapt={self.adapt_step_size})"
        )
