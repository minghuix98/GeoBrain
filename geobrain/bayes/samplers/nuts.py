"""
No-U-Turn Sampler (NUTS).

An adaptive extension of HMC that automatically tunes the number
of leapfrog steps using recursive tree building and the U-turn
criterion. Step size is adapted via dual averaging.

Reference:
    Hoffman & Gelman (2014). The No-U-Turn Sampler: Adaptively
    Setting Path Lengths in Hamiltonian Monte Carlo. JMLR.

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


class NUTS(BaseSampler):
    """
    No-U-Turn Sampler.

    NUTS automatically determines the number of leapfrog steps by
    building a balanced binary tree of trajectory states until a
    U-turn is detected. Step size is adapted using dual averaging.

    Args:
        target: Target distribution (must implement log_prob and score).
        target_accept_rate: Target acceptance rate (delta). Default: 0.65.
        max_tree_depth: Maximum binary tree depth. Number of leapfrog
            steps = 2^depth. Default: 10.
        max_delta_h: Maximum acceptable Hamiltonian drift. Default: 1000.
        mass_matrix: Mass matrix M. Can be:
            - None: identity (default)
            - 1D tensor: diagonal
            - 2D tensor: full
        n_adapt: Number of adaptation iterations. Default: 100.
        step_size: Initial step size. If None, found automatically.
        bounds: Box constraints [dim, 2]. None = unconstrained.
        device: Computation device.

    Example:
        >>> target = Gaussian(torch.zeros(10), torch.eye(10))
        >>> nuts = NUTS(target=target, target_accept_rate=0.8)
        >>> result = nuts.run(n_samples=1, n_steps=1000)
    """

    def __init__(
        self,
        target: Optional[Distribution] = None,
        target_accept_rate: float = 0.65,
        max_tree_depth: int = 10,
        max_delta_h: float = 1000.0,
        mass_matrix: Optional[torch.Tensor] = None,
        n_adapt: int = 100,
        step_size: Optional[float] = None,
        bounds: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(target, device)
        self.target_accept_rate = target_accept_rate
        self.max_tree_depth = max_tree_depth
        self.max_delta_h = max_delta_h
        self.n_adapt = n_adapt
        self._initial_step_size = step_size
        self.step_size = step_size or 1.0
        self.bounds = bounds

        # Mass matrix (deferred setup)
        self._mass_matrix_raw = mass_matrix
        self._inv_mass_matrix = None
        self._chol_mass_matrix = None

        # Dual averaging parameters
        self._gamma = 0.05
        self._t0 = 10.0
        self._kappa = 0.75
        self._mu = 0.0
        self._h_bar = 0.0
        self._eps_bar = 1.0

    # ------------------------------------------------------------------
    # Mass matrix helpers
    # ------------------------------------------------------------------
    def _setup_mass_matrix(self, dim: int) -> None:
        """Initialize mass matrix and decomposition."""
        M = self._mass_matrix_raw
        if M is None:
            self._inv_mass_matrix = torch.eye(dim, device=self.device)
            self._chol_mass_matrix = torch.eye(dim, device=self.device)
        elif M.dim() == 1:
            M = M.to(self.device)
            self._inv_mass_matrix = torch.diag(1.0 / M)
            self._chol_mass_matrix = torch.diag(torch.sqrt(M))
        else:
            M = M.to(self.device)
            self._inv_mass_matrix = torch.linalg.inv(M)
            self._chol_mass_matrix = torch.linalg.cholesky(M)

    def _draw_momentum(self, dim: int) -> torch.Tensor:
        z = torch.randn(dim, device=self.device)
        return self._chol_mass_matrix @ z

    # ------------------------------------------------------------------
    # Energy computations
    # ------------------------------------------------------------------
    def _potential_energy(self, q: torch.Tensor) -> float:
        return -self.target.log_prob(q.unsqueeze(0)).squeeze().item()

    def _grad_potential(self, q: torch.Tensor) -> torch.Tensor:
        return -self.target.score(q.unsqueeze(0)).squeeze(0)

    def _kinetic_energy(self, p: torch.Tensor) -> float:
        return 0.5 * torch.dot(p, self._inv_mass_matrix @ p).item()

    def _grad_kinetic(self, p: torch.Tensor) -> torch.Tensor:
        return self._inv_mass_matrix @ p

    def _hamiltonian(self, q: torch.Tensor, p: torch.Tensor) -> float:
        return self._potential_energy(q) + self._kinetic_energy(p)

    # ------------------------------------------------------------------
    # Constraint reflection
    # ------------------------------------------------------------------
    def _reflect(
        self, q: torch.Tensor, p: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.bounds is None:
            return q, p
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        for _ in range(100):
            below = q < lower
            above = q > upper
            if not (below.any() or above.any()):
                break
            q = torch.where(below, lower + (lower - q), q)
            p = torch.where(below, -p, p)
            q = torch.where(above, upper - (q - upper), q)
            p = torch.where(above, -p, p)
        return q, p

    # ------------------------------------------------------------------
    # Leapfrog (velocity Verlet)
    # ------------------------------------------------------------------
    def _leapfrog(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One leapfrog step with step size eps (can be negative)."""
        q = q.clone()
        p = p.clone()

        # Half step momentum
        q = q + (eps / 2.0) * self._grad_kinetic(p)
        if self.bounds is not None:
            q, p = self._reflect(q, p)

        # Full step momentum
        p = p - eps * self._grad_potential(q)

        # Half step position
        q = q + (eps / 2.0) * self._grad_kinetic(p)
        if self.bounds is not None:
            q, p = self._reflect(q, p)

        return q, p

    # ------------------------------------------------------------------
    # Find reasonable initial epsilon
    # ------------------------------------------------------------------
    def _find_reasonable_epsilon(self, q: torch.Tensor) -> float:
        """Heuristic for finding a reasonable initial step size."""
        eps = 1.0
        p = self._draw_momentum(q.shape[0])
        H0 = self._hamiltonian(q, p)

        q1, p1 = self._leapfrog(q, p, eps)
        H1 = self._hamiltonian(q1, p1)

        # Direction: double or halve
        a = 2.0 * float((-H1 + H0) > math.log(0.5)) - 1.0

        while a * (-H1 + H0) > -a * math.log(2.0):
            eps = (2.0 ** a) * eps
            if eps > 1e4:
                eps = 1e4
                break
            if eps < 1e-10:
                eps = 1e-10
                break
            q1, p1 = self._leapfrog(q, p, eps)
            H1 = self._hamiltonian(q1, p1)

        return eps

    # ------------------------------------------------------------------
    # Build tree (recursive)
    # ------------------------------------------------------------------
    def _build_tree(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        log_u: float,
        v: int,
        depth: int,
        eps: float,
        H0: float,
    ) -> Tuple:
        """
        Recursively build the NUTS binary tree.

        Returns:
            (q_minus, p_minus, q_plus, p_plus, q_prime, n_prime,
             s_prime, alpha, n_alpha)
        """
        if depth == 0:
            # Base case: take one leapfrog step
            q1, p1 = self._leapfrog(q, p, v * eps)
            H1 = self._hamiltonian(q1, p1)

            # Is this state in the slice?
            n1 = int(log_u <= -H1)
            # Is this state valid (not too far from initial)?
            s1 = int(log_u <= (self.max_delta_h - H1))

            alpha1 = min(1.0, math.exp(-H1 + H0))
            if math.isnan(alpha1):
                alpha1 = 0.0
            n1_alpha = 1

            return q1, p1, q1, p1, q1, n1, s1, alpha1, n1_alpha

        else:
            # Recursion: build left subtree
            (q_minus, p_minus, q_plus, p_plus,
             q_prime, n_prime, s_prime,
             alpha_prime, n_alpha_prime) = self._build_tree(
                q, p, log_u, v, depth - 1, eps, H0
            )

            if s_prime == 1:
                # Build right subtree
                if v == -1:
                    (q_minus, p_minus, _, _,
                     q_dprime, n_dprime, s_dprime,
                     alpha_dprime, n_alpha_dprime) = self._build_tree(
                        q_minus, p_minus, log_u, v, depth - 1, eps, H0
                    )
                else:
                    (_, _, q_plus, p_plus,
                     q_dprime, n_dprime, s_dprime,
                     alpha_dprime, n_alpha_dprime) = self._build_tree(
                        q_plus, p_plus, log_u, v, depth - 1, eps, H0
                    )

                # Metropolis-like selection
                total_n = n_prime + n_dprime
                if total_n > 0 and torch.rand(1).item() < n_dprime / total_n:
                    q_prime = q_dprime

                alpha_prime = alpha_prime + alpha_dprime
                n_alpha_prime = n_alpha_prime + n_alpha_dprime

                # U-turn check
                dq = q_plus - q_minus
                dot1 = torch.dot(dq, p_minus).item() >= 0
                dot2 = torch.dot(dq, p_plus).item() >= 0
                s_prime = s_dprime * int(dot1) * int(dot2)
                n_prime = total_n

            return (q_minus, p_minus, q_plus, p_plus,
                    q_prime, n_prime, s_prime,
                    alpha_prime, n_alpha_prime)

    # ------------------------------------------------------------------
    # Dual averaging for step-size adaptation
    # ------------------------------------------------------------------
    def _dual_averaging_update(
        self,
        alpha: float,
        n_alpha: int,
        iteration: int,
    ) -> None:
        """Update step size using dual averaging (Hoffman & Gelman 2014)."""
        if iteration > self.n_adapt:
            self.step_size = self._eps_bar
            return

        alpha_ave = alpha / max(n_alpha, 1)

        m = iteration
        tmp1 = 1.0 / (m + self._t0)
        self._h_bar = (1.0 - tmp1) * self._h_bar + tmp1 * (
            self.target_accept_rate - alpha_ave
        )

        log_eps = self._mu - (math.sqrt(m) / self._gamma) * self._h_bar
        self.step_size = math.exp(log_eps)

        tmp2 = m ** (-self._kappa)
        self._eps_bar = math.exp(
            tmp2 * math.log(self.step_size)
            + (1.0 - tmp2) * math.log(self._eps_bar)
        )

    # ------------------------------------------------------------------
    # Single NUTS iteration
    # ------------------------------------------------------------------
    def step(
        self,
        samples: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Perform a single NUTS step for each chain.

        Args:
            samples: Current positions [n_chains, dim].

        Returns:
            Tuple of (updated_samples, info_dict).
        """
        n_chains, dim = samples.shape

        if self._inv_mass_matrix is None:
            self._setup_mass_matrix(dim)

        new_samples = samples.clone()
        total_accept = 0.0
        total_depth = 0
        total_leapfrog = 0

        for i in range(n_chains):
            q_cur = samples[i]
            p_cur = self._draw_momentum(dim)

            H0 = self._hamiltonian(q_cur, p_cur)
            log_u = -H0 + math.log(max(torch.rand(1).item(), 1e-300))

            # Initialize tree endpoints
            q_minus = q_cur.clone()
            q_plus = q_cur.clone()
            p_minus = p_cur.clone()
            p_plus = p_cur.clone()
            q_new = q_cur.clone()

            j = 0  # tree depth
            n = 1.0
            s = 1
            alpha_total = 0.0
            n_alpha_total = 0
            accepted = False

            while s == 1:
                # Random direction
                v = 2 * int(torch.rand(1).item() > 0.5) - 1

                if v == -1:
                    (q_minus, p_minus, _, _,
                     q_prime, n_prime, s_prime,
                     alpha, n_alpha) = self._build_tree(
                        q_minus, p_minus, log_u, v, j,
                        self.step_size, H0
                    )
                else:
                    (_, _, q_plus, p_plus,
                     q_prime, n_prime, s_prime,
                     alpha, n_alpha) = self._build_tree(
                        q_plus, p_plus, log_u, v, j,
                        self.step_size, H0
                    )

                if s_prime == 1:
                    if n > 0 and torch.rand(1).item() < min(1.0, n_prime / n):
                        q_new = q_prime
                        accepted = True

                n += n_prime

                # U-turn check on full tree
                dq = q_plus - q_minus
                dot1 = torch.dot(dq, p_minus).item() >= 0
                dot2 = torch.dot(dq, p_plus).item() >= 0
                s = s_prime * int(dot1) * int(dot2)

                alpha_total += alpha
                n_alpha_total += n_alpha

                j += 1
                if j > self.max_tree_depth:
                    break

            new_samples[i] = q_new
            total_accept += float(accepted)
            total_depth += j
            total_leapfrog += 2 * (2 ** (j - 1)) if j > 0 else 1

            # Store for adaptation
            self._last_alpha = alpha_total
            self._last_n_alpha = n_alpha_total

        info = {
            'accept_rate': total_accept / n_chains,
            'mean_tree_depth': total_depth / n_chains,
            'mean_leapfrog_steps': total_leapfrog / n_chains,
            'step_size': self.step_size,
            'potential_energy': self._potential_energy(new_samples[0]),
        }

        return new_samples, info

    # ------------------------------------------------------------------
    # Run full NUTS chain
    # ------------------------------------------------------------------
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
        Run NUTS sampling.

        Args:
            n_samples: Number of parallel chains (typically 1 for NUTS).
            n_steps: Number of NUTS iterations.
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

        # Setup
        self._setup_mass_matrix(dim)
        if self.bounds is not None:
            self.bounds = self.bounds.to(self.device)

        # Find or set initial step size
        if self._initial_step_size is None:
            self.step_size = self._find_reasonable_epsilon(samples[0])
        else:
            self.step_size = self._initial_step_size

        # Initialize dual averaging
        self._mu = math.log(10.0 * self.step_size)
        self._h_bar = 0.0
        self._eps_bar = self.step_size

        # Tracking
        log_prob_history: List[float] = []
        trajectory: List[torch.Tensor] = [] if return_trajectory else None
        self._last_alpha = 0.0
        self._last_n_alpha = 1

        start_time = time.time()

        if verbose:
            logger.info(f"NUTS: {n_samples} chains, {dim} dims, {n_steps} steps")
            logger.info(f"  initial eps={self.step_size:.4f}, "
                        f"max_depth={self.max_tree_depth}")
            logger.info("-" * 60)

        for step in range(n_steps):
            samples, info = self.step(samples)

            # Step-size adaptation
            if step < self.n_adapt:
                self._dual_averaging_update(
                    self._last_alpha,
                    self._last_n_alpha,
                    step + 1,
                )

            log_prob_history.append(-info['potential_energy'])

            if return_trajectory and step % trajectory_every == 0:
                trajectory.append(samples.detach().cpu().clone())

            if verbose and (step % print_every == 0 or step == n_steps - 1):
                elapsed = time.time() - start_time
                logger.info(
                    f"[{step+1:5d}/{n_steps}] "
                    f"accept: {info['accept_rate']:.3f} | "
                    f"depth: {info['mean_tree_depth']:.1f} | "
                    f"lf: {info['mean_leapfrog_steps']:.0f} | "
                    f"eps: {info['step_size']:.4f} | "
                    f"time: {elapsed:.1f}s"
                )

            if not math.isfinite(info['potential_energy']):
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
            algorithm='NUTS',
            n_steps=step + 1,
            converged=False,
            total_time=total_time,
            trajectory=trajectory_tensor,
            config={
                'n_samples': n_samples,
                'dim': dim,
                'step_size': self.step_size,
                'max_tree_depth': self.max_tree_depth,
                'target_accept_rate': self.target_accept_rate,
                'n_adapt': self.n_adapt,
                'n_burnin': n_burnin,
            }
        )

        if verbose:
            logger.info("-" * 60)
            logger.info(f"Completed in {total_time:.2f}s, "
                        f"final eps={self.step_size:.4f}")

        return result

    def __repr__(self) -> str:
        return (
            f"NUTS(step_size={self.step_size:.4f}, "
            f"max_depth={self.max_tree_depth}, "
            f"delta={self.target_accept_rate})"
        )
