"""
Gauss-Newton optimizer for nonlinear least squares.

Solves inverse problems of the form:

    min_x  0.5 * ||F(x) - d||^2_{C_d^{-1}} + 0.5 * ||x - x_prior||^2_{C_m^{-1}}

using the Gauss-Newton approximation to the Hessian:
    H_GN = J^T C_d^{-1} J + C_m^{-1}

with strong Wolfe line search and optional box constraints.

Reference:
    Tarantola (2005). Inverse Problem Theory.
    Nocedal & Wright (2006). Numerical Optimization, 2nd ed.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import logging
import time
import torch
from typing import Callable, Optional, Tuple, List
from dataclasses import dataclass, field

from .line_search import line_search_wolfe

logger = logging.getLogger(__name__)


@dataclass
class GaussNewtonResult:
    """
    Result container for Gauss-Newton optimization.

    Attributes:
        x: Final solution.
        f_history: Objective value at each iteration.
        converged: Whether gradient convergence was achieved.
        n_iter: Number of iterations completed.
        total_time: Total wall-clock time.
        x_history: Solution at each iteration (if tracked).
    """
    x: torch.Tensor
    f_history: List[float]
    converged: bool = False
    n_iter: int = 0
    total_time: float = 0.0
    x_history: Optional[List[torch.Tensor]] = None


class GaussNewton:
    """
    Gauss-Newton optimizer for nonlinear least squares.

    Designed for inverse problems where the objective has the form:

        f(x) = 0.5 * (F(x)-d)^T C_d^{-1} (F(x)-d)
             + 0.5 * (x-x_prior)^T C_m^{-1} (x-x_prior)

    The Gauss-Newton method approximates the Hessian as:
        H = J^T C_d^{-1} J + C_m^{-1}

    which avoids computing second derivatives.

    Args:
        forward_fn: Forward model function ``u = forward_fn(x)``.
        jacobian_fn: Jacobian function ``J = jacobian_fn(x)``
            returning matrix of shape [n_data, n_params].
        obs_data: Observed data vector [n_data].
        inv_Cd: Inverse data covariance [n_data, n_data].
        inv_Cm: Inverse model covariance [n_params, n_params].
        x_prior: Prior model [n_params].
        max_iter: Maximum number of iterations. Default: 50.
        tol_grad: Gradient convergence tolerance (relative). Default: 1e-7.
        initial_step: Initial step size for first iteration. Default: 1.0.
        c1, c2: Wolfe parameters. Default: 1e-4, 0.9.
        bounds: Box constraints [n_params, 2]. Default: None.
        track_history: Save x at each iteration. Default: False.

    Example:
        >>> gn = GaussNewton(
        ...     forward_fn=forward, jacobian_fn=jacobian,
        ...     obs_data=d_obs, inv_Cd=torch.eye(N),
        ...     inv_Cm=0.01*torch.eye(M), x_prior=x0,
        ... )
        >>> result = gn.solve(x0)
    """

    def __init__(
        self,
        forward_fn: Callable,
        jacobian_fn: Callable,
        obs_data: torch.Tensor,
        inv_Cd: torch.Tensor,
        inv_Cm: torch.Tensor,
        x_prior: torch.Tensor,
        max_iter: int = 50,
        tol_grad: float = 1e-7,
        initial_step: float = 1.0,
        c1: float = 1e-4,
        c2: float = 0.9,
        max_iter_wolfe: int = 10,
        max_iter_zoom: int = 10,
        bounds: Optional[torch.Tensor] = None,
        track_history: bool = False,
    ):
        self.forward_fn = forward_fn
        self.jacobian_fn = jacobian_fn
        self.obs_data = obs_data
        self.inv_Cd = inv_Cd
        self.inv_Cm = inv_Cm
        self.x_prior = x_prior
        self.max_iter = max_iter
        self.tol_grad = tol_grad
        self.initial_step = initial_step
        self.c1 = c1
        self.c2 = c2
        self.max_iter_wolfe = max_iter_wolfe
        self.max_iter_zoom = max_iter_zoom
        self.bounds = bounds
        self.track_history = track_history

    def _objective_and_gradient(
        self,
        x: torch.Tensor,
    ) -> Tuple[float, torch.Tensor]:
        """
        Compute objective value and gradient.

        f = 0.5 * r_d^T C_d^{-1} r_d + 0.5 * r_m^T C_m^{-1} r_m
        g = J^T C_d^{-1} r_d + C_m^{-1} r_m

        where r_d = F(x) - d and r_m = x - x_prior.
        """
        u_calc = self.forward_fn(x)
        J = self.jacobian_fn(x)

        # Data residual
        r_d = u_calc - self.obs_data
        # Model residual
        r_m = x - self.x_prior

        # Objective
        f = 0.5 * torch.dot(r_d, self.inv_Cd @ r_d).item()
        f += 0.5 * torch.dot(r_m, self.inv_Cm @ r_m).item()

        # Gradient
        grad = J.T @ (self.inv_Cd @ r_d) + self.inv_Cm @ r_m

        return f, grad

    def _project_bounds(self, x: torch.Tensor) -> None:
        """Project x onto bounds (in-place)."""
        if self.bounds is not None:
            x.clamp_(min=self.bounds[:, 0], max=self.bounds[:, 1])

    def solve(
        self,
        x0: Optional[torch.Tensor] = None,
        verbose: bool = True,
        print_every: int = 1,
    ) -> GaussNewtonResult:
        """
        Run Gauss-Newton optimization.

        Args:
            x0: Initial guess. If None, uses x_prior.
            verbose: Whether to print progress.
            print_every: Print frequency.

        Returns:
            GaussNewtonResult with solution and diagnostics.
        """
        start_time = time.time()

        x = x0.clone() if x0 is not None else self.x_prior.clone()
        if self.bounds is not None:
            self.bounds = self.bounds.to(x.device)
            self._project_bounds(x)

        # Move data to same device
        device = x.device
        self.obs_data = self.obs_data.to(device)
        self.inv_Cd = self.inv_Cd.to(device)
        self.inv_Cm = self.inv_Cm.to(device)
        self.x_prior = self.x_prior.to(device)

        # Initial evaluation
        f_val, grad = self._objective_and_gradient(x)
        g0_norm = grad.norm().item()

        f_history = [f_val]
        x_history = [x.cpu().clone()] if self.track_history else None

        if verbose:
            logger.info(f"Gauss-Newton: dim={x.shape[0]}")
            logger.info(f"  Initial f = {f_val:.6e}")
            logger.info("-" * 55)

        converged = False
        k = 0

        for k in range(self.max_iter):
            # Compute Jacobian
            J = self.jacobian_fn(x)

            # Gauss-Newton Hessian approximation
            # H = J^T C_d^{-1} J + C_m^{-1}
            invCd_J = self.inv_Cd @ J
            H = J.T @ invCd_J + self.inv_Cm

            # Solve for Gauss-Newton direction
            # H * p_gn = -grad
            try:
                p_gn = torch.linalg.solve(H, -grad)
            except torch.linalg.LinAlgError:
                if verbose:
                    logger.warning(f"Hessian singular at iter {k+1}, "
                                   "using gradient descent")
                p_gn = -grad

            # Line search
            alpha0 = self.initial_step if k == 0 else 1.0

            alpha, f_new, grad_new, ls_success = line_search_wolfe(
                self._objective_and_gradient,
                x, grad, p_gn, f_val,
                alpha0=alpha0,
                c1=self.c1, c2=self.c2,
                bounds=self.bounds,
                max_iter_wolfe=self.max_iter_wolfe,
                max_iter_zoom=self.max_iter_zoom,
            )

            if not ls_success and verbose:
                logger.warning(f"line search did not converge at iter {k+1}")

            # Update solution
            x = x + alpha * p_gn
            if self.bounds is not None:
                self._project_bounds(x)

            f_val = f_new
            grad = grad_new

            f_history.append(f_val)
            if self.track_history:
                x_history.append(x.cpu().clone())

            if verbose and (k % print_every == 0 or k == self.max_iter - 1):
                g_norm = grad.norm().item()
                logger.info(
                    f"[{k+1:4d}/{self.max_iter}] "
                    f"f: {f_val:.6e} | "
                    f"|g|: {g_norm:.3e} | "
                    f"alpha: {alpha:.4f}"
                )

            # Convergence check
            g_norm = grad.norm().item()
            if g0_norm > 0 and abs(g_norm / g0_norm) < self.tol_grad:
                converged = True
                if verbose:
                    logger.info(f"  Gradient converged at iter {k+1}")
                break

        total_time = time.time() - start_time

        if verbose:
            logger.info("-" * 55)
            logger.info(f"Completed in {total_time:.2f}s, "
                        f"converged={converged}")

        return GaussNewtonResult(
            x=x,
            f_history=f_history,
            converged=converged,
            n_iter=k + 1,
            total_time=total_time,
            x_history=x_history,
        )

    def __repr__(self) -> str:
        return (
            f"GaussNewton(max_iter={self.max_iter}, "
            f"tol_grad={self.tol_grad})"
        )
