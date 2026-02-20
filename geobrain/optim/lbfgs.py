"""
Limited-memory BFGS (L-BFGS) optimizer.

Custom L-BFGS implementation with two-loop recursion, strong Wolfe
line search, and optional box constraints. Follows Nocedal & Wright
(2006) Algorithm 7.4 & 7.5.

Reference:
    Nocedal & Wright (2006). Numerical Optimization, 2nd ed.
    Springer. Chapter 7.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import logging
import time
import math
import torch
import numpy as np
from typing import Callable, Optional, Tuple, List, Union
from collections import deque
from dataclasses import dataclass, field

from .line_search import line_search_wolfe

logger = logging.getLogger(__name__)


@dataclass
class LBFGSResult:
    """
    Result container for L-BFGS optimization.

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


class LBFGS:
    """
    Limited-memory BFGS optimizer.

    Uses the L-BFGS two-loop recursion to approximate the inverse
    Hessian using only the last ``memory`` gradient/step pairs.
    Combined with a strong Wolfe line search for robust convergence.

    Args:
        memory: Number of correction pairs to store. Default: 10.
        max_iter: Maximum number of iterations. Default: 100.
        tol_grad: Gradient norm convergence tolerance (relative
            to initial gradient). Default: 1e-7.
        initial_step: Initial step size for the first iteration.
            Default: 1.0.
        c1: Wolfe sufficient decrease parameter. Default: 1e-4.
        c2: Wolfe curvature parameter. Default: 0.9.
        max_iter_wolfe: Max line search iterations. Default: 20.
        max_iter_zoom: Max zoom iterations. Default: 10.
        bounds: Box constraints as tensor [dim, 2]. Default: None.
        track_history: Whether to save x at each iteration.

    Example:
        >>> def objective_and_gradient(x):
        ...     f = 0.5 * (x ** 2).sum()
        ...     g = x.clone()
        ...     return f.item(), g
        >>>
        >>> optimizer = LBFGS(memory=10, max_iter=100)
        >>> result = optimizer.minimize(objective_and_gradient, x0)
        >>> print(f"Converged: {result.converged}, f={result.f_history[-1]}")

        With separate f and grad functions:
        >>> result = optimizer.minimize_fg(f_fn, grad_fn, x0)
    """

    def __init__(
        self,
        memory: int = 10,
        max_iter: int = 100,
        tol_grad: float = 1e-7,
        initial_step: float = 1.0,
        c1: float = 1e-4,
        c2: float = 0.9,
        max_iter_wolfe: int = 20,
        max_iter_zoom: int = 10,
        bounds: Optional[torch.Tensor] = None,
        track_history: bool = False,
    ):
        assert memory > 0
        self.memory = memory
        self.max_iter = max_iter
        self.tol_grad = tol_grad
        self.initial_step = initial_step
        self.c1 = c1
        self.c2 = c2
        self.max_iter_wolfe = max_iter_wolfe
        self.max_iter_zoom = max_iter_zoom
        self.bounds = bounds
        self.track_history = track_history

    def _two_loop_recursion(
        self,
        grad: torch.Tensor,
        s_history: List[torch.Tensor],
        y_history: List[torch.Tensor],
        rho_history: List[float],
        gamma: float,
    ) -> torch.Tensor:
        """
        L-BFGS two-loop recursion (Algorithm 7.4).

        Computes H_k * grad using stored correction pairs,
        where H_k is the L-BFGS approximation to the inverse Hessian.

        Args:
            grad: Current gradient.
            s_history: List of s_i = x_{i+1} - x_i (most recent first).
            y_history: List of y_i = g_{i+1} - g_i.
            rho_history: List of rho_i = 1 / (y_i^T s_i).
            gamma: Scaling factor for initial Hessian (gamma * I).

        Returns:
            Descent direction: -H_k * grad.
        """
        m = len(s_history)
        if m == 0:
            return -gamma * grad

        q = grad.clone()
        alpha = [0.0] * m

        # Forward loop: k-1, k-2, ..., k-m
        for i in range(m):
            alpha[i] = rho_history[i] * torch.dot(s_history[i], q).item()
            q = q - alpha[i] * y_history[i]

        # Initial Hessian approximation: H0 = gamma * I
        r = gamma * q

        # Backward loop: k-m, ..., k-2, k-1
        for i in range(m - 1, -1, -1):
            beta = rho_history[i] * torch.dot(y_history[i], r).item()
            r = r + s_history[i] * (alpha[i] - beta)

        return -r

    def _project_bounds(self, x: torch.Tensor) -> None:
        """Project x onto bounds (in-place)."""
        if self.bounds is not None:
            x.clamp_(min=self.bounds[:, 0], max=self.bounds[:, 1])

    def _line_search(
        self,
        fgrad_fn: Callable,
        x: torch.Tensor,
        grad: torch.Tensor,
        p: torch.Tensor,
        f_val: float,
        alpha0: float = 1.0,
    ) -> Tuple[float, float, torch.Tensor, bool]:
        """Perform line search appropriate for the problem type.

        Uses strong Wolfe for unconstrained problems. For bounded problems,
        uses backtracking Armijo (avoids inconsistent projected directional
        derivatives that break the Wolfe curvature condition).
        """
        if self.bounds is None:
            return line_search_wolfe(
                fgrad_fn, x, grad, p, f_val,
                alpha0=alpha0,
                c1=self.c1, c2=self.c2,
                bounds=None,
                max_iter_wolfe=self.max_iter_wolfe,
                max_iter_zoom=self.max_iter_zoom,
            )
        else:
            return self._backtracking_armijo(
                fgrad_fn, x, grad, p, f_val, alpha0=alpha0,
            )

    def _backtracking_armijo(
        self,
        fgrad_fn: Callable,
        x: torch.Tensor,
        grad: torch.Tensor,
        p: torch.Tensor,
        f_val: float,
        alpha0: float = 1.0,
        max_trials: int = 30,
        shrink: float = 0.5,
    ) -> Tuple[float, float, torch.Tensor, bool]:
        """Backtracking Armijo line search.

        Finds alpha satisfying: f(x + alpha*p) <= f(x) + c1*alpha*p'g

        Args:
            fgrad_fn: Objective and gradient function.
            x: Current point.
            grad: Current gradient.
            p: Search direction.
            f_val: Current objective value.
            alpha0: Initial step size.
            max_trials: Maximum backtracking steps.
            shrink: Step reduction factor per trial.

        Returns:
            (alpha, f_new, grad_new, success)
        """
        dphi0 = torch.dot(grad, p).item()
        if dphi0 >= 0:
            return 0.0, f_val, grad.clone(), False

        alpha = alpha0
        f_new = f_val
        g_new = grad.clone()
        for _ in range(max_trials):
            x_new = x + alpha * p
            self._project_bounds(x_new)
            f_new, g_new = fgrad_fn(x_new)
            if f_new <= f_val + self.c1 * alpha * dphi0:
                return alpha, f_new, g_new, True
            alpha *= shrink

        return alpha, f_new, g_new, False

    def minimize(
        self,
        fgrad_fn: Callable,
        x0: torch.Tensor,
        verbose: bool = True,
        print_every: int = 1,
    ) -> LBFGSResult:
        """
        Minimize objective using L-BFGS.

        Args:
            fgrad_fn: Function computing (objective_value, gradient).
                Signature: ``f, grad = fgrad_fn(x)``
                where f is a float and grad is a tensor.
            x0: Initial guess.
            verbose: Whether to print progress.
            print_every: Print frequency.

        Returns:
            LBFGSResult with solution and diagnostics.
        """
        start_time = time.time()

        x = x0.clone()
        if self.bounds is not None:
            self.bounds = self.bounds.to(device=x.device, dtype=x.dtype)
            self._project_bounds(x)

        # Memory buffers (most recent first)
        s_history: List[torch.Tensor] = []
        y_history: List[torch.Tensor] = []
        rho_history: List[float] = []

        # Initial evaluation
        f_val, grad = fgrad_fn(x)
        g0_norm = grad.norm().item()

        f_history = [f_val]
        x_history = [x.cpu().clone()] if self.track_history else None

        if verbose:
            constrained = "constrained" if self.bounds is not None else "unconstrained"
            logger.info(f"L-BFGS ({constrained}): dim={x.shape[0]}, "
                        f"memory={self.memory}")
            logger.info(f"  Initial f = {f_val:.6e}")
            logger.info("-" * 55)

        converged = False
        k = 0

        for k in range(self.max_iter):
            # Compute scaling gamma (with safeguards)
            if len(s_history) > 0:
                ys = torch.dot(s_history[0], y_history[0]).item()
                yy = torch.dot(y_history[0], y_history[0]).item()
                if yy > 1e-30 and ys > 0:
                    gamma = ys / yy
                else:
                    gamma = 1.0
            else:
                gamma = 1.0

            # Two-loop recursion for descent direction
            p = self._two_loop_recursion(
                grad, s_history, y_history, rho_history, gamma
            )

            # Verify descent direction; fall back to steepest descent
            g_dot_p = torch.dot(grad, p).item()
            g_norm_val = grad.norm().item()
            p_norm_val = p.norm().item()
            if g_norm_val > 0 and p_norm_val > 0:
                cos_angle = g_dot_p / (g_norm_val * p_norm_val)
                if cos_angle >= -1e-8:
                    p = -grad
                    s_history.clear()
                    y_history.clear()
                    rho_history.clear()
            elif g_dot_p >= 0:
                p = -grad

            # Step size: use 1.0 for L-BFGS (quasi-Newton step)
            alpha0 = self.initial_step if k == 0 else 1.0

            # Line search
            grad_old = grad.clone()

            alpha, f_new, grad_new, ls_success = self._line_search(
                fgrad_fn, x, grad, p, f_val, alpha0=alpha0,
            )

            if not ls_success:
                if verbose:
                    logger.warning(f"Line search failed at iter {k+1}, "
                                   f"trying steepest descent")
                # Reset memory and try steepest descent
                s_history.clear()
                y_history.clear()
                rho_history.clear()

                # Switch to steepest descent direction
                p = -grad
                alpha, f_new, grad_new, bt_success = self._backtracking_armijo(
                    fgrad_fn, x, grad, p, f_val,
                )
                if not bt_success and verbose:
                    logger.warning(f"  Steepest descent also failed at iter {k+1}")

            # Update solution
            x_old = x.clone()
            x = x_old + alpha * p
            if self.bounds is not None:
                self._project_bounds(x)

            f_val = f_new
            grad = grad_new

            f_history.append(f_val)
            if self.track_history:
                x_history.append(x.cpu().clone())

            # Compute s and y
            s_new = x - x_old
            y_new = grad - grad_old

            # Check for zero update
            s_norm = s_new.norm().item()
            if s_norm == 0:
                if verbose:
                    logger.info(f"  Zero update at iter {k+1}, stopping")
                break

            # Curvature condition: only update memory if y^T s > 0
            # (required for positive-definite Hessian approximation)
            ys = torch.dot(y_new, s_new).item()
            if ys > 1e-10 * s_norm * y_new.norm().item():
                rho_new = 1.0 / ys

                # Update memory (prepend = most recent first)
                s_history.insert(0, s_new)
                y_history.insert(0, y_new)
                rho_history.insert(0, rho_new)

                # Trim memory
                if len(s_history) > self.memory:
                    s_history.pop()
                    y_history.pop()
                    rho_history.pop()

            # Progress
            if verbose and (k % print_every == 0 or k == self.max_iter - 1):
                g_norm = grad.norm().item()
                logger.info(
                    f"[{k+1:4d}/{self.max_iter}] "
                    f"f: {f_val:.6e} | "
                    f"|g|: {g_norm:.3e} | "
                    f"alpha: {alpha:.4f}"
                )

            # Gradient convergence check
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

        return LBFGSResult(
            x=x,
            f_history=f_history,
            converged=converged,
            n_iter=k + 1,
            total_time=total_time,
            x_history=x_history,
        )

    def minimize_fg(
        self,
        f_fn: Callable,
        grad_fn: Callable,
        x0: torch.Tensor,
        **kwargs,
    ) -> LBFGSResult:
        """
        Minimize with separate objective and gradient functions.

        Args:
            f_fn: Objective function: ``f = f_fn(x)``.
            grad_fn: Gradient function: ``g = grad_fn(x)``.
            x0: Initial guess.
            **kwargs: Passed to minimize().

        Returns:
            LBFGSResult with solution and diagnostics.
        """
        def fgrad_fn(x):
            f = f_fn(x)
            g = grad_fn(x)
            return f, g

        return self.minimize(fgrad_fn, x0, **kwargs)

    def __repr__(self) -> str:
        return (
            f"LBFGS(memory={self.memory}, max_iter={self.max_iter}, "
            f"tol_grad={self.tol_grad})"
        )
