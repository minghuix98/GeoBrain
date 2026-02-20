"""
Strong Wolfe line search with zoom and cubic interpolation.

Implementation follows Nocedal & Wright (2006), Algorithms 3.5 & 3.6,
with support for box constraints.

Reference:
    Nocedal & Wright (2006). Numerical Optimization, 2nd ed.
    Springer. Chapter 3.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import math
import torch
from typing import Callable, Optional, Tuple


def _cubic_interpolation(
    a: float, b: float,
    phi_a: float, phi_b: float,
    dphi_a: float, dphi_b: float,
) -> float:
    """
    Cubic interpolation between two points to find minimum.

    Args:
        a, b: Interval endpoints.
        phi_a, phi_b: Function values at a and b.
        dphi_a, dphi_b: Directional derivatives at a and b.

    Returns:
        Interpolated step size.
    """
    if a == b:
        return a

    d1 = dphi_a + dphi_b - 3.0 * (phi_a - phi_b) / (a - b)
    disc = d1 * d1 - dphi_a * dphi_b

    if disc < 0:
        # No real root, return midpoint
        return 0.5 * (a + b)

    d2 = math.sqrt(disc)
    alpha = b - (b - a) * ((dphi_b + d2 - d1) / (dphi_b - dphi_a + 2.0 * d2))

    # Ensure alpha is within [min(a,b), max(a,b)]
    lo, hi = min(a, b), max(a, b)
    alpha = max(lo, min(alpha, hi))

    return alpha


def _project_bounds(
    x: torch.Tensor,
    p: torch.Tensor,
    bounds: torch.Tensor,
) -> torch.Tensor:
    """
    Project model to bounds and zero out gradient at active constraints.

    Args:
        x: Current point (modified in-place to clip to bounds).
        p: Search direction.
        bounds: [dim, 2] lower and upper bounds.

    Returns:
        Modified search direction with zeros at active constraints.
    """
    lower = bounds[:, 0]
    upper = bounds[:, 1]

    below = x < lower
    above = x > upper

    x.clamp_(min=lower, max=upper)

    p_bound = p.clone()
    p_bound[below] = 0.0
    p_bound[above] = 0.0

    return p_bound


def _zoom(
    fgrad_fn: Callable,
    x0: torch.Tensor,
    p: torch.Tensor,
    c1: float,
    c2: float,
    phi0: float,
    dphi0: float,
    alpha_lo: float,
    alpha_hi: float,
    phi_lo: float,
    phi_hi: float,
    dphi_lo: float,
    dphi_hi: float,
    bounds: Optional[torch.Tensor] = None,
    max_iter: int = 10,
) -> Tuple[float, float, torch.Tensor, bool]:
    """
    Zoom phase of Wolfe line search (Algorithm 3.6).

    Narrows the bracket [alpha_lo, alpha_hi] until strong Wolfe
    conditions are satisfied.

    Args:
        fgrad_fn: Function computing (objective, gradient) given x.
        x0: Starting point.
        p: Search direction.
        c1, c2: Wolfe parameters.
        phi0: Objective at alpha=0.
        dphi0: Directional derivative at alpha=0.
        alpha_lo, alpha_hi: Bracket endpoints.
        phi_lo, phi_hi: Objective at bracket endpoints.
        dphi_lo, dphi_hi: Directional derivative at bracket endpoints.
        bounds: Optional box constraints [dim, 2].
        max_iter: Maximum zoom iterations.

    Returns:
        (alpha, phi, grad, success).
    """
    grad = torch.zeros_like(x0)
    alpha = 0.0
    phi_trial = 0.0

    for _ in range(max_iter):
        # Cubic interpolation to find trial alpha
        if alpha_lo < alpha_hi:
            alpha = _cubic_interpolation(
                alpha_lo, alpha_hi, phi_lo, phi_hi, dphi_lo, dphi_hi
            )
        elif alpha_lo == alpha_hi:
            alpha = alpha_lo
        else:
            alpha = _cubic_interpolation(
                alpha_hi, alpha_lo, phi_hi, phi_lo, dphi_hi, dphi_lo
            )

        # Evaluate at trial point
        x_trial = x0 + alpha * p
        if bounds is not None:
            p_bound = _project_bounds(x_trial, p, bounds)
        else:
            p_bound = p

        phi_trial, grad = fgrad_fn(x_trial)
        dphi_trial = torch.dot(grad, p_bound).item()

        if phi_trial > phi0 + c1 * alpha * dphi0 or phi_trial >= phi_lo:
            alpha_hi = alpha
            dphi_hi = dphi_trial
            phi_hi = phi_trial
        else:
            if abs(dphi_trial) <= -c2 * dphi0:
                return alpha, phi_trial, grad, True

            if dphi_trial * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
                dphi_hi = dphi_lo
                phi_hi = phi_lo

            alpha_lo = alpha
            dphi_lo = dphi_trial
            phi_lo = phi_trial

    return alpha, phi_trial, grad, False


def line_search_wolfe(
    fgrad_fn: Callable,
    x0: torch.Tensor,
    grad0: torch.Tensor,
    p: torch.Tensor,
    phi0: float,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    c2: float = 0.9,
    bounds: Optional[torch.Tensor] = None,
    max_iter_wolfe: int = 10,
    max_iter_zoom: int = 10,
) -> Tuple[float, float, torch.Tensor, bool]:
    """
    Strong Wolfe line search (Algorithm 3.5).

    Finds a step size alpha satisfying the strong Wolfe conditions:
        1. Sufficient decrease: f(x+ap) <= f(x) + c1*a*p'g
        2. Curvature: |p'g(x+ap)| <= c2*|p'g(x)|

    Args:
        fgrad_fn: Function computing (objective, gradient) given x.
            Signature: ``phi, grad = fgrad_fn(x)``
        x0: Starting point.
        grad0: Gradient at x0.
        p: Search direction (should satisfy p'g < 0).
        phi0: Objective value at x0.
        alpha0: Initial step size guess. Default: 1.0.
        c1: Sufficient decrease parameter. Default: 1e-4.
        c2: Curvature parameter. Default: 0.9.
        bounds: Optional box constraints [dim, 2].
        max_iter_wolfe: Max iterations for main loop.
        max_iter_zoom: Max iterations for zoom phase.

    Returns:
        Tuple of (alpha, phi, grad, success):
            - alpha: Step size satisfying Wolfe conditions.
            - phi: Objective value at x0 + alpha * p.
            - grad: Gradient at x0 + alpha * p.
            - success: Whether line search succeeded.

    Example:
        >>> def fgrad_fn(x):
        ...     f = 0.5 * (x ** 2).sum()
        ...     g = x.clone()
        ...     return f.item(), g
        >>> alpha, phi, grad, ok = line_search_wolfe(
        ...     fgrad_fn, x0, grad0, p=-grad0, phi0=f0
        ... )
    """
    assert 0.0 < c1 < c2 < 1.0

    dphi0 = torch.dot(grad0, p).item()

    alpha_old = 0.0
    phi_old = phi0
    dphi_old = dphi0

    alpha = alpha0
    phi = 0.0
    grad = torch.zeros_like(x0)

    for i in range(max_iter_wolfe):
        x_trial = x0 + alpha * p
        if bounds is not None:
            p_bound = _project_bounds(x_trial, p, bounds)
        else:
            p_bound = p

        phi, grad = fgrad_fn(x_trial)
        dphi = torch.dot(grad, p_bound).item()

        # Wolfe condition 1 violated, or function increased
        if phi > phi0 + c1 * alpha * dphi0 or (phi >= phi_old and i > 0):
            alpha_out, phi_out, grad_out, success = _zoom(
                fgrad_fn, x0, p, c1, c2,
                phi0, dphi0,
                alpha_old, alpha,
                phi_old, phi,
                dphi_old, dphi,
                bounds=bounds,
                max_iter=max_iter_zoom,
            )
            return alpha_out, phi_out, grad_out, success

        # Strong Wolfe condition 2 satisfied
        if abs(dphi) <= -c2 * dphi0:
            return alpha, phi, grad, True

        # Overshoot
        if dphi >= 0.0:
            alpha_out, phi_out, grad_out, success = _zoom(
                fgrad_fn, x0, p, c1, c2,
                phi0, dphi0,
                alpha, alpha_old,
                phi, phi_old,
                dphi, dphi_old,
                bounds=bounds,
                max_iter=max_iter_zoom,
            )
            return alpha_out, phi_out, grad_out, success

        # Update for next iteration
        alpha_old = alpha
        phi_old = phi
        dphi_old = dphi

        # Increase alpha
        ratio = dphi0 / (dphi0 - dphi)
        if ratio > 1.0:
            alpha = ratio * alpha_old
        else:
            alpha = 2.0 * alpha_old

    return alpha_old, phi, grad, False
