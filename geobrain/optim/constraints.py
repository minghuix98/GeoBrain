"""
Constraint functions for model parameters.

Provides both hard constraints (projection-based) and soft constraints
(differentiable transforms) for constrained optimization.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
from typing import Optional, Callable


# =============================================================================
# Hard Constraints (Projection)
# =============================================================================

def clip_constraint(
    x: torch.Tensor, 
    min_val: float, 
    max_val: float
) -> torch.Tensor:
    """
    Clip tensor values to range [min_val, max_val].
    
    Args:
        x: Input tensor.
        min_val: Minimum allowed value.
        max_val: Maximum allowed value.
        
    Returns:
        Clipped tensor.
    """
    return torch.clamp(x, min=min_val, max=max_val)


def positive_constraint(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Enforce strict positivity.
    
    Args:
        x: Input tensor.
        eps: Minimum value (small positive number).
        
    Returns:
        Positive tensor.
    """
    return torch.clamp(x, min=eps)


def bound_constraint(
    min_val: Optional[float] = None,
    max_val: Optional[float] = None
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Create a bound constraint function.
    
    Returns a function that clips values to [min_val, max_val].
    
    Args:
        min_val: Minimum bound (None for no lower bound).
        max_val: Maximum bound (None for no upper bound).
        
    Returns:
        Constraint function.
    
    Example:
        >>> constraint = bound_constraint(1500, 6000)
        >>> velocity = constraint(raw_velocity)  # Always in [1500, 6000]
    """
    def fn(x: torch.Tensor) -> torch.Tensor:
        if min_val is not None:
            x = torch.clamp(x, min=min_val)
        if max_val is not None:
            x = torch.clamp(x, max=max_val)
        return x
    
    # Store bounds as attributes for inspection
    fn.min_val = min_val
    fn.max_val = max_val
    fn.__doc__ = f"Bound constraint: [{min_val}, {max_val}]"
    
    return fn


# =============================================================================
# Soft Constraints (Differentiable Transforms)
# =============================================================================

def sigmoid_constraint(
    min_val: float,
    max_val: float,
    temperature: float = 1.0
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Smooth constraint using sigmoid transformation.

    Maps unconstrained values to [min_val, max_val] differentiably.
    This allows gradients to flow through the constraint.

    Args:
        min_val: Minimum output value.
        max_val: Maximum output value.
        temperature: Controls sharpness of the sigmoid.

    Returns:
        Constraint function.

    Example:
        >>> constraint = sigmoid_constraint(1500, 6000)
        >>> velocity = constraint(unconstrained_params)  # Always in [1500, 6000]
    """
    range_val = max_val - min_val

    def fn(x: torch.Tensor) -> torch.Tensor:
        return min_val + range_val * torch.sigmoid(x / temperature)

    fn.min_val = min_val
    fn.max_val = max_val
    fn.__doc__ = f"Sigmoid constraint: [{min_val}, {max_val}]"

    return fn


def softplus_transform(
    min_val: float = 0.0,
    beta: float = 1.0
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Softplus constraint for smooth positivity.
    
    Ensures values are greater than min_val with smooth gradients.
    
    Args:
        min_val: Minimum output value.
        beta: Sharpness parameter.
        
    Returns:
        Transform function.
    
    Example:
        >>> transform = softplus_transform(min_val=0.1)
        >>> positive_params = transform(raw_params)
    """
    def fn(x: torch.Tensor) -> torch.Tensor:
        return min_val + torch.nn.functional.softplus(x, beta=beta)
    
    fn.min_val = min_val
    fn.__doc__ = f"Softplus transform: [{min_val}, inf)"
    
    return fn


def exp_transform(
    scale: float = 1.0
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Exponential transform for positive parameters.
    
    Maps real values to positive values via exp().
    Useful for parameters that must be strictly positive.
    
    Args:
        scale: Scaling factor applied before exp.
        
    Returns:
        Transform function.
    
    Example:
        >>> transform = exp_transform()
        >>> positive_params = transform(log_params)
    """
    def fn(x: torch.Tensor) -> torch.Tensor:
        return torch.exp(scale * x)
    
    fn.__doc__ = f"Exp transform with scale={scale}"
    
    return fn


def tanh_transform(
    min_val: float,
    max_val: float
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Tanh-based transform to bounded range.
    
    Similar to sigmoid but symmetric around zero input.
    
    Args:
        min_val: Minimum output value.
        max_val: Maximum output value.
        
    Returns:
        Transform function.
    """
    center = (max_val + min_val) / 2
    half_range = (max_val - min_val) / 2
    
    def fn(x: torch.Tensor) -> torch.Tensor:
        return center + half_range * torch.tanh(x)
    
    fn.min_val = min_val
    fn.max_val = max_val
    fn.__doc__ = f"Tanh transform: [{min_val}, {max_val}]"
    
    return fn


# =============================================================================
# Composite Constraints
# =============================================================================

def compose_constraints(*constraints: Callable) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Compose multiple constraint functions.
    
    Applies constraints in order (first to last).
    
    Args:
        *constraints: Constraint functions to compose.
        
    Returns:
        Composed constraint function.
    
    Example:
        >>> constraint = compose_constraints(
        ...     positive_constraint,
        ...     lambda x: torch.clamp(x, max=6000)
        ... )
    """
    def fn(x: torch.Tensor) -> torch.Tensor:
        for constraint in constraints:
            x = constraint(x)
        return x
    
    return fn
