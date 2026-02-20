"""
Transform functions and classes for geological properties.

This module provides utilities for transforming geological property data
to ensure values remain within physically plausible ranges and to prepare
data for neural network training.

Two APIs are provided:
    1. Functional API: Direct transform functions (clamp_properties, etc.)
    2. Class API: Composable transform objects (Clamp, Normalize, etc.)

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PropertyBounds:
    """
    Configuration for geological property bounds.

    Centralizes default ranges for common geological properties
    to avoid repetition across transform functions.

    Args:
        porosity: (min, max) bounds for porosity.
        shale_volume: (min, max) bounds for shale volume.
        saturation: (min, max) bounds for water saturation.
        permeability_log: (min, max) bounds for log permeability.

    Example:
        >>> bounds = PropertyBounds()
        >>> print(bounds.as_list())  # [[0.0, 0.4], [0.0, 1.0], [0.0, 1.0]]
        >>> bounds = PropertyBounds(porosity=(0.05, 0.35))
    """

    porosity: Tuple[float, float] = (0.0, 0.4)
    shale_volume: Tuple[float, float] = (0.0, 1.0)
    saturation: Tuple[float, float] = (0.0, 1.0)
    permeability_log: Tuple[float, float] = (-3.0, 3.0)

    def as_list(self, properties: List[str] = None) -> List[List[float]]:
        """
        Get bounds as list of [min, max] pairs.

        Args:
            properties: List of property names to include.
                       Defaults to ['porosity', 'shale_volume', 'saturation'].

        Returns:
            List of [min, max] bounds for each property.
        """
        if properties is None:
            properties = ['porosity', 'shale_volume', 'saturation']

        bounds_map = {
            'porosity': self.porosity,
            'shale_volume': self.shale_volume,
            'saturation': self.saturation,
            'permeability_log': self.permeability_log,
        }

        return [list(bounds_map.get(p, (0.0, 1.0))) for p in properties]


# Default bounds instance
DEFAULT_BOUNDS = PropertyBounds()


# =============================================================================
# Transform Classes (Object-Oriented API)
# =============================================================================

class Transform(ABC):
    """
    Abstract base class for transforms.

    Transforms are callable objects that can be composed into pipelines.
    Each transform should implement forward() and optionally inverse().

    Example:
        >>> class MyTransform(Transform):
        ...     def forward(self, x):
        ...         return x * 2
        ...     def inverse(self, x):
        ...         return x / 2
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward transform."""
        pass

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Apply inverse transform (optional)."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support inverse transform"
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward transform."""
        return self.forward(x)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Compose(Transform):
    """
    Compose multiple transforms into a pipeline.

    Args:
        transforms: List of transforms to apply in sequence.

    Example:
        >>> pipeline = Compose([Normalize(), Clamp()])
        >>> output = pipeline(input_data)
        >>> recovered = pipeline.inverse(output)
    """

    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all transforms in sequence."""
        for t in self.transforms:
            x = t(x)
        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Apply inverse transforms in reverse order."""
        for t in reversed(self.transforms):
            x = t.inverse(x)
        return x

    def __repr__(self) -> str:
        transforms_str = ", ".join(repr(t) for t in self.transforms)
        return f"Compose([{transforms_str}])"


class Normalize(Transform):
    """
    Normalize by subtracting mean and dividing by std.

    Args:
        means: Mean values per channel.
        stds: Standard deviation values per channel.

    Example:
        >>> norm = Normalize(means=[0.2, 0.3], stds=[0.1, 0.2])
        >>> normalized = norm(data)
        >>> original = norm.inverse(normalized)
    """

    def __init__(
        self,
        means: Optional[List[float]] = None,
        stds: Optional[List[float]] = None
    ):
        self.means = means or [0.2, 0.3, 0.5]
        self.stds = stds or [0.1, 0.2, 0.2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return normalize_properties(x, self.means, self.stds)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return denormalize_properties(x, self.means, self.stds)

    def __repr__(self) -> str:
        return f"Normalize(means={self.means}, stds={self.stds})"


class Denormalize(Transform):
    """
    Denormalize by multiplying by std and adding mean.

    Inverse of Normalize.
    """

    def __init__(
        self,
        means: Optional[List[float]] = None,
        stds: Optional[List[float]] = None
    ):
        self.means = means or [0.2, 0.3, 0.5]
        self.stds = stds or [0.1, 0.2, 0.2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return denormalize_properties(x, self.means, self.stds)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return normalize_properties(x, self.means, self.stds)


class Clamp(Transform):
    """
    Clamp values to specified bounds.

    Args:
        bounds: List of [min, max] bounds per channel.

    Example:
        >>> clamp = Clamp(bounds=[[0, 0.4], [0, 1]])
        >>> clamped = clamp(data)
    """

    def __init__(self, bounds: Optional[List[List[float]]] = None):
        self.bounds = bounds or DEFAULT_BOUNDS.as_list()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return clamp_properties(x, self.bounds)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        # Clamp is not invertible, return as-is
        return x

    def __repr__(self) -> str:
        return f"Clamp(bounds={self.bounds})"


class Sigmoid(Transform):
    """
    Apply sigmoid transform to map unconstrained values to bounded range.

    Args:
        ranges: List of [min, max] ranges per channel.

    Example:
        >>> sigmoid = Sigmoid(ranges=[[0, 0.4], [0, 1]])
        >>> bounded = sigmoid(unbounded_data)
        >>> unbounded = sigmoid.inverse(bounded)
    """

    def __init__(
        self,
        ranges: Optional[List[List[float]]] = None,
        eps: float = 1e-6
    ):
        self.ranges = ranges or DEFAULT_BOUNDS.as_list()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sigmoid_transform(x, self.ranges)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return logit_transform(x, self.ranges, self.eps)

    def __repr__(self) -> str:
        return f"Sigmoid(ranges={self.ranges})"


class Logit(Transform):
    """
    Apply logit (inverse sigmoid) transform.

    Maps bounded values to unconstrained range.
    """

    def __init__(
        self,
        ranges: Optional[List[List[float]]] = None,
        eps: float = 1e-6
    ):
        self.ranges = ranges or DEFAULT_BOUNDS.as_list()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return logit_transform(x, self.ranges, self.eps)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return sigmoid_transform(x, self.ranges)


class Exp(Transform):
    """
    Apply exponential transform for log-normal properties.

    Args:
        shift: Shift values per channel.
        scale: Scale values per channel.
    """

    def __init__(
        self,
        shift: Optional[List[float]] = None,
        scale: Optional[List[float]] = None,
        eps: float = 1e-6
    ):
        self.shift = shift or [0.0, 0.0, 0.0]
        self.scale = scale or [1.0, 1.0, 1.0]
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return exp_transform(x, self.shift, self.scale)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return log_transform(x, self.shift, self.scale, self.eps)


class Log(Transform):
    """
    Apply logarithmic transform for log-normal properties.

    Inverse of Exp transform.
    """

    def __init__(
        self,
        shift: Optional[List[float]] = None,
        scale: Optional[List[float]] = None,
        eps: float = 1e-6
    ):
        self.shift = shift or [0.0, 0.0, 0.0]
        self.scale = scale or [1.0, 1.0, 1.0]
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return log_transform(x, self.shift, self.scale, self.eps)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return exp_transform(x, self.shift, self.scale)


# =============================================================================
# Functional API
# =============================================================================


def clamp_properties(
        properties: torch.Tensor,
        bounds: Optional[List[List[float]]] = None
) -> torch.Tensor:
    """
    Clamp geological properties to physically plausible ranges.

    Args:
        properties (torch.Tensor): Properties tensor of shape [n_channels, ...]
        bounds (list, optional): List of [min, max] bounds for each channel
                               Defaults to standard geological property bounds

    Returns:
        torch.Tensor: Clamped properties tensor
    """
    if bounds is None:
        bounds = [
            [0.0, 0.4],  # Porosity range
            [0.0, 1.0],  # Shale volume range
            [0.0, 1.0],  # Water saturation range
        ]

    result = properties.clone()
    for i, (min_val, max_val) in enumerate(bounds):
        if i < properties.shape[0]:
            result[i] = torch.clamp(properties[i], min=min_val, max=max_val)

    return result


def sigmoid_transform(
        properties: torch.Tensor,
        ranges: Optional[List[List[float]]] = None
) -> torch.Tensor:
    """
    Transform unconstrained values to specified ranges using sigmoid function.

    This is useful when generating properties from unconstrained outputs
    (e.g., from neural networks).

    Args:
        properties (torch.Tensor): Properties tensor of shape [n_channels, ...]
        ranges (list, optional): List of [min, max] ranges for each channel
                               Defaults to standard geological property ranges

    Returns:
        torch.Tensor: Transformed properties tensor
    """
    if ranges is None:
        ranges = [
            [0.0, 0.4],  # Porosity range
            [0.0, 1.0],  # Shale volume range
            [0.0, 1.0],  # Water saturation range
        ]

    result = properties.clone()
    for i, (min_val, max_val) in enumerate(ranges):
        if i < properties.shape[0]:
            scale = max_val - min_val
            result[i] = min_val + scale * torch.sigmoid(properties[i])

    return result


def logit_transform(
        properties: torch.Tensor,
        ranges: Optional[List[List[float]]] = None,
        eps: float = 1e-6
) -> torch.Tensor:
    """
    Apply inverse sigmoid (logit) transform to convert bounded values
    to unconstrained values.

    This is the inverse of sigmoid_transform and useful for preparing
    bounded data for neural network training.

    Args:
        properties (torch.Tensor): Properties tensor of shape [n_channels, ...]
        ranges (list, optional): List of [min, max] ranges for each channel
        eps (float): Small value to prevent numerical instability

    Returns:
        torch.Tensor: Logit-transformed properties tensor
    """
    if ranges is None:
        ranges = [
            [0.0, 0.4],  # Porosity range
            [0.0, 1.0],  # Shale volume range
            [0.0, 1.0],  # Water saturation range
        ]

    result = properties.clone()
    for i, (min_val, max_val) in enumerate(ranges):
        if i < properties.shape[0]:
            # First normalize to [0, 1]
            normalized = (properties[i] - min_val) / (max_val - min_val)
            # Clip to prevent numerical issues
            normalized = torch.clamp(normalized, eps, 1.0 - eps)
            # Apply logit transform
            result[i] = torch.log(normalized / (1.0 - normalized))

    return result


def normalize_properties(
        properties: torch.Tensor,
        means: Optional[List[float]] = None,
        stds: Optional[List[float]] = None
) -> torch.Tensor:
    """
    Normalize properties by subtracting mean and dividing by standard deviation.

    Args:
        properties (torch.Tensor): Properties tensor of shape [n_channels, ...]
        means (list, optional): List of mean values for each channel
        stds (list, optional): List of standard deviation values for each channel

    Returns:
        torch.Tensor: Normalized properties tensor
    """
    if means is None:
        means = [0.2, 0.3, 0.5]  # Default means

    if stds is None:
        stds = [0.1, 0.2, 0.2]  # Default standard deviations

    result = properties.clone()
    for i, (mean, std) in enumerate(zip(means, stds)):
        if i < properties.shape[0]:
            result[i] = (properties[i] - mean) / std

    return result


def denormalize_properties(
        properties: torch.Tensor,
        means: Optional[List[float]] = None,
        stds: Optional[List[float]] = None
) -> torch.Tensor:
    """
    Denormalize properties by multiplying by standard deviation and adding mean.

    This is the inverse of normalize_properties.

    Args:
        properties (torch.Tensor): Normalized properties tensor of shape [n_channels, ...]
        means (list, optional): List of mean values for each channel
        stds (list, optional): List of standard deviation values for each channel

    Returns:
        torch.Tensor: Denormalized properties tensor
    """
    if means is None:
        means = [0.2, 0.3, 0.5]  # Default means

    if stds is None:
        stds = [0.1, 0.2, 0.2]  # Default standard deviations

    result = properties.clone()
    for i, (mean, std) in enumerate(zip(means, stds)):
        if i < properties.shape[0]:
            result[i] = properties[i] * std + mean

    return result


def exp_transform(
        properties: torch.Tensor,
        shift: Optional[List[float]] = None,
        scale: Optional[List[float]] = None
) -> torch.Tensor:
    """
    Apply exponential transform to handle properties with log-normal distributions
    (like permeability).

    Args:
        properties (torch.Tensor): Properties tensor of shape [n_channels, ...]
        shift (list, optional): List of shift values for each channel
        scale (list, optional): List of scale values for each channel

    Returns:
        torch.Tensor: Exponentially transformed properties tensor
    """
    if shift is None:
        shift = [0.0, 0.0, 0.0]  # Default shifts

    if scale is None:
        scale = [1.0, 1.0, 1.0]  # Default scales

    result = properties.clone()
    for i, (s, sc) in enumerate(zip(shift, scale)):
        if i < properties.shape[0]:
            result[i] = torch.exp(properties[i] * sc + s)

    return result


def log_transform(
        properties: torch.Tensor,
        shift: Optional[List[float]] = None,
        scale: Optional[List[float]] = None,
        eps: float = 1e-6
) -> torch.Tensor:
    """
    Apply logarithmic transform for properties with log-normal distributions.

    This is the inverse of exp_transform.

    Args:
        properties (torch.Tensor): Properties tensor of shape [n_channels, ...]
        shift (list, optional): List of shift values for each channel
        scale (list, optional): List of scale values for each channel
        eps (float): Small value to prevent log(0)

    Returns:
        torch.Tensor: Logarithmically transformed properties tensor
    """
    if shift is None:
        shift = [0.0, 0.0, 0.0]  # Default shifts

    if scale is None:
        scale = [1.0, 1.0, 1.0]  # Default scales

    result = properties.clone()
    for i, (s, sc) in enumerate(zip(shift, scale)):
        if i < properties.shape[0]:
            # Add small epsilon to avoid log(0)
            result[i] = (torch.log(properties[i] + eps) - s) / sc

    return result


def compute_property_stats(
        properties: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean and standard deviation for each property channel.

    Args:
        properties (torch.Tensor): Properties tensor of shape [n_channels, ...]

    Returns:
        tuple: (means, stds) - Mean and standard deviation tensors
    """
    # Flatten all dimensions except channel dimension
    flat_shape = (properties.shape[0], -1)
    flat_props = properties.reshape(flat_shape)

    # Compute statistics
    means = torch.mean(flat_props, dim=1)
    stds = torch.std(flat_props, dim=1)

    return means, stds