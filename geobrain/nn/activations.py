"""
Custom activation functions for GeoBrain.

Provides activation functions compatible with Bayesian neural
networks that require KL divergence tracking.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReLU(nn.Module):
    """
    ReLU activation compatible with Bayesian Neural Networks.

    Returns both the activated tensor and a KL divergence of 0,
    maintaining compatibility with BNN layers that track KL divergence.

    Args:
        inplace: Whether to perform the operation in-place.
            Default: False.

    Example:
        >>> relu = ReLU()
        >>> output, kl = relu((input_tensor,))
        >>> print(kl)  # 0

    Note:
        Input should be a tuple where the first element is the tensor
        to activate, matching the output format of Flipout layers.
    """

    __constants__ = ['inplace']

    def __init__(self, inplace: bool = False):
        super(ReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        """
        Apply ReLU activation.

        Args:
            input: Tuple where input[0] is the tensor to activate.

        Returns:
            Tuple of (activated_tensor, kl_divergence) where
            kl_divergence is always 0.
        """
        kl = 0
        return F.relu(input[0], inplace=self.inplace), kl

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


class ClippedLinearActivation(nn.Module):
    """
    Clipped linear activation that constrains outputs to [0, 1].

    Implements a piecewise linear function:
        - x ≤ 0: output = 0 (lower bound)
        - 0 < x ≤ 1: output = x (identity)
        - x > 1: output = 1 (upper bound)

    Useful for parameters that must be constrained to a [0, 1] range,
    such as probabilities or normalized coefficients.

    Example:
        >>> activation = ClippedLinearActivation()
        >>> x = torch.tensor([-0.5, 0.3, 0.8, 1.5])
        >>> output = activation(x)
        >>> print(output)  # tensor([0.0, 0.3, 0.8, 1.0])
    """

    def __init__(self):
        super(ClippedLinearActivation, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply clipped linear activation.

        Args:
            x: Input tensor of any shape.

        Returns:
            Tensor with values clipped to [0, 1].
        """
        return torch.where(
            x <= 0,
            torch.zeros_like(x),
            torch.where(x <= 1, x, torch.ones_like(x))
        )