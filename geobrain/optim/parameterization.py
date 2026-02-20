"""
Model parameterization strategies for inverse problems.

Three strategies are provided:
    1. ExplicitParameterization - Direct optimization of model parameters
    2. LatentParameterization - Optimize latent code with fixed decoder
    3. NetworkParameterization - Optimize network weights with fixed input (DIP)

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
import torch.nn as nn
from abc import abstractmethod
from typing import List, Tuple, Optional, Callable


class Parameterization(nn.Module):
    """
    Abstract base class for model parameterization strategies.

    Parameterization defines how model parameters are represented
    and optimized. Different strategies suit different inverse problems.

    Inherits from nn.Module to leverage PyTorch's parameter management,
    device handling (.to(), .cuda(), .cpu()), and serialization
    (state_dict / load_state_dict).

    All parameterizations must implement:
        - forward(): Returns (model, regularization_term)
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate model parameters and optional regularization term.

        Returns:
            Tuple of (model, regularization_term).
            regularization_term is None if no regularizer is defined.
        """
        pass

    def optimizable_parameters(self) -> List[nn.Parameter]:
        """
        Return parameters that should be passed to the optimizer.

        By default, returns all parameters with requires_grad=True.
        This filters out frozen parameters (e.g., decoder weights in
        LatentParameterization).

        Returns:
            List of trainable parameters.
        """
        return [p for p in self.parameters() if p.requires_grad]

    @property
    def device(self) -> torch.device:
        """Get the device of the parameterization."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            try:
                return next(self.buffers()).device
            except StopIteration:
                return torch.device('cpu')

    def reset(self) -> None:
        """Reset parameters to initial state. Override in subclasses."""
        pass


class ExplicitParameterization(Parameterization):
    """
    Direct optimization of model parameters.
    
    The simplest and most common approach: directly optimize the
    model parameters (e.g., velocity, density) with optional
    regularization and constraints.
    
    Args:
        initial_model: Initial model parameters.
        regularizer: Optional regularization function f(model) -> scalar.
        transform: Optional differentiable transform (e.g., exp for log-domain).
        constraints: Optional projection function for hard constraints.
        device: Computation device.
    
    Example:
        >>> from geobrain.optim import ExplicitParameterization, total_variation, bound_constraint
        >>> 
        >>> param = ExplicitParameterization(
        ...     initial_model=velocity_model,
        ...     regularizer=total_variation,
        ...     constraints=bound_constraint(1500, 6000)
        ... )
        >>> inverter = Inverter(param, forward_fn)
    """

    def __init__(
        self,
        initial_model: torch.Tensor,
        regularizer: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        constraints: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        device: Optional[torch.device] = None
    ):
        super().__init__()

        self.model = nn.Parameter(initial_model.clone(), requires_grad=True)
        self.register_buffer('_initial_model', initial_model.clone().detach())

        self.regularizer = regularizer
        self.transform = transform
        self.constraints = constraints

        if device is not None:
            self.to(device)

    def forward(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Apply hard constraints (projection)
        if self.constraints is not None:
            with torch.no_grad():
                self.model.data = self.constraints(self.model.data)

        # Apply differentiable transformation
        model = self.transform(self.model) if self.transform else self.model

        # Compute regularization term
        reg_term = self.regularizer(self.model) if self.regularizer else None

        return model, reg_term

    def reset(self) -> None:
        """Reset model to initial values."""
        with torch.no_grad():
            self.model.data.copy_(self._initial_model)

    @property
    def shape(self) -> torch.Size:
        """Shape of the model parameters."""
        return self.model.shape

    @property
    def numel(self) -> int:
        """Number of model parameters."""
        return self.model.numel()

    def __repr__(self) -> str:
        return (
            f"ExplicitParameterization(shape={tuple(self.shape)}, "
            f"has_regularizer={self.regularizer is not None}, "
            f"has_constraints={self.constraints is not None})"
        )


class LatentParameterization(Parameterization):
    """
    Optimize latent code with fixed pre-trained decoder.
    
    Instead of optimizing model parameters directly, optimize a
    low-dimensional latent code that is decoded to model space.
    This provides implicit regularization through the decoder.
    
    Args:
        decoder: Pre-trained decoder network (will be frozen).
        initial_latent: Initial latent code.
        regularizer: Optional regularization on latent code.
        latent_constraints: Optional constraints on latent code.
        device: Computation device.
    
    Example:
        >>> param = LatentParameterization(
        ...     decoder=pretrained_vae_decoder,
        ...     initial_latent=torch.randn(1, 128)
        ... )
        >>> inverter = Inverter(param, forward_fn)
    """

    def __init__(
        self,
        decoder: nn.Module,
        initial_latent: torch.Tensor,
        regularizer: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        latent_constraints: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        device: Optional[torch.device] = None
    ):
        super().__init__()

        # Freeze decoder weights (registered as submodule, moved by .to())
        self.decoder = decoder
        self.decoder.eval()
        for p in self.decoder.parameters():
            p.requires_grad = False

        # Trainable latent code
        self.latent = nn.Parameter(initial_latent.clone(), requires_grad=True)
        self.register_buffer('_initial_latent', initial_latent.clone().detach())

        self.regularizer = regularizer
        self.latent_constraints = latent_constraints

        if device is not None:
            self.to(device)

    def forward(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Apply latent constraints
        if self.latent_constraints is not None:
            with torch.no_grad():
                self.latent.data = self.latent_constraints(self.latent.data)

        # Generate model through decoder
        model = self.decoder(self.latent)

        # Regularization on latent code (e.g., L2 for standard normal prior)
        reg_term = self.regularizer(self.latent) if self.regularizer else None

        return model, reg_term

    def reset(self) -> None:
        """Reset latent code to initial values."""
        with torch.no_grad():
            self.latent.data.copy_(self._initial_latent)

    @property
    def latent_dim(self) -> int:
        """Dimensionality of latent space."""
        return self.latent.numel()

    def __repr__(self) -> str:
        return (
            f"LatentParameterization(latent_dim={self.latent_dim}, "
            f"has_regularizer={self.regularizer is not None})"
        )


class NetworkParameterization(Parameterization):
    """
    Optimize network weights with fixed input (Deep Image Prior).
    
    The model is generated by a neural network with fixed random input.
    Optimization is performed over network weights, providing implicit
    regularization through the network architecture.
    
    Reference:
        Ulyanov et al. (2018). Deep Image Prior. CVPR.
    
    Args:
        network: Neural network to optimize.
        fixed_input: Fixed input tensor (typically random noise).
        regularizer: Optional regularization on network weights.
        output_transform: Optional transform applied to network output.
        device: Computation device.
    
    Example:
        >>> network = UNet(in_channels=1, out_channels=1)
        >>> fixed_input = torch.randn(1, 1, 64, 64)
        >>> param = NetworkParameterization(network, fixed_input)
        >>> inverter = Inverter(param, forward_fn)
    """

    def __init__(
        self,
        network: nn.Module,
        fixed_input: torch.Tensor,
        regularizer: Optional[Callable[[nn.Module], torch.Tensor]] = None,
        output_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        device: Optional[torch.device] = None
    ):
        super().__init__()

        self.network = network
        self.register_buffer('fixed_input', fixed_input.detach().clone())

        # Store initial state for reset
        self._initial_state = {
            k: v.clone().detach()
            for k, v in self.network.state_dict().items()
        }

        self.regularizer = regularizer
        self.output_transform = output_transform

        if device is not None:
            self.to(device)

    def forward(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Generate model through network
        model = self.network(self.fixed_input)

        # Apply output transformation (e.g., sigmoid for bounded output)
        if self.output_transform is not None:
            model = self.output_transform(model)

        # Regularization on network weights
        reg_term = self.regularizer(self.network) if self.regularizer else None

        return model, reg_term

    def reset(self) -> None:
        """Reset network to initial weights."""
        device = next(self.network.parameters()).device
        state = {k: v.to(device) for k, v in self._initial_state.items()}
        self.network.load_state_dict(state)

    @property
    def n_parameters(self) -> int:
        """Number of network parameters."""
        return sum(p.numel() for p in self.network.parameters())

    def __repr__(self) -> str:
        return (
            f"NetworkParameterization(n_parameters={self.n_parameters}, "
            f"input_shape={tuple(self.fixed_input.shape)}, "
            f"has_regularizer={self.regularizer is not None})"
        )
