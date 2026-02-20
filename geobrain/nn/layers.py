"""
Neural network layers for GeoBrain.

Provides custom layer implementations including basic utility layers
and Bayesian neural network layers using the Flipout reparameterization
technique for efficient weight sampling.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Reshape(nn.Module):
    """
    Reshape layer for changing tensor dimensions.

    Args:
        *args: Target shape for the tensor (including batch dimension).

    Example:
        >>> reshape = Reshape(-1, 16, 8, 8)
        >>> x = torch.randn(32, 1024)
        >>> output = reshape(x)
        >>> print(output.shape)  # torch.Size([32, 16, 8, 8])
    """

    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape input tensor.

        Args:
            x: Input tensor.

        Returns:
            Reshaped tensor with target shape.
        """
        return x.view(self.shape)


class BaseVariationalLayer(nn.Module):
    """
    Base class for variational Bayesian layers.

    Implements common functionality for computing KL divergence
    between posterior and prior distributions, and managing
    DNN to BNN conversion flags.

    Attributes:
        _dnn_to_bnn_flag: Flag indicating DNN to BNN conversion mode.

    Example:
        Subclass implementation:
        >>> class MyBayesianLayer(BaseVariationalLayer):
        ...     def __init__(self):
        ...         super().__init__()
        ...         # Initialize variational parameters
        ...
        ...     def forward(self, x, return_kl=True):
        ...         # Implement forward with KL tracking
        ...         kl = self.kl_div(mu_q, sigma_q, mu_p, sigma_p)
        ...         return output, kl
    """

    def __init__(self):
        super().__init__()
        self._dnn_to_bnn_flag = False

    @property
    def dnn_to_bnn_flag(self) -> bool:
        """Flag for DNN to BNN conversion mode."""
        return self._dnn_to_bnn_flag

    @dnn_to_bnn_flag.setter
    def dnn_to_bnn_flag(self, value: bool):
        self._dnn_to_bnn_flag = value

    def kl_div(
        self,
        mu_q: torch.Tensor,
        sigma_q: torch.Tensor,
        mu_p: torch.Tensor,
        sigma_p: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence between two Gaussian distributions.

        Computes KL(Q || P) where Q ~ N(mu_q, sigma_q²) and
        P ~ N(mu_p, sigma_p²).

        Args:
            mu_q: Mean of distribution Q.
            sigma_q: Standard deviation of distribution Q.
            mu_p: Mean of distribution P.
            sigma_p: Standard deviation of distribution P.

        Returns:
            Mean KL divergence (scalar tensor).
        """
        kl = (
            torch.log(sigma_p) - torch.log(sigma_q) +
            (sigma_q ** 2 + (mu_q - mu_p) ** 2) / (2 * sigma_p ** 2) - 0.5
        )
        return kl.mean()


class LinearFlipout(BaseVariationalLayer):
    """
    Linear layer with Flipout reparameterization.

    Implements a variational linear layer using the Flipout technique
    for more efficient gradient estimation through decorrelated
    weight perturbations.

    The layer maintains a posterior distribution over weights:
        w ~ N(mu_weight, sigma_weight²)

    And computes KL divergence against a prior:
        p(w) ~ N(prior_mean, prior_variance)

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        prior_mean: Mean of the prior distribution. Default: 0.0.
        prior_variance: Variance of the prior distribution. Default: 1.0.
        posterior_mu_init: Initial mean for posterior. Default: 0.0.
        posterior_rho_init: Initial rho for posterior (controls initial
            variance via softplus). Default: -3.0.
        bias: Whether to include a bias term. Default: True.

    Example:
        >>> layer = LinearFlipout(256, 128)
        >>> x = torch.randn(32, 256)
        >>> output, kl = layer(x)
        >>> print(output.shape)  # torch.Size([32, 128])
        >>> print(kl)  # KL divergence scalar
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_mean: float = 0.0,
        prior_variance: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -3.0,
        bias: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init

        # Weight parameters
        self.mu_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.rho_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer(
            'eps_weight',
            torch.Tensor(out_features, in_features),
            persistent=False
        )
        self.register_buffer(
            'prior_weight_mu',
            torch.Tensor(out_features, in_features),
            persistent=False
        )
        self.register_buffer(
            'prior_weight_sigma',
            torch.Tensor(out_features, in_features),
            persistent=False
        )

        # Bias parameters
        if bias:
            self.mu_bias = nn.Parameter(torch.Tensor(out_features))
            self.rho_bias = nn.Parameter(torch.Tensor(out_features))
            self.register_buffer(
                'prior_bias_mu',
                torch.Tensor(out_features),
                persistent=False
            )
            self.register_buffer(
                'prior_bias_sigma',
                torch.Tensor(out_features),
                persistent=False
            )
            self.register_buffer(
                'eps_bias',
                torch.Tensor(out_features),
                persistent=False
            )
        else:
            self.register_buffer('prior_bias_mu', None, persistent=False)
            self.register_buffer('prior_bias_sigma', None, persistent=False)
            self.register_parameter('mu_bias', None)
            self.register_parameter('rho_bias', None)
            self.register_buffer('eps_bias', None, persistent=False)

        self.init_parameters()

    def init_parameters(self) -> None:
        """Initialize layer parameters."""
        # Initialize prior
        self.prior_weight_mu.fill_(self.prior_mean)
        self.prior_weight_sigma.fill_(self.prior_variance)

        # Initialize posterior
        self.mu_weight.data.normal_(mean=self.posterior_mu_init, std=0.1)
        self.rho_weight.data.normal_(mean=self.posterior_rho_init, std=0.1)

        if self.mu_bias is not None:
            self.prior_bias_mu.fill_(self.prior_mean)
            self.prior_bias_sigma.fill_(self.prior_variance)
            self.mu_bias.data.normal_(mean=self.posterior_mu_init, std=0.1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init, std=0.1)

    def kl_loss(self) -> torch.Tensor:
        """
        Compute KL divergence loss for this layer.

        Returns:
            KL divergence between posterior and prior.
        """
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        kl = self.kl_div(
            self.mu_weight, sigma_weight,
            self.prior_weight_mu, self.prior_weight_sigma
        )
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            kl += self.kl_div(
                self.mu_bias, sigma_bias,
                self.prior_bias_mu, self.prior_bias_sigma
            )
        return kl

    def forward(self, x: torch.Tensor, return_kl: bool = True):
        """
        Forward pass with Flipout sampling.

        Uses Flipout technique for decorrelated weight perturbations,
        providing lower variance gradient estimates than naive sampling.

        Args:
            x: Input tensor with shape (batch, in_features).
            return_kl: Whether to return KL divergence. Default: True.

        Returns:
            If return_kl is True, returns tuple (output, kl_divergence).
            Otherwise, returns only output tensor.
        """
        if self.dnn_to_bnn_flag:
            return_kl = False

        # Sample weight perturbation
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        delta_weight = sigma_weight * self.eps_weight.data.normal_()

        # Compute KL divergence
        if return_kl:
            kl = self.kl_div(
                self.mu_weight, sigma_weight,
                self.prior_weight_mu, self.prior_weight_sigma
            )

        # Sample bias perturbation
        bias = None
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            bias = sigma_bias * self.eps_bias.data.normal_()
            if return_kl:
                kl = kl + self.kl_div(
                    self.mu_bias, sigma_bias,
                    self.prior_bias_mu, self.prior_bias_sigma
                )

        # Compute mean output
        outputs = F.linear(x, self.mu_weight, self.mu_bias)

        # Flipout: decorrelated perturbations
        sign_input = x.clone().uniform_(-1, 1).sign()
        sign_output = outputs.clone().uniform_(-1, 1).sign()
        perturbed_outputs = F.linear(x * sign_input, delta_weight, bias) * sign_output

        if return_kl:
            return outputs + perturbed_outputs, kl
        return outputs + perturbed_outputs


class Conv2dFlipout(BaseVariationalLayer):
    """
    2D convolutional layer with Flipout reparameterization.

    Implements a variational 2D convolutional layer using Flipout
    for efficient gradient estimation in Bayesian neural networks.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution. Default: 1.
        padding: Padding added to input. Default: 0.
        dilation: Spacing between kernel elements. Default: 1.
        groups: Number of blocked connections. Default: 1.
        prior_mean: Mean of the prior distribution. Default: 0.0.
        prior_variance: Variance of the prior distribution. Default: 1.0.
        posterior_mu_init: Initial mean for posterior. Default: 0.0.
        posterior_rho_init: Initial rho for posterior. Default: -3.0.
        bias: Whether to include a bias term. Default: True.

    Example:
        >>> layer = Conv2dFlipout(3, 32, kernel_size=3, padding=1)
        >>> x = torch.randn(4, 3, 64, 64)
        >>> output, kl = layer(x)
        >>> print(output.shape)  # torch.Size([4, 32, 64, 64])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        prior_mean: float = 0.0,
        prior_variance: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -3.0,
        bias: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias

        self.kl = 0

        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init

        # Kernel shape: [out_channels, in_channels/groups, kH, kW]
        kernel_shape = (
            out_channels,
            in_channels // groups,
            kernel_size,
            kernel_size,
        )

        # Kernel parameters
        self.mu_kernel = nn.Parameter(torch.Tensor(*kernel_shape))
        self.rho_kernel = nn.Parameter(torch.Tensor(*kernel_shape))
        self.register_buffer('eps_kernel', torch.Tensor(*kernel_shape), persistent=False)
        self.register_buffer('prior_weight_mu', torch.Tensor(*kernel_shape), persistent=False)
        self.register_buffer('prior_weight_sigma', torch.Tensor(*kernel_shape), persistent=False)

        # Bias parameters
        if self.use_bias:
            self.mu_bias = nn.Parameter(torch.Tensor(out_channels))
            self.rho_bias = nn.Parameter(torch.Tensor(out_channels))
            self.register_buffer('eps_bias', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_mu', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_sigma', torch.Tensor(out_channels), persistent=False)
        else:
            self.register_parameter('mu_bias', None)
            self.register_parameter('rho_bias', None)
            self.register_buffer('eps_bias', None, persistent=False)
            self.register_buffer('prior_bias_mu', None, persistent=False)
            self.register_buffer('prior_bias_sigma', None, persistent=False)

        self.init_parameters()

    def init_parameters(self) -> None:
        """Initialize layer parameters."""
        # Initialize prior
        self.prior_weight_mu.data.fill_(self.prior_mean)
        self.prior_weight_sigma.data.fill_(self.prior_variance)

        # Initialize posterior
        self.mu_kernel.data.normal_(mean=self.posterior_mu_init, std=0.1)
        self.rho_kernel.data.normal_(mean=self.posterior_rho_init, std=0.1)

        if self.use_bias:
            self.mu_bias.data.normal_(mean=self.posterior_mu_init, std=0.1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init, std=0.1)
            self.prior_bias_mu.data.fill_(self.prior_mean)
            self.prior_bias_sigma.data.fill_(self.prior_variance)

    def kl_loss(self) -> torch.Tensor:
        """
        Compute KL divergence loss for this layer.

        Returns:
            KL divergence between posterior and prior.
        """
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        kl = self.kl_div(
            self.mu_kernel, sigma_weight,
            self.prior_weight_mu, self.prior_weight_sigma
        )
        if self.use_bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            kl += self.kl_div(
                self.mu_bias, sigma_bias,
                self.prior_bias_mu, self.prior_bias_sigma
            )
        return kl

    def forward(self, x: torch.Tensor, return_kl: bool = True):
        """
        Forward pass with Flipout sampling.

        Args:
            x: Input tensor with shape (batch, in_channels, H, W).
            return_kl: Whether to return KL divergence. Default: True.

        Returns:
            If return_kl is True, returns tuple (output, kl_divergence).
            Otherwise, returns only output tensor.
        """
        if self.dnn_to_bnn_flag:
            return_kl = False

        # Compute mean output
        outputs = F.conv2d(
            x,
            weight=self.mu_kernel,
            bias=self.mu_bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        # Flipout: sample perturbation signs
        sign_input = x.clone().uniform_(-1, 1).sign()
        sign_output = outputs.clone().uniform_(-1, 1).sign()

        # Sample weight perturbation
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        eps_kernel = self.eps_kernel.data.normal_()
        delta_kernel = sigma_weight * eps_kernel

        if return_kl:
            kl = self.kl_div(
                self.mu_kernel, sigma_weight,
                self.prior_weight_mu, self.prior_weight_sigma
            )

        # Sample bias perturbation
        bias = None
        if self.use_bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            eps_bias = self.eps_bias.data.normal_()
            bias = sigma_bias * eps_bias
            if return_kl:
                kl = kl + self.kl_div(
                    self.mu_bias, sigma_bias,
                    self.prior_bias_mu, self.prior_bias_sigma
                )

        # Compute perturbed output
        perturbed_outputs = F.conv2d(
            x * sign_input,
            weight=delta_kernel,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        ) * sign_output

        self.kl = kl if return_kl else 0

        if return_kl:
            return outputs + perturbed_outputs, kl
        return outputs + perturbed_outputs


class Conv3dFlipout(BaseVariationalLayer):
    """
    3D convolutional layer with Flipout reparameterization.

    Implements a variational 3D convolutional layer using Flipout
    for efficient gradient estimation in Bayesian neural networks.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution. Default: 1.
        padding: Padding added to input. Default: 0.
        dilation: Spacing between kernel elements. Default: 1.
        groups: Number of blocked connections. Default: 1.
        prior_mean: Mean of the prior distribution. Default: 0.0.
        prior_variance: Variance of the prior distribution. Default: 1.0.
        posterior_mu_init: Initial mean for posterior. Default: 0.0.
        posterior_rho_init: Initial rho for posterior. Default: -3.0.
        bias: Whether to include a bias term. Default: True.

    Example:
        >>> layer = Conv3dFlipout(1, 32, kernel_size=3, padding=1)
        >>> x = torch.randn(4, 1, 16, 16, 16)
        >>> output, kl = layer(x)
        >>> print(output.shape)  # torch.Size([4, 32, 16, 16, 16])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        prior_mean: float = 0.0,
        prior_variance: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -3.0,
        bias: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.kl = 0

        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init

        # Kernel shape
        kernel_shape = (
            out_channels,
            in_channels // groups,
            kernel_size,
            kernel_size,
            kernel_size,
        )

        # Kernel parameters
        self.mu_kernel = nn.Parameter(torch.Tensor(*kernel_shape))
        self.rho_kernel = nn.Parameter(torch.Tensor(*kernel_shape))
        self.register_buffer('eps_kernel', torch.Tensor(*kernel_shape), persistent=False)
        self.register_buffer('prior_weight_mu', torch.Tensor(*kernel_shape), persistent=False)
        self.register_buffer('prior_weight_sigma', torch.Tensor(*kernel_shape), persistent=False)

        # Bias parameters
        if self.bias:
            self.mu_bias = nn.Parameter(torch.Tensor(out_channels))
            self.rho_bias = nn.Parameter(torch.Tensor(out_channels))
            self.register_buffer('eps_bias', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_mu', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_sigma', torch.Tensor(out_channels), persistent=False)
        else:
            self.register_parameter('mu_bias', None)
            self.register_parameter('rho_bias', None)
            self.register_buffer('eps_bias', None, persistent=False)
            self.register_buffer('prior_bias_mu', None, persistent=False)
            self.register_buffer('prior_bias_sigma', None, persistent=False)

        self.init_parameters()

    def init_parameters(self) -> None:
        """Initialize layer parameters."""
        # Initialize prior
        self.prior_weight_mu.data.fill_(self.prior_mean)
        self.prior_weight_sigma.data.fill_(self.prior_variance)

        # Initialize posterior
        self.mu_kernel.data.normal_(mean=self.posterior_mu_init, std=0.1)
        self.rho_kernel.data.normal_(mean=self.posterior_rho_init, std=0.1)

        if self.bias:
            self.mu_bias.data.normal_(mean=self.posterior_mu_init, std=0.1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init, std=0.1)
            self.prior_bias_mu.data.fill_(self.prior_mean)
            self.prior_bias_sigma.data.fill_(self.prior_variance)

    def kl_loss(self) -> torch.Tensor:
        """
        Compute KL divergence loss for this layer.

        Returns:
            KL divergence between posterior and prior.
        """
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        kl = self.kl_div(
            self.mu_kernel, sigma_weight,
            self.prior_weight_mu, self.prior_weight_sigma
        )
        if self.bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            kl += self.kl_div(
                self.mu_bias, sigma_bias,
                self.prior_bias_mu, self.prior_bias_sigma
            )
        return kl

    def forward(self, x: torch.Tensor, return_kl: bool = True):
        """
        Forward pass with Flipout sampling.

        Args:
            x: Input tensor with shape (batch, in_channels, D, H, W).
            return_kl: Whether to return KL divergence. Default: True.

        Returns:
            If return_kl is True, returns tuple (output, kl_divergence).
            Otherwise, returns only output tensor.
        """
        if self.dnn_to_bnn_flag:
            return_kl = False

        # Compute mean output
        outputs = F.conv3d(
            x,
            weight=self.mu_kernel,
            bias=self.mu_bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        # Flipout: sample perturbation signs
        sign_input = x.clone().uniform_(-1, 1).sign()
        sign_output = outputs.clone().uniform_(-1, 1).sign()

        # Sample weight perturbation
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        eps_kernel = self.eps_kernel.data.normal_()
        delta_kernel = sigma_weight * eps_kernel

        if return_kl:
            kl = self.kl_div(
                self.mu_kernel, sigma_weight,
                self.prior_weight_mu, self.prior_weight_sigma
            )

        # Sample bias perturbation
        bias = None
        if self.bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            eps_bias = self.eps_bias.data.normal_()
            bias = sigma_bias * eps_bias
            if return_kl:
                kl = kl + self.kl_div(
                    self.mu_bias, sigma_bias,
                    self.prior_bias_mu, self.prior_bias_sigma
                )

        # Compute perturbed output
        perturbed_outputs = F.conv3d(
            x * sign_input,
            weight=delta_kernel,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        ) * sign_output

        self.kl = kl if return_kl else 0

        if return_kl:
            return outputs + perturbed_outputs, kl
        return outputs + perturbed_outputs


# =============================================================================
# Utility Functions
# =============================================================================

def get_kl_loss(model: nn.Module) -> torch.Tensor:
    """
    Collect total KL divergence from all variational layers in a model.

    Recursively traverses the model and sums KL divergence from all
    layers that have a `kl_loss` method (e.g., LinearFlipout, Conv2dFlipout).

    Args:
        model: PyTorch model containing variational layers.

    Returns:
        Total KL divergence (scalar tensor).

    Example:
        >>> model = nn.Sequential(
        ...     Conv2dFlipout(3, 32, 3, padding=1),
        ...     nn.ReLU(),
        ...     LinearFlipout(32*64*64, 10)
        ... )
        >>> x = torch.randn(4, 3, 64, 64)
        >>> # Forward pass (KL computed internally)
        >>> output = model(x)
        >>> # Collect KL for loss
        >>> kl = get_kl_loss(model)
        >>> loss = nll_loss + kl_weight * kl
    """
    kl = torch.tensor(0.0)
    device = None

    for module in model.modules():
        if hasattr(module, 'kl_loss') and callable(module.kl_loss):
            layer_kl = module.kl_loss()
            if device is None and layer_kl.device.type != 'cpu':
                device = layer_kl.device
            kl = kl + layer_kl.cpu()

    if device is not None:
        kl = kl.to(device)

    return kl


def count_variational_parameters(model: nn.Module) -> int:
    """
    Count total number of variational parameters in a model.

    Args:
        model: PyTorch model containing variational layers.

    Returns:
        Total number of variational parameters.

    Example:
        >>> model = LinearFlipout(256, 128)
        >>> n_params = count_variational_parameters(model)
    """
    total = 0
    for module in model.modules():
        if isinstance(module, BaseVariationalLayer):
            for param in module.parameters():
                total += param.numel()
    return total