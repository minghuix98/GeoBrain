"""
Abstract base classes for geological model simulators.

Two base classes are provided:
    1. GeostatSimulator - For geostatistical methods (FFT-MA, SGS, etc.)
    2. GenerativeSimulator - For generative AI models (GAN, VAE, Diffusion)

Extension Guide:
    1. Geostatistical methods: Inherit GeostatSimulator, implement _simulate()
    2. Generative methods: Inherit GenerativeSimulator, implement _simulate()
    3. Use @register_model decorator to register with factory

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import fields
from typing import Optional, Dict, Union, Callable
import warnings


class BaseSimulator(ABC):
    """
    Abstract base class for all geological model simulators.
    
    Provides common functionality for configuration handling, random seed
    management, and output transformation. Subclasses must implement
    the _simulate() method.
    
    Args:
        transform: Optional output transform function (e.g., torch.exp for log-normal).
    
    All subclasses must implement:
        - _simulate(config): Core simulation logic returning Tensor or Dict[str, Tensor]
    """
    
    def __init__(
        self,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.transform = transform
    
    def simulate(
        self, 
        config=None, 
        **overrides
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Run simulation.
        
        This is the main entry point for all simulators. It handles
        configuration, random seeding, and output transformation.
        
        Args:
            config: Configuration object (uses default if None).
            **overrides: Override specific configuration parameters.
            
        Returns:
            Simulated field as Tensor, or Dict[str, Tensor] for co-simulation.
        
        Example:
            >>> simulator = FFTMASimulator()
            >>> field = simulator.simulate(config)
            >>> # Or with overrides
            >>> field = simulator.simulate(config, seed=42, n_realizations=5)
        """
        # Get default config if not provided
        if config is None:
            config = self._get_default_config()
        
        # Apply parameter overrides
        if overrides:
            config = self._apply_overrides(config, overrides)
        
        # Set random seed
        self._set_seed(config)
        
        # Run simulation
        result = self._simulate(config)
        
        # Apply output transform
        result = self._apply_transform(result)
        
        return result
    
    @abstractmethod
    def _simulate(self, config) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Core simulation logic - must be implemented by subclasses.
        
        Args:
            config: Configuration object.
            
        Returns:
            Simulated field(s).
        """
        pass
    
    def _get_default_config(self):
        """Get default configuration. Override in subclasses if needed."""
        from .config import SimulationConfig
        return SimulationConfig()
    
    def _apply_overrides(self, config, overrides: dict):
        """Apply parameter overrides to configuration."""
        # Use dataclass fields to get valid field names
        cfg_dict = {f.name: getattr(config, f.name) for f in fields(config)}
        cfg_dict.update(overrides)
        return config.__class__(**cfg_dict)
    
    def _set_seed(self, config) -> None:
        """Set random seed for reproducibility."""
        seed = getattr(config, 'seed', None)
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
    
    def _apply_transform(
        self, 
        result: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply output transform if defined."""
        if self.transform is None:
            return result
        
        if isinstance(result, dict):
            return {k: self.transform(v) for k, v in result.items()}
        return self.transform(result)
    
    def __call__(
        self,
        config=None,
        **overrides
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Shortcut for simulate() - enables simulator(config) syntax.

        Args:
            config: Configuration object (uses default if None).
            **overrides: Override specific configuration parameters.

        Returns:
            Simulated field as Tensor, or Dict[str, Tensor] for co-simulation.

        Example:
            >>> simulator = FFTMASimulator()
            >>> field = simulator(config)
            >>> # Or with parameter overrides
            >>> field = simulator(config, n_realizations=10, seed=42)
        """
        return self.simulate(config, **overrides)

    def simulate_batch(
        self,
        n_batches: int,
        config=None,
        **overrides
    ) -> torch.Tensor:
        """
        Generate multiple independent simulation batches.

        Useful for generating a large ensemble of realizations
        with different random seeds.

        Args:
            n_batches: Number of batches to generate.
            config: Configuration object (uses default if None).
            **overrides: Override specific configuration parameters.

        Returns:
            Stacked tensor of shape [n_batches, n_realizations, *shape]
            or [n_batches * n_realizations, *shape] if flattened.

        Example:
            >>> simulator = FFTMASimulator()
            >>> # Generate 5 batches of 10 realizations each
            >>> fields = simulator.simulate_batch(5, config, n_realizations=10)
        """
        if config is None:
            config = self._get_default_config()

        results = []
        base_seed = getattr(config, 'seed', None)

        for i in range(n_batches):
            # Use different seed for each batch
            if base_seed is not None:
                batch_overrides = {**overrides, 'seed': base_seed + i}
            else:
                batch_overrides = overrides

            result = self.simulate(config, **batch_overrides)

            # Handle dict results (co-simulation)
            if isinstance(result, dict):
                result = torch.stack(list(result.values()), dim=0)

            results.append(result)

        return torch.stack(results, dim=0)

    @property
    def supports_conditioning(self) -> bool:
        """Whether this simulator supports conditioning data."""
        return False

    @property
    def is_differentiable(self) -> bool:
        """Whether this simulator supports gradient computation."""
        return False


class GeostatSimulator(BaseSimulator):
    """
    Base class for geostatistical simulation methods.
    
    Provides support for conditioning data and variogram/correlation models.
    Suitable for methods like FFT-MA, SGS, Turning Bands, etc.
    
    Args:
        correlation_model: Correlation model type ('spherical', 'exponential', 'gaussian').
        conditioning_data: Conditioning data values at known locations.
        conditioning_locations: Coordinates of conditioning data points.
        transform: Optional output transform function.
    
    Example:
        Extending with a new geostatistical method:
        
        >>> class TurningBandsSimulator(GeostatSimulator):
        ...     @property
        ...     def supports_conditioning(self):
        ...         return False
        ...     
        ...     def _simulate(self, config):
        ...         # Implementation here
        ...         pass
    """
    
    SUPPORTED_CORRELATION_MODELS = ("spherical", "exponential", "gaussian")
    
    def __init__(
        self,
        correlation_model: str = "spherical",
        conditioning_data: Optional[torch.Tensor] = None,
        conditioning_locations: Optional[torch.Tensor] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(transform=transform)
        
        # Validate correlation model
        if correlation_model not in self.SUPPORTED_CORRELATION_MODELS:
            raise ValueError(
                f"Unknown correlation model: '{correlation_model}'. "
                f"Supported: {self.SUPPORTED_CORRELATION_MODELS}"
            )
        
        self.correlation_model = correlation_model
        self.conditioning_data = conditioning_data
        self.conditioning_locations = conditioning_locations
        
        # Validate conditioning data
        self._validate_conditioning()
    
    def _validate_conditioning(self) -> None:
        """Validate conditioning data consistency."""
        if self.conditioning_data is None:
            return
        
        if self.conditioning_locations is None:
            raise ValueError(
                "conditioning_locations required when conditioning_data is provided"
            )
        
        if len(self.conditioning_data) != len(self.conditioning_locations):
            raise ValueError(
                f"Length mismatch: conditioning_data ({len(self.conditioning_data)}) "
                f"vs conditioning_locations ({len(self.conditioning_locations)})"
            )
        
        if not self.supports_conditioning:
            warnings.warn(
                f"{self.__class__.__name__} does not support conditioning. "
                "Conditioning data will be ignored.",
                UserWarning
            )
    
    def set_conditioning(
        self,
        data: torch.Tensor,
        locations: torch.Tensor
    ) -> 'GeostatSimulator':
        """
        Set conditioning data (chainable).
        
        Args:
            data: Conditioning data values.
            locations: Coordinates of conditioning points (n_points, ndim).
            
        Returns:
            Self for method chaining.
        
        Example:
            >>> simulator = SGSSimulator().set_conditioning(well_data, well_coords)
            >>> field = simulator.simulate(config)
        """
        self.conditioning_data = data
        self.conditioning_locations = locations
        self._validate_conditioning()
        return self
    
    @property
    def is_differentiable(self) -> bool:
        """Geostatistical methods using PyTorch operations are differentiable."""
        return True


class GenerativeSimulator(BaseSimulator):
    """
    Base class for generative AI simulation methods.
    
    Provides support for neural network models with checkpoint loading
    and latent space sampling. Suitable for GAN, VAE, Diffusion, etc.
    
    Args:
        model: Pre-trained generator/decoder network.
        checkpoint_path: Path to saved model checkpoint.
        latent_dim: Dimension of latent space for sampling.
        transform: Optional output transform function.
    
    Example:
        Extending with a new generative method:
        
        >>> class FlowSimulator(GenerativeSimulator):
        ...     def _simulate(self, config):
        ...         model = self._ensure_model_loaded(config.torch_device)
        ...         z = self._sample_latent(config.n_realizations, config.torch_device)
        ...         return model(z)
        ...     
        ...     def _load_checkpoint(self, device):
        ...         self._model = MyFlowModel()
        ...         self._model.load_state_dict(torch.load(self.checkpoint_path))
    """
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        checkpoint_path: Optional[str] = None,
        latent_dim: Optional[int] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        conditioning_data: Optional[torch.Tensor] = None,
    ):
        super().__init__(transform=transform)

        self._model = model
        self.checkpoint_path = checkpoint_path
        self.latent_dim = latent_dim
        self.conditioning_data = conditioning_data
    
    def _get_default_config(self):
        """Get default configuration for generative models."""
        from .config import GenerativeConfig
        return GenerativeConfig()
    
    def _ensure_model_loaded(self, device: torch.device) -> nn.Module:
        """
        Ensure model is loaded and on correct device.
        
        Args:
            device: Target device.
            
        Returns:
            Loaded model in eval mode.
            
        Raises:
            ValueError: If no model or checkpoint available.
        """
        if self._model is None:
            if self.checkpoint_path is None:
                raise ValueError(
                    "No model available. Provide 'model' or 'checkpoint_path' "
                    "to the constructor."
                )
            self._load_checkpoint(device)
        
        self._model = self._model.to(device).eval()
        return self._model
    
    def _load_checkpoint(self, device: torch.device) -> None:
        """
        Load model from checkpoint - subclasses must implement.
        
        Args:
            device: Target device for loaded model.
            
        Raises:
            NotImplementedError: Subclass must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _load_checkpoint() "
            "to load model from checkpoint_path."
        )
    
    def _sample_latent(self, n: int, device: torch.device) -> torch.Tensor:
        """
        Sample latent vectors from standard normal distribution.
        
        Args:
            n: Number of samples.
            device: Target device.
            
        Returns:
            Latent vectors of shape (n, latent_dim).
            
        Raises:
            ValueError: If latent_dim not specified.
        """
        if self.latent_dim is None:
            raise ValueError(
                "latent_dim not specified. Provide it to the constructor."
            )
        return torch.randn(n, self.latent_dim, device=device)
    
    def set_conditioning(
        self,
        data: torch.Tensor
    ) -> 'GenerativeSimulator':
        """
        Set conditioning data for conditional generation (chainable).

        For generative models, conditioning can be class labels, embeddings,
        or any tensor that guides the generation process.

        Args:
            data: Conditioning data tensor.

        Returns:
            Self for method chaining.

        Example:
            >>> cgan = CGANSimulator(model=generator, latent_dim=128)
            >>> # Set class labels for conditional generation
            >>> field = cgan.set_conditioning(class_labels).simulate(config)
        """
        self.conditioning_data = data
        return self

    @property
    def decoder(self) -> Optional[nn.Module]:
        """
        Get decoder network for integration with optim module.

        Returns:
            Decoder module if available, None otherwise.

        Example:
            >>> vae_sim = VAESimulator(model=vae, latent_dim=128)
            >>> inverter = Inverter.create_latent(
            ...     decoder=vae_sim.decoder,
            ...     initial_latent=torch.zeros(1, 128),
            ...     forward_fn=forward
            ... )
        """
        if self._model is None:
            return None
        if hasattr(self._model, 'decoder'):
            return self._model.decoder
        return self._model

    @property
    def is_differentiable(self) -> bool:
        """Generative models are differentiable by design."""
        return True
