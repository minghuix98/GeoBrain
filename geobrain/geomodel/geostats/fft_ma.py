"""
FFT-based Moving Average (FFT-MA) for spatially correlated field generation.

FFT-MA generates unconditional Gaussian random fields with specified
spatial correlation structure using frequency domain filtering.

Features:
    - 2D and 3D field generation
    - Anisotropic correlation (different horizontal/vertical ranges)
    - Multi-variable co-simulation via Cholesky decomposition
    - GPU acceleration with automatic device handling
    - Batch generation for multiple realizations

Theory:
    FFT-MA converts white noise to spatially correlated fields via
    frequency domain multiplication: Z = IFFT(sqrt(S) * FFT(W))
    where S is the power spectrum and W is white noise.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
from typing import Optional, Union, Dict, Tuple
import warnings

from ..base import GeostatSimulator
from ..config import SimulationConfig, CoSimConfig
from ..registry import register_model


class FFTMovingAverage:
    """
    Core FFT-MA algorithm implementation.

    Generates spatially correlated Gaussian random fields using
    FFT-based filtering of white noise.

    Args:
        lh: Horizontal correlation length.
        lv: Vertical correlation length.
        shape: Grid dimensions (nx, ny) or (nx, ny, nz).
        correlation_model: Correlation model type ('spherical', 'exponential', 'gaussian').

    Example:
        >>> fft_ma = FFTMovingAverage(lh=20, lv=5, shape=(64, 64, 128))
        >>> field = fft_ma.generate(device=torch.device('cuda'))
        >>> print(field.shape)  # (64, 64, 128)

        >>> # With exponential model
        >>> fft_ma = FFTMovingAverage(lh=20, lv=5, shape=(64, 64),
        ...                           correlation_model='exponential')
    """

    def __init__(
        self,
        lh: float,
        lv: float,
        shape: Tuple[int, ...],
        correlation_model: str = "spherical",
    ):
        self.lh = lh
        self.lv = lv
        self.shape = shape
        self.ndim = len(shape)
        self.correlation_model = correlation_model

        if correlation_model not in ("spherical", "exponential", "gaussian"):
            raise ValueError(
                f"Unknown correlation model: '{correlation_model}'. "
                f"Supported: spherical, exponential, gaussian"
            )

        # Cache for FFT of correlation function
        self._sqrt_spectrum: Optional[torch.Tensor] = None
    
    def _build_correlation(
        self,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Build correlation function with exponential taper.

        Supports spherical, exponential, and gaussian correlation models.

        Args:
            device: Computation device.
            dtype: Data type.

        Returns:
            Correlation field tensor.
        """
        # Taper parameters for smooth boundary handling
        order, decay = 3, 0.25

        if self.ndim == 3:
            nx, ny, nz = self.shape
            x = torch.arange(nx, device=device, dtype=dtype) - nx / 2
            y = torch.arange(ny, device=device, dtype=dtype) - ny / 2
            z = torch.arange(nz, device=device, dtype=dtype) - nz / 2
            X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

            # Anisotropic normalized distance
            r = torch.sqrt(
                (X / (2 * self.lh)) ** 2 +
                (Y / (2 * self.lh)) ** 2 +
                (Z / (2 * self.lv)) ** 2
            )

            # Exponential taper
            taper = torch.exp(-(
                (torch.abs(X) / (decay * nx)) ** order +
                (torch.abs(Y) / (decay * ny)) ** order +
                (torch.abs(Z) / (decay * nz)) ** order
            ))
        else:
            nx, ny = self.shape
            x = torch.arange(nx, device=device, dtype=dtype) - nx / 2
            y = torch.arange(ny, device=device, dtype=dtype) - ny / 2
            X, Y = torch.meshgrid(x, y, indexing='ij')

            # Isotropic in 2D using horizontal correlation length
            r = torch.sqrt(
                (X / (2 * self.lh)) ** 2 +
                (Y / (2 * self.lh)) ** 2
            )

            taper = torch.exp(-(
                (torch.abs(X) / (decay * nx)) ** order +
                (torch.abs(Y) / (decay * ny)) ** order
            ))

        # Compute correlation based on model type
        if self.correlation_model == "spherical":
            # C(r) = 1 - 1.5r + 0.5r^3 for r < 1, 0 otherwise
            corr = torch.where(
                r < 1.0,
                1.0 - 1.5 * r + 0.5 * r ** 3,
                torch.zeros_like(r)
            )
        elif self.correlation_model == "exponential":
            # C(r) = exp(-3r)  (practical range at r=1)
            corr = torch.exp(-3.0 * r)
        elif self.correlation_model == "gaussian":
            # C(r) = exp(-3r^2)  (practical range at r=1)
            corr = torch.exp(-3.0 * r ** 2)
        else:
            raise ValueError(f"Unknown correlation model: {self.correlation_model}")

        return corr * taper
    
    def _get_sqrt_spectrum(
        self, 
        device: torch.device, 
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Get square root of power spectrum (with caching).
        
        Args:
            device: Computation device.
            dtype: Data type.
            
        Returns:
            Square root of power spectrum.
        """
        # Check cache validity
        if (self._sqrt_spectrum is not None and 
            self._sqrt_spectrum.device == device):
            return self._sqrt_spectrum
        
        # Build correlation and compute FFT
        corr = self._build_correlation(device, dtype)
        spectrum = torch.fft.fftn(corr)
        self._sqrt_spectrum = torch.sqrt(torch.abs(spectrum))
        
        return self._sqrt_spectrum
    
    def generate(
        self,
        n: int = 1,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Generate spatially correlated random field(s).
        
        Args:
            n: Number of realizations to generate.
            device: Computation device.
            dtype: Data type.
            
        Returns:
            Generated field(s). Shape is self.shape if n=1, 
            otherwise (n, *self.shape).
        
        Example:
            >>> fft_ma = FFTMovingAverage(lh=20, lv=5, shape=(64, 64))
            >>> # Single realization
            >>> field = fft_ma.generate()
            >>> # Multiple realizations (batch)
            >>> fields = fft_ma.generate(n=10, device=torch.device('cuda'))
        """
        if device is None:
            device = torch.device('cpu')
        
        sqrt_spectrum = self._get_sqrt_spectrum(device, dtype)
        
        if n == 1:
            # Single realization
            noise = torch.randn(self.shape, device=device, dtype=dtype)
            noise_fft = torch.fft.fftn(noise)
            filtered = torch.fft.ifftn(sqrt_spectrum * noise_fft)
            return torch.real(filtered)
        else:
            # Batch generation
            noise = torch.randn(n, *self.shape, device=device, dtype=dtype)
            fft_dims = tuple(range(1, self.ndim + 1))
            noise_fft = torch.fft.fftn(noise, dim=fft_dims)
            filtered = torch.fft.ifftn(sqrt_spectrum * noise_fft, dim=fft_dims)
            return torch.real(filtered)
    
    def clear_cache(self) -> None:
        """Clear cached spectrum."""
        self._sqrt_spectrum = None


@register_model(
    'fft_ma',
    category='geostat',
    implemented=True,
    description='FFT-based Moving Average (unconditional, fast, GPU-accelerated)'
)
class FFTMASimulator(GeostatSimulator):
    """
    FFT-MA simulator for geological field generation.
    
    Generates unconditional Gaussian random fields with specified
    spatial correlation structure. Supports single-field and
    multi-field co-simulation.
    
    Note:
        FFT-MA does not support conditioning data. For conditional
        simulation, use SGS or other conditional methods.
    
    Args:
        correlation_model: Correlation model (only 'spherical' currently).
        transform: Optional output transform (e.g., torch.exp for log-normal).
    
    Example:
        Single field simulation:
        
        >>> from geobrain.geomodel import Simulator, SimulationConfig
        >>> 
        >>> config = SimulationConfig(shape=(64, 64, 128), lh=20, lv=5)
        >>> simulator = Simulator.create('fft_ma')
        >>> field = simulator.simulate(config)
        
        Co-simulation with correlated variables:
        
        >>> from geobrain.geomodel import CoSimConfig
        >>> import torch
        >>> 
        >>> config = CoSimConfig(
        ...     shape=(64, 64),
        ...     n_variables=2,
        ...     correlation_matrix=torch.tensor([[1.0, 0.7], [0.7, 1.0]]),
        ...     field_names=['porosity', 'permeability']
        ... )
        >>> fields = simulator.simulate(config)
        >>> print(fields.keys())  # dict_keys(['porosity', 'permeability'])
    """
    
    def __init__(
        self,
        correlation_model: str = "spherical",
        conditioning_data: Optional[torch.Tensor] = None,
        conditioning_locations: Optional[torch.Tensor] = None,
        transform=None,
    ):
        # Warn if conditioning data provided
        if conditioning_data is not None:
            warnings.warn(
                "FFT-MA does not support conditioning. Use SGS for "
                "conditional simulation. Conditioning data will be ignored.",
                UserWarning
            )
        
        super().__init__(
            correlation_model=correlation_model,
            conditioning_data=None,  # Force None
            conditioning_locations=None,
            transform=transform,
        )
    
    def _simulate(
        self, 
        config: Union[SimulationConfig, CoSimConfig]
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Execute simulation based on config type."""
        if isinstance(config, CoSimConfig):
            return self._cosimulate(config)
        return self._simulate_single(config)
    
    def _simulate_single(self, config: SimulationConfig) -> torch.Tensor:
        """
        Single field simulation.
        
        Args:
            config: Simulation configuration.
            
        Returns:
            Simulated field. Shape is config.shape if n_realizations=1,
            otherwise (n_realizations, *config.shape).
        """
        device = config.torch_device
        dtype = config.torch_dtype
        
        fft_ma = FFTMovingAverage(
            config.lh, config.lv, config.shape,
            correlation_model=self.correlation_model,
        )

        # Generate realizations
        fields = fft_ma.generate(n=config.n_realizations, device=device, dtype=dtype)
        
        # Apply mean and std transformation
        mean = config.mean
        std = config.std
        
        if isinstance(mean, torch.Tensor):
            mean = mean.to(device)
        if isinstance(std, torch.Tensor):
            std = std.to(device)
        
        fields = mean + std * fields
        
        return fields
    
    def _cosimulate(self, config: CoSimConfig) -> Dict[str, torch.Tensor]:
        """
        Co-simulation via Cholesky decomposition.
        
        Generates multiple correlated fields using LU transformation
        of independent fields: Z = L @ U, where L is the Cholesky
        factor of the covariance matrix.
        
        Args:
            config: Co-simulation configuration.
            
        Returns:
            Dictionary mapping field names to tensors.
        """
        device = config.torch_device
        dtype = config.torch_dtype
        n_vars = config.n_variables
        
        if config.correlation_matrix is None:
            raise ValueError(
                "Co-simulation requires correlation_matrix in config"
            )
        
        # Build covariance matrix from correlation and std
        corr = config.correlation_matrix.to(device=device, dtype=dtype)
        
        if isinstance(config.std, torch.Tensor) and len(config.std) == n_vars:
            # Variable-specific standard deviations
            S = torch.diag(config.std.to(device=device, dtype=dtype))
            cov = S @ corr @ S
        else:
            # Uniform standard deviation
            std_val = config.std if isinstance(config.std, (int, float)) else config.std.item()
            cov = std_val ** 2 * corr
        
        # Cholesky decomposition
        L = torch.linalg.cholesky(cov)
        
        # Initialize FFT-MA generator
        fft_ma = FFTMovingAverage(
            config.lh, config.lv, config.shape,
            correlation_model=self.correlation_model,
        )
        
        # Generate for each realization
        results = {name: [] for name in config.field_names}
        
        for _ in range(config.n_realizations):
            # Generate independent fields
            U = torch.stack([
                fft_ma.generate(n=1, device=device, dtype=dtype)
                for _ in range(n_vars)
            ], dim=0)
            
            # Transform to correlated fields: Z = L @ U
            flat_shape = U.shape[1:]
            U_flat = U.view(n_vars, -1)
            Z_flat = L @ U_flat
            Z = Z_flat.view(n_vars, *flat_shape)
            
            # Apply mean
            mean = config.mean
            if isinstance(mean, torch.Tensor):
                mean = mean.to(device=device, dtype=dtype)
                if mean.ndim == 1 and len(mean) == n_vars:
                    mean = mean.view(-1, *([1] * len(flat_shape)))
            Z = Z + mean
            
            # Store results
            for i, name in enumerate(config.field_names):
                results[name].append(Z[i])
        
        # Format output
        output = {}
        for name in config.field_names:
            if config.n_realizations == 1:
                output[name] = results[name][0]
            else:
                output[name] = torch.stack(results[name], dim=0)
        
        return output
    
    @property
    def supports_conditioning(self) -> bool:
        """FFT-MA does not support conditioning."""
        return False
