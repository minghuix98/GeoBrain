"""
Configuration classes for geological model simulation.

Three configuration classes are provided:
    1. SimulationConfig - Single field geostatistical simulation
    2. CoSimConfig - Multi-field co-simulation with correlations
    3. GenerativeConfig - Generative model (GAN, VAE, Diffusion) generation

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
from dataclasses import dataclass, field
from typing import Optional, Union, List, Tuple


@dataclass
class SimulationConfig:
    """
    Configuration for geostatistical simulation.
    
    Controls grid dimensions, correlation structure, and output statistics
    for methods like FFT-MA and SGS.
    
    Args:
        shape: Grid dimensions (nx, ny) for 2D or (nx, ny, nz) for 3D.
        lh: Horizontal correlation length.
        lv: Vertical correlation length.
        mean: Mean value (scalar or tensor for spatially varying).
        std: Standard deviation (scalar or tensor).
        n_realizations: Number of realizations to generate.
        seed: Random seed for reproducibility.
        device: Compute device ('cpu', 'cuda', or 'auto').
        dtype: Data type ('float32' or 'float64').
    
    Example:
        >>> config = SimulationConfig(
        ...     shape=(64, 64, 128),
        ...     lh=20.0,
        ...     lv=5.0,
        ...     mean=0.25,
        ...     std=0.05
        ... )
        >>> print(config.is_3d)  # True
    """
    
    shape: Tuple[int, ...] = (64, 64, 128)
    lh: float = 20.0
    lv: float = 5.0
    mean: Union[float, torch.Tensor] = 0.0
    std: Union[float, torch.Tensor] = 1.0
    n_realizations: int = 1
    seed: Optional[int] = None
    device: str = "auto"
    dtype: str = "float32"
    
    def __post_init__(self):
        # Auto-select device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Validate shape
        if len(self.shape) not in (2, 3):
            raise ValueError(f"shape must be 2D or 3D, got {len(self.shape)}D: {self.shape}")
        
        # Validate correlation lengths
        if self.lh <= 0 or self.lv <= 0:
            raise ValueError(
                f"Correlation lengths must be positive: lh={self.lh}, lv={self.lv}"
            )
    
    @property
    def is_3d(self) -> bool:
        """Whether this is a 3D configuration."""
        return len(self.shape) == 3
    
    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""
        return len(self.shape)
    
    @property
    def torch_device(self) -> torch.device:
        """Get torch.device object."""
        return torch.device(self.device)
    
    @property
    def torch_dtype(self) -> torch.dtype:
        """Get torch.dtype object."""
        return torch.float64 if self.dtype == "float64" else torch.float32


@dataclass
class CoSimConfig(SimulationConfig):
    """
    Configuration for co-simulation (multi-variable correlated fields).
    
    Extends SimulationConfig with support for multiple correlated variables.
    Uses Cholesky decomposition to generate fields with specified
    inter-variable correlations.
    
    Args:
        n_variables: Number of variables to co-simulate.
        correlation_matrix: Inter-variable correlation matrix (n x n).
        field_names: Names for each output field.
        
    Example:
        >>> import torch
        >>> config = CoSimConfig(
        ...     shape=(64, 64),
        ...     n_variables=2,
        ...     correlation_matrix=torch.tensor([[1.0, 0.7], [0.7, 1.0]]),
        ...     field_names=['porosity', 'permeability']
        ... )
    """
    
    n_variables: int = 2
    correlation_matrix: Optional[torch.Tensor] = None
    field_names: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        super().__post_init__()
        
        # Validate n_variables
        if self.n_variables < 2:
            raise ValueError(f"n_variables must be >= 2, got {self.n_variables}")
        
        # Auto-generate field names
        if not self.field_names:
            self.field_names = [f"field_{i}" for i in range(self.n_variables)]
        
        # Validate field_names length
        if len(self.field_names) != self.n_variables:
            raise ValueError(
                f"field_names length ({len(self.field_names)}) must match "
                f"n_variables ({self.n_variables})"
            )
        
        # Validate correlation matrix
        if self.correlation_matrix is not None:
            self._validate_correlation_matrix()
    
    def _validate_correlation_matrix(self) -> None:
        """Validate correlation matrix properties."""
        C = self.correlation_matrix
        n = self.n_variables

        # Check shape
        if C.shape != (n, n):
            raise ValueError(
                f"Correlation matrix shape {C.shape} doesn't match "
                f"n_variables ({n}, {n})"
            )

        # Check symmetry
        if not torch.allclose(C, C.T, atol=1e-6):
            raise ValueError("Correlation matrix must be symmetric")

        # Check positive definiteness
        eigenvalues = torch.linalg.eigvalsh(C)
        if eigenvalues.min() < -1e-6:
            raise ValueError(
                "Correlation matrix must be positive semi-definite. "
                f"Min eigenvalue: {eigenvalues.min().item():.6f}"
            )

    @classmethod
    def from_correlations(
        cls,
        correlations: List[Tuple[int, int, float]],
        n_variables: int,
        field_names: Optional[List[str]] = None,
        **kwargs
    ) -> 'CoSimConfig':
        """
        Create config from pairwise correlations.

        Args:
            correlations: List of (i, j, rho) tuples for correlation between
                         variable i and j.
            n_variables: Number of variables.
            field_names: Optional names for fields.
            **kwargs: Other SimulationConfig parameters.

        Returns:
            CoSimConfig with constructed correlation matrix.

        Example:
            >>> config = CoSimConfig.from_correlations(
            ...     correlations=[(0, 1, 0.7), (0, 2, 0.3), (1, 2, 0.5)],
            ...     n_variables=3,
            ...     field_names=['porosity', 'permeability', 'saturation'],
            ...     shape=(64, 64),
            ... )
        """
        # Build correlation matrix
        C = torch.eye(n_variables)
        for i, j, rho in correlations:
            if not -1 <= rho <= 1:
                raise ValueError(f"Correlation must be in [-1, 1], got {rho}")
            C[i, j] = rho
            C[j, i] = rho

        return cls(
            n_variables=n_variables,
            correlation_matrix=C,
            field_names=field_names or [],
            **kwargs
        )


@dataclass
class GenerativeConfig:
    """
    Configuration for generative models (GAN, VAE, Diffusion).

    Controls latent space sampling and model inference settings
    for neural network-based geological field generation.

    Args:
        shape: Output grid dimensions.
        latent_dim: Dimension of latent space.
        n_realizations: Number of samples to generate.
        checkpoint: Path to model checkpoint file.
        seed: Random seed for reproducibility.
        device: Compute device ('cpu', 'cuda', or 'auto').
        dtype: Data type ('float32' or 'float64').
        use_amp: Whether to use automatic mixed precision.

    Example:
        >>> config = GenerativeConfig(
        ...     shape=(64, 64, 64),
        ...     latent_dim=256,
        ...     n_realizations=10,
        ...     checkpoint='models/vae_geological.pt'
        ... )
    """

    shape: Tuple[int, ...] = (64, 64, 64)
    latent_dim: int = 512
    n_realizations: int = 1
    checkpoint: Optional[str] = None
    seed: Optional[int] = None
    device: str = "auto"
    dtype: str = "float32"
    use_amp: bool = False

    def __post_init__(self):
        # Auto-select device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Validate latent_dim
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""
        return len(self.shape)

    @property
    def torch_device(self) -> torch.device:
        """Get torch.device object."""
        return torch.device(self.device)

    @property
    def torch_dtype(self) -> torch.dtype:
        """Get torch.dtype object."""
        return torch.float64 if self.dtype == "float64" else torch.float32
