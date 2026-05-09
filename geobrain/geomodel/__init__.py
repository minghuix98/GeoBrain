"""
GeoBrain Geological Model Generation Module.

A framework for generating geological models using geostatistical
and generative AI methods.

Quick Start:
    >>> from geobrain.geomodel import Simulator, SimulationConfig
    >>>
    >>> config = SimulationConfig(shape=(64, 64, 128), lh=20, lv=5)
    >>> field = Simulator.create('fft_ma').simulate(config)

Co-simulation:
    >>> from geobrain.geomodel import CoSimConfig
    >>> import torch
    >>>
    >>> config = CoSimConfig(
    ...     shape=(64, 64),
    ...     n_variables=2,
    ...     correlation_matrix=torch.tensor([[1.0, 0.7], [0.7, 1.0]]),
    ...     field_names=['porosity', 'permeability']
    ... )
    >>> fields = Simulator.create('fft_ma').simulate(config)
    >>> print(fields.keys())  # dict_keys(['porosity', 'permeability'])

Available Methods:
    Implemented:
        - fft_ma: FFT-based Moving Average (unconditional, fast, GPU)

    Implemented (generative):
        - diffusion: Latent Diffusion Model (VAE + UNet + DDPM/DDIM)
        - diffsim: DiffSim facies diffusion models (2D/3D, DDPM/DDIM)
        - vae: 3D AutoencoderKL
        - gan: DCGAN-style 3D facies generation

Extending with Custom Methods:
    >>> from geobrain.geomodel import GeostatSimulator, register_model
    >>>
    >>> @register_model('my_method', category='geostat', implemented=True)
    ... class MySimulator(GeostatSimulator):
    ...     def _simulate(self, config):
    ...         # Your implementation
    ...         pass
    >>>
    >>> # Now available via factory
    >>> simulator = Simulator.create('my_method')

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "Mingliang Liu"
__email__ = "mingliangliu@sdu.edu.cn"

# =============================================================================
# Core Components
# =============================================================================

from .base import BaseSimulator, GeostatSimulator, GenerativeSimulator
from .config import SimulationConfig, CoSimConfig, GenerativeConfig
from .registry import registry, register_model

# =============================================================================
# Import Submodules (triggers registration)
# =============================================================================

from . import geostats
from . import geogen
from . import implicit

# Explicit imports for convenience
from .geostats import FFTMovingAverage, FFTMASimulator, SGSSimulator
from .geostats import VariogramModel, VariogramStructure
from .geogen import GANSimulator, VAESimulator, DiffusionSimulator, DiffSimSimulator
from .implicit import ImplicitSimulator, ImplicitModel, ImplicitModelConfig


# =============================================================================
# Factory Class
# =============================================================================

class Simulator:
    """
    Factory class for creating geological model simulators.
    
    Provides a unified interface to create simulators by name,
    list available methods, and query implementation status.
    
    Example:
        >>> # Create simulator
        >>> simulator = Simulator.create('fft_ma')
        >>> field = simulator.simulate(config)
        >>> 
        >>> # List available methods
        >>> print(Simulator.list_methods())
        ['fft_ma', 'sgs', 'gan', 'vae', 'diffusion', 'diffsim']
        >>> 
        >>> # List implemented methods
        >>> print(Simulator.list_implemented())
        ['fft_ma']
        >>> 
        >>> # Get method info
        >>> info = Simulator.get_info('fft_ma')
        >>> print(info['description'])
    """
    
    @staticmethod
    def create(method: str, **kwargs) -> BaseSimulator:
        """
        Create simulator instance by method name.
        
        Args:
            method: Method name ('fft_ma', 'sgs', 'gan', 'vae', 'diffusion', 'diffsim').
            **kwargs: Arguments passed to simulator constructor.
            
        Returns:
            Simulator instance.
            
        Raises:
            KeyError: If method name not found.
        
        Example:
            >>> simulator = Simulator.create('fft_ma')
            >>> simulator = Simulator.create('sgs', 
            ...     conditioning_data=well_data,
            ...     conditioning_locations=well_coords
            ... )
        """
        return registry.create(method, **kwargs)
    
    @staticmethod
    def list_methods(category: str = None) -> list:
        """
        List available simulation methods.
        
        Args:
            category: Filter by category ('geostat' or 'generative').
                     None returns all methods.
            
        Returns:
            List of method names.
        """
        return registry.list_methods(category)
    
    @staticmethod
    def list_implemented() -> list:
        """
        List fully implemented methods.
        
        Returns:
            List of implemented method names.
        """
        return registry.list_implemented()
    
    @staticmethod
    def get_info(method: str) -> dict:
        """
        Get information about a simulation method.
        
        Args:
            method: Method name.
            
        Returns:
            Dictionary with keys: category, implemented, description, class_name.
        """
        return registry.get_info(method)
    
    @staticmethod
    def is_implemented(method: str) -> bool:
        """
        Check if a method is fully implemented.
        
        Args:
            method: Method name.
            
        Returns:
            True if implemented, False otherwise.
        """
        return registry.is_implemented(method)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",

    # Main entry points
    "Simulator",
    "register_model",

    # Configuration
    "SimulationConfig",
    "CoSimConfig",
    "GenerativeConfig",

    # Base classes (for extension)
    "BaseSimulator",
    "GeostatSimulator",
    "GenerativeSimulator",

    # Geostatistical methods
    "FFTMovingAverage",
    "FFTMASimulator",
    "SGSSimulator",
    "VariogramModel",
    "VariogramStructure",

    # Generative methods
    "GANSimulator",
    "VAESimulator",
    "DiffusionSimulator",
    "DiffSimSimulator",

    # Implicit modeling
    "ImplicitSimulator",
    "ImplicitModel",
    "ImplicitModelConfig",
]
