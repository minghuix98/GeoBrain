"""
Generative AI methods for geological modeling.

Implemented:
    - DiffusionSimulator: Latent Diffusion Model (VAE + UNet + DDPM/DDIM)
    - VAESimulator: 3D AutoencoderKL (encode, decode, interpolate)
    - GANSimulator: DCGAN-style 3D generator for facies

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

from .gans import GANSimulator, Generator3D
from .vae import VAESimulator
from .diffusion import DiffusionSimulator

__all__ = [
    "GANSimulator",
    "Generator3D",
    "VAESimulator",
    "DiffusionSimulator",
]
