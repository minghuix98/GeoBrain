"""
DiffSim-backed diffusion simulator for geological modeling.

Provides DiffSimSimulator that wraps the vendored DiffSim package for both
unconditional and conditional (inpainting) diffusion generation.

Supports:
    - 2D and 3D geological facies generation
    - DDPM (full timesteps) and DDIM (accelerated) sampling
    - Unconditional generation from noise
    - Conditional generation with partial observations (inpainting)

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
        Minghui Xu (minghuix@stanford.edu)
Version: 0.2.0
"""

import json
import torch
import torch.nn as nn
from typing import Optional, Literal, Callable, Union, Dict, Any
from pathlib import Path

from ..base import GenerativeSimulator
from ..config import GenerativeConfig
from ..registry import register_model


@register_model(
    'diffsim',
    category='generative',
    implemented=True,
    description='DiffSim denoising diffusion model for geological facies modeling (2D/3D)'
)
class DiffSimSimulator(GenerativeSimulator):
    """
    DiffSim diffusion model simulator for geological field generation.

    Wraps the DiffSim package to provide both unconditional and conditional
    (inpainting) generation using denoising diffusion models.

    Args:
        model: Pre-trained denoising network (UNet for unconditional, or None for conditional).
        checkpoint_path: Path to saved model checkpoint.
        model_type: Model dimensionality ('2d' or '3d').
        mode: Generation mode ('unconditional' or 'conditional').
        timesteps: Number of diffusion timesteps (default: 1500).
        beta_schedule: Noise schedule type ('linear', 'cosine', 'quadratic', 'sigmoid').
        sampler: Default sampling method ('ddpm' or 'ddim').
        channels: Number of output channels (default: 1 for facies).
        image_size: Output image size (int for 2D, tuple for 3D).
        unet_config: UNet configuration dict for conditional mode.
        beta_schedule_config: Beta schedule config dict for conditional mode.
        device: Default generation device ('auto', 'cuda', or 'cpu').
        transform: Optional output transform function.

    Example (unconditional):
        >>> from geobrain.geomodel import Simulator
        >>> sim = Simulator.create(
        ...     'diffsim',
        ...     checkpoint_path='checkpoints/diffsim/case1_geomodeling/unconditional.pth',
        ...     model_type='2d',
        ...     mode='unconditional',
        ...     image_size=64
        ... )
        >>> config = GenerativeConfig(shape=(64, 64), n_realizations=16)
        >>> samples = sim.simulate(config)

    Example (conditional/inpainting):
        >>> sim = Simulator.create(
        ...     'diffsim',
        ...     checkpoint_path='checkpoints/diffsim/case1_geomodeling/conditional.pth',
        ...     model_type='2d',
        ...     mode='conditional',
        ...     unet_config={...},
        ...     beta_schedule_config={...}
        ... )
        >>> output = sim.generate_conditional(y_cond, y_0, mask)
    """

    SUPPORTED_SCHEDULES = ("linear", "cosine", "quadratic", "sigmoid")
    SUPPORTED_SAMPLERS = ("ddpm", "ddim")
    SUPPORTED_MODES = ("unconditional", "conditional")
    SUPPORTED_TYPES = ("2d", "3d")

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        checkpoint_path: Optional[str] = None,
        model_type: Literal["2d", "3d"] = "2d",
        mode: Literal["unconditional", "conditional"] = "unconditional",
        timesteps: int = 1500,
        beta_schedule: Literal["linear", "cosine", "quadratic", "sigmoid"] = "linear",
        sampler: Literal["ddpm", "ddim"] = "ddpm",
        channels: int = 1,
        image_size: Optional[Union[int, tuple]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        unet_config: Optional[Dict[str, Any]] = None,
        beta_schedule_config: Optional[Dict[str, Any]] = None,
        module_name: Optional[str] = None,
        predict_type: Literal["epsilon", "x_start"] = "epsilon",
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        device: Union[str, torch.device] = "auto",
    ):
        super().__init__(
            model=model,
            checkpoint_path=checkpoint_path,
            latent_dim=None,  # Diffusion doesn't use latent vectors
            transform=transform,
        )

        # Validate parameters
        if model_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Unknown model_type: '{model_type}'. "
                f"Supported: {self.SUPPORTED_TYPES}"
            )

        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unknown mode: '{mode}'. "
                f"Supported: {self.SUPPORTED_MODES}"
            )

        if beta_schedule not in self.SUPPORTED_SCHEDULES:
            raise ValueError(
                f"Unknown beta_schedule: '{beta_schedule}'. "
                f"Supported: {self.SUPPORTED_SCHEDULES}"
            )

        if sampler not in self.SUPPORTED_SAMPLERS:
            raise ValueError(
                f"Unknown sampler: '{sampler}'. "
                f"Supported: {self.SUPPORTED_SAMPLERS}"
            )

        self.model_type = model_type
        self.mode = mode
        self.timesteps = timesteps
        self.beta_schedule = beta_schedule
        self.sampler = sampler
        self.channels = channels
        self.image_size = image_size
        self.model_config = model_config or {}
        self.unet_config = unet_config
        self.beta_schedule_config = beta_schedule_config
        self.module_name = module_name
        self.predict_type = predict_type
        self.default_device = self._resolve_device(device)

        # Diffusion process handler (for unconditional)
        self._diffusion = None
        # Network wrapper (for conditional)
        self._network = None

    @classmethod
    def from_config(
        cls,
        config_path: str,
        mode: Literal["unconditional", "conditional"] = "unconditional",
        checkpoint_dir: Optional[str] = None,
        sampler: Literal["ddpm", "ddim"] = "ddpm",
        device: Union[str, torch.device] = "auto",
    ) -> "DiffSimSimulator":
        """
        Create a DiffSimSimulator from a JSON config file.

        Args:
            config_path: Path to the JSON config file (e.g., configs/diffsim/case1_geomodeling.json).
            mode: Generation mode ('unconditional' or 'conditional').
            checkpoint_dir: Directory containing checkpoints. If None, uses path relative to config.
            sampler: Default sampling method ('ddpm' or 'ddim').
            device: Default generation device. Use "auto" to select CUDA when
                available and CPU otherwise.

        Returns:
            Configured DiffSimSimulator ready for generation.

        Example:
            >>> sim = DiffSimSimulator.from_config(
            ...     'configs/diffsim/case1_geomodeling.json',
            ...     mode='unconditional'
            ... )
            >>> samples = sim.generate_unconditional(n_samples=16)

            >>> sim_cond = DiffSimSimulator.from_config(
            ...     'configs/diffsim/case1_geomodeling.json',
            ...     mode='conditional'
            ... )
            >>> output, _ = sim_cond.generate_conditional(y_cond, y_0, mask)
        """
        config_path = Path(config_path).expanduser()
        if not config_path.is_absolute():
            config_path = (Path.cwd() / config_path).resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            config = json.load(f)

        # Extract common parameters
        model_type = config.get('type', '2d')
        image_size = config.get('image_size', 64)

        # Resolve checkpoint path
        if checkpoint_dir is None:
            # Default: checkpoints are relative to config file's parent directory
            checkpoint_dir = config_path.parent.parent.parent / 'checkpoints' / 'diffsim'

        checkpoint_dir = Path(checkpoint_dir)
        case_name = config.get('name', config_path.stem)

        if mode == 'unconditional':
            # Extract unconditional config
            uncond = config.get('unconditional', {})

            # Build checkpoint path
            ckpt_rel = config.get('checkpoints', {}).get('unconditional')
            if ckpt_rel:
                checkpoint_path = checkpoint_dir.parent.parent / ckpt_rel
            else:
                checkpoint_path = checkpoint_dir / case_name / 'unconditional.pth'

            return cls(
                checkpoint_path=str(checkpoint_path),
                model_type=model_type,
                mode='unconditional',
                timesteps=uncond.get('timesteps', 1500),
                beta_schedule=uncond.get('beta_schedule', 'linear'),
                sampler=sampler,
                channels=uncond.get('channels', 1),
                image_size=image_size,
                model_config=uncond,
                device=device,
            )

        else:  # conditional
            # Extract conditional config
            cond = config.get('conditional', {})

            # Build checkpoint path
            ckpt_rel = config.get('checkpoints', {}).get('conditional')
            if ckpt_rel:
                checkpoint_path = checkpoint_dir.parent.parent / ckpt_rel
            else:
                checkpoint_path = checkpoint_dir / case_name / 'conditional.pth'

            # Build unet_config for Network
            unet_config = {
                "image_size": image_size,
                "in_channel": cond.get('in_channel', 5),
                "out_channel": cond.get('out_channel', 1),
                "inner_channel": cond.get('inner_channel', 64),
                "channel_mults": cond.get('channel_mults', [1, 2, 4]),
                "attn_res": cond.get('attn_res', [16]),
                "num_head_channels": cond.get('num_head_channels', 32),
                "res_blocks": cond.get('res_blocks', 2),
                "dropout": cond.get('dropout', 0.2),
            }

            # Beta schedule config
            beta_schedule_config = cond.get('beta_schedule', {
                'train': {'schedule': 'cosine', 'n_timestep': 1500, 'linear_start': 1e-6, 'linear_end': 0.01},
                'test': {'schedule': 'cosine', 'n_timestep': 1500, 'linear_start': 1e-4, 'linear_end': 0.09}
            })

            # Get predict_type (epsilon or x_start)
            predict_type = cond.get('predict_type', 'epsilon')

            return cls(
                checkpoint_path=str(checkpoint_path),
                model_type=model_type,
                mode='conditional',
                timesteps=cond.get('timesteps', 1500),
                sampler=sampler,
                channels=cond.get('out_channel', 1),
                image_size=image_size,
                unet_config=unet_config,
                beta_schedule_config=beta_schedule_config,
                module_name=cond.get('module_name'),
                predict_type=predict_type,
                device=device,
            )

    def _get_diffusion(self):
        """Get or create Diffusion handler for unconditional generation."""
        if self._diffusion is None:
            from .diffsim import Diffusion
            self._diffusion = Diffusion(
                timesteps=self.timesteps,
                beta_schedule=self.beta_schedule
            )
        return self._diffusion

    @staticmethod
    def _resolve_device(device: Union[str, torch.device]) -> torch.device:
        """Resolve "auto" to an available torch device."""
        if isinstance(device, torch.device):
            return device
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _resolve_generation_device(self, device: Union[str, torch.device]) -> torch.device:
        """Resolve a per-call device, falling back to the simulator default."""
        if device == "auto":
            return self.default_device
        return self._resolve_device(device)

    def _set_unconditional_ddpm_steps(self, ddpm_steps: Optional[int]) -> None:
        """Update unconditional DDPM schedule length before sampler creation."""
        if ddpm_steps is None:
            return

        ddpm_steps = int(ddpm_steps)
        if ddpm_steps <= 0:
            raise ValueError("ddpm_steps must be a positive integer")

        if self.timesteps != ddpm_steps:
            self.timesteps = ddpm_steps
            self._diffusion = None

    def _set_conditional_ddpm_steps(
        self,
        ddpm_steps: Optional[int],
        device: torch.device
    ) -> None:
        """Update conditional test noise schedule length before sampling."""
        if ddpm_steps is None:
            return

        if self.beta_schedule_config is None:
            raise ValueError("beta_schedule_config must be provided for conditional DDPM steps")

        ddpm_steps = int(ddpm_steps)
        if ddpm_steps <= 0:
            raise ValueError("ddpm_steps must be a positive integer")

        test_schedule = self.beta_schedule_config.setdefault('test', {})
        test_schedule['n_timestep'] = ddpm_steps

        if self._network is not None:
            self._network.beta_schedule = self.beta_schedule_config
            if getattr(self._network, 'num_timesteps', None) != ddpm_steps:
                self._network.set_new_noise_schedule(device=device, phase='test')

    def _simulate(self, config: GenerativeConfig) -> torch.Tensor:
        """
        Generate geological fields using diffusion model.

        Args:
            config: Generation configuration with shape, n_realizations, device, etc.

        Returns:
            Generated fields of shape (n_realizations, channels, *shape).
        """
        device = config.torch_device

        if self.mode == 'unconditional':
            return self._simulate_unconditional(config, device)
        else:
            raise ValueError(
                "For conditional generation, use generate_conditional() method directly. "
                "The simulate() method is for unconditional generation only."
            )

    def _simulate_unconditional(
        self,
        config: GenerativeConfig,
        device: torch.device
    ) -> torch.Tensor:
        """Generate samples using unconditional diffusion."""
        # Ensure model is loaded
        model = self._ensure_model_loaded(device)

        # Get diffusion handler
        diffusion = self._get_diffusion()

        # Determine image size from config or stored value
        if config.shape is not None:
            image_size = config.shape
        elif self.image_size is not None:
            image_size = self.image_size
        else:
            raise ValueError("image_size must be provided in config.shape or constructor")

        # Generate samples
        if self.sampler == 'ddpm':
            samples = diffusion.sample(
                model=model,
                image_size=image_size,
                batch_size=config.n_realizations,
                channels=self.channels,
                return_all_steps=False
            )
        else:  # ddim
            samples = diffusion.sample_ddim(
                model=model,
                image_size=image_size,
                batch_size=config.n_realizations,
                channels=self.channels,
                ddim_steps=50,
                eta=0.0,
                return_all_steps=False
            )

        return samples

    def _load_checkpoint(self, device: torch.device) -> None:
        """
        Load model from checkpoint.

        For unconditional mode: loads UNet weights
        For conditional mode: loads Network weights
        """
        if self.checkpoint_path is None:
            raise ValueError("checkpoint_path must be provided")

        checkpoint_path = Path(self.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if self.mode == 'unconditional':
            self._load_unconditional_checkpoint(device)
        else:
            self._load_conditional_checkpoint(device)

    def _load_unconditional_checkpoint(self, device: torch.device) -> None:
        """Load unconditional UNet model."""
        dim = self.model_config.get('dim', 64)
        dim_mults = tuple(self.model_config.get('dim_mults', [1, 2, 4]))

        if self.model_type == '2d':
            from .diffsim import Unet
            self._model = Unet(
                dim=dim,
                channels=self.channels,
                dim_mults=dim_mults
            )
        else:  # 3d
            from .diffsim import Unet3D
            self._model = Unet3D(
                dim=dim,
                channels=self.channels,
                dim_mults=dim_mults
            )

        # Load weights
        state_dict = torch.load(self.checkpoint_path, map_location=device)
        self._model.load_state_dict(state_dict)
        self._model = self._model.to(device).eval()

    def _load_conditional_checkpoint(self, device: torch.device) -> None:
        """Load conditional Network model."""
        if self.unet_config is None or self.beta_schedule_config is None:
            raise ValueError(
                "unet_config and beta_schedule_config must be provided for conditional mode"
            )

        from .diffsim import Network

        # Determine module name based on config or model type
        module_name = self.module_name or (
            'guided_diffusion' if self.model_type == '2d' else 'guided_diffusion_3d'
        )

        self._network = Network(
            unet=self.unet_config,
            beta_schedule=self.beta_schedule_config,
            module_name=module_name,
            predict_type=self.predict_type
        )

        # Load weights
        state_dict = torch.load(self.checkpoint_path, map_location=device, weights_only=False)
        self._network.load_state_dict(state_dict, strict=False)
        self._network = self._network.to(device).eval()

        # Set noise schedule for inference
        self._network.set_new_noise_schedule(device=device, phase='test')

    def generate_unconditional(
        self,
        n_samples: int = 16,
        image_size: Optional[Union[int, tuple]] = None,
        device: Union[str, torch.device] = 'auto',
        sampler: Optional[str] = None,
        ddpm_steps: Optional[int] = None,
        ddim_steps: int = 50,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        Generate samples using unconditional diffusion.

        Args:
            n_samples: Number of samples to generate.
            image_size: Output size (int for 2D square, tuple for 3D).
            device: Device to run on ('auto', 'cuda', or 'cpu').
            sampler: Sampling method ('ddpm' or 'ddim'). Defaults to self.sampler.
            ddpm_steps: Number of DDPM/test schedule steps.
            ddim_steps: Number of DDIM steps (if using DDIM).
            eta: DDIM stochasticity (0 = deterministic).

        Returns:
            Generated samples tensor.
        """
        device = self._resolve_generation_device(device)
        self._set_unconditional_ddpm_steps(ddpm_steps)
        model = self._ensure_model_loaded(device)
        diffusion = self._get_diffusion()

        size = image_size or self.image_size
        if size is None:
            raise ValueError("image_size must be provided")

        sampler = sampler or self.sampler

        if sampler == 'ddpm':
            return diffusion.sample(
                model=model,
                image_size=size,
                batch_size=n_samples,
                channels=self.channels,
                return_all_steps=False
            )
        else:
            return diffusion.sample_ddim(
                model=model,
                image_size=size,
                batch_size=n_samples,
                channels=self.channels,
                ddim_steps=ddim_steps,
                eta=eta,
                return_all_steps=False
            )

    def generate_conditional(
        self,
        y_cond: torch.Tensor,
        y_t: Optional[torch.Tensor] = None,
        y_0: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        device: Union[str, torch.device] = 'auto',
        sampler: Optional[str] = None,
        ddpm_steps: Optional[int] = None,
        ddim_steps: int = 50,
        eta: float = 0.0,
        sample_num: int = 8
    ) -> tuple:
        """
        Generate samples using conditional diffusion (inpainting).

        Args:
            y_cond: Conditioning input (e.g., one-hot encoded facies).
            y_t: Initial noisy input (optional, generated from noise if None).
            y_0: Ground truth for inpainting (regions to keep).
            mask: Binary mask (1 = generate, 0 = keep from y_0).
            device: Device to run on ('auto', 'cuda', or 'cpu').
            sampler: Sampling method ('ddpm' or 'ddim').
            ddpm_steps: Number of DDPM/test schedule steps.
            ddim_steps: Number of DDIM steps.
            eta: DDIM stochasticity.
            sample_num: Number of intermediate samples to return.

        Returns:
            Tuple of (final_sample, intermediate_samples).
        """
        if self.mode != 'conditional':
            raise ValueError("generate_conditional() requires mode='conditional'")

        device = self._resolve_generation_device(device)
        self._set_conditional_ddpm_steps(ddpm_steps, device)

        # Load network if needed
        if self._network is None:
            self._load_conditional_checkpoint(device)

        # Move inputs to device
        y_cond = y_cond.to(device)
        if y_t is not None:
            y_t = y_t.to(device)
        if y_0 is not None:
            y_0 = y_0.to(device)
        if mask is not None:
            mask = mask.to(device)

        sampler = sampler or self.sampler

        with torch.no_grad():
            if sampler == 'ddpm':
                output, intermediates = self._network.restoration(
                    y_cond=y_cond,
                    y_t=y_t,
                    y_0=y_0,
                    mask=mask,
                    sample_num=sample_num
                )
            else:
                output, intermediates = self._network.restoration_ddim(
                    y_cond=y_cond,
                    y_t=y_t,
                    y_0=y_0,
                    mask=mask,
                    ddim_steps=ddim_steps,
                    eta=eta,
                    sample_num=sample_num
                )

        return output, intermediates

    @property
    def network(self) -> Optional[nn.Module]:
        """Get the conditional Network (for advanced usage)."""
        return self._network

    @property
    def diffusion(self):
        """Get the Diffusion handler (for advanced usage)."""
        return self._get_diffusion()


# Convenience functions for direct access
def create_diffsim_unconditional_simulator(
    checkpoint_path: str,
    model_type: str = '2d',
    image_size: Union[int, tuple] = 64,
    timesteps: int = 1500,
    beta_schedule: str = 'linear',
    sampler: str = 'ddpm',
    channels: int = 1
) -> DiffSimSimulator:
    """
    Create an unconditional diffusion simulator.

    Args:
        checkpoint_path: Path to trained model checkpoint.
        model_type: '2d' or '3d'.
        image_size: Output size.
        timesteps: Number of diffusion timesteps.
        beta_schedule: Noise schedule type.
        sampler: Default sampler ('ddpm' or 'ddim').
        channels: Number of output channels.

    Returns:
        Configured DiffSimSimulator.
    """
    return DiffSimSimulator(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        mode='unconditional',
        timesteps=timesteps,
        beta_schedule=beta_schedule,
        sampler=sampler,
        channels=channels,
        image_size=image_size
    )


def create_diffsim_conditional_simulator(
    checkpoint_path: str,
    unet_config: Dict[str, Any],
    beta_schedule_config: Dict[str, Any],
    model_type: str = '2d',
    sampler: str = 'ddpm'
) -> DiffSimSimulator:
    """
    Create a conditional diffusion simulator for inpainting.

    Args:
        checkpoint_path: Path to trained model checkpoint.
        unet_config: UNet architecture configuration dict.
        beta_schedule_config: Beta schedule configuration dict.
        model_type: '2d' or '3d'.
        sampler: Default sampler ('ddpm' or 'ddim').

    Returns:
        Configured DiffSimSimulator.
    """
    return DiffSimSimulator(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        mode='conditional',
        unet_config=unet_config,
        beta_schedule_config=beta_schedule_config,
        sampler=sampler
    )
