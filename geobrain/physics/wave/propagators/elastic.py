"""
Elastic wave equation propagator.

High-level interface for elastic wave simulation combining
velocity model, survey geometry, and FDTD kernels.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
from torch import Tensor
from typing import Optional, Dict

from ..core import AbstractPropagator, WaveResult
from ..models import IsotropicElasticModel
from ..survey import Survey
from ..boundary import create_boundary
from ..kernels import elastic_forward_kernel
from ..kernels.elastic_3d import elastic_forward_kernel_3d
from ..utils import numpy2tensor


class ElasticPropagator(AbstractPropagator):
    """
    Elastic wave equation propagator.

    Solves the elastic wave equation using finite difference
    method with 4th-order staggered grid scheme. Supports
    both 2D and 3D simulations.

    Args:
        model: Elastic velocity model (IsotropicElasticModel).
        survey: Survey geometry (source and receivers).
        device: Computation device. Default: 'cpu'.
        dtype: Data type. Default: torch.float32.

    Attributes:
        survey: Survey geometry object.
        nt: Number of time steps.
        dt: Time step size (s).
        damp: Boundary damping coefficients.
        src_x, src_z: Source positions (grid indices).
        rcv_x, rcv_z: Receiver positions (grid indices).

    Example:
        >>> model = IsotropicElasticModel(grid, boundary, vp, vs, rho)
        >>> survey = Survey(source, receiver)
        >>> propagator = ElasticPropagator(model, survey, device='cuda')
        >>> result = propagator.forward(checkpoint_segments=4)
        >>> pressure = result.p  # Shape: (n_shots, nt, n_receivers)
    """

    def __init__(
        self,
        model: IsotropicElasticModel,
        survey: Survey,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(model, device, dtype)

        if not isinstance(model, IsotropicElasticModel):
            raise TypeError(
                f"Expected IsotropicElasticModel, got {type(model).__name__}"
            )
        if not isinstance(survey, Survey):
            raise TypeError(
                f"Expected Survey, got {type(survey).__name__}"
            )

        self.survey = survey

        # Time parameters
        self.nt = survey.nt
        self.dt = survey.dt
        self.f0 = survey.f0

        # Initialize boundary condition
        self.damp = None
        self._init_boundary()

        # Parse source parameters
        self.source = survey.source
        src_loc = self.source.get_loc()
        self.src_n = self.source.num
        self.wavelet = numpy2tensor(self.source.get_wavelet(), dtype).to(device)

        if self.model.ndim == 3:
            # 3D: src_loc columns are (x, y, z)
            self.src_x = numpy2tensor(src_loc[:, 0], torch.long).to(device)
            self.src_y = numpy2tensor(src_loc[:, 1], torch.long).to(device)
            self.src_z = numpy2tensor(src_loc[:, 2], torch.long).to(device)
        else:
            # 2D: src_loc columns are (x, z)
            self.src_x = numpy2tensor(src_loc[:, 0], torch.long).to(device)
            self.src_z = numpy2tensor(src_loc[:, 1], torch.long).to(device)

        # Parse receiver parameters
        self.receiver = survey.receiver
        rcv_loc = self.receiver.get_loc()
        self.rcv_n = self.receiver.num

        if self.model.ndim == 3:
            # 3D: rcv_loc columns are (x, y, z)
            self.rcv_x = numpy2tensor(rcv_loc[:, 0], torch.long).to(device)
            self.rcv_y = numpy2tensor(rcv_loc[:, 1], torch.long).to(device)
            self.rcv_z = numpy2tensor(rcv_loc[:, 2], torch.long).to(device)
        else:
            # 2D: rcv_loc columns are (x, z)
            self.rcv_x = numpy2tensor(rcv_loc[:, 0], torch.long).to(device)
            self.rcv_z = numpy2tensor(rcv_loc[:, 1], torch.long).to(device)

        # Receiver masks
        self.receiver_masks = survey.receiver_masks

    def _init_boundary(self, vmax: Optional[float] = None) -> None:
        """
        Initialize boundary condition arrays.

        Args:
            vmax: Maximum velocity for PML scaling. Default: max(vp).
        """
        if vmax is None:
            vmax = float(self.model.vp.detach().cpu().max())

        bc_kwargs = dict(
            bc_type=self.abc_type,
            nx=self.nx,
            nz=self.nz,
            dx=self.dx,
            dz=self.dz,
            n_layers=self.nabc,
            vmax=vmax,
            alpha=self.model.abc_alpha,
            free_surface=False,  # Handled separately in kernel
        )
        if self.model.ndim == 3:
            bc_kwargs['ny'] = self.ny
            bc_kwargs['dy'] = self.dy

        damp = create_boundary(**bc_kwargs)

        self.damp = numpy2tensor(damp, self.dtype).to(self.device)

    def forward(
        self,
        model: Optional[IsotropicElasticModel] = None,
        shot_index: Optional[int] = None,
        checkpoint_segments: int = 1,
        snapshot_every: int = 0,
    ) -> WaveResult:
        """
        Run forward wave propagation.

        Args:
            model: Model to use. If None, uses instance model.
            shot_index: Specific shot index to simulate.
                If None, simulates all shots. Default: None.
            checkpoint_segments: Number of segments for gradient
                checkpointing. Higher values reduce memory but
                increase computation. Default: 1.
            snapshot_every: Save pressure wavefield snapshot every N
                steps. 0 disables. When > 0, result.snapshots contains
                a tensor of shape (n_snaps, nz, nx). Default: 0.

        Returns:
            WaveResult with fields:
                - p: Pressure seismograms, shape (n_shots, nt, n_rcv)
                - vx: Horizontal velocity seismograms
                - vz: Vertical velocity seismograms
                - forward_wavefield_p: Pressure wavefield energy
                - forward_wavefield_vx: Horizontal velocity energy
                - forward_wavefield_vz: Vertical velocity energy
                - snapshots: Wavefield snapshots (when snapshot_every > 0)
                For 3D, additionally includes vy and forward_wavefield_vy.

        Example:
            >>> result = propagator.forward(checkpoint_segments=4)
            >>> seismograms = result.p
        """
        model = model if model is not None else self.model
        model.forward()  # Prepare model (computes Lame parameters)

        # Skip checkpointing when no parameter requires gradient
        if checkpoint_segments > 1 and not any(
            p.requires_grad for p in model.parameters()
        ):
            checkpoint_segments = 1

        # Select shots
        if shot_index is not None:
            src_x = self.src_x[shot_index:shot_index + 1]
            src_z = self.src_z[shot_index:shot_index + 1]
            src_n = 1
            wavelet = self.wavelet[shot_index:shot_index + 1]
            if self.model.ndim == 3:
                src_y = self.src_y[shot_index:shot_index + 1]
        else:
            src_x = self.src_x
            src_z = self.src_z
            src_n = self.src_n
            wavelet = self.wavelet
            if self.model.ndim == 3:
                src_y = self.src_y

        # Run forward kernel
        if self.model.ndim == 3:
            result = elastic_forward_kernel_3d(
                nx=self.nx,
                ny=self.ny,
                nz=self.nz,
                dx=self.dx,
                dy=self.dy,
                dz=self.dz,
                nt=self.nt,
                dt=self.dt,
                nabc=self.nabc,
                free_surface=self.free_surface,
                src_x=src_x,
                src_y=src_y,
                src_z=src_z,
                src_n=src_n,
                src_v=wavelet,
                rcv_x=self.rcv_x,
                rcv_y=self.rcv_y,
                rcv_z=self.rcv_z,
                rcv_n=self.rcv_n,
                damp=self.damp,
                vp=model.vp,
                vs=model.vs,
                rho=model.rho,
                checkpoint_segments=checkpoint_segments,
                device=torch.device(self.device),
                dtype=self.dtype,
            )
        else:
            result = elastic_forward_kernel(
                nx=self.nx,
                nz=self.nz,
                dx=self.dx,
                dz=self.dz,
                nt=self.nt,
                dt=self.dt,
                nabc=self.nabc,
                free_surface=self.free_surface,
                src_x=src_x,
                src_z=src_z,
                src_n=src_n,
                src_v=wavelet,
                rcv_x=self.rcv_x,
                rcv_z=self.rcv_z,
                rcv_n=self.rcv_n,
                damp=self.damp,
                vp=model.vp,
                vs=model.vs,
                rho=model.rho,
                checkpoint_segments=checkpoint_segments,
                device=torch.device(self.device),
                dtype=self.dtype,
                snapshot_every=snapshot_every,
            )

        return WaveResult.from_dict(result)

    def __call__(
        self,
        model: Optional[IsotropicElasticModel] = None,
        shot_index: Optional[int] = None,
        checkpoint_segments: int = 1,
        snapshot_every: int = 0,
    ) -> WaveResult:
        """
        Shortcut for forward() - enables propagator(...) syntax.

        Args:
            model: Model to use. If None, uses instance model.
            shot_index: Specific shot index to simulate.
            checkpoint_segments: Number of segments for gradient checkpointing.
            snapshot_every: Save wavefield snapshot every N steps.

        Returns:
            WaveResult containing seismograms and wavefields.
        """
        return self.forward(
            model=model,
            shot_index=shot_index,
            checkpoint_segments=checkpoint_segments,
            snapshot_every=snapshot_every,
        )
