"""
Acoustic wave equation FDTD kernels.

Implements finite-difference time-domain kernels for acoustic
wave propagation using the stress-velocity formulation with
4th-order staggered grid scheme.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from typing import Dict, Tuple


@torch.jit.script
def pad_model(
    v: Tensor,
    pml: int,
    nz: int,
    nx: int,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """
    Pad velocity/density model for PML boundary.
    
    Extends the model by replicating edge values into the
    absorbing boundary region.
    
    Args:
        v: Input model with shape (nz, nx).
        pml: Number of PML layers on each side.
        nz: Number of grid points in z.
        nx: Number of grid points in x.
        device: Computation device. Default: 'cpu'.

    Returns:
        Padded model with shape (nz + 2*pml, nx + 2*pml).

    Example:
        >>> vp = torch.ones(100, 200) * 2500.0
        >>> vp_padded = pad_model(vp, pml=20, nz=100, nx=200)
        >>> print(vp_padded.shape)  # (140, 240)
    """
    nz_pml = nz + 2 * pml
    nx_pml = nx + 2 * pml
    cc = torch.zeros((nz_pml, nx_pml), device=device)

    # Copy original tensor to center
    cc[pml:nz_pml - pml, pml:nx_pml - pml] = v

    # Top boundary (extend)
    cc[:pml, pml:pml + nx] = cc[pml, pml:pml + nx].expand(pml, -1)

    # Bottom boundary (extend)
    cc[nz_pml - pml:nz_pml, pml:pml + nx] = cc[nz_pml - pml - 1, pml:pml + nx].expand(pml, -1)

    # Left boundary (extend)
    cc[:, :pml] = cc[:, [pml]].expand(-1, pml)

    # Right boundary (extend)
    cc[:, nx_pml - pml:nx_pml] = cc[:, [nx_pml - pml - 1]].expand(-1, pml)

    return cc


@torch.jit.script
def pad_model_3d(
    v: Tensor,
    pml: int,
    nz: int,
    ny: int,
    nx: int,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """
    Pad 3D velocity/density model for PML boundary.

    Extends the model by replicating edge values into the
    absorbing boundary region on all six faces.

    Args:
        v: Input model with shape (nz, ny, nx).
        pml: Number of PML layers on each side.
        nz: Number of grid points in z.
        ny: Number of grid points in y.
        nx: Number of grid points in x.
        device: Computation device. Default: 'cpu'.

    Returns:
        Padded model with shape (nz + 2*pml, ny + 2*pml, nx + 2*pml).

    Example:
        >>> vp = torch.ones(64, 64, 64) * 2500.0
        >>> vp_padded = pad_model_3d(vp, pml=10, nz=64, ny=64, nx=64)
        >>> print(vp_padded.shape)  # (84, 84, 84)
    """
    nz_pml = nz + 2 * pml
    ny_pml = ny + 2 * pml
    nx_pml = nx + 2 * pml
    cc = torch.zeros((nz_pml, ny_pml, nx_pml), device=device)

    # Copy original tensor to center
    cc[pml:pml + nz, pml:pml + ny, pml:pml + nx] = v

    # Top/bottom z-faces (extend along z, only in the y-x center region)
    cc[:pml, pml:pml + ny, pml:pml + nx] = cc[pml, pml:pml + ny, pml:pml + nx].unsqueeze(0).expand(pml, -1, -1)
    cc[pml + nz:, pml:pml + ny, pml:pml + nx] = cc[pml + nz - 1, pml:pml + ny, pml:pml + nx].unsqueeze(0).expand(pml, -1, -1)

    # Front/back y-faces (extend along y, full z range now)
    cc[:, :pml, pml:pml + nx] = cc[:, pml, pml:pml + nx].unsqueeze(1).expand(-1, pml, -1)
    cc[:, pml + ny:, pml:pml + nx] = cc[:, pml + ny - 1, pml:pml + nx].unsqueeze(1).expand(-1, pml, -1)

    # Left/right x-faces (extend along x, full z and y range now)
    cc[:, :, :pml] = cc[:, :, pml].unsqueeze(2).expand(-1, -1, pml)
    cc[:, :, pml + nx:] = cc[:, :, pml + nx - 1].unsqueeze(2).expand(-1, -1, pml)

    return cc


@torch.jit.script
def acoustic_step_forward(
    nx: int,
    nz: int,
    dx: float,
    dz: float,
    dt: float,
    nabc: int,
    free_surface: bool,
    src_x: Tensor,
    src_z: Tensor,
    src_n: int,
    src_v: Tensor,
    rcv_x: Tensor,
    rcv_z: Tensor,
    rcv_n: int,
    kappa1: Tensor,
    alpha1: Tensor,
    kappa2: Tensor,
    alpha2: Tensor,
    kappa3: Tensor,
    c1: float,
    c2: float,
    p: Tensor,
    vx: Tensor,
    vz: Tensor,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Single time step segment for acoustic wave equation.

    Uses 4th-order staggered grid finite difference scheme.
    This function processes multiple time steps for a wavelet segment.

    Args:
        nx: Number of grid points in x-direction.
        nz: Number of grid points in z-direction.
        dx: Grid spacing in x-direction (m).
        dz: Grid spacing in z-direction (m).
        dt: Time step (s).
        nabc: Number of absorbing boundary layers.
        free_surface: Whether to use free surface at top.
        src_x: Source x-positions (grid indices).
        src_z: Source z-positions (grid indices).
        src_n: Number of sources.
        src_v: Source wavelet values for this segment.
        rcv_x: Receiver x-positions (grid indices).
        rcv_z: Receiver z-positions (grid indices).
        rcv_n: Number of receivers.
        kappa1: Precomputed damping coefficient for pressure.
        alpha1: Precomputed coefficient for pressure update.
        kappa2: Precomputed damping coefficient for vx velocity.
        alpha2: Precomputed coefficient for velocity update.
        kappa3: Precomputed damping coefficient for vz velocity.
        c1: FD stencil coefficient (9/8).
        c2: FD stencil coefficient (-1/24).
        p: Pressure field with shape (src_n, nz_pml, nx_pml).
        vx: Horizontal velocity field with shape (src_n, nz_pml, nx_pml-1).
        vz: Vertical velocity field with shape (src_n, nz_pml-1, nx_pml).
        device: Computation device. Default: 'cpu'.
        dtype: Data type. Default: torch.float32.

    Returns:
        Tuple of 9 tensors:
            - p, vx, vz: Updated pressure and velocity fields
            - rcv_p, rcv_vx, rcv_vz: Recorded seismograms at receivers
            - forward_wavefield_p/vx/vz: Accumulated wavefield energy
    """
    p = p.clone()
    vx = vx.clone()
    vz = vz.clone()

    nt = src_v.shape[-1]
    free_surface_start = nabc if free_surface else 1
    nx_pml = nx + 2 * nabc
    nz_pml = nz + 2 * nabc

    # Initialize recorded values
    rcv_p = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_vx = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_vz = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)

    # Initialize forward wavefield accumulators
    forward_wavefield_p = torch.zeros((nz, nx), dtype=dtype, device=device)
    forward_wavefield_vx = torch.zeros((nz, nx), dtype=dtype, device=device)
    forward_wavefield_vz = torch.zeros((nz, nx), dtype=dtype, device=device)

    for it in range(nt):
        # Update pressure field
        p[:, free_surface_start + 1:nz_pml - 2, 2:nx_pml - 2] = (
            (1.0 - kappa1[free_surface_start + 1:nz_pml - 2, 2:nx_pml - 2]) *
            p[:, free_surface_start + 1:nz_pml - 2, 2:nx_pml - 2] -
            alpha1[free_surface_start + 1:nz_pml - 2, 2:nx_pml - 2] * (
                c1 * (vx[:, free_surface_start + 1:nz_pml - 2, 2:nx_pml - 2] -
                      vx[:, free_surface_start + 1:nz_pml - 2, 1:nx_pml - 3] +
                      vz[:, free_surface_start + 1:nz_pml - 2, 2:nx_pml - 2] -
                      vz[:, free_surface_start:nz_pml - 3, 2:nx_pml - 2]) +
                c2 * (vx[:, free_surface_start + 1:nz_pml - 2, 3:nx_pml - 1] -
                      vx[:, free_surface_start + 1:nz_pml - 2, 0:nx_pml - 4] +
                      vz[:, free_surface_start + 2:nz_pml - 1, 2:nx_pml - 2] -
                      vz[:, free_surface_start - 1:nz_pml - 4, 2:nx_pml - 2])
            )
        )

        # Add source
        if src_z.dim() == 1:
            src_update = dt * (src_v[it] if len(src_v.shape) == 1 else src_v[:, it])
            p[torch.arange(src_n), src_z, src_x] = (
                p[torch.arange(src_n), src_z, src_x] + src_update
            )
        else:
            for i in range(src_n):
                src_update = dt * (src_v[i, it] if len(src_v.shape) == 2 else src_v[i, :, it])
                p[i, src_z[i], src_x[i]] = p[i, src_z[i], src_x[i]] + src_update

        # Free surface condition for pressure
        if free_surface:
            p[:, free_surface_start - 1, :] = -p[:, free_surface_start + 1, :]

        # Update horizontal particle velocity (vx)
        vx[:, free_surface_start:nz_pml - 1, 1:nx_pml - 2] = (
            (1.0 - kappa2[free_surface_start:nz_pml - 1, 1:nx_pml - 2]) *
            vx[:, free_surface_start:nz_pml - 1, 1:nx_pml - 2] -
            alpha2[free_surface_start:nz_pml - 1, 1:nx_pml - 2] * (
                c1 * (p[:, free_surface_start:nz_pml - 1, 2:nx_pml - 1] -
                      p[:, free_surface_start:nz_pml - 1, 1:nx_pml - 2]) +
                c2 * (p[:, free_surface_start:nz_pml - 1, 3:nx_pml] -
                      p[:, free_surface_start:nz_pml - 1, 0:nx_pml - 3])
            )
        )

        # Update vertical particle velocity (vz)
        vz[:, free_surface_start:nz_pml - 2, 1:nx_pml - 1] = (
            (1.0 - kappa3[free_surface_start:nz_pml - 2, 1:nx_pml - 1]) *
            vz[:, free_surface_start:nz_pml - 2, 1:nx_pml - 1] -
            alpha2[free_surface_start:nz_pml - 2, 1:nx_pml - 1] * (
                c1 * (p[:, free_surface_start + 1:nz_pml - 1, 1:nx_pml - 1] -
                      p[:, free_surface_start:nz_pml - 2, 1:nx_pml - 1]) +
                c2 * (p[:, free_surface_start + 2:nz_pml, 1:nx_pml - 1] -
                      p[:, free_surface_start - 1:nz_pml - 3, 1:nx_pml - 1])
            )
        )

        # Free surface condition for vertical velocity
        if free_surface:
            vz[:, free_surface_start - 1, :] = vz[:, free_surface_start, :]

        # Record seismograms at receivers
        rcv_p[:, it, :] = p[:, rcv_z, rcv_x]
        rcv_vx[:, it, :] = vx[:, rcv_z, rcv_x]
        rcv_vz[:, it, :] = vz[:, rcv_z, rcv_x]

        # Accumulate forward wavefield energy
        forward_wavefield_p += torch.sum(
            p * p, dim=0
        )[nabc:nabc + nz, nabc:nabc + nx].detach()
        forward_wavefield_vx += torch.sum(
            vx * vx, dim=0
        )[nabc:nabc + nz, nabc:nabc + nx].detach()
        forward_wavefield_vz += torch.sum(
            vz * vz, dim=0
        )[nabc:nabc + nz, nabc:nabc + nx].detach()

    return (
        p, vx, vz,
        rcv_p, rcv_vx, rcv_vz,
        forward_wavefield_p, forward_wavefield_vx, forward_wavefield_vz
    )


def acoustic_forward_kernel(
    nx: int,
    nz: int,
    dx: float,
    dz: float,
    nt: int,
    dt: float,
    nabc: int,
    free_surface: bool,
    src_x: Tensor,
    src_z: Tensor,
    src_n: int,
    src_v: Tensor,
    rcv_x: Tensor,
    rcv_z: Tensor,
    rcv_n: int,
    damp: Tensor,
    vp: Tensor,
    rho: Tensor,
    checkpoint_segments: int = 1,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
    snapshot_every: int = 0,
) -> Dict[str, Tensor]:
    """
    Forward simulation kernel for acoustic wave equation.

    Main entry point for acoustic wave propagation. Handles model
    padding, coefficient computation, and time stepping with
    optional gradient checkpointing.

    Args:
        nx: Number of grid points in x-direction.
        nz: Number of grid points in z-direction.
        dx: Grid spacing in x-direction (m).
        dz: Grid spacing in z-direction (m).
        nt: Number of time steps.
        dt: Time step (s).
        nabc: Number of absorbing boundary layers.
        free_surface: Whether to use free surface at top.
        src_x: Source x-positions (grid indices).
        src_z: Source z-positions (grid indices).
        src_n: Number of sources.
        src_v: Source wavelets with shape (src_n, nt) or (nt,).
        rcv_x: Receiver x-positions (grid indices).
        rcv_z: Receiver z-positions (grid indices).
        rcv_n: Number of receivers.
        damp: Damping coefficients from boundary module.
        vp: P-wave velocity model (m/s) with shape (nz, nx).
        rho: Density model (kg/m³) with shape (nz, nx).
        checkpoint_segments: Number of segments for gradient checkpointing.
            Higher values reduce memory but increase computation. Default: 1.
        device: Computation device. Default: 'cpu'.
        dtype: Data type. Default: torch.float32.
        snapshot_every: Save pressure wavefield snapshot every N steps.
            0 disables snapshots. When > 0, checkpointing is adjusted
            to capture snapshots at segment boundaries. Default: 0.

    Returns:
        Dictionary containing:
            - 'p': Pressure seismograms with shape (src_n, nt, rcv_n)
            - 'vx': Horizontal velocity seismograms
            - 'vz': Vertical velocity seismograms
            - 'forward_wavefield_p': Pressure wavefield energy (nz, nx)
            - 'forward_wavefield_vx': Horizontal velocity energy
            - 'forward_wavefield_vz': Vertical velocity energy
            - 'snapshots': Wavefield snapshots (n_snaps, nz, nx),
              only present when snapshot_every > 0.

    Example:
        >>> result = acoustic_forward_kernel(
        ...     nx=200, nz=100, dx=10.0, dz=10.0,
        ...     nt=1000, dt=0.001, nabc=20, free_surface=True,
        ...     src_x=src_x, src_z=src_z, src_n=1, src_v=wavelet,
        ...     rcv_x=rcv_x, rcv_z=rcv_z, rcv_n=100,
        ...     damp=damp, vp=vp, rho=rho
        ... )
        >>> seismogram = result['p']
    """
    # Pad model parameters
    c = pad_model(vp, nabc, nz, nx, device=device)
    den = pad_model(rho, nabc, nz, nx, device=device)

    free_surface_start = nabc if free_surface else 1
    nx_pml = nx + 2 * nabc
    nz_pml = nz + 2 * nabc

    # Adjust source and receiver positions for padding
    src_x = src_x + nabc
    src_z = src_z + nabc
    rcv_x = rcv_x + nabc
    rcv_z = rcv_z + nabc

    # Initialize fields
    p = torch.zeros((src_n, nz_pml, nx_pml), dtype=dtype, device=device)
    vx = torch.zeros((src_n, nz_pml, nx_pml - 1), dtype=dtype, device=device)
    vz = torch.zeros((src_n, nz_pml - 1, nx_pml), dtype=dtype, device=device)

    # Initialize recorded waveforms
    rcv_p = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_vx = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_vz = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)

    # Initialize wavefield accumulators
    forward_wavefield_p = torch.zeros((nz, nx), dtype=dtype, device=device)
    forward_wavefield_vx = torch.zeros((nz, nx), dtype=dtype, device=device)
    forward_wavefield_vz = torch.zeros((nz, nx), dtype=dtype, device=device)

    # 4th-order FD coefficients
    c1 = 9.0 / 8.0
    c2 = -1.0 / 24.0

    # Precompute coefficients
    alpha1 = den * c * c * dt / dz
    kappa1 = damp * dt

    alpha2 = dt / (den * dz)
    kappa2 = torch.zeros_like(damp, device=device)
    kappa2[:, 1:nx_pml - 2] = 0.5 * (
        damp[:, 1:nx_pml - 2] + damp[:, 2:nx_pml - 1]
    ) * dt

    kappa3 = torch.zeros_like(damp, device=device)
    kappa3[free_surface_start:nz_pml - 2, :] = 0.5 * (
        damp[free_surface_start:nz_pml - 2, :] +
        damp[free_surface_start + 1:nz_pml - 1, :]
    ) * dt

    # Build chunk list
    if snapshot_every > 0:
        # Fixed-size chunks for snapshot collection
        chunk_list = []
        for t0 in range(0, nt, snapshot_every):
            chunk_list.append(src_v[..., t0:min(t0 + snapshot_every, nt)])
        snapshots = []
    else:
        chunk_list = list(torch.chunk(src_v, checkpoint_segments, dim=-1))
        snapshots = None

    # Time stepping with checkpointing
    k = 0
    for i, chunk in enumerate(chunk_list):
        if snapshot_every > 0:
            # Direct call (no checkpointing) for snapshot mode
            (p, vx, vz, rcv_p_temp, rcv_vx_temp, rcv_vz_temp,
             fwd_p_temp, fwd_vx_temp, fwd_vz_temp) = acoustic_step_forward(
                nx, nz, dx, dz, dt,
                nabc, free_surface,
                src_x, src_z, src_n, chunk,
                rcv_x, rcv_z, rcv_n,
                kappa1, alpha1, kappa2, alpha2, kappa3, c1, c2,
                p, vx, vz,
                device, dtype,
            )
        elif checkpoint_segments > 1:
            (p, vx, vz, rcv_p_temp, rcv_vx_temp, rcv_vz_temp,
             fwd_p_temp, fwd_vx_temp, fwd_vz_temp) = checkpoint(
                acoustic_step_forward,
                nx, nz, dx, dz, dt,
                nabc, free_surface,
                src_x, src_z, src_n, chunk,
                rcv_x, rcv_z, rcv_n,
                kappa1, alpha1, kappa2, alpha2, kappa3, c1, c2,
                p, vx, vz,
                device, dtype,
                use_reentrant=True
            )
        else:
            (p, vx, vz, rcv_p_temp, rcv_vx_temp, rcv_vz_temp,
             fwd_p_temp, fwd_vx_temp, fwd_vz_temp) = acoustic_step_forward(
                nx, nz, dx, dz, dt,
                nabc, free_surface,
                src_x, src_z, src_n, chunk,
                rcv_x, rcv_z, rcv_n,
                kappa1, alpha1, kappa2, alpha2, kappa3, c1, c2,
                p, vx, vz,
                device, dtype,
            )

        # Save recorded waveforms
        rcv_p[:, k:k + chunk.shape[-1]] = rcv_p_temp
        rcv_vx[:, k:k + chunk.shape[-1]] = rcv_vx_temp
        rcv_vz[:, k:k + chunk.shape[-1]] = rcv_vz_temp

        # Accumulate wavefield
        forward_wavefield_p += fwd_p_temp.detach()
        forward_wavefield_vx += fwd_vx_temp.detach()
        forward_wavefield_vz += fwd_vz_temp.detach()

        # Capture snapshot (first shot pressure field)
        if snapshot_every > 0:
            snapshots.append(
                p[0, nabc:nabc + nz, nabc:nabc + nx].detach().cpu().clone()
            )

        k += chunk.shape[-1]

    out = {
        'p': rcv_p,
        'vx': rcv_vx,
        'vz': rcv_vz,
        'forward_wavefield_p': forward_wavefield_p,
        'forward_wavefield_vx': forward_wavefield_vx,
        'forward_wavefield_vz': forward_wavefield_vz,
    }
    if snapshot_every > 0:
        out['snapshots'] = torch.stack(snapshots, dim=0)
    return out