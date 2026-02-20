"""
3D Acoustic wave equation FDTD kernels.

Implements finite-difference time-domain kernels for 3D acoustic
wave propagation using the stress-velocity formulation with
4th-order staggered grid scheme.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from typing import Dict, Tuple

from .acoustic_2d import pad_model_3d


@torch.jit.script
def acoustic_step_forward_3d(
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    dy: float,
    dz: float,
    dt: float,
    nabc: int,
    free_surface: bool,
    src_x: Tensor,
    src_y: Tensor,
    src_z: Tensor,
    src_n: int,
    src_v: Tensor,
    rcv_x: Tensor,
    rcv_y: Tensor,
    rcv_z: Tensor,
    rcv_n: int,
    kappa1: Tensor,
    alpha1: Tensor,
    kappa2: Tensor,
    alpha2: Tensor,
    kappa3: Tensor,
    kappa4: Tensor,
    c1: float,
    c2: float,
    p: Tensor,
    vx: Tensor,
    vy: Tensor,
    vz: Tensor,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Single time step segment for 3D acoustic wave equation.

    Uses 4th-order staggered grid finite difference scheme.
    This function processes multiple time steps for a wavelet segment.

    Args:
        nx: Number of grid points in x-direction.
        ny: Number of grid points in y-direction.
        nz: Number of grid points in z-direction.
        dx: Grid spacing in x-direction (m).
        dy: Grid spacing in y-direction (m).
        dz: Grid spacing in z-direction (m).
        dt: Time step (s).
        nabc: Number of absorbing boundary layers.
        free_surface: Whether to use free surface at top.
        src_x: Source x-positions (grid indices).
        src_y: Source y-positions (grid indices).
        src_z: Source z-positions (grid indices).
        src_n: Number of sources.
        src_v: Source wavelet values for this segment.
        rcv_x: Receiver x-positions (grid indices).
        rcv_y: Receiver y-positions (grid indices).
        rcv_z: Receiver z-positions (grid indices).
        rcv_n: Number of receivers.
        kappa1: Precomputed damping coefficient for pressure.
        alpha1: Precomputed coefficient for pressure update.
        kappa2: Precomputed damping coefficient for vx velocity.
        alpha2: Precomputed coefficient for velocity update.
        kappa3: Precomputed damping coefficient for vy velocity.
        kappa4: Precomputed damping coefficient for vz velocity.
        c1: FD stencil coefficient (9/8).
        c2: FD stencil coefficient (-1/24).
        p: Pressure field (src_n, nz_pml, ny_pml, nx_pml).
        vx: X-velocity field (src_n, nz_pml, ny_pml, nx_pml-1).
        vy: Y-velocity field (src_n, nz_pml, ny_pml-1, nx_pml).
        vz: Z-velocity field (src_n, nz_pml-1, ny_pml, nx_pml).
        device: Computation device.
        dtype: Data type.

    Returns:
        Tuple of 13 tensors:
            - p, vx, vy, vz: Updated fields
            - rcv_p, rcv_vx, rcv_vy, rcv_vz: Recorded seismograms
            - fwd_p, fwd_vx, fwd_vy, fwd_vz: Accumulated wavefield energy
    """
    p = p.clone()
    vx = vx.clone()
    vy = vy.clone()
    vz = vz.clone()

    nt = src_v.shape[-1]
    free_surface_start = nabc if free_surface else 1
    nx_pml = nx + 2 * nabc
    ny_pml = ny + 2 * nabc
    nz_pml = nz + 2 * nabc

    # Initialize recorded values
    rcv_p = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_vx = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_vy = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_vz = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)

    # Initialize forward wavefield accumulators
    forward_wavefield_p = torch.zeros((nz, ny, nx), dtype=dtype, device=device)
    forward_wavefield_vx = torch.zeros((nz, ny, nx), dtype=dtype, device=device)
    forward_wavefield_vy = torch.zeros((nz, ny, nx), dtype=dtype, device=device)
    forward_wavefield_vz = torch.zeros((nz, ny, nx), dtype=dtype, device=device)

    for it in range(nt):
        # ====== Update pressure field ======
        # p -= alpha1 * (c1*(dvx/dx + dvy/dy + dvz/dz) + c2*(...))
        # Interior region: z in [fs+1, nz_pml-2], y in [2, ny_pml-2], x in [2, nx_pml-2]
        zs = free_surface_start + 1
        ze = nz_pml - 2
        ys = 2
        ye = ny_pml - 2
        xs = 2
        xe = nx_pml - 2

        # dvx/dx (forward difference at staggered position)
        dvx_dx = (
            c1 * (vx[:, zs:ze, ys:ye, xs:xe] -
                  vx[:, zs:ze, ys:ye, xs - 1:xe - 1]) +
            c2 * (vx[:, zs:ze, ys:ye, xs + 1:xe + 1] -
                  vx[:, zs:ze, ys:ye, xs - 2:xe - 2])
        )

        # dvy/dy (forward difference at staggered position)
        dvy_dy = (
            c1 * (vy[:, zs:ze, ys:ye, xs:xe] -
                  vy[:, zs:ze, ys - 1:ye - 1, xs:xe]) +
            c2 * (vy[:, zs:ze, ys + 1:ye + 1, xs:xe] -
                  vy[:, zs:ze, ys - 2:ye - 2, xs:xe])
        )

        # dvz/dz (forward difference at staggered position)
        dvz_dz = (
            c1 * (vz[:, zs:ze, ys:ye, xs:xe] -
                  vz[:, zs - 1:ze - 1, ys:ye, xs:xe]) +
            c2 * (vz[:, zs + 1:ze + 1, ys:ye, xs:xe] -
                  vz[:, zs - 2:ze - 2, ys:ye, xs:xe])
        )

        p[:, zs:ze, ys:ye, xs:xe] = (
            (1.0 - kappa1[zs:ze, ys:ye, xs:xe]) *
            p[:, zs:ze, ys:ye, xs:xe] -
            alpha1[zs:ze, ys:ye, xs:xe] * (dvx_dx + dvy_dy + dvz_dz)
        )

        # ====== Add source ======
        if src_z.dim() == 1:
            src_update = dt * (src_v[it] if len(src_v.shape) == 1 else src_v[:, it])
            p[torch.arange(src_n), src_z, src_y, src_x] = (
                p[torch.arange(src_n), src_z, src_y, src_x] + src_update
            )
        else:
            for i in range(src_n):
                src_update = dt * (src_v[i, it] if len(src_v.shape) == 2 else src_v[i, :, it])
                p[i, src_z[i], src_y[i], src_x[i]] = (
                    p[i, src_z[i], src_y[i], src_x[i]] + src_update
                )

        # ====== Free surface condition for pressure ======
        if free_surface:
            p[:, free_surface_start - 1, :, :] = -p[:, free_surface_start + 1, :, :]

        # ====== Update vx (x-velocity) ======
        # vx staggered at (iz, iy, ix+1/2): size nx_pml-1 in x
        vx_zs = free_surface_start
        vx_ze = nz_pml - 1
        vx_ys = 1
        vx_ye = ny_pml - 1
        vx_xs = 1
        vx_xe = nx_pml - 2

        vx[:, vx_zs:vx_ze, vx_ys:vx_ye, vx_xs:vx_xe] = (
            (1.0 - kappa2[vx_zs:vx_ze, vx_ys:vx_ye, vx_xs:vx_xe]) *
            vx[:, vx_zs:vx_ze, vx_ys:vx_ye, vx_xs:vx_xe] -
            alpha2[vx_zs:vx_ze, vx_ys:vx_ye, vx_xs:vx_xe] * (
                c1 * (p[:, vx_zs:vx_ze, vx_ys:vx_ye, vx_xs + 1:vx_xe + 1] -
                      p[:, vx_zs:vx_ze, vx_ys:vx_ye, vx_xs:vx_xe]) +
                c2 * (p[:, vx_zs:vx_ze, vx_ys:vx_ye, vx_xs + 2:vx_xe + 2] -
                      p[:, vx_zs:vx_ze, vx_ys:vx_ye, vx_xs - 1:vx_xe - 1])
            )
        )

        # ====== Update vy (y-velocity) ======
        # vy staggered at (iz, iy+1/2, ix): size ny_pml-1 in y
        vy_zs = free_surface_start
        vy_ze = nz_pml - 1
        vy_ys = 1
        vy_ye = ny_pml - 2
        vy_xs = 1
        vy_xe = nx_pml - 1

        vy[:, vy_zs:vy_ze, vy_ys:vy_ye, vy_xs:vy_xe] = (
            (1.0 - kappa3[vy_zs:vy_ze, vy_ys:vy_ye, vy_xs:vy_xe]) *
            vy[:, vy_zs:vy_ze, vy_ys:vy_ye, vy_xs:vy_xe] -
            alpha2[vy_zs:vy_ze, vy_ys:vy_ye, vy_xs:vy_xe] * (
                c1 * (p[:, vy_zs:vy_ze, vy_ys + 1:vy_ye + 1, vy_xs:vy_xe] -
                      p[:, vy_zs:vy_ze, vy_ys:vy_ye, vy_xs:vy_xe]) +
                c2 * (p[:, vy_zs:vy_ze, vy_ys + 2:vy_ye + 2, vy_xs:vy_xe] -
                      p[:, vy_zs:vy_ze, vy_ys - 1:vy_ye - 1, vy_xs:vy_xe])
            )
        )

        # ====== Update vz (z-velocity) ======
        # vz staggered at (iz+1/2, iy, ix): size nz_pml-1 in z
        vz_zs = free_surface_start
        vz_ze = nz_pml - 2
        vz_ys = 1
        vz_ye = ny_pml - 1
        vz_xs = 1
        vz_xe = nx_pml - 1

        vz[:, vz_zs:vz_ze, vz_ys:vz_ye, vz_xs:vz_xe] = (
            (1.0 - kappa4[vz_zs:vz_ze, vz_ys:vz_ye, vz_xs:vz_xe]) *
            vz[:, vz_zs:vz_ze, vz_ys:vz_ye, vz_xs:vz_xe] -
            alpha2[vz_zs:vz_ze, vz_ys:vz_ye, vz_xs:vz_xe] * (
                c1 * (p[:, vz_zs + 1:vz_ze + 1, vz_ys:vz_ye, vz_xs:vz_xe] -
                      p[:, vz_zs:vz_ze, vz_ys:vz_ye, vz_xs:vz_xe]) +
                c2 * (p[:, vz_zs + 2:vz_ze + 2, vz_ys:vz_ye, vz_xs:vz_xe] -
                      p[:, vz_zs - 1:vz_ze - 1, vz_ys:vz_ye, vz_xs:vz_xe])
            )
        )

        # ====== Free surface condition for vz ======
        if free_surface:
            vz[:, free_surface_start - 1, :, :] = vz[:, free_surface_start, :, :]

        # ====== Record seismograms ======
        rcv_p[:, it, :] = p[:, rcv_z, rcv_y, rcv_x]
        rcv_vx[:, it, :] = vx[:, rcv_z, rcv_y, rcv_x]
        rcv_vy[:, it, :] = vy[:, rcv_z, rcv_y, rcv_x]
        rcv_vz[:, it, :] = vz[:, rcv_z, rcv_y, rcv_x]

        # ====== Accumulate forward wavefield energy ======
        forward_wavefield_p += torch.sum(
            p * p, dim=0
        )[nabc:nabc + nz, nabc:nabc + ny, nabc:nabc + nx].detach()
        forward_wavefield_vx += torch.sum(
            vx * vx, dim=0
        )[nabc:nabc + nz, nabc:nabc + ny, nabc:nabc + nx].detach()
        forward_wavefield_vy += torch.sum(
            vy * vy, dim=0
        )[nabc:nabc + nz, nabc:nabc + ny, nabc:nabc + nx].detach()
        forward_wavefield_vz += torch.sum(
            vz * vz, dim=0
        )[nabc:nabc + nz, nabc:nabc + ny, nabc:nabc + nx].detach()

    return (
        p, vx, vy, vz,
        rcv_p, rcv_vx, rcv_vy, rcv_vz,
        forward_wavefield_p, forward_wavefield_vx,
        forward_wavefield_vy, forward_wavefield_vz,
        torch.tensor(0),  # placeholder for TorchScript tuple length consistency
    )


def acoustic_forward_kernel_3d(
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    dy: float,
    dz: float,
    nt: int,
    dt: float,
    nabc: int,
    free_surface: bool,
    src_x: Tensor,
    src_y: Tensor,
    src_z: Tensor,
    src_n: int,
    src_v: Tensor,
    rcv_x: Tensor,
    rcv_y: Tensor,
    rcv_z: Tensor,
    rcv_n: int,
    damp: Tensor,
    vp: Tensor,
    rho: Tensor,
    checkpoint_segments: int = 1,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Tensor]:
    """
    Forward simulation kernel for 3D acoustic wave equation.

    Main entry point for 3D acoustic wave propagation. Handles model
    padding, coefficient computation, and time stepping with
    optional gradient checkpointing.

    Args:
        nx: Number of grid points in x-direction.
        ny: Number of grid points in y-direction.
        nz: Number of grid points in z-direction.
        dx: Grid spacing in x-direction (m).
        dy: Grid spacing in y-direction (m).
        dz: Grid spacing in z-direction (m).
        nt: Number of time steps.
        dt: Time step (s).
        nabc: Number of absorbing boundary layers.
        free_surface: Whether to use free surface at top.
        src_x: Source x-positions (grid indices).
        src_y: Source y-positions (grid indices).
        src_z: Source z-positions (grid indices).
        src_n: Number of sources.
        src_v: Source wavelets with shape (src_n, nt) or (nt,).
        rcv_x: Receiver x-positions (grid indices).
        rcv_y: Receiver y-positions (grid indices).
        rcv_z: Receiver z-positions (grid indices).
        rcv_n: Number of receivers.
        damp: 3D damping coefficients from boundary module.
        vp: P-wave velocity model (m/s) with shape (nz, ny, nx).
        rho: Density model (kg/m³) with shape (nz, ny, nx).
        checkpoint_segments: Number of segments for gradient checkpointing.
        device: Computation device.
        dtype: Data type.

    Returns:
        Dictionary containing:
            - 'p': Pressure seismograms (src_n, nt, rcv_n)
            - 'vx': X-velocity seismograms
            - 'vy': Y-velocity seismograms
            - 'vz': Z-velocity seismograms
            - 'forward_wavefield_p': Pressure energy (nz, ny, nx)
            - 'forward_wavefield_vx': X-velocity energy
            - 'forward_wavefield_vy': Y-velocity energy
            - 'forward_wavefield_vz': Z-velocity energy
    """
    # Pad model parameters
    c = pad_model_3d(vp, nabc, nz, ny, nx, device=device)
    den = pad_model_3d(rho, nabc, nz, ny, nx, device=device)

    free_surface_start = nabc if free_surface else 1
    nx_pml = nx + 2 * nabc
    ny_pml = ny + 2 * nabc
    nz_pml = nz + 2 * nabc

    # Adjust source and receiver positions for padding
    src_x = src_x + nabc
    src_y = src_y + nabc
    src_z = src_z + nabc
    rcv_x = rcv_x + nabc
    rcv_y = rcv_y + nabc
    rcv_z = rcv_z + nabc

    # Initialize fields
    p = torch.zeros((src_n, nz_pml, ny_pml, nx_pml), dtype=dtype, device=device)
    vx = torch.zeros((src_n, nz_pml, ny_pml, nx_pml - 1), dtype=dtype, device=device)
    vy = torch.zeros((src_n, nz_pml, ny_pml - 1, nx_pml), dtype=dtype, device=device)
    vz = torch.zeros((src_n, nz_pml - 1, ny_pml, nx_pml), dtype=dtype, device=device)

    # Initialize recorded waveforms
    rcv_p = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_vx = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_vy = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_vz = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)

    # Initialize wavefield accumulators
    forward_wavefield_p = torch.zeros((nz, ny, nx), dtype=dtype, device=device)
    forward_wavefield_vx = torch.zeros((nz, ny, nx), dtype=dtype, device=device)
    forward_wavefield_vy = torch.zeros((nz, ny, nx), dtype=dtype, device=device)
    forward_wavefield_vz = torch.zeros((nz, ny, nx), dtype=dtype, device=device)

    # 4th-order FD coefficients
    c1 = 9.0 / 8.0
    c2 = -1.0 / 24.0

    # Precompute coefficients
    # Pressure update: alpha1 = rho * vp^2 * dt / dz
    alpha1 = den * c * c * dt / dz
    kappa1 = damp * dt

    # Velocity update: alpha2 = dt / (rho * dz)
    alpha2 = dt / (den * dz)

    # kappa2 for vx: average damp at staggered x-positions
    kappa2 = torch.zeros_like(damp, device=device)
    kappa2[:, :, 1:nx_pml - 2] = 0.5 * (
        damp[:, :, 1:nx_pml - 2] + damp[:, :, 2:nx_pml - 1]
    ) * dt

    # kappa3 for vy: average damp at staggered y-positions
    kappa3 = torch.zeros_like(damp, device=device)
    kappa3[:, 1:ny_pml - 2, :] = 0.5 * (
        damp[:, 1:ny_pml - 2, :] + damp[:, 2:ny_pml - 1, :]
    ) * dt

    # kappa4 for vz: average damp at staggered z-positions
    kappa4 = torch.zeros_like(damp, device=device)
    kappa4[free_surface_start:nz_pml - 2, :, :] = 0.5 * (
        damp[free_surface_start:nz_pml - 2, :, :] +
        damp[free_surface_start + 1:nz_pml - 1, :, :]
    ) * dt

    # Time stepping with checkpointing
    k = 0
    for i, chunk in enumerate(torch.chunk(src_v, checkpoint_segments, dim=-1)):
        if checkpoint_segments > 1:
            result = checkpoint(
                acoustic_step_forward_3d,
                nx, ny, nz, dx, dy, dz, dt,
                nabc, free_surface,
                src_x, src_y, src_z, src_n, chunk,
                rcv_x, rcv_y, rcv_z, rcv_n,
                kappa1, alpha1, kappa2, alpha2, kappa3, kappa4, c1, c2,
                p, vx, vy, vz,
                device, dtype,
                use_reentrant=True
            )
        else:
            result = acoustic_step_forward_3d(
                nx, ny, nz, dx, dy, dz, dt,
                nabc, free_surface,
                src_x, src_y, src_z, src_n, chunk,
                rcv_x, rcv_y, rcv_z, rcv_n,
                kappa1, alpha1, kappa2, alpha2, kappa3, kappa4, c1, c2,
                p, vx, vy, vz,
                device, dtype,
            )

        p = result[0]
        vx = result[1]
        vy = result[2]
        vz = result[3]
        rcv_p_temp = result[4]
        rcv_vx_temp = result[5]
        rcv_vy_temp = result[6]
        rcv_vz_temp = result[7]
        fwd_p_temp = result[8]
        fwd_vx_temp = result[9]
        fwd_vy_temp = result[10]
        fwd_vz_temp = result[11]

        # Save recorded waveforms
        chunk_len = chunk.shape[-1]
        rcv_p[:, k:k + chunk_len] = rcv_p_temp
        rcv_vx[:, k:k + chunk_len] = rcv_vx_temp
        rcv_vy[:, k:k + chunk_len] = rcv_vy_temp
        rcv_vz[:, k:k + chunk_len] = rcv_vz_temp

        # Accumulate wavefield
        forward_wavefield_p += fwd_p_temp.detach()
        forward_wavefield_vx += fwd_vx_temp.detach()
        forward_wavefield_vy += fwd_vy_temp.detach()
        forward_wavefield_vz += fwd_vz_temp.detach()

        k += chunk_len

    return {
        'p': rcv_p,
        'vx': rcv_vx,
        'vy': rcv_vy,
        'vz': rcv_vz,
        'forward_wavefield_p': forward_wavefield_p,
        'forward_wavefield_vx': forward_wavefield_vx,
        'forward_wavefield_vy': forward_wavefield_vy,
        'forward_wavefield_vz': forward_wavefield_vz,
    }
