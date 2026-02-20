"""
3D Elastic wave equation FDTD kernels.

Implements finite-difference time-domain kernels for 3D elastic
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
def elastic_step_forward_3d(
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
    kappa_s: Tensor,
    kappa_sxy: Tensor,
    kappa_sxz: Tensor,
    kappa_syz: Tensor,
    kappa_vx: Tensor,
    kappa_vy: Tensor,
    kappa_vz: Tensor,
    alpha_lamu: Tensor,
    alpha_lam: Tensor,
    alpha_mu_xy: Tensor,
    alpha_mu_xz: Tensor,
    alpha_mu_yz: Tensor,
    alpha_bx: Tensor,
    alpha_by: Tensor,
    alpha_bz: Tensor,
    c1: float,
    c2: float,
    sigma_xx: Tensor,
    sigma_yy: Tensor,
    sigma_zz: Tensor,
    sigma_xy: Tensor,
    sigma_xz: Tensor,
    sigma_yz: Tensor,
    vx: Tensor,
    vy: Tensor,
    vz: Tensor,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Time-stepping segment for 3D elastic wave equation.

    Uses 4th-order staggered grid finite difference scheme
    with velocity-stress formulation.

    Returns:
        Tuple of 18 tensors:
            - 9 fields: sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_xz,
              sigma_yz, vx, vy, vz
            - 4 seismograms: rcv_p, rcv_vx, rcv_vy, rcv_vz
            - 4 wavefields: fwd_p, fwd_vx, fwd_vy, fwd_vz
            - 1 placeholder for TorchScript consistency
    """
    sigma_xx = sigma_xx.clone()
    sigma_yy = sigma_yy.clone()
    sigma_zz = sigma_zz.clone()
    sigma_xy = sigma_xy.clone()
    sigma_xz = sigma_xz.clone()
    sigma_yz = sigma_yz.clone()
    vx = vx.clone()
    vy = vy.clone()
    vz = vz.clone()

    nt = src_v.shape[-1]
    fs = nabc if free_surface else 1
    nx_pml = nx + 2 * nabc
    ny_pml = ny + 2 * nabc
    nz_pml = nz + 2 * nabc

    # Initialize recorded values
    rcv_p = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_vx = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_vy = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_vz = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)

    # Initialize forward wavefield accumulators
    fwd_p = torch.zeros((nz, ny, nx), dtype=dtype, device=device)
    fwd_vx = torch.zeros((nz, ny, nx), dtype=dtype, device=device)
    fwd_vy = torch.zeros((nz, ny, nx), dtype=dtype, device=device)
    fwd_vz = torch.zeros((nz, ny, nx), dtype=dtype, device=device)

    # Normal stress update range
    szs = fs + 1
    sze = nz_pml - 2
    sys_ = 2
    sye = ny_pml - 2
    sxs = 2
    sxe = nx_pml - 2

    # Shear stress ranges
    xz_zs = fs
    if xz_zs < 1:
        xz_zs = 1
    xz_ze = nz_pml - 3

    for it in range(nt):
        # ====== 1. Update normal stresses ======
        dvx_dx = (
            c1 * (vx[:, szs:sze, sys_:sye, sxs:sxe] -
                  vx[:, szs:sze, sys_:sye, sxs - 1:sxe - 1]) +
            c2 * (vx[:, szs:sze, sys_:sye, sxs + 1:sxe + 1] -
                  vx[:, szs:sze, sys_:sye, sxs - 2:sxe - 2])
        )
        dvy_dy = (
            c1 * (vy[:, szs:sze, sys_:sye, sxs:sxe] -
                  vy[:, szs:sze, sys_ - 1:sye - 1, sxs:sxe]) +
            c2 * (vy[:, szs:sze, sys_ + 1:sye + 1, sxs:sxe] -
                  vy[:, szs:sze, sys_ - 2:sye - 2, sxs:sxe])
        )
        dvz_dz = (
            c1 * (vz[:, szs:sze, sys_:sye, sxs:sxe] -
                  vz[:, szs - 1:sze - 1, sys_:sye, sxs:sxe]) +
            c2 * (vz[:, szs + 1:sze + 1, sys_:sye, sxs:sxe] -
                  vz[:, szs - 2:sze - 2, sys_:sye, sxs:sxe])
        )

        div_v = dvx_dx + dvy_dy + dvz_dz

        sigma_xx[:, szs:sze, sys_:sye, sxs:sxe] = (
            (1.0 - kappa_s[szs:sze, sys_:sye, sxs:sxe]) *
            sigma_xx[:, szs:sze, sys_:sye, sxs:sxe] +
            alpha_lamu[szs:sze, sys_:sye, sxs:sxe] * dvx_dx +
            alpha_lam[szs:sze, sys_:sye, sxs:sxe] * (dvy_dy + dvz_dz)
        )
        sigma_yy[:, szs:sze, sys_:sye, sxs:sxe] = (
            (1.0 - kappa_s[szs:sze, sys_:sye, sxs:sxe]) *
            sigma_yy[:, szs:sze, sys_:sye, sxs:sxe] +
            alpha_lamu[szs:sze, sys_:sye, sxs:sxe] * dvy_dy +
            alpha_lam[szs:sze, sys_:sye, sxs:sxe] * (dvx_dx + dvz_dz)
        )
        sigma_zz[:, szs:sze, sys_:sye, sxs:sxe] = (
            (1.0 - kappa_s[szs:sze, sys_:sye, sxs:sxe]) *
            sigma_zz[:, szs:sze, sys_:sye, sxs:sxe] +
            alpha_lamu[szs:sze, sys_:sye, sxs:sxe] * dvz_dz +
            alpha_lam[szs:sze, sys_:sye, sxs:sxe] * (dvx_dx + dvy_dy)
        )

        # ====== 2. Update shear stresses ======
        # sigma_xy at (i, j+1/2, k+1/2)
        xy_ys = 1
        xy_ye = ny_pml - 3
        xy_xs = 1
        xy_xe = nx_pml - 3
        dvx_dy = (
            c1 * (vx[:, szs:sze, xy_ys + 1:xy_ye + 1, xy_xs:xy_xe] -
                  vx[:, szs:sze, xy_ys:xy_ye, xy_xs:xy_xe]) +
            c2 * (vx[:, szs:sze, xy_ys + 2:xy_ye + 2, xy_xs:xy_xe] -
                  vx[:, szs:sze, xy_ys - 1:xy_ye - 1, xy_xs:xy_xe])
        )
        dvy_dx = (
            c1 * (vy[:, szs:sze, xy_ys:xy_ye, xy_xs + 1:xy_xe + 1] -
                  vy[:, szs:sze, xy_ys:xy_ye, xy_xs:xy_xe]) +
            c2 * (vy[:, szs:sze, xy_ys:xy_ye, xy_xs + 2:xy_xe + 2] -
                  vy[:, szs:sze, xy_ys:xy_ye, xy_xs - 1:xy_xe - 1])
        )
        sigma_xy[:, szs:sze, xy_ys:xy_ye, xy_xs:xy_xe] = (
            (1.0 - kappa_sxy[szs:sze, xy_ys:xy_ye, xy_xs:xy_xe]) *
            sigma_xy[:, szs:sze, xy_ys:xy_ye, xy_xs:xy_xe] +
            alpha_mu_xy[szs:sze, xy_ys:xy_ye, xy_xs:xy_xe] * (dvx_dy + dvy_dx)
        )

        # sigma_xz at (i+1/2, j, k+1/2)
        xz_xs = 1
        xz_xe = nx_pml - 3
        dvx_dz = (
            c1 * (vx[:, xz_zs + 1:xz_ze + 1, sys_:sye, xz_xs:xz_xe] -
                  vx[:, xz_zs:xz_ze, sys_:sye, xz_xs:xz_xe]) +
            c2 * (vx[:, xz_zs + 2:xz_ze + 2, sys_:sye, xz_xs:xz_xe] -
                  vx[:, xz_zs - 1:xz_ze - 1, sys_:sye, xz_xs:xz_xe])
        )
        dvz_dx = (
            c1 * (vz[:, xz_zs:xz_ze, sys_:sye, xz_xs + 1:xz_xe + 1] -
                  vz[:, xz_zs:xz_ze, sys_:sye, xz_xs:xz_xe]) +
            c2 * (vz[:, xz_zs:xz_ze, sys_:sye, xz_xs + 2:xz_xe + 2] -
                  vz[:, xz_zs:xz_ze, sys_:sye, xz_xs - 1:xz_xe - 1])
        )
        sigma_xz[:, xz_zs:xz_ze, sys_:sye, xz_xs:xz_xe] = (
            (1.0 - kappa_sxz[xz_zs:xz_ze, sys_:sye, xz_xs:xz_xe]) *
            sigma_xz[:, xz_zs:xz_ze, sys_:sye, xz_xs:xz_xe] +
            alpha_mu_xz[xz_zs:xz_ze, sys_:sye, xz_xs:xz_xe] * (dvx_dz + dvz_dx)
        )

        # sigma_yz at (i+1/2, j+1/2, k)
        yz_ys = 1
        yz_ye = ny_pml - 3
        yz_xs = 1
        yz_xe = nx_pml - 2
        dvy_dz = (
            c1 * (vy[:, xz_zs + 1:xz_ze + 1, yz_ys:yz_ye, yz_xs:yz_xe] -
                  vy[:, xz_zs:xz_ze, yz_ys:yz_ye, yz_xs:yz_xe]) +
            c2 * (vy[:, xz_zs + 2:xz_ze + 2, yz_ys:yz_ye, yz_xs:yz_xe] -
                  vy[:, xz_zs - 1:xz_ze - 1, yz_ys:yz_ye, yz_xs:yz_xe])
        )
        dvz_dy = (
            c1 * (vz[:, xz_zs:xz_ze, yz_ys + 1:yz_ye + 1, yz_xs:yz_xe] -
                  vz[:, xz_zs:xz_ze, yz_ys:yz_ye, yz_xs:yz_xe]) +
            c2 * (vz[:, xz_zs:xz_ze, yz_ys + 2:yz_ye + 2, yz_xs:yz_xe] -
                  vz[:, xz_zs:xz_ze, yz_ys - 1:yz_ye - 1, yz_xs:yz_xe])
        )
        sigma_yz[:, xz_zs:xz_ze, yz_ys:yz_ye, yz_xs:yz_xe] = (
            (1.0 - kappa_syz[xz_zs:xz_ze, yz_ys:yz_ye, yz_xs:yz_xe]) *
            sigma_yz[:, xz_zs:xz_ze, yz_ys:yz_ye, yz_xs:yz_xe] +
            alpha_mu_yz[xz_zs:xz_ze, yz_ys:yz_ye, yz_xs:yz_xe] * (dvy_dz + dvz_dy)
        )

        # ====== 3. Source injection (isotropic explosion) ======
        if src_z.dim() == 1:
            src_update = dt * (src_v[it] if len(src_v.shape) == 1 else src_v[:, it])
            sigma_xx[torch.arange(src_n), src_z, src_y, src_x] = (
                sigma_xx[torch.arange(src_n), src_z, src_y, src_x] + src_update
            )
            sigma_yy[torch.arange(src_n), src_z, src_y, src_x] = (
                sigma_yy[torch.arange(src_n), src_z, src_y, src_x] + src_update
            )
            sigma_zz[torch.arange(src_n), src_z, src_y, src_x] = (
                sigma_zz[torch.arange(src_n), src_z, src_y, src_x] + src_update
            )
        else:
            for i in range(src_n):
                src_update = dt * (src_v[i, it] if len(src_v.shape) == 2 else src_v[i, :, it])
                sigma_xx[i, src_z[i], src_y[i], src_x[i]] = (
                    sigma_xx[i, src_z[i], src_y[i], src_x[i]] + src_update
                )
                sigma_yy[i, src_z[i], src_y[i], src_x[i]] = (
                    sigma_yy[i, src_z[i], src_y[i], src_x[i]] + src_update
                )
                sigma_zz[i, src_z[i], src_y[i], src_x[i]] = (
                    sigma_zz[i, src_z[i], src_y[i], src_x[i]] + src_update
                )

        # ====== 4. Free surface BC for stresses ======
        if free_surface:
            sigma_zz[:, fs, :, :] = 0
            sigma_zz[:, :fs, :, :] = 0
            if fs >= 1:
                sigma_xz[:, :fs, :, :] = 0
                sigma_yz[:, :fs, :, :] = 0

        # ====== 5. Update vx ======
        vx_zs = fs
        if vx_zs < 2:
            vx_zs = 2
        vx_ze = nz_pml - 2  # sigma_xz z+1 must be <= nz_pml-2
        vx_ys = 2   # sigma_xy y-2 must be >= 0
        vx_ye = ny_pml - 2  # sigma_xy y+1 must be <= ny_pml-2
        vx_xs = 1
        vx_xe = nx_pml - 2

        dsxx_dx = (
            c1 * (sigma_xx[:, vx_zs:vx_ze, vx_ys:vx_ye, vx_xs + 1:vx_xe + 1] -
                  sigma_xx[:, vx_zs:vx_ze, vx_ys:vx_ye, vx_xs:vx_xe]) +
            c2 * (sigma_xx[:, vx_zs:vx_ze, vx_ys:vx_ye, vx_xs + 2:vx_xe + 2] -
                  sigma_xx[:, vx_zs:vx_ze, vx_ys:vx_ye, vx_xs - 1:vx_xe - 1])
        )
        dsxy_dy = (
            c1 * (sigma_xy[:, vx_zs:vx_ze, vx_ys:vx_ye, vx_xs:vx_xe] -
                  sigma_xy[:, vx_zs:vx_ze, vx_ys - 1:vx_ye - 1, vx_xs:vx_xe]) +
            c2 * (sigma_xy[:, vx_zs:vx_ze, vx_ys + 1:vx_ye + 1, vx_xs:vx_xe] -
                  sigma_xy[:, vx_zs:vx_ze, vx_ys - 2:vx_ye - 2, vx_xs:vx_xe])
        )
        dsxz_dz = (
            c1 * (sigma_xz[:, vx_zs:vx_ze, vx_ys:vx_ye, vx_xs:vx_xe] -
                  sigma_xz[:, vx_zs - 1:vx_ze - 1, vx_ys:vx_ye, vx_xs:vx_xe]) +
            c2 * (sigma_xz[:, vx_zs + 1:vx_ze + 1, vx_ys:vx_ye, vx_xs:vx_xe] -
                  sigma_xz[:, vx_zs - 2:vx_ze - 2, vx_ys:vx_ye, vx_xs:vx_xe])
        )

        vx[:, vx_zs:vx_ze, vx_ys:vx_ye, vx_xs:vx_xe] = (
            (1.0 - kappa_vx[vx_zs:vx_ze, vx_ys:vx_ye, vx_xs:vx_xe]) *
            vx[:, vx_zs:vx_ze, vx_ys:vx_ye, vx_xs:vx_xe] +
            alpha_bx[vx_zs:vx_ze, vx_ys:vx_ye, vx_xs:vx_xe] * (dsxx_dx + dsxy_dy + dsxz_dz)
        )

        # ====== 6. Update vy ======
        vy_zs = fs
        if vy_zs < 2:
            vy_zs = 2
        vy_ze = nz_pml - 2  # sigma_yz z+1 must be <= nz_pml-2
        vy_ys = 1
        vy_ye = ny_pml - 2
        vy_xs = 2   # sigma_xy x-2 must be >= 0
        vy_xe = nx_pml - 2  # sigma_xy x+1 must be <= nx_pml-2

        dsxy_dx = (
            c1 * (sigma_xy[:, vy_zs:vy_ze, vy_ys:vy_ye, vy_xs:vy_xe] -
                  sigma_xy[:, vy_zs:vy_ze, vy_ys:vy_ye, vy_xs - 1:vy_xe - 1]) +
            c2 * (sigma_xy[:, vy_zs:vy_ze, vy_ys:vy_ye, vy_xs + 1:vy_xe + 1] -
                  sigma_xy[:, vy_zs:vy_ze, vy_ys:vy_ye, vy_xs - 2:vy_xe - 2])
        )
        dsyy_dy = (
            c1 * (sigma_yy[:, vy_zs:vy_ze, vy_ys + 1:vy_ye + 1, vy_xs:vy_xe] -
                  sigma_yy[:, vy_zs:vy_ze, vy_ys:vy_ye, vy_xs:vy_xe]) +
            c2 * (sigma_yy[:, vy_zs:vy_ze, vy_ys + 2:vy_ye + 2, vy_xs:vy_xe] -
                  sigma_yy[:, vy_zs:vy_ze, vy_ys - 1:vy_ye - 1, vy_xs:vy_xe])
        )
        dsyz_dz = (
            c1 * (sigma_yz[:, vy_zs:vy_ze, vy_ys:vy_ye, vy_xs:vy_xe] -
                  sigma_yz[:, vy_zs - 1:vy_ze - 1, vy_ys:vy_ye, vy_xs:vy_xe]) +
            c2 * (sigma_yz[:, vy_zs + 1:vy_ze + 1, vy_ys:vy_ye, vy_xs:vy_xe] -
                  sigma_yz[:, vy_zs - 2:vy_ze - 2, vy_ys:vy_ye, vy_xs:vy_xe])
        )

        vy[:, vy_zs:vy_ze, vy_ys:vy_ye, vy_xs:vy_xe] = (
            (1.0 - kappa_vy[vy_zs:vy_ze, vy_ys:vy_ye, vy_xs:vy_xe]) *
            vy[:, vy_zs:vy_ze, vy_ys:vy_ye, vy_xs:vy_xe] +
            alpha_by[vy_zs:vy_ze, vy_ys:vy_ye, vy_xs:vy_xe] * (dsxy_dx + dsyy_dy + dsyz_dz)
        )

        # ====== 7. Update vz ======
        vz_zs = fs
        vz_ze = nz_pml - 2
        vz_ys = 2   # sigma_yz y-2 must be >= 0
        vz_ye = ny_pml - 2  # sigma_yz y+1 must be <= ny_pml-2
        vz_xs = 2   # sigma_xz x-2 must be >= 0
        vz_xe = nx_pml - 2  # sigma_xz x+1 must be <= nx_pml-2

        dsxz_dx = (
            c1 * (sigma_xz[:, vz_zs:vz_ze, vz_ys:vz_ye, vz_xs:vz_xe] -
                  sigma_xz[:, vz_zs:vz_ze, vz_ys:vz_ye, vz_xs - 1:vz_xe - 1]) +
            c2 * (sigma_xz[:, vz_zs:vz_ze, vz_ys:vz_ye, vz_xs + 1:vz_xe + 1] -
                  sigma_xz[:, vz_zs:vz_ze, vz_ys:vz_ye, vz_xs - 2:vz_xe - 2])
        )
        dsyz_dy = (
            c1 * (sigma_yz[:, vz_zs:vz_ze, vz_ys:vz_ye, vz_xs:vz_xe] -
                  sigma_yz[:, vz_zs:vz_ze, vz_ys - 1:vz_ye - 1, vz_xs:vz_xe]) +
            c2 * (sigma_yz[:, vz_zs:vz_ze, vz_ys + 1:vz_ye + 1, vz_xs:vz_xe] -
                  sigma_yz[:, vz_zs:vz_ze, vz_ys - 2:vz_ye - 2, vz_xs:vz_xe])
        )
        dszz_dz = (
            c1 * (sigma_zz[:, vz_zs + 1:vz_ze + 1, vz_ys:vz_ye, vz_xs:vz_xe] -
                  sigma_zz[:, vz_zs:vz_ze, vz_ys:vz_ye, vz_xs:vz_xe]) +
            c2 * (sigma_zz[:, vz_zs + 2:vz_ze + 2, vz_ys:vz_ye, vz_xs:vz_xe] -
                  sigma_zz[:, vz_zs - 1:vz_ze - 1, vz_ys:vz_ye, vz_xs:vz_xe])
        )

        vz[:, vz_zs:vz_ze, vz_ys:vz_ye, vz_xs:vz_xe] = (
            (1.0 - kappa_vz[vz_zs:vz_ze, vz_ys:vz_ye, vz_xs:vz_xe]) *
            vz[:, vz_zs:vz_ze, vz_ys:vz_ye, vz_xs:vz_xe] +
            alpha_bz[vz_zs:vz_ze, vz_ys:vz_ye, vz_xs:vz_xe] * (dsxz_dx + dsyz_dy + dszz_dz)
        )

        # ====== 8. Free surface BC for vz (mirror) ======
        if free_surface:
            vz[:, fs - 1, :, :] = vz[:, fs, :, :]

        # ====== 9. Record seismograms ======
        rcv_p[:, it, :] = -(
            sigma_xx[:, rcv_z, rcv_y, rcv_x] +
            sigma_yy[:, rcv_z, rcv_y, rcv_x] +
            sigma_zz[:, rcv_z, rcv_y, rcv_x]
        ) / 3.0
        rcv_vx[:, it, :] = vx[:, rcv_z, rcv_y, rcv_x]
        rcv_vy[:, it, :] = vy[:, rcv_z, rcv_y, rcv_x]
        rcv_vz[:, it, :] = vz[:, rcv_z, rcv_y, rcv_x]

        # ====== 10. Accumulate forward wavefield energy ======
        p_field = -(sigma_xx + sigma_yy + sigma_zz) / 3.0
        fwd_p += torch.sum(
            p_field * p_field, dim=0
        )[nabc:nabc + nz, nabc:nabc + ny, nabc:nabc + nx].detach()
        fwd_vx += torch.sum(
            vx * vx, dim=0
        )[nabc:nabc + nz, nabc:nabc + ny, nabc:nabc + nx].detach()
        fwd_vy += torch.sum(
            vy * vy, dim=0
        )[nabc:nabc + nz, nabc:nabc + ny, nabc:nabc + nx].detach()
        fwd_vz += torch.sum(
            vz * vz, dim=0
        )[nabc:nabc + nz, nabc:nabc + ny, nabc:nabc + nx].detach()

    return (
        sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_xz, sigma_yz,
        vx, vy, vz,
        rcv_p, rcv_vx, rcv_vy, rcv_vz,
        fwd_p, fwd_vx, fwd_vy, fwd_vz,
        torch.tensor(0),  # placeholder for TorchScript tuple length consistency
    )


def elastic_forward_kernel_3d(
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
    vs: Tensor,
    rho: Tensor,
    checkpoint_segments: int = 1,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Tensor]:
    """
    Forward simulation kernel for 3D elastic wave equation.

    Main entry point for 3D elastic wave propagation. Handles model
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
        vs: S-wave velocity model (m/s) with shape (nz, ny, nx).
        rho: Density model (kg/m^3) with shape (nz, ny, nx).
        checkpoint_segments: Number of segments for gradient checkpointing.
        device: Computation device.
        dtype: Data type.

    Returns:
        Dictionary containing:
            - 'p': Pressure seismograms (src_n, nt, rcv_n)
            - 'vx', 'vy', 'vz': Velocity seismograms
            - 'forward_wavefield_p/vx/vy/vz': Wavefield energy (nz, ny, nx)
    """
    # Pad model parameters
    vp_pad = pad_model_3d(vp, nabc, nz, ny, nx, device=device)
    vs_pad = pad_model_3d(vs, nabc, nz, ny, nx, device=device)
    rho_pad = pad_model_3d(rho, nabc, nz, ny, nx, device=device)

    nx_pml = nx + 2 * nabc
    ny_pml = ny + 2 * nabc
    nz_pml = nz + 2 * nabc
    fs = nabc if free_surface else 1

    # Adjust source and receiver positions for padding
    src_x = src_x + nabc
    src_y = src_y + nabc
    src_z = src_z + nabc
    rcv_x = rcv_x + nabc
    rcv_y = rcv_y + nabc
    rcv_z = rcv_z + nabc

    # Compute Lame parameters on padded grid
    mu = vs_pad ** 2 * rho_pad
    lamu = vp_pad ** 2 * rho_pad
    lam = lamu - 2 * mu
    b = 1.0 / rho_pad

    # Staggered averages for mu at shear stress positions (harmonic mean)
    # mu_xz at (i+1/2, j, k+1/2): shape (nz_pml-1, ny_pml, nx_pml-1)
    mu_xz = torch.zeros(nz_pml - 1, ny_pml, nx_pml - 1, dtype=dtype, device=device)
    m00 = mu[:nz_pml - 1, :, :nx_pml - 1]
    m10 = mu[1:nz_pml, :, :nx_pml - 1]
    m01 = mu[:nz_pml - 1, :, 1:nx_pml]
    m11 = mu[1:nz_pml, :, 1:nx_pml]
    nz_mask = (m00 > 0) & (m10 > 0) & (m01 > 0) & (m11 > 0)
    inv_s = torch.zeros_like(m00)
    inv_s[nz_mask] = 1.0 / m00[nz_mask] + 1.0 / m10[nz_mask] + 1.0 / m01[nz_mask] + 1.0 / m11[nz_mask]
    mu_xz[nz_mask] = 4.0 / inv_s[nz_mask]

    # mu_xy at (i, j+1/2, k+1/2): shape (nz_pml, ny_pml-1, nx_pml-1)
    mu_xy = torch.zeros(nz_pml, ny_pml - 1, nx_pml - 1, dtype=dtype, device=device)
    m00 = mu[:, :ny_pml - 1, :nx_pml - 1]
    m10 = mu[:, 1:ny_pml, :nx_pml - 1]
    m01 = mu[:, :ny_pml - 1, 1:nx_pml]
    m11 = mu[:, 1:ny_pml, 1:nx_pml]
    nz_mask = (m00 > 0) & (m10 > 0) & (m01 > 0) & (m11 > 0)
    inv_s = torch.zeros_like(m00)
    inv_s[nz_mask] = 1.0 / m00[nz_mask] + 1.0 / m10[nz_mask] + 1.0 / m01[nz_mask] + 1.0 / m11[nz_mask]
    mu_xy[nz_mask] = 4.0 / inv_s[nz_mask]

    # mu_yz at (i+1/2, j+1/2, k): shape (nz_pml-1, ny_pml-1, nx_pml)
    mu_yz = torch.zeros(nz_pml - 1, ny_pml - 1, nx_pml, dtype=dtype, device=device)
    m00 = mu[:nz_pml - 1, :ny_pml - 1, :]
    m10 = mu[1:nz_pml, :ny_pml - 1, :]
    m01 = mu[:nz_pml - 1, 1:ny_pml, :]
    m11 = mu[1:nz_pml, 1:ny_pml, :]
    nz_mask = (m00 > 0) & (m10 > 0) & (m01 > 0) & (m11 > 0)
    inv_s = torch.zeros_like(m00)
    inv_s[nz_mask] = 1.0 / m00[nz_mask] + 1.0 / m10[nz_mask] + 1.0 / m01[nz_mask] + 1.0 / m11[nz_mask]
    mu_yz[nz_mask] = 4.0 / inv_s[nz_mask]

    # Buoyancy staggered averages
    # bx at (i, j, k+1/2)
    bx = torch.zeros(nz_pml, ny_pml, nx_pml, dtype=dtype, device=device)
    bx[:, :, :nx_pml - 1] = 0.5 * (b[:, :, :nx_pml - 1] + b[:, :, 1:nx_pml])

    # by at (i, j+1/2, k)
    by = torch.zeros(nz_pml, ny_pml, nx_pml, dtype=dtype, device=device)
    by[:, :ny_pml - 1, :] = 0.5 * (b[:, :ny_pml - 1, :] + b[:, 1:ny_pml, :])

    # bz at (i+1/2, j, k)
    bz = torch.zeros(nz_pml, ny_pml, nx_pml, dtype=dtype, device=device)
    bz[:nz_pml - 1, :, :] = 0.5 * (b[:nz_pml - 1, :, :] + b[1:nz_pml, :, :])

    # 4th-order FD coefficients
    c1 = 9.0 / 8.0
    c2 = -1.0 / 24.0

    # Precompute alpha coefficients
    alpha_lamu = lamu * dt / dz
    alpha_lam = lam * dt / dz
    alpha_mu_xy = mu_xy * dt / dz
    alpha_mu_xz = mu_xz * dt / dz
    alpha_mu_yz = mu_yz * dt / dz
    alpha_bx = bx * dt / dz
    alpha_by = by * dt / dz
    alpha_bz = bz * dt / dz

    # Precompute kappa (damping) coefficients
    kappa_s = damp * dt

    # kappa for shear stresses (average damp at staggered positions)
    # kappa_sxy at (i, j+1/2, k+1/2)
    kappa_sxy = torch.zeros(nz_pml, ny_pml - 1, nx_pml - 1, dtype=dtype, device=device)
    kappa_sxy = 0.25 * (
        damp[:, :ny_pml - 1, :nx_pml - 1] + damp[:, 1:ny_pml, :nx_pml - 1] +
        damp[:, :ny_pml - 1, 1:nx_pml] + damp[:, 1:ny_pml, 1:nx_pml]
    ) * dt

    # kappa_sxz at (i+1/2, j, k+1/2)
    kappa_sxz = torch.zeros(nz_pml - 1, ny_pml, nx_pml - 1, dtype=dtype, device=device)
    kappa_sxz = 0.25 * (
        damp[:nz_pml - 1, :, :nx_pml - 1] + damp[1:nz_pml, :, :nx_pml - 1] +
        damp[:nz_pml - 1, :, 1:nx_pml] + damp[1:nz_pml, :, 1:nx_pml]
    ) * dt

    # kappa_syz at (i+1/2, j+1/2, k)
    kappa_syz = torch.zeros(nz_pml - 1, ny_pml - 1, nx_pml, dtype=dtype, device=device)
    kappa_syz = 0.25 * (
        damp[:nz_pml - 1, :ny_pml - 1, :] + damp[1:nz_pml, :ny_pml - 1, :] +
        damp[:nz_pml - 1, 1:ny_pml, :] + damp[1:nz_pml, 1:ny_pml, :]
    ) * dt

    # kappa_vx at (i, j, k+1/2)
    kappa_vx = torch.zeros_like(damp, device=device)
    kappa_vx[:, :, 1:nx_pml - 2] = 0.5 * (
        damp[:, :, 1:nx_pml - 2] + damp[:, :, 2:nx_pml - 1]
    ) * dt

    # kappa_vy at (i, j+1/2, k)
    kappa_vy = torch.zeros_like(damp, device=device)
    kappa_vy[:, 1:ny_pml - 2, :] = 0.5 * (
        damp[:, 1:ny_pml - 2, :] + damp[:, 2:ny_pml - 1, :]
    ) * dt

    # kappa_vz at (i+1/2, j, k)
    kappa_vz = torch.zeros_like(damp, device=device)
    kappa_vz[fs:nz_pml - 2, :, :] = 0.5 * (
        damp[fs:nz_pml - 2, :, :] + damp[fs + 1:nz_pml - 1, :, :]
    ) * dt

    # Initialize fields
    sigma_xx = torch.zeros((src_n, nz_pml, ny_pml, nx_pml), dtype=dtype, device=device)
    sigma_yy = torch.zeros((src_n, nz_pml, ny_pml, nx_pml), dtype=dtype, device=device)
    sigma_zz = torch.zeros((src_n, nz_pml, ny_pml, nx_pml), dtype=dtype, device=device)
    sigma_xy = torch.zeros((src_n, nz_pml, ny_pml - 1, nx_pml - 1), dtype=dtype, device=device)
    sigma_xz = torch.zeros((src_n, nz_pml - 1, ny_pml, nx_pml - 1), dtype=dtype, device=device)
    sigma_yz = torch.zeros((src_n, nz_pml - 1, ny_pml - 1, nx_pml), dtype=dtype, device=device)
    vx_field = torch.zeros((src_n, nz_pml, ny_pml, nx_pml - 1), dtype=dtype, device=device)
    vy_field = torch.zeros((src_n, nz_pml, ny_pml - 1, nx_pml), dtype=dtype, device=device)
    vz_field = torch.zeros((src_n, nz_pml - 1, ny_pml, nx_pml), dtype=dtype, device=device)

    # Initialize recorded waveforms
    rcv_p = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_vx = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_vy = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_vz = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)

    # Initialize wavefield accumulators
    fwd_p = torch.zeros((nz, ny, nx), dtype=dtype, device=device)
    fwd_vx = torch.zeros((nz, ny, nx), dtype=dtype, device=device)
    fwd_vy = torch.zeros((nz, ny, nx), dtype=dtype, device=device)
    fwd_vz = torch.zeros((nz, ny, nx), dtype=dtype, device=device)

    # Time stepping with checkpointing
    k = 0
    for i, chunk in enumerate(torch.chunk(src_v, checkpoint_segments, dim=-1)):
        if checkpoint_segments > 1:
            result = checkpoint(
                elastic_step_forward_3d,
                nx, ny, nz, dx, dy, dz, dt,
                nabc, free_surface,
                src_x, src_y, src_z, src_n, chunk,
                rcv_x, rcv_y, rcv_z, rcv_n,
                kappa_s, kappa_sxy, kappa_sxz, kappa_syz,
                kappa_vx, kappa_vy, kappa_vz,
                alpha_lamu, alpha_lam,
                alpha_mu_xy, alpha_mu_xz, alpha_mu_yz,
                alpha_bx, alpha_by, alpha_bz,
                c1, c2,
                sigma_xx, sigma_yy, sigma_zz,
                sigma_xy, sigma_xz, sigma_yz,
                vx_field, vy_field, vz_field,
                device, dtype,
                use_reentrant=True
            )
        else:
            result = elastic_step_forward_3d(
                nx, ny, nz, dx, dy, dz, dt,
                nabc, free_surface,
                src_x, src_y, src_z, src_n, chunk,
                rcv_x, rcv_y, rcv_z, rcv_n,
                kappa_s, kappa_sxy, kappa_sxz, kappa_syz,
                kappa_vx, kappa_vy, kappa_vz,
                alpha_lamu, alpha_lam,
                alpha_mu_xy, alpha_mu_xz, alpha_mu_yz,
                alpha_bx, alpha_by, alpha_bz,
                c1, c2,
                sigma_xx, sigma_yy, sigma_zz,
                sigma_xy, sigma_xz, sigma_yz,
                vx_field, vy_field, vz_field,
                device, dtype,
            )

        sigma_xx = result[0]
        sigma_yy = result[1]
        sigma_zz = result[2]
        sigma_xy = result[3]
        sigma_xz = result[4]
        sigma_yz = result[5]
        vx_field = result[6]
        vy_field = result[7]
        vz_field = result[8]
        rcv_p_temp = result[9]
        rcv_vx_temp = result[10]
        rcv_vy_temp = result[11]
        rcv_vz_temp = result[12]
        fwd_p_temp = result[13]
        fwd_vx_temp = result[14]
        fwd_vy_temp = result[15]
        fwd_vz_temp = result[16]

        # Save recorded waveforms
        chunk_len = chunk.shape[-1]
        rcv_p[:, k:k + chunk_len] = rcv_p_temp
        rcv_vx[:, k:k + chunk_len] = rcv_vx_temp
        rcv_vy[:, k:k + chunk_len] = rcv_vy_temp
        rcv_vz[:, k:k + chunk_len] = rcv_vz_temp

        # Accumulate wavefield
        fwd_p += fwd_p_temp.detach()
        fwd_vx += fwd_vx_temp.detach()
        fwd_vy += fwd_vy_temp.detach()
        fwd_vz += fwd_vz_temp.detach()

        k += chunk_len

    return {
        'p': rcv_p,
        'vx': rcv_vx,
        'vy': rcv_vy,
        'vz': rcv_vz,
        'forward_wavefield_p': fwd_p,
        'forward_wavefield_vx': fwd_vx,
        'forward_wavefield_vy': fwd_vy,
        'forward_wavefield_vz': fwd_vz,
    }
