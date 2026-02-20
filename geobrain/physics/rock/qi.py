#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quantitative Interpretation (QI) models for rock physics.

This module provides composite models for rock physics diagnostics
and quantitative interpretation, including:
    - Elastic screening bounds (soft sand, contact cement, MUHS)
    - Constant cement velocity modeling
    - Cement volume estimation from well logs
    - Rock physics templates (RPT)

All classes follow the CompositeModel pattern for PyTorch compatibility.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
from torch import Tensor
from typing import Tuple

from .core import CompositeModel, as_tensor, EPS
from .utils import v_from_moduli, matrix_modulus, den_matrix, normalize_kde
from .models.granular import SoftSand, ContactCement, ContactCementFull, ConstantCement, MUHS
from .models.fluid import Gassmann
from .models.effective import VRH


class Screening(CompositeModel):
    """
    Rock physics screening bounds.

    Computes three diagnostic velocity-porosity trends:
        - Model 1 (soft sand): Friable sand lower bound
        - Model 2 (MUHS): Increasing cement upper bound (low porosity)
        - Model 3 (contact cement): Contact cement upper bound (high porosity)

    These bounds are used to classify rock quality and cementation
    in porosity-velocity crossplots.

    References:
        Dvorkin, J. & Nur, A. (1996). Elasticity of high-porosity sandstones.
        Avseth, P. et al. (2005). Quantitative Seismic Interpretation.

    Args:
        phi_c: Critical porosity. Default: 0.4
        Cn: Coordination number. Default: 8.6
        scheme: Cementation scheme (1=contacts, 2=surfaces). Default: 2
        f: Friction coefficient for soft sand. Default: 0.5

    Example:
        >>> scr = Screening()
        >>> phi, vp1, vp2, vp3, vs1, vs2, vs3 = scr(
        ...     K_grain=36.6, G_grain=45.0, rho_grain=2.65,
        ...     K_sh=21.0, G_sh=7.0, rho_sh=2.58,
        ...     K_cem=36.6, G_cem=45.0, rho_cem=2.65,
        ...     K_fl=2.25, rho_fl=1.04
        ... )
    """

    def __init__(self, phi_c: float = 0.4, Cn: float = 8.6,
                 scheme: int = 2, f: float = 0.5):
        super().__init__()
        self.phi_c = phi_c
        self.Cn = Cn
        self.scheme = scheme
        self.f = f

    def forward(
            self,
            K_grain: Tensor,
            G_grain: Tensor,
            rho_grain: Tensor,
            K_sh: Tensor,
            G_sh: Tensor,
            rho_sh: Tensor,
            K_cem: Tensor,
            G_cem: Tensor,
            rho_cem: Tensor,
            K_fl: Tensor,
            rho_fl: Tensor,
            vsh: Tensor = 0.0,
            P: Tensor = 20.0,
            phi_b: Tensor = 0.3,
            n_points: int = 100
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Compute screening velocity bounds.

        Args:
            K_grain: Grain bulk modulus (GPa)
            G_grain: Grain shear modulus (GPa)
            rho_grain: Grain density (g/cm³)
            K_sh: Shale bulk modulus (GPa)
            G_sh: Shale shear modulus (GPa)
            rho_sh: Shale density (g/cm³)
            K_cem: Cement bulk modulus (GPa)
            G_cem: Cement shear modulus (GPa)
            rho_cem: Cement density (g/cm³)
            K_fl: Fluid bulk modulus (GPa)
            rho_fl: Fluid density (g/cm³)
            vsh: Shale volume fraction. Default: 0.0
            P: Effective pressure (MPa). Default: 20.0
            phi_b: Porosity boundary between cement models. Default: 0.3
            n_points: Number of porosity points. Default: 100

        Returns:
            Tuple of (phi, vp1, vp2, vp3, vs1, vs2, vs3):
                - phi: Porosity array
                - vp1, vs1: Soft sand (friable) velocities (m/s)
                - vp2, vs2: MUHS (increasing cement) velocities (m/s)
                - vp3, vs3: Contact cement velocities (m/s)
        """
        K_grain, G_grain = as_tensor(K_grain), as_tensor(G_grain)
        rho_grain = as_tensor(rho_grain)
        K_sh, G_sh, rho_sh = as_tensor(K_sh), as_tensor(G_sh), as_tensor(rho_sh)
        K_cem, G_cem, rho_cem = as_tensor(K_cem), as_tensor(G_cem), as_tensor(rho_cem)
        K_fl, rho_fl = as_tensor(K_fl), as_tensor(rho_fl)
        vsh = as_tensor(vsh)
        P, phi_b = as_tensor(P), as_tensor(phi_b)
        phi_c = as_tensor(self.phi_c)
        Cn = as_tensor(self.Cn)

        # Effective grain moduli via VRH
        vrh = VRH()
        M_K = torch.stack([K_sh.squeeze(), K_grain.squeeze()])
        M_G = torch.stack([G_sh.squeeze(), G_grain.squeeze()])
        f_vol = torch.stack([vsh.squeeze(), (1.0 - vsh).squeeze()])
        _, _, K0 = vrh(M_K, f_vol)
        _, _, G0 = vrh(M_G, f_vol)
        rho0 = rho_sh.squeeze() * vsh.squeeze() + rho_grain.squeeze() * (1.0 - vsh.squeeze())

        # Porosity range
        phi = torch.linspace(1e-7, self.phi_c, n_points)

        # -- Model 1: Soft sand (friable, lower bound) --
        soft = SoftSand()
        Kdry1, Gdry1 = soft(K0, G0, phi, phi_c, Cn, P, self.f)

        gass = Gassmann()
        Ksat1, Gsat1 = gass(Kdry1, Gdry1, K0, K_fl, phi)
        rho = rho0 * (1.0 - phi) + rho_fl.squeeze() * phi
        vp1, vs1 = v_from_moduli(Ksat1, Gsat1, rho)

        # -- Model 3: Contact cement (upper bound, high porosity) --
        ccm = ContactCementFull(scheme=self.scheme)
        Kdry3, Gdry3 = ccm(K0, G0, K_cem, G_cem, phi, phi_c, Cn)

        K_mat = matrix_modulus(vsh, phi_c, phi, K_sh, K_grain, K_cem)
        Den_mat = den_matrix(vsh, phi_c, phi, rho_sh, rho_grain, rho_cem)

        Ksat3, Gsat3 = gass(Kdry3, Gdry3, K_mat, K_fl, phi)
        rho_cst = Den_mat * (1.0 - phi) + rho_fl.squeeze() * phi
        vp3, vs3 = v_from_moduli(Ksat3, Gsat3, rho_cst)

        # Contact cement invalid for phi < phi_b
        mask3 = phi < phi_b
        nan_val = torch.tensor(float('nan'))
        vp3 = torch.where(mask3, nan_val, vp3)
        vs3 = torch.where(mask3, nan_val, vs3)

        # -- Model 2: MUHS (increasing cement, low porosity) --
        # Use rockphypy-style HS upper bound with K0, G0 as stiff reference
        ccm_b = ContactCementFull(scheme=self.scheme)
        K_b, G_b = ccm_b(K0, G0, K_cem, G_cem, phi_b, phi_c, Cn)
        phi_ratio = phi / phi_b
        Kdry2 = -4/3*G0 + ((phi_ratio)/(K_b + 4/3*G0) + (1-phi_ratio)/(K0 + 4/3*G0))**(-1)
        zeta_g = G0/6 * (9*K0 + 8*G0) / (K0 + 2*G0 + EPS)
        Gdry2 = -zeta_g + ((phi_ratio)/(G_b + zeta_g) + (1-phi_ratio)/(G0 + zeta_g))**(-1)

        Ksat2, Gsat2 = gass(Kdry2, Gdry2, K0, K_fl, phi)
        vp2, vs2 = v_from_moduli(Ksat2, Gsat2, rho)

        # MUHS invalid for phi > phi_b
        mask2 = phi > phi_b
        vp2 = torch.where(mask2, nan_val, vp2)
        vs2 = torch.where(mask2, nan_val, vs2)

        return phi, vp1, vp2, vp3, vs1, vs2, vs3


class ConstantCementVelocity(CompositeModel):
    """
    Compute velocities using the constant cement model.

    Combines the ConstantCement dry rock model with Gassmann fluid
    substitution and density calculation to produce saturated velocities
    at a given cement amount (defined by phi_b).

    References:
        Avseth, P. et al. (2000). Rock physics diagnostic of North Sea sands.

    Example:
        >>> ccv = ConstantCementVelocity()
        >>> Vp, Vs = ccv(
        ...     phi_b=0.3, K0=36.6, G0=45.0, rho0=2.65,
        ...     K_cem=36.6, G_cem=45.0,
        ...     K_fl=2.25, rho_fl=1.04,
        ...     phi=torch.linspace(0.01, 0.29, 50)
        ... )
    """

    def __init__(self):
        super().__init__()

    def forward(
            self,
            phi_b: Tensor,
            K0: Tensor,
            G0: Tensor,
            rho0: Tensor,
            K_cem: Tensor,
            G_cem: Tensor,
            K_fl: Tensor,
            rho_fl: Tensor,
            phi: Tensor,
            phi_c: float = 0.4,
            Cn: float = 8.6,
            scheme: int = 2,
            vsh: float = 0.0,
            rho_sh: float = 2.7,
            rho_grain: float = 2.65,
            rho_cem: float = 2.65
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute saturated velocities from constant cement model.

        Args:
            phi_b: Reference cemented porosity (high-porosity end member)
            K0: Effective grain bulk modulus (GPa)
            G0: Effective grain shear modulus (GPa)
            rho0: Grain density (g/cm³)
            K_cem: Cement bulk modulus (GPa)
            G_cem: Cement shear modulus (GPa)
            K_fl: Fluid bulk modulus (GPa)
            rho_fl: Fluid density (g/cm³)
            phi: Target porosity array
            phi_c: Critical porosity. Default: 0.4
            Cn: Coordination number. Default: 8.6
            scheme: Cementation scheme. Default: 2
            vsh: Shale volume fraction. Default: 0.0
            rho_sh: Shale density (g/cm³). Default: 2.7
            rho_grain: Grain density (g/cm³). Default: 2.65
            rho_cem: Cement density (g/cm³). Default: 2.65

        Returns:
            Tuple of (Vp, Vs) in m/s. NaN where phi > phi_b.
        """
        phi_b = as_tensor(phi_b)
        K0, G0, rho0 = as_tensor(K0), as_tensor(G0), as_tensor(rho0)
        K_cem, G_cem = as_tensor(K_cem), as_tensor(G_cem)
        K_fl, rho_fl = as_tensor(K_fl), as_tensor(rho_fl)
        phi = as_tensor(phi)

        # Dry rock moduli via constant cement
        cc = ConstantCement()
        Kdry, Gdry = cc(phi_b, K0, G0, K_cem, G_cem, phi,
                         as_tensor(phi_c), as_tensor(Cn), scheme)

        # Matrix density at reference cemented porosity
        D = den_matrix(vsh, phi_c, phi_b, rho_sh, rho_grain, rho_cem)

        # Gassmann fluid substitution
        gass = Gassmann()
        Ksat, Gsat = gass(Kdry, Gdry, K0, K_fl, phi)

        # Bulk density
        rho = D * (1.0 - phi) + rho_fl * phi

        # Velocities
        Vp, Vs = v_from_moduli(Ksat, Gsat, rho)

        # Invalid where phi > phi_b
        mask = phi > phi_b
        nan_val = torch.tensor(float('nan'))
        Vp = torch.where(mask, nan_val, Vp)
        Vs = torch.where(mask, nan_val, Vs)

        return Vp, Vs


class CementEstimator(CompositeModel):
    """
    Estimate cement volume from well log data.

    Uses constant cement model lines at various cement amounts to
    bracket observed Vp values and interpolate the cement volume
    fraction for each data point.

    This is an interpolation-based method (no autodiff).

    References:
        Avseth, P. et al. (2005). Quantitative Seismic Interpretation.

    Example:
        >>> est = CementEstimator()
        >>> vcem = est(
        ...     Vp=data_vp, phi=data_phi,
        ...     vcem_seeds=torch.tensor([0.0, 0.02, 0.04, 0.06, 0.08, 0.1]),
        ...     K_grain=36.6, G_grain=45.0, K_sh=21.0, G_sh=7.0,
        ...     K_cem=36.6, G_cem=45.0,
        ...     K_fl=2.25, rho_fl=1.04,
        ...     rho_grain=2.65, rho_sh=2.58, rho_cem=2.65
        ... )
    """

    def __init__(self):
        super().__init__()

    def forward(
            self,
            Vp: Tensor,
            phi: Tensor,
            vcem_seeds: Tensor,
            K_grain: Tensor,
            G_grain: Tensor,
            K_sh: Tensor,
            G_sh: Tensor,
            K_cem: Tensor,
            G_cem: Tensor,
            K_fl: Tensor,
            rho_fl: Tensor,
            rho_grain: Tensor,
            rho_sh: Tensor,
            rho_cem: Tensor,
            phi_c: float = 0.4,
            Cn: float = 8.6,
            scheme: int = 2,
            vsh: float = 0.0
    ) -> Tensor:
        """
        Estimate cement volume for each data point.

        Args:
            Vp: Observed P-wave velocities (m/s)
            phi: Observed porosities
            vcem_seeds: Cement volume seeds to evaluate (ascending order)
            K_grain: Grain bulk modulus (GPa)
            G_grain: Grain shear modulus (GPa)
            K_sh: Shale bulk modulus (GPa)
            G_sh: Shale shear modulus (GPa)
            K_cem: Cement bulk modulus (GPa)
            G_cem: Cement shear modulus (GPa)
            K_fl: Fluid bulk modulus (GPa)
            rho_fl: Fluid density (g/cm³)
            rho_grain: Grain density (g/cm³)
            rho_sh: Shale density (g/cm³)
            rho_cem: Cement density (g/cm³)
            phi_c: Critical porosity. Default: 0.4
            Cn: Coordination number. Default: 8.6
            scheme: Cementation scheme. Default: 2
            vsh: Shale volume fraction. Default: 0.0

        Returns:
            Estimated cement volume fraction per data point
        """
        Vp = as_tensor(Vp)
        phi = as_tensor(phi)
        vcem_seeds = as_tensor(vcem_seeds)

        # Effective grain moduli via VRH
        vrh = VRH()
        vsh_t = as_tensor(vsh)
        M_K = torch.stack([as_tensor(K_sh).squeeze(), as_tensor(K_grain).squeeze()])
        M_G = torch.stack([as_tensor(G_sh).squeeze(), as_tensor(G_grain).squeeze()])
        f_vol = torch.stack([vsh_t.squeeze(), (1.0 - vsh_t).squeeze()])
        _, _, K0 = vrh(M_K, f_vol)
        _, _, G0 = vrh(M_G, f_vol)
        rho0 = as_tensor(rho_sh).squeeze() * vsh_t.squeeze() + \
               as_tensor(rho_grain).squeeze() * (1.0 - vsh_t.squeeze())

        n_data = Vp.shape[0]
        n_seeds = vcem_seeds.shape[0]

        # Compute Vp for each cement seed at data porosities
        ccv = ConstantCementVelocity()
        vp_curves = torch.empty(n_data, n_seeds)
        for j in range(n_seeds):
            seed = vcem_seeds[j]
            phi_b = phi_c - seed
            vp_j, _ = ccv(
                phi_b, K0, G0, rho0, K_cem, G_cem, K_fl, rho_fl, phi,
                phi_c=phi_c, Cn=Cn, scheme=scheme, vsh=vsh,
                rho_sh=float(as_tensor(rho_sh).squeeze()),
                rho_grain=float(as_tensor(rho_grain).squeeze()),
                rho_cem=float(as_tensor(rho_cem).squeeze()),
            )
            vp_curves[:, j] = vp_j

        # Interpolate cement volume for each data point
        vcem = torch.full((n_data,), float('nan'))
        for i in range(n_data):
            val = Vp[i].item()
            arr = vp_curves[i]

            valid_mask = ~torch.isnan(arr)
            if not valid_mask.any():
                continue

            arr_v = arr[valid_mask]
            seeds_v = vcem_seeds[valid_mask]

            if val <= arr_v[0].item():
                vcem[i] = 0.0
            elif val >= arr_v[-1].item():
                vcem[i] = seeds_v[-1]
            else:
                # Linear interpolation between bracketing curves
                for j in range(len(arr_v) - 1):
                    lo = arr_v[j].item()
                    hi = arr_v[j + 1].item()
                    if lo <= val <= hi:
                        t = (val - lo) / (hi - lo + 1e-10)
                        vcem[i] = seeds_v[j] + t * (seeds_v[j + 1] - seeds_v[j])
                        break

        return vcem


class RPT(CompositeModel):
    """
    Rock Physics Template computation.

    Computes acoustic impedance (IP) and Vp/Vs ratio (PS) grids
    over porosity and water saturation for creating RPT crossplots.

    References:
        Avseth, P. et al. (2005). Quantitative Seismic Interpretation.

    Example:
        >>> rpt = RPT()
        >>> IP, PS = rpt(
        ...     K_dry=Kdry, G_dry=Gdry, K0=36.6, rho0=2.65,
        ...     K_brine=2.25, rho_brine=1.04,
        ...     K_hc=0.02, rho_hc=0.1,
        ...     phi=torch.linspace(0.05, 0.35, 7),
        ...     Sw=torch.linspace(0.0, 1.0, 5)
        ... )
    """

    def __init__(self):
        super().__init__()

    def forward(
            self,
            K_dry: Tensor,
            G_dry: Tensor,
            K0: Tensor,
            rho0: Tensor,
            K_brine: Tensor,
            rho_brine: Tensor,
            K_hc: Tensor,
            rho_hc: Tensor,
            phi: Tensor,
            Sw: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute RPT grid of impedance and Vp/Vs.

        Args:
            K_dry: Dry rock bulk modulus (GPa), shape (n_phi,)
            G_dry: Dry rock shear modulus (GPa), shape (n_phi,)
            K0: Mineral bulk modulus (GPa)
            rho0: Mineral density (g/cm³)
            K_brine: Brine bulk modulus (GPa)
            rho_brine: Brine density (g/cm³)
            K_hc: Hydrocarbon bulk modulus (GPa)
            rho_hc: Hydrocarbon density (g/cm³)
            phi: Porosity array, shape (n_phi,)
            Sw: Water saturation array, shape (n_sw,)

        Returns:
            Tuple of (IP, PS):
                - IP: Acoustic impedance, shape (n_phi, n_sw)
                - PS: Vp/Vs ratio, shape (n_phi, n_sw)
        """
        K_dry, G_dry = as_tensor(K_dry), as_tensor(G_dry)
        K0, rho0 = as_tensor(K0), as_tensor(rho0)
        K_brine, rho_brine = as_tensor(K_brine), as_tensor(rho_brine)
        K_hc, rho_hc = as_tensor(K_hc), as_tensor(rho_hc)
        phi = as_tensor(phi)
        Sw = as_tensor(Sw)

        n_phi = phi.shape[0]
        n_sw = Sw.shape[0]

        IP = torch.empty(n_phi, n_sw)
        PS = torch.empty(n_phi, n_sw)

        gass = Gassmann()

        for i, sw_val in enumerate(Sw):
            # Wood's equation for fluid mixing
            Kf = (sw_val / (K_brine + EPS) + (1.0 - sw_val) / (K_hc + EPS)) ** (-1)
            Df = sw_val * rho_brine + (1.0 - sw_val) * rho_hc

            # Gassmann fluid substitution
            Ksat, Gsat = gass(K_dry, G_dry, K0, Kf, phi)

            # Bulk density
            rho = rho0 * (1.0 - phi) + Df * phi

            # Velocities
            Vp, Vs = v_from_moduli(Ksat, Gsat, rho)

            IP[:, i] = Vp * rho
            PS[:, i] = Vp / (Vs + EPS)

        return IP, PS


# =============================================================================
# Visualization functions (optional, require matplotlib)
# =============================================================================

def screening_plot(phi, vp1, vp2, vp3,
                   data_phi=None, data_Vp=None, data_Vsh=None):
    """
    Plot rock physics screening crossplot.

    Args:
        phi: Porosity array from Screening model
        vp1: Soft sand Vp (lower bound)
        vp2: MUHS Vp (increasing cement)
        vp3: Contact cement Vp (upper bound)
        data_phi: Observed porosity data (optional)
        data_Vp: Observed Vp data (optional)
        data_Vsh: Shale volume for coloring (optional)

    Returns:
        matplotlib Figure, or None if matplotlib unavailable
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    phi_np = phi.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(phi_np, vp3.detach().cpu().numpy(), '-k', lw=4, alpha=0.7, label='Contact cement')
    ax.plot(phi_np, vp1.detach().cpu().numpy(), '--k', lw=2, alpha=0.7, label='Soft sand')
    ax.plot(phi_np, vp2.detach().cpu().numpy(), '-k', lw=4, alpha=0.7, label='MUHS')
    ax.set_ylabel('Vp (m/s)')
    ax.set_xlabel('Porosity')
    ax.grid(ls='--', alpha=0.7)

    if data_phi is not None and data_Vp is not None:
        dp = data_phi.detach().cpu().numpy() if isinstance(data_phi, Tensor) else data_phi
        dv = data_Vp.detach().cpu().numpy() if isinstance(data_Vp, Tensor) else data_Vp
        c = None
        if data_Vsh is not None:
            c = 1 - (data_Vsh.detach().cpu().numpy() if isinstance(data_Vsh, Tensor) else data_Vsh)
        sc = ax.scatter(dp, dv, c=c, vmin=0, vmax=1,
                        edgecolors='grey', s=100, alpha=1, cmap='viridis')
        if c is not None:
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label(r'$V_{Sand}$')

    ax.legend()
    return fig


def rpt_plot(IP, PS, phi, Sw):
    """
    Plot Rock Physics Template.

    Args:
        IP: Acoustic impedance grid, shape (n_phi, n_sw)
        PS: Vp/Vs grid, shape (n_phi, n_sw)
        phi: Porosity array
        Sw: Water saturation array

    Returns:
        matplotlib Figure, or None if matplotlib unavailable
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    IP_np = IP.detach().cpu().numpy()
    PS_np = PS.detach().cpu().numpy()

    fig = plt.figure(figsize=(10, 8))
    plt.plot(IP_np.T, PS_np.T, '-ok', mec='k', ms=10, mfc='yellow')
    plt.plot(IP_np[:, -1], PS_np[:, -1], '-ok', mec='k', ms=10, mfc='blue')
    plt.xlabel('Acoustic Impedance')
    plt.ylabel('Vp/Vs')
    return fig


def cement_diagnostic_plot(phi, vp_bounds, vp_cst_lines,
                           data_phi, data_Vp, vcem):
    """
    Plot cement diagnostic crossplot.

    Args:
        phi: Porosity array for model curves
        vp_bounds: Tuple of (vp_soft, vp_cement) bound curves
        vp_cst_lines: List of constant cement Vp curves
        data_phi: Observed porosity
        data_Vp: Observed Vp
        vcem: Estimated cement volume per data point

    Returns:
        matplotlib Figure, or None if matplotlib unavailable
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    phi_np = phi.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 6))

    vp_soft, vp_cem = vp_bounds
    ax.plot(phi_np, vp_soft.detach().cpu().numpy(), '-k', lw=3, alpha=0.7)
    ax.plot(phi_np, vp_cem.detach().cpu().numpy(), '-k', lw=4, alpha=0.7)

    for vp_cst in vp_cst_lines:
        ax.plot(phi_np, vp_cst.detach().cpu().numpy(), '--k', lw=2, alpha=0.7)

    dp = data_phi.detach().cpu().numpy() if isinstance(data_phi, Tensor) else data_phi
    dv = data_Vp.detach().cpu().numpy() if isinstance(data_Vp, Tensor) else data_Vp
    vc = vcem.detach().cpu().numpy() if isinstance(vcem, Tensor) else vcem

    sc = ax.scatter(dp, dv, c=vc, edgecolors='grey', s=100,
                    vmin=0, vmax=0.1, alpha=0.7, cmap='RdYlGn_r')
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r'$V_{CEM}$')
    ax.set_xlabel('Porosity')
    ax.set_ylabel('Vp (m/s)')
    ax.grid()
    return fig
