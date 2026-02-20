#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fluid models for rock physics.

This module provides models for fluid substitution and fluid property
calculations in porous rocks, including Gassmann equations, fluid
mixing laws, and empirical fluid property correlations.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
from torch import Tensor
from typing import Tuple, Optional

from ..core import FluidModel, register, as_tensor, ensure_same_device, EPS, PI
from ..utils import write_iso_matrix


@register(name="Gassmann", category="fluid")
class Gassmann(FluidModel):
    """
    Gassmann fluid substitution for isotropic porous media.

    Computes saturated rock moduli from dry rock moduli using
    Gassmann's equations:

        K_sat = K_dry + (1 - K_dry/K0)² / (φ/K_fl + (1-φ)/K0 - K_dry/K0²)
        G_sat = G_dry

    Key assumptions:
        - Isotropic, homogeneous rock frame
        - Connected pore space in pressure equilibrium
        - Fluid does not interact with rock frame
        - Low frequency (quasi-static)

    References:
        Gassmann, F. (1951). Uber die Elastizitat poroser Medien.
        Vierteljahrsschrift der Naturforschenden Gesellschaft in Zurich,
        96, 1-23.

    Args:
        None (stateless model)

    Example:
        >>> gassmann = Gassmann()
        >>> K_sat, G_sat = gassmann(
        ...     K_dry=15.0,   # Dry bulk modulus (GPa)
        ...     G_dry=12.0,   # Dry shear modulus (GPa)
        ...     K0=36.6,   # Matrix bulk modulus (GPa)
        ...     K_fl=2.25,    # Fluid bulk modulus (GPa)
        ...     phi=0.25      # Porosity
        ... )
        >>> print(f"K_sat = {K_sat.item():.2f} GPa")
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "Gassmann fluid substitution")

    def forward(
            self,
            K_dry: Tensor,
            G_dry: Tensor,
            K0: Tensor,
            K_fl: Tensor,
            phi: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute saturated moduli from dry rock properties.

        Args:
            K_dry: Dry rock bulk modulus (GPa)
            G_dry: Dry rock shear modulus (GPa)
            K0: Mineral matrix bulk modulus (GPa)
            K_fl: Pore fluid bulk modulus (GPa)
            phi: Porosity (fraction, 0-1)

        Returns:
            Tuple of (K_sat, G_sat):
                - K_sat: Saturated bulk modulus (GPa)
                - G_sat: Saturated shear modulus (GPa, equals G_dry)
        """
        K_dry, G_dry = as_tensor(K_dry), as_tensor(G_dry)
        K0, K_fl, phi = as_tensor(K0), as_tensor(K_fl), as_tensor(phi)
        K_dry, G_dry, K0, K_fl, phi = ensure_same_device(K_dry, G_dry, K0, K_fl, phi)

        A = (1 - K_dry / K0) ** 2
        B = phi / K_fl + (1 - phi) / K0 - K_dry / (K0 ** 2 + EPS)
        K_sat = K_dry + A / (B + EPS)

        return K_sat, G_dry


@register(name="GassmannInverse", category="fluid")
class GassmannInverse(FluidModel):
    """
    Inverse Gassmann to compute dry rock moduli from saturated.

    Inverts the Gassmann equation to obtain dry rock properties
    from measured saturated rock properties.

    References:
        Gassmann, F. (1951). Uber die Elastizitat poroser Medien.
        Vierteljahrsschrift der Naturforschenden Gesellschaft in Zurich,
        96, 1-23.

    Args:
        None (stateless model)

    Example:
        >>> inv_gass = GassmannInverse()
        >>> K_dry, G_dry = inv_gass(
        ...     K_sat=20.0, G_sat=12.0, K0=36.6, K_fl=2.25, phi=0.25
        ... )
    """

    def __init__(self):
        super().__init__()

    def forward(
            self,
            K_sat: Tensor,
            G_sat: Tensor,
            K0: Tensor,
            K_fl: Tensor,
            phi: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute dry moduli from saturated rock properties.

        Args:
            K_sat: Saturated bulk modulus (GPa)
            G_sat: Saturated shear modulus (GPa)
            K0: Matrix bulk modulus (GPa)
            K_fl: Fluid bulk modulus (GPa)
            phi: Porosity (fraction)

        Returns:
            Tuple of (K_dry, G_dry)
        """
        K_sat, G_sat = as_tensor(K_sat), as_tensor(G_sat)
        K0, K_fl, phi = as_tensor(K0), as_tensor(K_fl), as_tensor(phi)

        A = K_sat * (phi * K0 / K_fl + 1 - phi) - K0
        B = phi * K0 / K_fl + K_sat / K0 - 1 - phi
        K_dry = A / (B + EPS)

        return K_dry, G_sat


@register(name="GassmannFluidSub", category="fluid", aliases=["FluidSub"])
class GassmannFluidSub(FluidModel):
    """
    Gassmann fluid-to-fluid substitution.

    Performs fluid substitution by first computing dry rock properties
    from initial saturation, then computing new saturated properties
    with the replacement fluid.

    References:
        Smith, T.M., Sondergeld, C.H., & Rai, C.S. (2003). Gassmann fluid
        substitutions: A tutorial. Geophysics, 68(2), 430-440.

    Args:
        None (stateless model)

    Example:
        >>> fluidsub = GassmannFluidSub()
        >>> K_sat2, G_sat2 = fluidsub(
        ...     K_sat1=20.0, G_sat1=12.0,  # Brine-saturated
        ...     K0=36.6,
        ...     K_fl1=2.80,                 # Brine
        ...     K_fl2=0.02,                 # Gas
        ...     phi=0.25
        ... )
    """

    def __init__(self):
        super().__init__()
        self._inv = GassmannInverse()
        self._fwd = Gassmann()

    def forward(
            self,
            K_sat1: Tensor,
            G_sat1: Tensor,
            K0: Tensor,
            K_fl1: Tensor,
            K_fl2: Tensor,
            phi: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Substitute fluid 1 with fluid 2.

        Args:
            K_sat1: Initial saturated bulk modulus (GPa)
            G_sat1: Initial saturated shear modulus (GPa)
            K0: Matrix bulk modulus (GPa)
            K_fl1: Initial fluid bulk modulus (GPa)
            K_fl2: New fluid bulk modulus (GPa)
            phi: Porosity (fraction)

        Returns:
            Tuple of (K_sat2, G_sat2) with new fluid
        """
        K_dry, G_dry = self._inv(K_sat1, G_sat1, K0, K_fl1, phi)
        return self._fwd(K_dry, G_dry, K0, K_fl2, phi)


@register(name="Wood", category="fluid", aliases=["ReussFluid"])
class Wood(FluidModel):
    """
    Wood's equation for fluid mixing (Reuss average).

    Computes effective fluid properties for a mixture using
    iso-stress (Reuss) averaging:

        1/K_eff = Σ(f_i / K_i)
        ρ_eff = Σ(f_i * ρ_i)

    References:
        Wood, A.B. (1955). A Textbook of Sound. Bell and Sons, London.

    Args:
        None (stateless model)

    Example:
        >>> wood = Wood()
        >>> K = torch.tensor([2.25, 0.02])    # Water, gas moduli
        >>> rho = torch.tensor([1.0, 0.1])    # Water, gas densities
        >>> f = torch.tensor([0.8, 0.2])      # Saturations
        >>> K_eff, rho_eff = wood(K, rho, f)
    """

    def __init__(self):
        super().__init__()

    def forward(
            self,
            K: Tensor,
            rho: Tensor,
            f: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute effective fluid properties.

        Args:
            K: Bulk moduli of fluid phases (GPa)
            rho: Densities of fluid phases (g/cm³)
            f: Volume fractions (should sum to 1)

        Returns:
            Tuple of (K_eff, rho_eff)
        """
        K, rho, f = as_tensor(K), as_tensor(rho), as_tensor(f)
        return 1.0 / torch.sum(f / (K + EPS)), torch.sum(f * rho)

    def compute_fluid_mix(self, fl1_K, fl2_K, fl1_rho, fl2_rho, Sw, **params):
        K = torch.tensor([fl1_K, fl2_K])
        rho = torch.tensor([fl1_rho, fl2_rho])
        f = torch.stack([as_tensor(Sw), 1 - as_tensor(Sw)])
        return self.forward(K, rho, f)


@register(name="Brie", category="fluid")
class Brie(FluidModel):
    """
    Brie mixing law for patchy saturation.

    Empirical mixing law that accounts for patchy fluid distribution:

        K_eff = K_gas + (K_water - K_gas) * Sw^e

    The exponent e controls the transition between uniform (e=1)
    and patchy (e>1) saturation.

    References:
        Brie, A., Pampuri, F., Marsala, A.F., & Meazza, O. (1995). Shear
        sonic interpretation in gas-bearing sands. SPE Annual Technical
        Conference, Paper SPE-30595.

    Args:
        e: Brie exponent. Default: 3.0

    Example:
        >>> brie = Brie(e=3.0)
        >>> K_eff, rho_eff = brie(
        ...     K_w=2.25, K_g=0.02,
        ...     rho_w=1.0, rho_g=0.1,
        ...     Sw=0.8
        ... )
    """

    def __init__(self, e: float = 3.0):
        super().__init__()
        self.e = e

    def forward(
            self,
            K_w: Tensor,
            K_g: Tensor,
            rho_w: Tensor,
            rho_g: Tensor,
            Sw: Tensor,
            e: Tensor = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute effective fluid properties using Brie equation.

        Args:
            K_w: Water bulk modulus (GPa)
            K_g: Gas bulk modulus (GPa)
            rho_w: Water density (g/cm³)
            rho_g: Gas density (g/cm³)
            Sw: Water saturation (fraction, 0-1)
            e: Brie exponent (optional, uses self.e if None)

        Returns:
            Tuple of (K_eff, rho_eff)
        """
        K_w, K_g = as_tensor(K_w), as_tensor(K_g)
        rho_w, rho_g = as_tensor(rho_w), as_tensor(rho_g)
        Sw = as_tensor(Sw)
        e = as_tensor(e if e is not None else self.e)

        K_eff = K_g + (K_w - K_g) * Sw ** e
        rho_eff = Sw * rho_w + (1 - Sw) * rho_g

        return K_eff, rho_eff

    def compute_fluid_mix(self, fl1_K, fl2_K, fl1_rho, fl2_rho, Sw, **params):
        return self.forward(fl1_K, fl2_K, fl1_rho, fl2_rho, Sw)


@register(name="BatzleWang", category="fluid", aliases=["BW"])
class BatzleWang(FluidModel):
    """
    Batzle-Wang fluid properties as function of temperature and pressure.

    Empirical correlations for computing bulk modulus and density
    of reservoir fluids (brine, gas, oil) as functions of temperature,
    pressure, and fluid composition.

    References:
        Batzle, M. & Wang, Z. (1992). Seismic properties of pore fluids.
        Geophysics, 57(11), 1396-1408.

    Args:
        None (stateless model)

    Example:
        >>> bw = BatzleWang()
        >>> rho, K = bw.brine(T=80, P=30, S=0.035)  # 80°C, 30 MPa, 3.5% salinity
        >>> print(f"Brine: ρ={rho.item():.3f} g/cc, K={K.item():.2f} GPa")

        >>> rho, K = bw(T=80, P=30, fluid_type='gas', G=0.6)
    """

    def __init__(self):
        super().__init__()

    def brine(
            self,
            T: Tensor,
            P: Tensor,
            S: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute brine properties.

        Args:
            T: Temperature (°C)
            P: Pressure (MPa)
            S: Salinity (weight fraction, e.g., 0.035 for 35,000 ppm)

        Returns:
            Tuple of (density in g/cm³, bulk modulus in GPa)
        """
        T, P, S = as_tensor(T), as_tensor(P), as_tensor(S)

        # Water density
        rho_w = 1 + 1e-6 * (-80 * T - 3.3 * T ** 2 + 0.00175 * T ** 3 + 489 * P
                            - 2 * T * P + 0.016 * P * T ** 2 - 1.3e-5 * T ** 3 * P - 0.333 * P ** 2 - 0.002 * T * P ** 2)

        # Water velocity coefficients
        w = torch.tensor([[1402.85, 1.524, 3.437e-3, -1.197e-5],
                          [4.871, -0.0111, 1.739e-4, -1.628e-6],
                          [-0.04783, 2.747e-4, -2.135e-6, 1.237e-8],
                          [1.487e-4, -6.503e-7, -1.455e-8, 1.327e-10],
                          [-2.197e-7, 7.987e-10, 5.23e-11, -4.614e-13]])
        v_w = sum(w[i, j] * T ** i * P ** j for i in range(5) for j in range(4))

        # Brine corrections
        x = 300 * P - 2400 * P * S + T * (80 + 3 * T - 3300 * S - 13 * P + 47 * P * S)
        rho_b = rho_w + S * (0.668 + 0.44 * S + 1e-6 * x)
        s1 = 1170 - 9.6 * T + 0.055 * T ** 2 - 8.5e-5 * T ** 3 + 2.6 * P - 0.0029 * T * P - 0.0476 * P ** 2
        s15 = 780 - 10 * P + 0.16 * P ** 2
        v_b = v_w + s1 * S + s15 * S ** 1.5 - 820 * S ** 2

        return rho_b, rho_b * v_b ** 2 * 1e-6

    def gas(
            self,
            T: Tensor,
            P: Tensor,
            G: Tensor = 0.6
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute gas properties.

        Args:
            T: Temperature (°C)
            P: Pressure (MPa)
            G: Gas specific gravity (air = 1.0). Default: 0.6

        Returns:
            Tuple of (density in g/cm³, bulk modulus in GPa)
        """
        T, P, G = as_tensor(T), as_tensor(P), as_tensor(G)
        R = 8.3145
        Ta = T + 273.15

        # Pseudo-reduced properties
        P_pr = P / (4.892 - 0.4048 * G)
        T_pr = Ta / (94.72 + 170.75 * G)

        # Compressibility factor
        E = 0.109 * (3.85 - T_pr) ** 2 * torch.exp(-(0.45 + 8 * (0.56 - 1 / (T_pr + EPS)) ** 2) * P_pr ** 1.2 / (T_pr + EPS))
        Z = (0.03 + 0.00527 * (3.5 - T_pr) ** 3) * P_pr + (0.642 * T_pr - 0.007 * T_pr ** 4 - 0.52) + E

        # Density
        rho = 28.8 * G * P / (Z * R * Ta + EPS)

        # Bulk modulus
        dzdp = (0.03 + 0.00527 * (3.5 - T_pr) ** 3) + \
               0.109 * (3.85 - T_pr) ** 2 * 1.2 * P_pr ** 0.2 * \
               (-(0.45 + 8 * (0.56 - 1 / (T_pr + EPS)) ** 2) / (T_pr + EPS)) * \
               torch.exp(-(0.45 + 8 * (0.56 - 1 / (T_pr + EPS)) ** 2) * P_pr ** 1.2 / (T_pr + EPS))
        r0 = 0.85 + 5.6 / (P_pr + 2) + 27.1 / (P_pr + 3.5) ** 2 - 8.7 * torch.exp(-0.65 * (P_pr + 1))
        K = P / (1 - P_pr * dzdp / (Z + EPS) + EPS) * r0 / 1000

        return rho, K

    def oil(
            self,
            T: Tensor,
            P: Tensor,
            rho_0: Tensor = 0.8
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute oil properties.

        Args:
            T: Temperature (°C)
            P: Pressure (MPa)
            rho_0: Reference density at surface conditions (g/cm³). Default: 0.8

        Returns:
            Tuple of (density in g/cm³, bulk modulus in GPa)
        """
        T, P, rho_0 = as_tensor(T), as_tensor(P), as_tensor(rho_0)

        rho_p = rho_0 + (0.00277 * P - 1.71e-7 * P ** 3) * (rho_0 - 1.15) ** 2 + 3.49e-4 * P
        rho = rho_p / (0.972 + 3.81e-4 * (T + 17.78) ** 1.175)
        v = 2096 * (rho_0 / (2.6 - rho_0 + EPS)) ** 0.5 - 3.7 * T + 4.64 * P + \
            0.0115 * (4.12 * torch.clamp(1.08 / (rho_0 + EPS) - 1, min=0.0) ** 0.5 - 1) * T * P

        return rho, rho * v ** 2 / 1e6

    def forward(
            self,
            T: Tensor,
            P: Tensor,
            fluid_type: str = 'brine',
            **kw
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute fluid properties based on type.

        Args:
            T: Temperature (°C)
            P: Pressure (MPa)
            fluid_type: One of 'brine', 'gas', 'oil'
            **kw: Additional parameters:
                - S: Salinity for brine (default: 0.035)
                - G: Specific gravity for gas (default: 0.6)
                - rho_0: Reference density for oil (default: 0.8)

        Returns:
            Tuple of (density, bulk modulus)

        Raises:
            ValueError: If fluid_type is not recognized
        """
        if fluid_type == 'brine':
            return self.brine(T, P, kw.get('S', 0.035))
        elif fluid_type == 'gas':
            return self.gas(T, P, kw.get('G', 0.6))
        elif fluid_type == 'oil':
            return self.oil(T, P, kw.get('rho_0', 0.8))
        raise ValueError(f"Unknown fluid type: {fluid_type}. Use 'brine', 'gas', or 'oil'.")


# =============================================================================
# Additional Fluid Models
# =============================================================================

@register(name="BiotHF", category="fluid", tags=["dispersion"])
class BiotHF(FluidModel):
    """
    Biot high-frequency limiting velocities.

    Computes the fast P-wave, slow P-wave, and S-wave velocities
    in the high-frequency (unrelaxed) limit of Biot theory.

    References:
        Biot, M.A. (1956). Theory of propagation of elastic waves in a
        fluid-saturated porous solid. II. Higher frequency range. Journal
        of the Acoustical Society of America, 28(2), 179-191.

    Args:
        None (stateless model)
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "Biot high-frequency velocities")

    def forward(
            self,
            K_dry: Tensor,
            G_dry: Tensor,
            K0: Tensor,
            K_fl: Tensor,
            rho0: Tensor,
            rho_fl: Tensor,
            phi: Tensor,
            alpha: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute high-frequency Biot velocities.

        Args:
            K_dry: Dry frame bulk modulus (GPa)
            G_dry: Dry frame shear modulus (GPa)
            K0: Mineral bulk modulus (GPa)
            K_fl: Fluid bulk modulus (GPa)
            rho0: Grain density (g/cm3)
            rho_fl: Fluid density (g/cm3)
            phi: Porosity
            alpha: Tortuosity (>=1)

        Returns:
            Tuple of (Vp_fast, Vp_slow, Vs) in m/s
        """
        K_dry, G_dry = as_tensor(K_dry), as_tensor(G_dry)
        K0, K_fl = as_tensor(K0), as_tensor(K_fl)
        rho0, rho_fl = as_tensor(rho0), as_tensor(rho_fl)
        phi, alpha = as_tensor(phi), as_tensor(alpha)

        rho = (1.0 - phi) * rho0 + phi * rho_fl
        rho12 = (1.0 - alpha) * phi * rho_fl
        rho22 = alpha * phi * rho_fl
        rho11 = (1.0 - phi) * rho0 - (1.0 - alpha) * phi * rho_fl

        T1 = 1.0 - phi - K_dry / (K0 + EPS)
        T2 = phi * K0 / (K_fl + EPS)
        R = phi ** 2 * K0 / (T1 + T2 + EPS)
        Q = T1 * phi * K0 / (T1 + T2 + EPS)
        P = ((1.0 - phi) * T1 * K0 + T2 * K_dry) / (T1 + T2 + EPS) + 4.0 * G_dry / 3.0

        Delta = P * rho22 + R * rho11 - 2.0 * Q * rho12
        T3 = rho11 * rho22 - rho12 ** 2
        T4 = P * R - Q ** 2

        disc = torch.clamp(Delta ** 2 - 4.0 * T3 * T4, min=EPS)
        Vp_fast = torch.sqrt((Delta + torch.sqrt(disc)) / (2.0 * T3 + EPS)) * 1000.0
        Vp_slow = torch.sqrt(torch.clamp((Delta - torch.sqrt(disc)) / (2.0 * T3 + EPS), min=EPS)) * 1000.0
        Vs = torch.sqrt(G_dry / (rho - phi * rho_fl / (alpha + EPS) + EPS)) * 1000.0

        return Vp_fast, Vp_slow, Vs


@register(name="BiotDispersion", category="fluid", tags=["dispersion"])
class BiotDispersion(FluidModel):
    """
    Full frequency-dependent Biot dispersion model.

    Computes velocities and attenuation as a function of frequency
    using the full Biot poroelastic theory with viscodynamic corrections.

    References:
        Biot, M.A. (1956). Theory of propagation of elastic waves in a
        fluid-saturated porous solid. I. Low-frequency range. Journal of
        the Acoustical Society of America, 28(2), 168-178.
        Also: Biot, M.A. (1956). II. Higher frequency range. JASA, 28(2),
        179-191.

    Args:
        None (stateless model)
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "Biot dispersion (full frequency)")

    def forward(
            self,
            K_dry: Tensor,
            G_dry: Tensor,
            K0: Tensor,
            K_fl: Tensor,
            rho0: Tensor,
            rho_fl: Tensor,
            eta: Tensor,
            phi: Tensor,
            kapa: Tensor,
            a: Tensor,
            alpha: Tensor,
            freq: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Compute Biot dispersion and attenuation.

        Args:
            K_dry: Dry frame bulk modulus (GPa)
            G_dry: Dry frame shear modulus (GPa)
            K0: Mineral bulk modulus (GPa)
            K_fl: Fluid bulk modulus (GPa)
            rho0: Grain density (g/cm3)
            rho_fl: Fluid density (g/cm3)
            eta: Pore fluid viscosity (Pa*s)
            phi: Porosity
            kapa: Absolute permeability (m^2)
            a: Pore-size parameter (m)
            alpha: Tortuosity (>=1)
            freq: Frequency array (Hz)

        Returns:
            Tuple of (Vp_fast, Vp_slow, Vs, QP1_inv, QP2_inv, Qs_inv)
            Velocities in m/s, Q inverse (attenuation)
        """
        K_dry, G_dry = as_tensor(K_dry), as_tensor(G_dry)
        K0, K_fl = as_tensor(K0), as_tensor(K_fl)
        rho0, rho_fl = as_tensor(rho0), as_tensor(rho_fl)
        eta, phi = as_tensor(eta), as_tensor(phi)
        kapa, a_p = as_tensor(kapa), as_tensor(a)
        alpha, freq = as_tensor(alpha), as_tensor(freq)

        rho = (1.0 - phi) * rho0 + phi * rho_fl

        # Poroelastic coefficients
        D = K0 * (1.0 + phi * (K0 / (K_fl + EPS) - 1.0))
        M = K0 ** 2 / (D - K_dry + EPS)
        C_coeff = (K0 - K_dry) * K0 / (D - K_dry + EPS)
        H = K_dry + 4.0 * G_dry / 3.0 + (K0 - K_dry) ** 2 / (D - K_dry + EPS)

        w = 2.0 * PI * freq

        # Viscodynamic operator (simplified high/low freq behavior)
        zeta = torch.sqrt(w * a_p ** 2 * rho_fl / (eta + EPS))

        # F factor: low frequency F=1, high frequency F~zeta/4
        F = torch.where(zeta < 0.1, torch.ones_like(zeta), zeta / 4.0)

        q = alpha * rho_fl / (phi + EPS) - 1j * eta * F / (w * kapa + EPS)

        # Complex slowness calculation
        q_c = q.to(torch.complex64)
        rho_c = as_tensor(rho).to(torch.complex64)
        rho_fl_c = as_tensor(rho_fl).to(torch.complex64)
        H_c = as_tensor(H).to(torch.complex64)
        C_c = as_tensor(C_coeff).to(torch.complex64)
        M_c = as_tensor(M).to(torch.complex64)
        G_dry_c = as_tensor(G_dry).to(torch.complex64)

        Ta = C_c ** 2 - M_c * H_c
        Tb = H_c * q_c + M_c * rho_c - 2.0 * C_c * rho_fl_c
        Tc = rho_fl_c ** 2 - rho_c * q_c

        disc = Tb ** 2 - 4.0 * Ta * Tc
        P1_s2 = (-Tb + torch.sqrt(disc)) / (2.0 * Ta + EPS)
        P2_s2 = (-Tb - torch.sqrt(disc)) / (2.0 * Ta + EPS)
        S_s2 = (rho_c * q_c - rho_fl_c ** 2) / (G_dry_c * q_c + EPS)

        Vp_fast = 1.0 / torch.sqrt(P1_s2).real * 1000.0
        Vp_slow = 1.0 / torch.sqrt(P2_s2).real * 1000.0
        Vs = 1.0 / torch.sqrt(S_s2).real * 1000.0

        QP1_inv = (1.0 / P1_s2).imag / ((1.0 / P1_s2).real + EPS)
        QP2_inv = (1.0 / P2_s2).imag / ((1.0 / P2_s2).real + EPS)
        Qs_inv = (1.0 / S_s2).imag / ((1.0 / S_s2).real + EPS)

        return Vp_fast, Vp_slow, Vs, QP1_inv, QP2_inv, Qs_inv


@register(name="GeertsmaSmitHF", category="fluid", tags=["dispersion"])
class GeertsmaSmitHF(FluidModel):
    """
    Geertsma-Smit high-frequency approximation.

    Approximation of the Biot high-frequency limit for the fast
    P-wave velocity. Typically 3-6% too high compared to exact limit.

    References:
        Geertsma, J. & Smit, D.C. (1961). Some aspects of elastic wave
        propagation in fluid-saturated porous solids. Geophysics, 26(2),
        169-181.

    Args:
        None (stateless model)
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "Geertsma-Smit HF approximation")

    def forward(
            self,
            K_dry: Tensor,
            G_dry: Tensor,
            K0: Tensor,
            K_fl: Tensor,
            rho0: Tensor,
            rho_fl: Tensor,
            phi: Tensor,
            alpha: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute HF approximate velocities.

        Args:
            K_dry: Dry frame bulk modulus (GPa)
            G_dry: Dry frame shear modulus (GPa)
            K0: Mineral bulk modulus (GPa)
            K_fl: Fluid bulk modulus (GPa)
            rho0: Grain density (g/cm3)
            rho_fl: Fluid density (g/cm3)
            phi: Porosity
            alpha: Tortuosity (>=1)

        Returns:
            Tuple of (Vp_fast, Vs) in m/s
        """
        K_dry, G_dry = as_tensor(K_dry), as_tensor(G_dry)
        K0, K_fl = as_tensor(K0), as_tensor(K_fl)
        rho0, rho_fl = as_tensor(rho0), as_tensor(rho_fl)
        phi, alpha = as_tensor(phi), as_tensor(alpha)

        rho = (1.0 - phi) * rho0 + phi * rho_fl
        rho_biot = rho0 * (1.0 - phi) + phi * rho_fl * (1.0 - 1.0 / (alpha + EPS))
        Hdry = K_dry + 4.0 * G_dry / 3.0
        T1 = phi * rho / (rho_fl * alpha + EPS)
        alpha_biot = 1.0 - K_dry / (K0 + EPS)

        Vp_fast = torch.sqrt(1.0 / (rho_biot + EPS) * (
                Hdry + (T1 + alpha_biot * (alpha_biot - 2.0 * phi / (alpha + EPS))) /
                ((alpha_biot - phi) / (K0 + EPS) + phi / (K_fl + EPS) + EPS)
        )) * 1000.0

        Vs = torch.sqrt(G_dry / (rho_biot + EPS)) * 1000.0
        return Vp_fast, Vs


@register(name="GeertsmaSmitLF", category="fluid", tags=["dispersion"])
class GeertsmaSmitLF(FluidModel):
    """
    Geertsma-Smit low/middle-frequency approximation.

    Frequency-dependent P-wave velocity interpolation between
    Gassmann low-frequency and Biot high-frequency limits.

    References:
        Geertsma, J. & Smit, D.C. (1961). Some aspects of elastic wave
        propagation in fluid-saturated porous solids. Geophysics, 26(2),
        169-181.

    Args:
        None (stateless model)
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "Geertsma-Smit LF interpolation")

    def forward(
            self,
            Vp0: Tensor,
            Vpinf: Tensor,
            freq: Tensor,
            phi: Tensor,
            rho_fl: Tensor,
            kapa: Tensor,
            eta: Tensor
    ) -> Tensor:
        """
        Compute frequency-dependent P-wave velocity.

        Args:
            Vp0: Low-frequency (Gassmann) P-wave velocity (m/s or km/s)
            Vpinf: High-frequency P-wave velocity (same units)
            freq: Frequency (Hz)
            phi: Porosity
            rho_fl: Fluid density (g/cm3)
            kapa: Absolute permeability (m^2)
            eta: Fluid viscosity (Pa*s)

        Returns:
            Frequency-dependent P-wave velocity (same units as input)
        """
        Vp0, Vpinf = as_tensor(Vp0), as_tensor(Vpinf)
        freq, phi = as_tensor(freq), as_tensor(phi)
        rho_fl, kapa, eta = as_tensor(rho_fl), as_tensor(kapa), as_tensor(eta)

        fc = phi * eta / (2.0 * PI * rho_fl * kapa + EPS)
        a_coeff = (fc / (freq + EPS)) ** 2
        Vp = torch.sqrt((Vpinf ** 4 + Vp0 ** 4 * a_coeff) / (Vpinf ** 2 + Vp0 ** 2 * a_coeff + EPS))
        return Vp


@register(name="BrownKorringaDry2Sat", category="fluid", tags=["anisotropic"])
class BrownKorringaDry2Sat(FluidModel):
    """
    Brown-Korringa dry-to-saturated fluid substitution.

    Computes saturated compliance from dry compliance for
    arbitrarily anisotropic rock using Brown & Korringa (1975).

    References:
        Brown, R.J.S. & Korringa, J. (1975). On the dependence of the
        elastic properties of a porous rock on the compressibility of the
        pore fluid. Geophysics, 40(4), 608-616.

    Args:
        None (stateless model)
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "Brown-Korringa dry→sat")

    def forward(
            self,
            Sdry: Tensor,
            K0: Tensor,
            G0: Tensor,
            K_fl: Tensor,
            phi: Tensor
    ) -> Tensor:
        """
        Compute saturated compliance from dry compliance.

        Args:
            Sdry: Dry compliance matrix (6x6)
            K0: Mineral bulk modulus (GPa)
            G0: Mineral shear modulus (GPa)
            K_fl: Fluid bulk modulus (GPa)
            phi: Porosity

        Returns:
            Saturated compliance matrix (6x6)
        """
        Sdry = as_tensor(Sdry)
        K0, G0 = as_tensor(K0), as_tensor(G0)
        K_fl, phi = as_tensor(K_fl), as_tensor(phi)

        beta0 = 1.0 / K0
        betadry = Sdry[0:3, 0:3].sum()
        betafl = 1.0 / K_fl

        S0 = torch.linalg.inv(write_iso_matrix(K0, G0))

        Sprime = Sdry[0:3, :].sum(dim=0) - S0[0:3, :].sum(dim=0)
        Sprime = Sprime.unsqueeze(0)

        denom = (betafl - beta0) * phi + (betadry - beta0)
        Ssat = Sdry - Sprime.T @ Sprime / (denom + EPS)
        return Ssat


@register(name="BrownKorringaSat2Dry", category="fluid", tags=["anisotropic"])
class BrownKorringaSat2Dry(FluidModel):
    """
    Brown-Korringa saturated-to-dry inverse fluid substitution.

    Computes dry compliance from saturated compliance for
    arbitrarily anisotropic rock.

    References:
        Brown, R.J.S. & Korringa, J. (1975). On the dependence of the
        elastic properties of a porous rock on the compressibility of the
        pore fluid. Geophysics, 40(4), 608-616.

    Args:
        None (stateless model)
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "Brown-Korringa sat→dry")

    def forward(
            self,
            Ssat: Tensor,
            K0: Tensor,
            G0: Tensor,
            K_fl: Tensor,
            phi: Tensor
    ) -> Tensor:
        """
        Compute dry compliance from saturated compliance.

        Args:
            Ssat: Saturated compliance matrix (6x6)
            K0: Mineral bulk modulus (GPa)
            G0: Mineral shear modulus (GPa)
            K_fl: Fluid bulk modulus (GPa)
            phi: Porosity

        Returns:
            Dry compliance matrix (6x6)
        """
        Ssat = as_tensor(Ssat)
        K0, G0 = as_tensor(K0), as_tensor(G0)
        K_fl, phi = as_tensor(K_fl), as_tensor(phi)

        beta0 = 1.0 / K0
        betasat = Ssat[0:3, 0:3].sum()
        betafl = 1.0 / K_fl

        S0 = torch.linalg.inv(write_iso_matrix(K0, G0))

        Sprime = Ssat[0:3, :].sum(dim=0) - S0[0:3, :].sum(dim=0)
        Sprime = Sprime.unsqueeze(0)

        denom = (betafl - beta0) * phi - (betasat - beta0)
        Sdry = Ssat + Sprime.T @ Sprime / (denom + EPS)
        return Sdry


@register(name="BrownKorringaSub", category="fluid", tags=["anisotropic"])
class BrownKorringaSub(FluidModel):
    """
    Brown-Korringa anisotropic fluid substitution.

    Full fluid substitution for arbitrarily anisotropic rock:
    fluid 1 → fluid 2 via dry intermediate.

    References:
        Brown, R.J.S. & Korringa, J. (1975). On the dependence of the
        elastic properties of a porous rock on the compressibility of the
        pore fluid. Geophysics, 40(4), 608-616.

    Args:
        None (stateless model)
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "Brown-Korringa fluid substitution")

    def forward(
            self,
            Csat: Tensor,
            K0: Tensor,
            G0: Tensor,
            K_fl1: Tensor,
            K_fl2: Tensor,
            phi: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Substitute fluid in anisotropic rock.

        Args:
            Csat: Original saturated stiffness matrix (6x6)
            K0: Mineral bulk modulus (GPa)
            G0: Mineral shear modulus (GPa)
            K_fl1: Original fluid bulk modulus (GPa)
            K_fl2: New fluid bulk modulus (GPa)
            phi: Porosity

        Returns:
            Tuple of (Csat2, Ssat2): new stiffness and compliance (6x6)
        """
        Csat = as_tensor(Csat)
        K0, G0 = as_tensor(K0), as_tensor(G0)
        K_fl1, K_fl2, phi = as_tensor(K_fl1), as_tensor(K_fl2), as_tensor(phi)

        Ssat = torch.linalg.inv(Csat)

        # sat → dry
        sat2dry = BrownKorringaSat2Dry()
        Sdry = sat2dry(Ssat, K0, G0, K_fl1, phi)

        # dry → new sat
        dry2sat = BrownKorringaDry2Sat()
        Ssat2 = dry2sat(Sdry, K0, G0, K_fl2, phi)

        Csat2 = torch.linalg.inv(Ssat2)
        return Csat2, Ssat2


@register(name="MavkoJizba", category="fluid", tags=["squirt"])
class MavkoJizba(FluidModel):
    """
    Mavko-Jizba squirt flow model.

    Predicts high-frequency saturated moduli from dry rock
    properties measured at different pressures.

    References:
        Mavko, G. & Jizba, D. (1991). Estimating grain-scale fluid effects
        on velocity dispersion in rocks. Geophysics, 56(12), 1940-1949.

    Args:
        None (stateless model)
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "Mavko-Jizba squirt flow")

    def forward(
            self,
            Vp_hs: Tensor,
            Vs_hs: Tensor,
            Vpdry: Tensor,
            Vsdry: Tensor,
            K0: Tensor,
            rhodry: Tensor,
            rho_fl: Tensor,
            K_fl: Tensor,
            phi: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Compute high-frequency saturated velocities.

        Args:
            Vp_hs: Dry Vp at high pressure (m/s)
            Vs_hs: Dry Vs at high pressure (m/s)
            Vpdry: Dry Vp at different pressures (m/s)
            Vsdry: Dry Vs at different pressures (m/s)
            K0: Mineral bulk modulus (GPa)
            rhodry: Dry rock bulk density (g/cm3)
            rho_fl: Fluid density (g/cm3)
            K_fl: Fluid bulk modulus (GPa)
            phi: Porosity

        Returns:
            Tuple of (Kuf_sat, Guf_sat, Vp_hf, Vs_hf)
            Moduli in GPa, velocities in m/s
        """
        Vp_hs, Vs_hs = as_tensor(Vp_hs), as_tensor(Vs_hs)
        Vpdry, Vsdry = as_tensor(Vpdry), as_tensor(Vsdry)
        K0 = as_tensor(K0)
        rhodry, rho_fl = as_tensor(rhodry), as_tensor(rho_fl)
        K_fl, phi = as_tensor(K_fl), as_tensor(phi)

        # Convert m/s to km/s
        vp_km = Vpdry / 1000.0
        vs_km = Vsdry / 1000.0
        K_dry = rhodry * vp_km ** 2 - 4.0 / 3.0 * rhodry * vs_km ** 2
        G_dry = rhodry * vs_km ** 2

        vp_hs_km = Vp_hs / 1000.0
        vs_hs_km = Vs_hs / 1000.0
        Khs = rhodry * vp_hs_km ** 2 - 4.0 / 3.0 * rhodry * vs_hs_km ** 2

        # High-frequency wet-frame modulus
        Kuf = Khs

        # Gassmann for high-frequency saturated bulk modulus
        A = (1.0 - Kuf / (K0 + EPS)) ** 2
        B = phi / (K_fl + EPS) + (1.0 - phi) / (K0 + EPS) - Kuf / (K0 ** 2 + EPS)
        Kuf_sat = Kuf + A / (B + EPS)

        # High-frequency saturated shear modulus
        Guf_sat_inv = 1.0 / (G_dry + EPS) - 4.0 / 15.0 * (1.0 / (K_dry + EPS) - 1.0 / (Kuf + EPS))
        Guf_sat = 1.0 / (Guf_sat_inv + EPS)

        # Predicted velocities
        rho_sat = rhodry + phi * rho_fl
        Vp_hf = torch.sqrt((Kuf_sat + 4.0 / 3.0 * Guf_sat) / (rho_sat + EPS)) * 1000.0
        Vs_hf = torch.sqrt(Guf_sat / (rho_sat + EPS)) * 1000.0

        return Kuf_sat, Guf_sat, Vp_hf, Vs_hf


@register(name="CO2Properties", category="fluid", tags=["co2"])
class CO2Properties(FluidModel):
    """
    CO2 density and bulk modulus using modified Batzle-Wang.

    Computes supercritical CO2 properties as a function of
    temperature and pressure using Xu (2006) modifications.

    References:
        Span, R. & Wagner, W. (1996). A new equation of state for carbon
        dioxide covering the fluid region from the triple-point temperature
        to 1100 K at pressures up to 800 MPa. Journal of Physical and
        Chemical Reference Data, 25(6), 1509-1596.

    Args:
        None (stateless model)
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "CO2 properties (Batzle-Wang)")

    def forward(
            self,
            P: Tensor,
            T: Tensor,
            G: Tensor = 1.5349
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute CO2 density and bulk modulus.

        Args:
            P: Pressure (MPa)
            T: Temperature (deg C)
            G: Gas gravity (default: 1.5349 for CO2)

        Returns:
            Tuple of (rho in g/cm3, K in GPa)
        """
        P, T = as_tensor(P), as_tensor(T)
        G = as_tensor(G)
        R = 8.3145

        Ta = T + 273.15
        P_pr = P / 7.4
        T_pr = Ta / (31.1 + 273.5)

        E = 0.109 * (3.85 - T_pr) ** 2 * torch.exp(
            -(0.45 + 8.0 * (0.56 - 1.0 / T_pr) ** 2) * P_pr ** 1.2 / T_pr)
        Z = (0.03 + 0.00527 * (3.5 - T_pr) ** 3) * P_pr + \
            (0.642 * T_pr - 0.007 * T_pr ** 4 - 0.52) + E
        rho = 28.8 * G * P / (Z * R * Ta + EPS)

        r_0 = 0.85 + 5.6 / (P_pr + 2) + 27.1 / (P_pr + 3.5) ** 2 - \
              8.7 * torch.exp(-0.65 * (P_pr + 1))
        dzdp = (0.03 + 0.00527 * (3.5 - T_pr) ** 3) + \
               0.109 * (3.85 - T_pr) ** 2 * 1.2 * P_pr ** 0.2 * \
               (-(0.45 + 8.0 * (0.56 - 1.0 / T_pr) ** 2) / T_pr) * \
               torch.exp(-(0.45 + 8.0 * (0.56 - 1.0 / T_pr) ** 2) * P_pr ** 1.2 / T_pr)
        K = P / (1.0 - P_pr * dzdp / (Z + EPS) + EPS) * r_0 / 1000.0

        return rho, K


@register(name="LiveOil", category="fluid", tags=["oil"])
class LiveOil(FluidModel):
    """
    Live oil (gas-saturated oil) properties.

    Computes density and bulk modulus of oil with dissolved gas
    using Batzle-Wang relations.

    References:
        Batzle, M. & Wang, Z. (1992). Seismic properties of pore fluids.
        Geophysics, 57(11), 1396-1408.

    Args:
        None (stateless model)
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "Live oil properties")

    def forward(
            self,
            P: Tensor,
            T: Tensor,
            den: Tensor,
            G: Tensor,
            Rg: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute live oil density and bulk modulus.

        Args:
            P: Pressure (MPa)
            T: Temperature (deg C)
            den: Oil density at surface (g/cm3)
            G: Gas gravity
            Rg: Gas-oil ratio (L/L). If None, computed from P,T,G,den.

        Returns:
            Tuple of (rho_g in g/cm3, K in GPa)
        """
        P, T = as_tensor(P), as_tensor(T)
        den, G = as_tensor(den), as_tensor(G)

        if Rg is None:
            Rg = 0.02123 * G * (P * torch.exp(4.072 / den - 0.00377 * T)) ** 1.205
        else:
            Rg = as_tensor(Rg)

        B = 0.972 + 0.00038 * (2.4 * Rg * (G / den) ** 0.5 + T + 17.8) ** 1.175
        rho_p = den * (1.0 + 0.001 * Rg) ** (-1) * B ** (-1)
        v = 2096.0 * (rho_p / (2.6 - rho_p + EPS)) ** 0.5 - 3.7 * T + 4.64 * P + \
            0.0115 * (4.12 * (1.08 / (rho_p + EPS) - 1.0) ** 0.5 - 1.0) * T * P

        rho_g = (den + 0.0012 * G * Rg) / (B + EPS)
        K = rho_g * v ** 2 / 1e6

        return rho_g, K


@register(name="CO2Brine", category="fluid", tags=["co2", "brine"])
class CO2Brine(FluidModel):
    """
    CO2-brine mixture properties.

    Computes effective density and bulk modulus of CO2-brine
    mixture using either Woods equation (uniform saturation)
    or Brie mixing (patchy saturation).

    References:
        Batzle, M. & Wang, Z. (1992). Seismic properties of pore fluids.
        Geophysics, 57(11), 1396-1408.

    Args:
        None (stateless model)
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "CO2-brine mixture properties")

    def forward(
            self,
            T: Tensor,
            P: Tensor,
            salinity: Tensor,
            Sco2: Tensor,
            brie_e: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute CO2-brine mixture properties.

        Args:
            T: Temperature (deg C)
            P: Pressure (MPa)
            salinity: NaCl weight fraction (e.g. 0.035)
            Sco2: CO2 saturation (fraction)
            brie_e: Brie exponent. If None, use Woods (uniform sat).

        Returns:
            Tuple of (den_mix in g/cm3, Kf_mix in GPa)
        """
        T, P = as_tensor(T), as_tensor(P)
        salinity, Sco2 = as_tensor(salinity), as_tensor(Sco2)

        # CO2 properties
        co2 = CO2Properties()
        rho_co2, K_co2 = co2(P, T)

        # Brine properties (using BatzleWang class)
        bw = BatzleWang()
        rho_brine, K_brine = bw.brine(T, P, salinity)

        # Mixture density
        den_mix = (1.0 - Sco2) * rho_brine + Sco2 * rho_co2

        if brie_e is None:
            # Woods equation (uniform saturation, Reuss average)
            Kf_mix = ((1.0 - Sco2) / (K_brine + EPS) + Sco2 / (K_co2 + EPS)) ** (-1)
        else:
            # Brie mixing (patchy saturation)
            brie_e = as_tensor(brie_e)
            Kf_mix = (K_brine - K_co2) * (1.0 - Sco2) ** brie_e + K_co2

        return den_mix, Kf_mix