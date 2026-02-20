#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Empirical models for rock physics.

This module provides empirical velocity-porosity, velocity-depth,
and density-velocity relationships commonly used in rock physics
and petrophysical analysis.

Categories:
    - Velocity-porosity relations (Han, Raymer, Wyllie)
    - Vp-Vs relations (Castagna, Greenberg-Castagna)
    - Density-velocity relations (Gardner)
    - Compaction trends (Storvoll, Japsen, Hillis)
    - Porosity-depth trends (Ehrenberg, Ramm, Sclater)

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
from torch import Tensor
from typing import Tuple

from ..core import EmpiricalModel, register, as_tensor, ensure_same_device, EPS


@register(name="Han", category="empirical")
class Han(EmpiricalModel):
    """
    Han velocity-porosity-clay relation for shaly sandstones.

    Empirical relations from ultrasonic measurements on water-saturated
    sandstones at 40 MPa effective pressure:

        Vp = 5.59 - 6.93φ - 2.81C  (km/s)
        Vs = 3.52 - 4.91φ - 1.89C  (km/s)

    References:
        Han, D., Nur, A., & Morgan, D. (1986). Effects of porosity and clay
        content on wave velocities in sandstones. Geophysics, 51(11), 2093-2107.

    Args:
        None (stateless model)

    Example:
        >>> han = Han()
        >>> Vp, Vs = han(phi=0.25, C=0.1)  # 25% porosity, 10% clay
        >>> print(f"Vp = {Vp.item():.0f} m/s")
    """

    def __init__(self):
        super().__init__()

    def forward(self, phi: Tensor, C: Tensor = 0.0) -> Tuple[Tensor, Tensor]:
        """
        Compute velocities from porosity and clay content.

        Args:
            phi: Porosity (fraction, 0-1)
            C: Clay volume fraction (fraction, 0-1). Default: 0.0

        Returns:
            Tuple of (Vp, Vs) in m/s
        """
        phi, C = as_tensor(phi), as_tensor(C)
        Vp = (5.59 - 6.93 * phi - 2.81 * C) * 1000
        Vs = (3.52 - 4.91 * phi - 1.89 * C) * 1000
        return Vp, Vs


@register(name="RaymerHuntGardner", category="empirical", aliases=["RHG"])
class RaymerHuntGardner(EmpiricalModel):
    """
    Raymer-Hunt-Gardner time average equation.

    Modified time-average equation for water-saturated rocks:

        Vp = (1 - φ)² * Vp_mat + φ * Vp_fl

    References:
        Raymer, L.L., Hunt, E.R., & Gardner, J.S. (1980). An improved sonic
        transit time-to-porosity transform. SPWLA 21st Annual Logging Symposium.

    Args:
        None (stateless model)

    Example:
        >>> rhg = RaymerHuntGardner()
        >>> Vp = rhg(phi=0.25, Vp_mat=5500, Vp_fl=1500)
    """

    def __init__(self):
        super().__init__()

    def forward(
            self,
            phi: Tensor,
            Vp_mat: Tensor = 5500.0,
            Vp_fl: Tensor = 1500.0
    ) -> Tensor:
        """
        Compute P-wave velocity.

        Args:
            phi: Porosity (fraction)
            Vp_mat: Matrix P-wave velocity (m/s). Default: 5500
            Vp_fl: Fluid P-wave velocity (m/s). Default: 1500

        Returns:
            P-wave velocity (m/s)
        """
        phi, Vp_mat, Vp_fl = as_tensor(phi), as_tensor(Vp_mat), as_tensor(Vp_fl)
        return (1 - phi) ** 2 * Vp_mat + phi * Vp_fl


@register(name="WyllieTimeAverage", category="empirical", aliases=["Wyllie"])
class WyllieTimeAverage(EmpiricalModel):
    """
    Wyllie time average equation.

    Classic time-average for sonic log interpretation:

        1/Vp = φ/Vp_fl + (1-φ)/Vp_mat

    References:
        Wyllie, M.R.J., Gregory, A.R., & Gardner, L.W. (1956). Elastic wave
        velocities in heterogeneous and porous media. Geophysics, 21(1), 41-70.

    Args:
        None (stateless model)

    Example:
        >>> wyllie = WyllieTimeAverage()
        >>> Vp = wyllie(phi=0.25)
    """

    def __init__(self):
        super().__init__()

    def forward(
            self,
            phi: Tensor,
            Vp_mat: Tensor = 5500.0,
            Vp_fl: Tensor = 1500.0
    ) -> Tensor:
        """
        Compute P-wave velocity using time average.

        Args:
            phi: Porosity (fraction)
            Vp_mat: Matrix velocity (m/s). Default: 5500
            Vp_fl: Fluid velocity (m/s). Default: 1500

        Returns:
            P-wave velocity (m/s)
        """
        phi, Vp_mat, Vp_fl = as_tensor(phi), as_tensor(Vp_mat), as_tensor(Vp_fl)
        return 1.0 / (phi / Vp_fl + (1 - phi) / Vp_mat)


@register(name="CastagnaMudrock", category="empirical", aliases=["Mudrock"])
class CastagnaMudrock(EmpiricalModel):
    """
    Castagna mudrock line for Vp-Vs relationship.

    Empirical relation for water-saturated mudrocks:

        Vs = 0.862 * Vp - 1.172  (km/s)

    References:
        Castagna, J.P., Batzle, M.L., & Eastwood, R.L. (1985). Relationships
        between compressional-wave and shear-wave velocities in clastic silicate
        rocks. Geophysics, 50(4), 571-581.

    Args:
        None (stateless model)

    Example:
        >>> mudrock = CastagnaMudrock()
        >>> Vs = mudrock(Vp=3500)  # Vp in m/s
    """

    def __init__(self):
        super().__init__()

    def forward(self, Vp: Tensor) -> Tensor:
        """
        Compute Vs from Vp using mudrock line.

        Args:
            Vp: P-wave velocity (m/s)

        Returns:
            S-wave velocity (m/s), clamped to non-negative
        """
        Vp = as_tensor(Vp) / 1000  # Convert to km/s
        Vs = 0.862 * Vp - 1.172
        return torch.clamp(Vs, min=0) * 1000


@register(name="GreenbergCastagna", category="empirical", aliases=["GC"])
class GreenbergCastagna(EmpiricalModel):
    """
    Greenberg-Castagna Vp-Vs relation with shale correction.

    Lithology-dependent Vp-Vs relations weighted by shale volume:

        Vs_sand = 0.80416 * Vp - 0.85588
        Vs_shale = 0.76969 * Vp - 0.86735
        Vs = (1 - Vsh) * Vs_sand + Vsh * Vs_shale

    References:
        Greenberg, M.L. & Castagna, J.P. (1992). Shear-wave velocity estimation
        in porous rocks. Geophysics, 57(6), 733-744.

    Args:
        None (stateless model)

    Example:
        >>> gc = GreenbergCastagna()
        >>> Vs = gc(Vp=3500, vsh=0.3)  # 30% shale
    """

    def __init__(self):
        super().__init__()

    def forward(self, Vp: Tensor, vsh: Tensor = 0.0) -> Tensor:
        """
        Compute Vs from Vp with shale volume correction.

        Args:
            Vp: P-wave velocity (m/s)
            vsh: Shale volume fraction (0-1). Default: 0.0

        Returns:
            S-wave velocity (m/s)
        """
        Vp, vsh = as_tensor(Vp) / 1000, as_tensor(vsh)
        Vs_sand = 0.80416 * Vp - 0.85588
        Vs_shale = 0.76969 * Vp - 0.86735
        Vs = (1 - vsh) * Vs_sand + vsh * Vs_shale
        return torch.clamp(Vs, min=0) * 1000


@register(name="Gardner", category="empirical")
class Gardner(EmpiricalModel):
    """
    Gardner density-velocity relation.

    Empirical relation between density and P-wave velocity:

        ρ = a * Vp^b

    Default coefficients (a=0.31, b=0.25) are for Vp in km/s and ρ in g/cm³.

    References:
        Gardner, G.H.F., Gardner, L.W., & Gregory, A.R. (1974). Formation
        velocity and density — the diagnostic basics for stratigraphic traps.
        Geophysics, 39(6), 770-780.

    Args:
        a: Coefficient. Default: 0.31
        b: Exponent. Default: 0.25

    Example:
        >>> gardner = Gardner()
        >>> rho = gardner(Vp=3500)  # Returns density in g/cm³

        Inverse (velocity from density):
        >>> Vp = gardner.inverse(rho=2.4)
    """

    def __init__(self, a: float = 0.31, b: float = 0.25):
        super().__init__()
        self.a, self.b = a, b

    def forward(self, Vp: Tensor) -> Tensor:
        """
        Compute density from P-wave velocity.

        Args:
            Vp: P-wave velocity (m/s)

        Returns:
            Density (g/cm³)
        """
        Vp = as_tensor(Vp) / 1000
        return self.a * Vp ** self.b

    def inverse(self, rho: Tensor) -> Tensor:
        """
        Compute P-wave velocity from density.

        Args:
            rho: Density (g/cm³)

        Returns:
            P-wave velocity (m/s)
        """
        rho = as_tensor(rho)
        return (rho / self.a) ** (1 / self.b) * 1000


@register(name="Krief", category="empirical")
class Krief(EmpiricalModel):
    """
    Krief porosity-modulus relation.

    Empirical relation for dry rock moduli:

        K_dry = K0 * (1 - a0)
        G_dry = G0 * (1 - a0)

    where a0 = 1 - (1 - φ)^(3/(1-φ))

    References:
        Krief, M., Garat, J., Stellingwerff, J., & Ventre, J. (1990). A
        petrophysical interpretation using the velocities of P and S waves.
        The Log Analyst, 31(6), 355-369.

    Args:
        None (stateless model)

    Example:
        >>> krief = Krief()
        >>> K_dry, G_dry = krief(phi=0.25, K0=36.6, G0=45.0)
    """

    def __init__(self):
        super().__init__()

    def forward(
            self,
            phi: Tensor,
            K0: Tensor,
            G0: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute dry rock moduli from porosity.

        Args:
            phi: Porosity (fraction)
            K0: Matrix bulk modulus (GPa)
            G0: Matrix shear modulus (GPa)

        Returns:
            Tuple of (K_dry, G_dry) in GPa
        """
        phi, K0, G0 = as_tensor(phi), as_tensor(K0), as_tensor(G0)
        phi_safe = torch.clamp(phi, max=0.99)
        exp = 3 / (1 - phi_safe + EPS)
        a0 = 1 - (1 - phi_safe) ** exp
        return K0 * (1 - a0), G0 * (1 - a0)


@register(name="Storvoll", category="empirical", tags=["compaction"])
class Storvoll(EmpiricalModel):
    """
    Storvoll velocity compaction trend.

    Linear velocity-depth trend for normally compacted sediments:

        Vp = Z/1.76 + 2600  (m/s)

    References:
        Storvoll, V., Bjorlykke, K., & Mondol, N.H. (2005). Velocity-depth
        trends in Mesozoic and Cenozoic sediments from the Norwegian Shelf.
        AAPG Bulletin, 89(3), 359-381.

    Args:
        None (stateless model)

    Example:
        >>> storvoll = Storvoll()
        >>> Vp = storvoll(Z=2000)  # Depth in meters
    """

    def __init__(self):
        super().__init__()

    def forward(self, Z: Tensor) -> Tensor:
        """
        Compute P-wave velocity from depth.

        Args:
            Z: Depth below seafloor (m)

        Returns:
            P-wave velocity (m/s)
        """
        return as_tensor(Z) / 1.76 + 2600


@register(name="Japsen", category="empirical", tags=["compaction"])
class Japsen(EmpiricalModel):
    """
    Japsen segmented velocity-depth trend.

    Piecewise linear velocity-depth relation with depth-dependent
    gradients for different compaction regimes.

    References:
        Japsen, P. (1998). Regional velocity-depth anomalies, North Sea chalk:
        A record of overpressure and Neogene uplift and erosion. AAPG Bulletin,
        82(11), 2031-2074.

    Args:
        None (stateless model)

    Example:
        >>> japsen = Japsen()
        >>> Vp = japsen(Z=2500)
    """

    def __init__(self):
        super().__init__()

    def forward(self, Z: Tensor) -> Tensor:
        """
        Compute P-wave velocity using segmented trend.

        Args:
            Z: Depth (m)

        Returns:
            P-wave velocity (m/s)
        """
        Z = as_tensor(Z)
        V = torch.zeros_like(Z)

        # Segment 1: Z < 1393 m
        V[Z < 1393] = 1550 + 0.6 * Z[Z < 1393]

        # Segment 2: 1393 <= Z < 2000 m
        mask2 = (Z >= 1393) & (Z < 2000)
        V[mask2] = -400 + 2 * Z[mask2]

        # Segment 3: 2000 <= Z < 3500 m
        mask3 = (Z >= 2000) & (Z < 3500)
        V[mask3] = 2600 + 0.5 * Z[mask3]

        # Segment 4: Z >= 3500 m
        V[Z >= 3500] = 3475 + 0.25 * Z[Z >= 3500]

        return V


@register(name="Hillis", category="empirical", tags=["compaction"])
class Hillis(EmpiricalModel):
    """
    Hillis compaction trend.

    Velocity from transit time compaction trend.

    References:
        Hillis, R.R. (1995). Quantification of Tertiary exhumation in the
        United Kingdom southern North Sea using sonic velocity data. AAPG
        Bulletin, 79(1), 130-152.

    Args:
        None (stateless model)

    Example:
        >>> hillis = Hillis()
        >>> Vp = hillis(Z=2000)
    """

    def __init__(self):
        super().__init__()

    def forward(self, Z: Tensor) -> Tensor:
        """
        Compute P-wave velocity from depth.

        Args:
            Z: Depth (m)

        Returns:
            P-wave velocity (m/s)
        """
        Z = as_tensor(Z)
        transit = 135.9 - 20.22 * Z / 1000
        return 304.8 * 1000 / transit


@register(name="Scherbaum", category="empirical", tags=["compaction"])
class Scherbaum(EmpiricalModel):
    """
    Scherbaum velocity-depth trend.

    Linear velocity-depth relation:

        Vp = 2325 + 0.51*Z

    References:
        Scherbaum, F. (1982). Seismic velocities in sedimentary rocks.
        Geophysical Prospecting, 30(5), 581-591.

    Args:
        None (stateless model)
    """

    def __init__(self):
        super().__init__()

    def forward(self, Z: Tensor) -> Tensor:
        """
        Compute P-wave velocity from depth.

        Args:
            Z: Depth (m)

        Returns:
            P-wave velocity (m/s)
        """
        return 2325 + 0.51 * as_tensor(Z)


@register(name="Ehrenberg", category="empirical", tags=["porosity-depth"])
class Ehrenberg(EmpiricalModel):
    """
    Ehrenberg porosity trend.

    Linear porosity-depth relation:

        φ = -0.0922*Z + 0.48  (Z in km)

    References:
        Ehrenberg, S.N. (1990). Relationship between diagenesis and reservoir
        quality in sandstones of the Garn Formation, Haltenbanken, mid-Norwegian
        continental shelf. AAPG Bulletin, 74(10), 1538-1558.

    Args:
        None (stateless model)

    Example:
        >>> ehrenberg = Ehrenberg()
        >>> phi = ehrenberg(Z=2000)  # Returns porosity at 2 km depth
    """

    def __init__(self):
        super().__init__()

    def forward(self, Z: Tensor) -> Tensor:
        """
        Compute porosity from depth.

        Args:
            Z: Depth (m)

        Returns:
            Porosity (fraction)
        """
        Z = as_tensor(Z) / 1000
        return -0.0922 * Z + 0.48


@register(name="RammPorosity", category="empirical", tags=["porosity-depth"])
class RammPorosity(EmpiricalModel):
    """
    Ramm & Bjørlykke porosity trend.

    Regional porosity-depth relations for Norwegian shelf sandstones.

    References:
        Ramm, M. & Bjørlykke, K. (1994). Porosity/depth trends in reservoir
        sandstones: Assessing the quantitative effects of varying pore-pressure,
        temperature history and mineralogy, Norwegian Shelf data. Clay Minerals,
        29(4), 475-490.

    Args:
        region: 'haltenbanken' or 'northern'. Default: 'haltenbanken'

    Example:
        >>> ramm = RammPorosity(region='haltenbanken')
        >>> phi = ramm(Z=3000)
    """

    def __init__(self, region: str = 'haltenbanken'):
        super().__init__()
        self.region = region

    def forward(self, Z: Tensor) -> Tensor:
        """
        Compute porosity from depth.

        Args:
            Z: Depth (m)

        Returns:
            Porosity (fraction)
        """
        Z = as_tensor(Z)
        if self.region == 'haltenbanken':
            return (46.4 - 0.0085 * Z) / 100
        return (42.7 - 0.0069 * Z) / 100


@register(name="Sclater", category="empirical", tags=["porosity-depth"])
class Sclater(EmpiricalModel):
    """
    Sclater-Christie exponential porosity-depth relation.

    Exponential compaction model:

        φ = φ0 * exp(-Z/c)

    where φ0 = 0.49 and c = 3.7 km for shales.

    References:
        Sclater, J.G. & Christie, P.A.F. (1980). Continental stretching: An
        explanation of the post-mid-Cretaceous subsidence of the central North
        Sea Basin. Journal of Geophysical Research, 85(B7), 3711-3739.

    Args:
        None (stateless model)

    Example:
        >>> sclater = Sclater()
        >>> Z = sclater(phi=0.25)     # Depth from porosity
        >>> phi = sclater.inverse(Z=2.0)  # Porosity from depth (km)
    """

    def __init__(self):
        super().__init__()

    def forward(self, phi: Tensor) -> Tensor:
        """
        Compute depth from porosity (forward model).

        Args:
            phi: Porosity (fraction)

        Returns:
            Depth (km)
        """
        phi = as_tensor(phi)
        return 3.7 * torch.log(0.49 / (phi + EPS))

    def inverse(self, Z: Tensor) -> Tensor:
        """
        Compute porosity from depth (inverse model).

        Args:
            Z: Depth (km)

        Returns:
            Porosity (fraction)
        """
        return 0.49 * torch.exp(-as_tensor(Z) / 3.7)


@register(name="Hjelstuen", category="empirical", tags=["compaction"])
class Hjelstuen(EmpiricalModel):
    """
    Hjelstuen velocity-depth relation.

    Linear velocity-depth trend:

        Vp = (1.87 + 0.55*Z) * 1000  (Z in km)

    References:
        Hjelstuen, B.O., Eldholm, O., & Faleide, J.I. (2007). Recurrent
        Pleistocene mega-failures on the SW Barents Sea margin. Earth and
        Planetary Science Letters, 258(3-4), 605-618.

    Args:
        None (stateless model)
    """

    def __init__(self):
        super().__init__()

    def forward(self, Z: Tensor) -> Tensor:
        """
        Compute P-wave velocity from depth.

        Args:
            Z: Depth (m)

        Returns:
            P-wave velocity (m/s)
        """
        Z = as_tensor(Z)
        return (1.87 + 0.55 * Z / 1000) * 1000


@register(name="StPeter", category="empirical", tags=["pressure-velocity"])
class StPeter(EmpiricalModel):
    """
    St. Peter sandstone pressure-velocity relation.

    Laboratory measurements of velocity vs effective pressure
    for St. Peter sandstone samples.

    References:
        Yin, H. (1992). Acoustic velocity and attenuation of rocks: Isotropy,
        intrinsic anisotropy, and stress-induced anisotropy. PhD thesis,
        Stanford University.

    Args:
        sample: Sample number (1 or 2). Default: 1

    Example:
        >>> stpeter = StPeter(sample=1)
        >>> Vp, Vs = stpeter(Pe=0.3)  # Pe in kbar (0.3 kbar = 30 MPa)
    """

    def __init__(self, sample: int = 1):
        super().__init__()
        self.sample = sample

    def forward(self, Pe: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute velocities from effective pressure.

        Args:
            Pe: Effective pressure (kbar, 1 kbar = 100 MPa)

        Returns:
            Tuple of (Vp, Vs) in m/s
        """
        Pe = as_tensor(Pe)
        if self.sample == 1:
            Vp = 4.21 + 0.187 * Pe - 0.746 * torch.exp(-24 * Pe)
            Vs = 2.58 + 0.160 * Pe - 0.781 * torch.exp(-20 * Pe)
        else:
            Vp = 4.55 + 0.298 * Pe - 0.8 * torch.exp(-19 * Pe)
            Vs = 2.83 + 0.195 * Pe - 0.741 * torch.exp(-16 * Pe)
        return Vp * 1000, Vs * 1000


@register(name="DensityModel", category="empirical", aliases=["Density", "BulkDensity"])
class DensityModel(EmpiricalModel):
    """
    Bulk density from porosity.

    Volume-weighted average density:

        ρ_bulk = (1-φ)*ρ_matrix + φ*ρ_fluid

    References:
        Gardner, G.H.F., Gardner, L.W., & Gregory, A.R. (1974). Formation
        velocity and density — the diagnostic basics for stratigraphic traps.
        Geophysics, 39(6), 770-780.

    Args:
        None (stateless model)

    Example:
        >>> density = DensityModel()
        >>> rho = density(phi=0.25, rho_matrix=2.65, rho_fluid=1.0)

        Inverse (porosity from density):
        >>> phi = density.porosity_from_density(rho_bulk=2.4)
    """

    def __init__(self):
        super().__init__()

    def forward(
            self,
            phi: Tensor,
            rho_matrix: Tensor = 2.65,
            rho_fluid: Tensor = 1.0
    ) -> Tensor:
        """
        Compute bulk density from porosity.

        Args:
            phi: Porosity (fraction)
            rho_matrix: Matrix density (g/cm³). Default: 2.65
            rho_fluid: Fluid density (g/cm³). Default: 1.0

        Returns:
            Bulk density (g/cm³)
        """
        phi, rho_matrix, rho_fluid = as_tensor(phi), as_tensor(rho_matrix), as_tensor(rho_fluid)
        phi, rho_matrix, rho_fluid = ensure_same_device(phi, rho_matrix, rho_fluid)
        return (1 - phi) * rho_matrix + phi * rho_fluid

    def porosity_from_density(
            self,
            rho_bulk: Tensor,
            rho_matrix: Tensor = 2.65,
            rho_fluid: Tensor = 1.0
    ) -> Tensor:
        """
        Compute porosity from bulk density (inverse).

        Args:
            rho_bulk: Bulk density (g/cm³)
            rho_matrix: Matrix density (g/cm³). Default: 2.65
            rho_fluid: Fluid density (g/cm³). Default: 1.0

        Returns:
            Porosity (fraction)
        """
        rho_bulk, rho_matrix, rho_fluid = as_tensor(rho_bulk), as_tensor(rho_matrix), as_tensor(rho_fluid)
        return (rho_matrix - rho_bulk) / (rho_matrix - rho_fluid + EPS)