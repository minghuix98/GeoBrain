#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Permeability models for rock physics.

This module provides models for estimating permeability of porous rocks
from porosity, grain size, and other petrophysical properties.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
from torch import Tensor
from typing import Tuple

from ..core import PermeabilityModel, register, as_tensor, EPS


@register(name="KozenyCarman", category="permeability")
class KozenyCarman(PermeabilityModel):
    """
    Kozeny-Carman permeability model.

    Describes permeability in a porous medium assuming tortuosity
    tau=sqrt(2) and 1/B=2.5 for unconsolidated monomodal sphere pack:

        k = d^2/180 * phi^3 / (1-phi)^2

    References:
        Kozeny, J. (1927). Ueber kapillare Leitung des Wassers im Boden.
            Sitzungsberichte der Akademie der Wissenschaften Wien, 136(2a),
            271-306.
        Carman, P.C. (1937). Fluid flow through granular beds. Transactions
            of the Institution of Chemical Engineers, 15, 150-166.

    Args:
        None (stateless model)

    Example:
        >>> kc = KozenyCarman()
        >>> k = kc(phi=0.3, d=0.0001)
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "Kozeny-Carman permeability")

    def forward(self, phi: Tensor, d: Tensor) -> Tensor:
        """
        Compute permeability.

        Args:
            phi: Porosity (fraction)
            d: Pore/grain diameter (m)

        Returns:
            Permeability k (m^2)
        """
        phi, d = as_tensor(phi), as_tensor(d)
        k = d ** 2 / 180.0 * phi ** 3 / ((1.0 - phi) ** 2 + EPS)
        return k


@register(name="KozenyCarmanPercolation", category="permeability")
class KozenyCarmanPercolation(PermeabilityModel):
    """
    Kozeny-Carman with percolation effect.

        k = B * d^2 * (phi - phi_c)^3 / (1 + phi_c - phi)^2

    References:
        Mavko, G. & Nur, A. (1997). The effect of a percolation threshold
            in the Kozeny-Carman relation. Geophysics, 62(5), 1480-1482.

    Args:
        None (stateless model)
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "Kozeny-Carman with percolation")

    def forward(
            self,
            phi: Tensor,
            phi_c: Tensor,
            d: Tensor,
            B: Tensor
    ) -> Tensor:
        """
        Compute permeability with percolation.

        Args:
            phi: Porosity (fraction)
            phi_c: Percolation porosity
            d: Pore diameter
            B: Geometric factor

        Returns:
            Permeability k (same units as d^2)
        """
        phi, phi_c = as_tensor(phi), as_tensor(phi_c)
        d, B = as_tensor(d), as_tensor(B)
        k = B * d ** 2 * (phi - phi_c) ** 3 / ((1.0 + phi_c - phi) ** 2 + EPS)
        return k


@register(name="Owolabi", category="permeability")
class Owolabi(PermeabilityModel):
    """
    Owolabi permeability model for unconsolidated sands.

    Estimates permeability from log-derived porosity and irreducible
    water saturation for Pleistocene to Oligocene age sands.

    References:
        Owolabi, O.O., Longe, T.A., & Ajienka, J.A. (1994). An empirical
            expression for permeability in unconsolidated sands of eastern
            Niger Delta. Journal of Petroleum Geology, 17(1), 111-116.

    Args:
        None (stateless model)
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "Owolabi permeability")

    def forward(self, phi: Tensor, Swi: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute permeability for oil and gas sands.

        Args:
            phi: Porosity (fraction)
            Swi: Irreducible water saturation

        Returns:
            Tuple of (k_oil, k_gas) in mD
        """
        phi, Swi = as_tensor(phi), as_tensor(Swi)
        k_oil = 307.0 + 26552.0 * phi ** 2 - 34540.0 * (phi * Swi) ** 2
        k_gas = 30.7 + 2655.0 * phi ** 2 - 3454.0 * (phi * Swi) ** 2
        return k_oil, k_gas


@register(name="PermLogs", category="permeability")
class PermLogs(PermeabilityModel):
    """
    Empirical permeability correlations from well logs.

    Implements Tixier, Timur, Coates, and Coates-Dumanoir models.

    References:
        Wyllie, M.R.J. & Rose, W.D. (1950). Some theoretical considerations
            related to the quantitative evaluation of the physical
            characteristics of reservoir rock from electrical log data.
            Journal of Petroleum Technology, 2(4), 105-118.

    Args:
        None (stateless model)
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "Log-based permeability correlations")

    def forward(
            self, phi: Tensor, Swi: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Compute permeability using multiple correlations.

        Args:
            phi: Porosity (fraction)
            Swi: Irreducible water saturation

        Returns:
            Tuple of (k_tixier, k_timur, k_coates, k_coates_dumanoir) in mD
        """
        phi, Swi = as_tensor(phi), as_tensor(Swi)
        k_tixier = 62500.0 * phi ** 6 / (Swi ** 2 + EPS)
        k_timur = 10000.0 * phi ** 4.5 / (Swi ** 2 + EPS)
        k_coates = 10000.0 * phi ** 4 * (1.0 - Swi) ** 2 / (Swi ** 2 + EPS)
        k_coates_dumanoir = 352.0 * phi ** 4 / (Swi ** 4 + EPS)
        return k_tixier, k_timur, k_coates, k_coates_dumanoir


@register(name="PandaLake", category="permeability")
class PandaLake(PermeabilityModel):
    """
    Modified Kozeny-Carman with grain size variation (Panda & Lake 1994).

        k = d^2 * phi^3 * (C^3*S + 3*C^2 + 1)^2 /
            (72 * tau * (1-phi)^2 * (C^2+1)^2)

    References:
        Panda, M.N. & Lake, L.W. (1994). Estimation of single-phase
            permeability from parameters of particle-size distribution.
            AAPG Bulletin, 78(7), 1028-1039.

    Args:
        None (stateless model)
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "Panda-Lake permeability")

    def forward(
            self,
            d: Tensor,
            C: Tensor,
            S: Tensor,
            tau: Tensor,
            phi: Tensor
    ) -> Tensor:
        """
        Compute permeability.

        Args:
            d: Mean particle size (um)
            C: Coefficient of variation of particle size distribution
            S: Skewness of particle size distribution
            tau: Tortuosity factor
            phi: Porosity (fraction)

        Returns:
            Permeability k (mD)
        """
        d, C, S = as_tensor(d), as_tensor(C), as_tensor(S)
        tau, phi = as_tensor(tau), as_tensor(phi)
        numer = d ** 2 * phi ** 3 * (C ** 3 * S + 3.0 * C ** 2 + 1.0) ** 2
        denom = 72.0 * tau * (1.0 - phi) ** 2 * (C ** 2 + 1.0) ** 2 + EPS
        return numer / denom


@register(name="PandaLakeCem", category="permeability")
class PandaLakeCem(PermeabilityModel):
    """
    Panda & Lake model for cemented sands.

        K = 3.34 * d^2 * phi^3 / (1-phi)^2

    References:
        Panda, M.N. & Lake, L.W. (1995). A physical model of cementation
            and its effects on single-phase permeability. AAPG Bulletin,
            79(3), 431-443.

    Args:
        None (stateless model)
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "Panda-Lake cement permeability")

    def forward(self, phi: Tensor, d: Tensor) -> Tensor:
        """
        Compute permeability for cemented sands.

        Args:
            phi: Porosity (fraction)
            d: Mean particle size (um)

        Returns:
            Permeability k (mD)
        """
        phi, d = as_tensor(phi), as_tensor(d)
        return 3.34 * d ** 2 * phi ** 3 / ((1.0 - phi) ** 2 + EPS)


@register(name="Revil", category="permeability")
class Revil(PermeabilityModel):
    """
    Revil et al. (1997) permeability model for shaly rocks.

        k = 1000 * d^2 * phi^4.5 / 24

    References:
        Revil, A. & Cathles, L.M. (1999). Permeability of shaly sands.
            Water Resources Research, 35(3), 651-662.

    Args:
        None (stateless model)
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "Revil shaly rock permeability")

    def forward(self, phi: Tensor, d: Tensor) -> Tensor:
        """
        Compute permeability for shaly rocks.

        Args:
            phi: Porosity (fraction)
            d: Mean particle size (um)

        Returns:
            Permeability k (mD)
        """
        phi, d = as_tensor(phi), as_tensor(d)
        return 1000.0 * d ** 2 * phi ** 4.5 / 24.0


@register(name="Fredrich", category="permeability")
class Fredrich(PermeabilityModel):
    """
    Fredrich et al. (1993) pore geometry permeability model.

        k = 1/(b*F) * (phi/Sv)^2
        F = 2.5/phi (formation factor)
        Sv = 6*(1-phi)*d (pore surface area per unit volume)

    References:
        Fredrich, J.T., Greaves, K.H., & Martin, J.W. (1993). Pore geometry
            and transport properties of Fontainebleau sandstone. International
            Journal of Rock Mechanics and Mining Sciences, 30(7), 691-697.

    Args:
        None (stateless model)
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "Fredrich pore geometry permeability")

    def forward(self, phi: Tensor, d: Tensor, b: Tensor) -> Tensor:
        """
        Compute permeability from pore geometry.

        Args:
            phi: Porosity (>10%)
            d: Grain size parameter
            b: Shape factor (2 for circular tubes, 3 for cracks)

        Returns:
            Permeability k (mD)
        """
        phi, d, b = as_tensor(phi), as_tensor(d), as_tensor(b)
        F = 2.5 / (phi + EPS)
        Sv = 6.0 * (1.0 - phi) * d
        k = 1.0 / (b * F + EPS) * (phi / (Sv + EPS)) ** 2
        return k


@register(name="Bloch", category="permeability")
class Bloch(PermeabilityModel):
    """
    Bloch empirical porosity and permeability prediction.

    Empirical relations from the Yacheng field for predicting porosity
    and permeability in sandstones prior to drilling.

    References:
        Bloch, S. (1991). Empirical prediction of porosity and permeability
            in sandstones. AAPG Bulletin, 75(7), 1145-1160.

    Args:
        None (stateless model)
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "Bloch empirical permeability")

    def forward(
            self, S: Tensor, C: Tensor, D: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Predict porosity and permeability.

        Args:
            S: Trask sorting coefficient
            C: Rigid grain content (fraction)
            D: Grain size (mm)

        Returns:
            Tuple of (phi, k): porosity (frac) and permeability (mD)
        """
        S, C, D = as_tensor(S), as_tensor(C), as_tensor(D)
        phi = -6.1 + 9.8 / (S + EPS) + 0.17 * C
        k = 10.0 ** (-4.67 + 1.34 * D + 4.08 / (S + EPS) + 3.42 * C / 100.0)
        return phi, k


@register(name="Bernabe", category="permeability")
class Bernabe(PermeabilityModel):
    """
    Bernabe (1991) dual porosity permeability model.

    Computes permeability from crack and tube pore components:

        k_crack = w^2 * phi_crack / 30
        k_tube = r^2 * phi_tube / 20
        k = k_crack + k_tube

    References:
        Bernabe, Y., Brace, W.F., & Evans, B. (1982). Permeability, porosity,
            and pore geometry of hot-pressed calcite. Mechanics of Materials,
            1(3), 173-183.

    Args:
        None (stateless model)
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "Bernabe dual porosity permeability")

    def forward(
            self,
            phi: Tensor,
            crf: Tensor,
            w: Tensor,
            r: Tensor
    ) -> Tensor:
        """
        Compute total permeability.

        Args:
            phi: Total porosity
            crf: Crack fraction in pore volume
            w: Crack aperture (um)
            r: Tube radius (um)

        Returns:
            Total permeability k (mD)
        """
        phi, crf = as_tensor(phi), as_tensor(crf)
        w, r = as_tensor(w), as_tensor(r)
        phi_crack = phi * crf
        phi_tube = phi - phi_crack
        k_crack = w ** 2 * phi_crack / 30.0
        k_tube = r ** 2 * phi_tube / 20.0
        return k_crack + k_tube
