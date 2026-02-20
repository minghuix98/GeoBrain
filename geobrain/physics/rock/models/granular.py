#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Granular medium models for rock physics.

This module provides models for computing elastic properties of
granular and cemented sediments, including contact theories,
sand models, and shale mixture models.

Models:
    - Hertz-Mindlin contact theory
    - Soft sand / Friable sand model
    - Stiff sand / Cemented sand model
    - Contact cement models
    - Patchy cement model (PCM)
    - Thomas-Stieber porosity model
    - Shaly sand models

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
from torch import Tensor
from typing import Tuple

from ..core import GranularModel, register, as_tensor, ensure_same_device, EPS, PI


@register(name="HertzMindlin", category="granular", aliases=["HM"])
class HertzMindlin(GranularModel):
    """
    Hertz-Mindlin contact theory for granular media.

    Computes effective moduli at critical porosity from grain contact
    stiffness under confining pressure:

        K_HM = [P*Cn²*(1-φc)²*G0² / (18π²*(1-ν)²)]^(1/3)

    References:
        Hertz, H. (1882). Ueber die Beruehrung fester elastischer Koerper.
            Journal fur die reine und angewandte Mathematik, 92, 156-171.
        Mindlin, R.D. (1949). Compliance of elastic bodies in contact.
            Journal of Applied Mechanics, 16, 259-268.

    Args:
        None (stateless model)

    Example:
        >>> hm = HertzMindlin()
        >>> K_HM, G_HM = hm(
        ...     K0=36.6, G0=45.0,     # Grain moduli
        ...     phi_c=0.36,            # Critical porosity
        ...     Cn=8.5,                # Coordination number
        ...     P=20.0,                # Pressure (MPa)
        ...     f=1.0                  # Friction coefficient
        ... )
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "Hertz-Mindlin contact theory")

    def forward(
            self,
            K0: Tensor,
            G0: Tensor,
            phi_c: Tensor = 0.36,
            Cn: Tensor = 8.5,
            P: Tensor = 20.0,
            f: Tensor = 1.0
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute Hertz-Mindlin moduli.

        Args:
            K0: Grain bulk modulus (GPa)
            G0: Grain shear modulus (GPa)
            phi_c: Critical porosity. Default: 0.36
            Cn: Coordination number. Default: 8.5
            P: Effective pressure (MPa). Default: 20.0
            f: Friction coefficient (0=smooth, 1=rough). Default: 1.0

        Returns:
            Tuple of (K_HM, G_HM) at critical porosity (GPa)
        """
        K0, G0 = as_tensor(K0), as_tensor(G0)
        phi_c, Cn, P, f = as_tensor(phi_c), as_tensor(Cn), as_tensor(P), as_tensor(f)
        K0, G0, phi_c, Cn, P, f = ensure_same_device(K0, G0, phi_c, Cn, P, f)
        sigma = P / 1000.0  # Convert MPa to GPa

        # Poisson's ratio
        nu = (3 * K0 - 2 * G0) / (6 * K0 + 2 * G0 + EPS)

        # Hertz-Mindlin moduli
        K_HM = (sigma * Cn ** 2 * (1 - phi_c) ** 2 * G0 ** 2 / (18 * PI ** 2 * (1 - nu) ** 2 + EPS)) ** (1 / 3)
        G_HM = ((2 + 3 * f - nu * (1 + 3 * f)) / (5 * (2 - nu) + EPS)) * \
               (3 * sigma * Cn ** 2 * (1 - phi_c) ** 2 * G0 ** 2 / (2 * PI ** 2 * (1 - nu) ** 2 + EPS)) ** (1 / 3)

        return K_HM, G_HM


@register(name="SoftSand", category="granular", aliases=["FriableSand"])
class SoftSand(GranularModel):
    """
    Soft sand model (modified Hashin-Shtrikman lower bound).

    Models unconsolidated or friable sands by interpolating between
    Hertz-Mindlin contact moduli and mineral moduli using the
    HS lower bound.

    References:
        Dvorkin, J. & Nur, A. (1996). Elasticity of high-porosity sandstones:
            Theory for two North Sea data sets. Geophysics, 61(5), 1363-1370.

    Args:
        None (stateless model)

    Example:
        >>> soft = SoftSand()
        >>> K_dry, G_dry = soft(
        ...     K0=36.6, G0=45.0,
        ...     phi=0.30,
        ...     phi_c=0.36
        ... )
    """

    def __init__(self):
        super().__init__()
        self._hm = HertzMindlin()

    def forward(
            self,
            K0: Tensor,
            G0: Tensor,
            phi: Tensor,
            phi_c: Tensor = 0.36,
            Cn: Tensor = 8.5,
            P: Tensor = 20.0,
            f: Tensor = 1.0
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute dry rock moduli using soft sand model.

        Args:
            K0: Grain bulk modulus (GPa)
            G0: Grain shear modulus (GPa)
            phi: Porosity (fraction)
            phi_c: Critical porosity. Default: 0.36
            Cn: Coordination number. Default: 8.5
            P: Effective pressure (MPa). Default: 20.0
            f: Friction coefficient. Default: 1.0

        Returns:
            Tuple of (K_dry, G_dry) in GPa
        """
        K0, G0, phi, phi_c = as_tensor(K0), as_tensor(G0), as_tensor(phi), as_tensor(phi_c)
        K0, G0, phi, phi_c = ensure_same_device(K0, G0, phi, phi_c)
        K_HM, G_HM = self._hm(K0, G0, phi_c, Cn, P, f)

        z = phi / phi_c

        # Modified HS lower bound
        gamma = K_HM + 4 * G_HM / 3
        K_dry = (z / gamma + (1 - z) / (K0 + 4 * G_HM / 3 + EPS)) ** (-1) - 4 * G_HM / 3

        zeta = G_HM / 6 * (9 * K_HM + 8 * G_HM) / (K_HM + 2 * G_HM + EPS)
        G_dry = (z / (G_HM + zeta) + (1 - z) / (G0 + zeta + EPS)) ** (-1) - zeta

        return K_dry, G_dry

    def compute_dry_rock(self, K0, G0, phi, **params):
        return self.forward(
            K0, G0, phi,
            phi_c=params.get('phi_c', 0.36),
            Cn=params.get('Cn', 8.5),
            P=params.get('P', 20.0),
        )


@register(name="StiffSand", category="granular", aliases=["CementedSand"])
class StiffSand(GranularModel):
    """
    Stiff sand model (modified Hashin-Shtrikman upper bound).

    Models well-cemented or compacted sands by interpolating between
    Hertz-Mindlin contact moduli and mineral moduli using the
    HS upper bound.

    References:
        Dvorkin, J. & Nur, A. (1996). Elasticity of high-porosity sandstones:
            Theory for two North Sea data sets. Geophysics, 61(5), 1363-1370.

    Args:
        None (stateless model)

    Example:
        >>> stiff = StiffSand()
        >>> K_dry, G_dry = stiff(K0=36.6, G0=45.0, phi=0.20)
    """

    def __init__(self):
        super().__init__()
        self._hm = HertzMindlin()

    def forward(
            self,
            K0: Tensor,
            G0: Tensor,
            phi: Tensor,
            phi_c: Tensor = 0.36,
            Cn: Tensor = 8.5,
            P: Tensor = 20.0,
            f: Tensor = 1.0
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute dry rock moduli using stiff sand model.

        Args:
            K0: Grain bulk modulus (GPa)
            G0: Grain shear modulus (GPa)
            phi: Porosity (fraction)
            phi_c: Critical porosity. Default: 0.36
            Cn: Coordination number. Default: 8.5
            P: Effective pressure (MPa). Default: 20.0
            f: Friction coefficient. Default: 1.0

        Returns:
            Tuple of (K_dry, G_dry) in GPa
        """
        K0, G0, phi, phi_c = as_tensor(K0), as_tensor(G0), as_tensor(phi), as_tensor(phi_c)
        K0, G0, phi, phi_c = ensure_same_device(K0, G0, phi, phi_c)
        K_HM, G_HM = self._hm(K0, G0, phi_c, Cn, P, f)

        z = phi / phi_c

        # Modified HS upper bound
        gamma = K0 + 4 * G0 / 3
        K_dry = (z / (K_HM + 4 * G0 / 3 + EPS) + (1 - z) / gamma) ** (-1) - 4 * G0 / 3

        zeta = G0 / 6 * (9 * K0 + 8 * G0) / (K0 + 2 * G0 + EPS)
        G_dry = (z / (G_HM + zeta + EPS) + (1 - z) / (G0 + zeta)) ** (-1) - zeta

        return K_dry, G_dry

    def compute_dry_rock(self, K0, G0, phi, **params):
        return self.forward(
            K0, G0, phi,
            phi_c=params.get('phi_c', 0.36),
            Cn=params.get('Cn', 8.5),
            P=params.get('P', 20.0),
        )


@register(name="ContactCement", category="granular")
class ContactCement(GranularModel):
    """
    Contact cement model (simplified).

    Models cemented granular rocks where cement is deposited at
    grain contacts. Uses simplified Sn, St coefficients.

    References:
        Dvorkin, J., Nur, A., & Yin, H. (1994). Effective properties of
            cemented granular materials. Mechanics of Materials, 18(4), 351-366.

    Args:
        scheme: Cementation scheme (1 or 2). Default: 2
            - 1: Cement deposited at grain contacts
            - 2: Cement deposited uniformly

    Example:
        >>> ccm = ContactCement(scheme=2)
        >>> K_dry, G_dry = ccm(
        ...     K0=36.6, G0=45.0,   # Grain moduli
        ...     Kc=36.6, Gc=45.0,   # Cement moduli
        ...     phi=0.30,           # Current porosity
        ...     phi_c=0.36          # Initial (critical) porosity
        ... )
    """

    def __init__(self, scheme: int = 2):
        super().__init__()
        self.scheme = scheme

    def forward(
            self,
            K0: Tensor,
            G0: Tensor,
            Kc: Tensor,
            Gc: Tensor,
            phi: Tensor,
            phi_c: Tensor = 0.36,
            Cn: Tensor = 8.5
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute dry rock moduli for cemented sand.

        Args:
            K0: Grain bulk modulus (GPa)
            G0: Grain shear modulus (GPa)
            Kc: Cement bulk modulus (GPa)
            Gc: Cement shear modulus (GPa)
            phi: Current porosity (fraction)
            phi_c: Critical porosity. Default: 0.36
            Cn: Coordination number. Default: 8.5

        Returns:
            Tuple of (K_dry, G_dry) in GPa
        """
        K0, G0, Kc, Gc = as_tensor(K0), as_tensor(G0), as_tensor(Kc), as_tensor(Gc)
        phi, phi_c, Cn = as_tensor(phi), as_tensor(phi_c), as_tensor(Cn)

        phi_cement = phi_c - phi

        # Contact radius from cement volume
        if self.scheme == 1:
            alpha = ((2 * phi_cement) / (3 * Cn * (1 - phi_c))) ** 0.25
        else:
            alpha = ((2 * phi_cement) / (3 * (1 - phi_c))) ** 0.5

        # Simplified Sn, St
        Sn = alpha ** 2 * 2 * Gc / (PI * G0 + EPS)
        St = alpha ** 2 * Gc / (PI * G0 + EPS)

        # Effective moduli
        Cn_term = Cn * (1 - phi_c)
        K_dry = Cn_term * (K0 + 4 * G0 / 3) * Sn / 6
        G_dry = 3 * K_dry / 5 + 3 * Cn_term * G0 * St / 20

        return K_dry, G_dry


@register(name="ContactCementFull", category="granular")
class ContactCementFull(GranularModel):
    """
    Full contact cement model with empirical Sn, St coefficients.

    Uses complete empirical expressions for normal and tangential
    stiffness coefficients from Dvorkin & Nur (1996).

    References:
        Dvorkin, J., Nur, A., & Yin, H. (1994). Effective properties of
            cemented granular materials. Mechanics of Materials, 18(4), 351-366.

    Args:
        scheme: Cementation scheme (1 or 2). Default: 2

    Example:
        >>> ccm = ContactCementFull(scheme=2)
        >>> K_dry, G_dry = ccm(K0=36.6, G0=45.0, Kc=36.6, Gc=45.0, phi=0.28)
    """

    def __init__(self, scheme: int = 2):
        super().__init__()
        self.scheme = scheme

    def forward(
            self,
            K0: Tensor,
            G0: Tensor,
            Kc: Tensor,
            Gc: Tensor,
            phi: Tensor,
            phi_c: Tensor = 0.36,
            Cn: Tensor = 8.5
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute dry rock moduli using full contact cement model.

        Args:
            K0: Grain bulk modulus (GPa)
            G0: Grain shear modulus (GPa)
            Kc: Cement bulk modulus (GPa)
            Gc: Cement shear modulus (GPa)
            phi: Current porosity (fraction)
            phi_c: Critical porosity. Default: 0.36
            Cn: Coordination number. Default: 8.5

        Returns:
            Tuple of (K_dry, G_dry) in GPa
        """
        K0, G0, Kc, Gc = as_tensor(K0), as_tensor(G0), as_tensor(Kc), as_tensor(Gc)
        phi, phi_c, Cn = as_tensor(phi), as_tensor(phi_c), as_tensor(Cn)

        # Poisson's ratios
        nu_0 = (3 * K0 - 2 * G0) / (6 * K0 + 2 * G0 + EPS)
        nu_c = (3 * Kc - 2 * Gc) / (6 * Kc + 2 * Gc + EPS)

        phi_cement = phi_c - phi

        # Contact radius
        if self.scheme == 1:
            alpha = ((2 * phi_cement) / (3 * Cn * (1 - phi_c))) ** 0.25
        else:
            alpha = ((2 * phi_cement) / (3 * (1 - phi_c))) ** 0.5

        # Dimensionless stiffness parameters
        LambdaN = 2 * Gc * (1 - nu_0) * (1 - nu_c) / (PI * G0 * (1 - 2 * nu_c) + EPS)
        LambdaT = Gc / (PI * G0 + EPS)

        # Empirical Sn coefficients
        N1 = -0.024153 * LambdaN ** (-1.3646)
        N2 = 0.20405 * LambdaN ** (-0.89008)
        N3 = 0.00024649 * LambdaN ** (-1.9864)
        Sn = N1 * alpha ** 2 + N2 * alpha + N3

        # Empirical St coefficients
        T1 = -1e-2 * (2.26 * nu_0 ** 2 + 2.07 * nu_0 + 2.3) * LambdaT ** (0.079 * nu_0 ** 2 + 0.1754 * nu_0 - 1.342)
        T2 = (0.0573 * nu_0 ** 2 + 0.0937 * nu_0 + 0.202) * LambdaT ** (0.0274 * nu_0 ** 2 + 0.0529 * nu_0 - 0.8765)
        T3 = 1e-4 * (9.654 * nu_0 ** 2 + 4.945 * nu_0 + 3.1) * LambdaT ** (0.01867 * nu_0 ** 2 + 0.4011 * nu_0 - 1.8186)
        St = T1 * alpha ** 2 + T2 * alpha + T3

        # Effective moduli
        K_dry = Cn * (1 - phi_c) * (Kc + 4 * Gc / 3) * Sn / 6
        G_dry = 3 * K_dry / 5 + 3 * Cn * (1 - phi_c) * Gc * St / 20

        return K_dry, G_dry


@register(name="Walton", category="granular")
class Walton(GranularModel):
    """
    Walton sphere pack model.

    Models random packing of identical elastic spheres under
    hydrostatic pressure with variable friction.

    References:
        Walton, K. (1987). The effective elastic moduli of a random packing of
            spheres. Journal of the Mechanics and Physics of Solids, 35(2),
            213-226.

    Args:
        None (stateless model)

    Example:
        >>> walton = Walton()
        >>> K, G = walton(K0=36.6, G0=45.0, phi_c=0.36, P=20.0, f=0.5)
    """

    def __init__(self):
        super().__init__()

    def forward(
            self,
            K0: Tensor,
            G0: Tensor,
            phi_c: Tensor = 0.36,
            Cn: Tensor = 8.5,
            P: Tensor = 20.0,
            f: Tensor = 1.0
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute Walton sphere pack moduli.

        Args:
            K0: Grain bulk modulus (GPa)
            G0: Grain shear modulus (GPa)
            phi_c: Porosity. Default: 0.36
            Cn: Coordination number. Default: 8.5
            P: Effective pressure (MPa). Default: 20.0
            f: Friction coefficient (0=smooth, 1=rough). Default: 1.0

        Returns:
            Tuple of (K_w, G_w) in GPa
        """
        K0, G0, phi_c, Cn = as_tensor(K0), as_tensor(G0), as_tensor(phi_c), as_tensor(Cn)
        P, f = as_tensor(P) / 1000, as_tensor(f)

        lamda = K0 - 2 * G0 / 3
        B = 1 / (4 * PI) * (1 / G0 + 1 / (G0 + lamda))

        K_w = 1 / 6 * (3 * (1 - phi_c) ** 2 * Cn ** 2 * P / (PI ** 4 * B ** 2 + EPS)) ** (1 / 3)

        nu = (3 * K0 - 2 * G0) / (6 * K0 + 2 * G0 + EPS)
        G_smooth = 3 / 5 * K_w
        G_rough = 3 / 5 * K_w * (5 - 4 * nu) / (2 - nu + EPS)
        G_w = f * G_rough + (1 - f) * G_smooth

        return K_w, G_w


@register(name="MUHS", category="granular", aliases=["IncreasingCement"])
class MUHS(GranularModel):
    """
    Modified Upper Hashin-Shtrikman for increasing cement.

    Extends contact cement model to allow porosity reduction
    beyond the cemented reference porosity.

    References:
        Dvorkin, J. & Nur, A. (1996). Elasticity of high-porosity sandstones:
            Theory for two North Sea data sets. Geophysics, 61(5), 1363-1370.

    Args:
        scheme: Cementation scheme (1 or 2). Default: 2

    Example:
        >>> muhs = MUHS()
        >>> K, G = muhs(
        ...     K0=36.6, G0=45.0, Kc=36.6, Gc=45.0,
        ...     phi=0.15, phi_i=0.28, phi_c=0.36
        ... )
    """

    def __init__(self, scheme: int = 2):
        super().__init__()
        self._ccm = ContactCementFull(scheme)

    def forward(
            self,
            K0: Tensor,
            G0: Tensor,
            Kc: Tensor,
            Gc: Tensor,
            phi: Tensor,
            phi_i: Tensor,
            phi_c: Tensor = 0.36,
            Cn: Tensor = 8.5
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute moduli for increasing cement model.

        Args:
            K0: Grain bulk modulus (GPa)
            G0: Grain shear modulus (GPa)
            Kc: Cement bulk modulus (GPa)
            Gc: Cement shear modulus (GPa)
            phi: Current porosity (fraction)
            phi_i: Initial cemented porosity (fraction)
            phi_c: Critical porosity. Default: 0.36
            Cn: Coordination number. Default: 8.5

        Returns:
            Tuple of (K_eff, G_eff) in GPa
        """
        K0, G0, Kc, Gc = as_tensor(K0), as_tensor(G0), as_tensor(Kc), as_tensor(Gc)
        phi, phi_i, phi_c = as_tensor(phi), as_tensor(phi_i), as_tensor(phi_c)

        # Contact cement moduli at initial porosity
        K_cc, G_cc = self._ccm(K0, G0, Kc, Gc, phi_i, phi_c, Cn)

        # Modified upper HS bound
        z = (phi_i - phi) / phi_i

        gamma = K_cc + 4 * G_cc / 3
        K_eff = (z / (K0 + 4 * G_cc / 3 + EPS) + (1 - z) / gamma) ** (-1) - 4 * G_cc / 3

        zeta = G_cc / 6 * (9 * K_cc + 8 * G_cc) / (K_cc + 2 * G_cc + EPS)
        G_eff = (z / (G0 + zeta + EPS) + (1 - z) / (G_cc + zeta)) ** (-1) - zeta

        return K_eff, G_eff


@register(name="PCM", category="granular", aliases=["PatchyCement"])
class PCM(GranularModel):
    """
    Patchy Cement Model.

    Models heterogeneous cementation where only a fraction of the
    rock is cemented (patchy distribution).

    References:
        Avseth, P., Dvorkin, J., Mavko, G., & Rykkje, J. (2000). Rock physics
            diagnostic of North Sea sands: Link between microstructure and
            seismic properties. Geophysical Research Letters, 27(17), 2761-2764.

    Args:
        scheme: Cementation scheme (1 or 2). Default: 2

    Example:
        >>> pcm = PCM()
        >>> K_dry, G_dry = pcm(
        ...     K0=36.6, G0=45.0, Kc=36.6, Gc=45.0,
        ...     phi=0.25, phi_c=0.36, v_cem=0.05, f_cem=0.3
        ... )
    """

    def __init__(self, scheme: int = 2):
        super().__init__()
        self._hm = HertzMindlin()
        self._ccm = ContactCementFull(scheme)

    def forward(
            self,
            K0: Tensor,
            G0: Tensor,
            Kc: Tensor,
            Gc: Tensor,
            phi: Tensor,
            phi_c: Tensor,
            v_cem: Tensor,
            f_cem: Tensor,
            Cn: Tensor = 8.5,
            P: Tensor = 20.0,
            f_hm: Tensor = 0.5,
            mode: str = 'stiff'
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute dry rock moduli using patchy cement model.

        Args:
            K0: Grain bulk modulus (GPa)
            G0: Grain shear modulus (GPa)
            Kc: Cement bulk modulus (GPa)
            Gc: Cement shear modulus (GPa)
            phi: Porosity (fraction)
            phi_c: Critical porosity
            v_cem: Cement volume fraction
            f_cem: Fraction of rock that is cemented
            Cn: Coordination number. Default: 8.5
            P: Effective pressure (MPa). Default: 20.0
            f_hm: HM friction coefficient. Default: 0.5
            mode: 'stiff' or 'soft' mixing. Default: 'stiff'

        Returns:
            Tuple of (K_dry, G_dry) in GPa
        """
        K0, G0, Kc, Gc = as_tensor(K0), as_tensor(G0), as_tensor(Kc), as_tensor(Gc)
        phi, phi_c, v_cem, f_cem = as_tensor(phi), as_tensor(phi_c), as_tensor(v_cem), as_tensor(f_cem)

        # Uncemented region moduli (Hertz-Mindlin)
        K_unc, G_unc = self._hm(K0, G0, phi_c, Cn, P, f_hm)

        # Cemented region moduli
        phi_cem = phi_c - v_cem
        K_cem, G_cem = self._ccm(K0, G0, Kc, Gc, phi_cem, phi_c, Cn)

        # Mix cemented and uncemented regions
        if mode == 'stiff':
            K_p = K_cem + (1 - f_cem) / (1 / (K_unc - K_cem + EPS) + f_cem / (K_cem + 4 * G_cem / 3 + EPS))
            temp = (K_cem + 2 * G_cem) / (5 * G_cem * (K_cem + 4 * G_cem / 3) + EPS)
            G_p = G_cem + (1 - f_cem) / (1 / (G_unc - G_cem + EPS) + 2 * f_cem * temp)
        else:
            K_p = K_unc + f_cem / (1 / (K_cem - K_unc + EPS) + (1 - f_cem) / (K_unc + 4 * G_unc / 3 + EPS))
            temp = (K_unc + 2 * G_unc) / (5 * G_unc * (K_unc + 4 * G_unc / 3) + EPS)
            G_p = G_unc + f_cem / (1 / (G_cem - G_unc + EPS) + 2 * (1 - f_cem) * temp)

        # Interpolate to target porosity
        z = phi / phi_c
        K_dry = -4 * G_p / 3 + (z / (K_p + 4 * G_p / 3 + EPS) + (1 - z) / (K0 + 4 * G_p / 3 + EPS)) ** (-1)

        zeta = G_p / 6 * (9 * K_p + 8 * G_p) / (K_p + 2 * G_p + EPS)
        G_dry = -zeta + (z / (G_p + zeta + EPS) + (1 - z) / (G0 + zeta + EPS)) ** (-1)

        return K_dry, G_dry


@register(name="ThomasStieber", category="granular")
class ThomasStieber(GranularModel):
    """
    Thomas-Stieber porosity model.

    Computes porosity bounds for sand-shale mixtures assuming
    dispersed or laminated shale distribution.

    References:
        Thomas, E.C. & Stieber, S.J. (1975). The distribution of shale in
            sandstones and its effect upon porosity. SPWLA 16th Annual Logging
            Symposium.

    Args:
        None (stateless model)

    Example:
        >>> ts = ThomasStieber()
        >>> phi_dispersed, phi_laminated = ts(
        ...     phi_sand=0.35, phi_sh=0.10, vsh=0.3
        ... )
    """

    def __init__(self):
        super().__init__()

    def forward(
            self,
            phi_sand: Tensor,
            phi_sh: Tensor,
            vsh: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute porosity for dispersed and laminated shale models.

        Args:
            phi_sand: Clean sand porosity (fraction)
            phi_sh: Shale porosity (fraction)
            vsh: Shale volume fraction (0-1)

        Returns:
            Tuple of (phi_dispersed, phi_laminated)
        """
        phi_sand, phi_sh, vsh = as_tensor(phi_sand), as_tensor(phi_sh), as_tensor(vsh)

        phi_dirty = phi_sand - (1 - phi_sh) * vsh
        phi_structural = phi_sh * vsh
        phi_dispersed = torch.maximum(phi_dirty, phi_structural)
        phi_laminated = phi_sand + (phi_sh - phi_sand) * vsh

        return phi_dispersed, phi_laminated


@register(name="SiltyShale", category="granular")
class SiltyShale(GranularModel):
    """
    Dvorkin-Gutierrez silty shale model.

    Models elastic properties of shale with dispersed silt using
    the Hashin-Shtrikman lower bound with shale as the matrix.

    References:
        Marion, D., Nur, A., Yin, H., & Han, D. (1992). Compressional velocity
            and porosity in sand-clay mixtures. Geophysics, 57(4), 554-563.

    Args:
        None (stateless model)

    Example:
        >>> ss = SiltyShale()
        >>> K_sat, G_sat = ss(
        ...     C=0.2,                    # Silt fraction
        ...     Kq=36.6, Gq=45.0,         # Silt (quartz) moduli
        ...     Ksh=21.0, Gsh=7.0         # Shale moduli
        ... )
    """

    def __init__(self):
        super().__init__()

    def forward(
            self,
            C: Tensor,
            Kq: Tensor,
            Gq: Tensor,
            Ksh: Tensor,
            Gsh: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute saturated moduli for silty shale.

        Args:
            C: Silt/quartz volume fraction
            Kq: Silt (quartz) bulk modulus (GPa)
            Gq: Silt (quartz) shear modulus (GPa)
            Ksh: Shale bulk modulus (GPa)
            Gsh: Shale shear modulus (GPa)

        Returns:
            Tuple of (K_sat, G_sat) in GPa
        """
        C, Kq, Gq, Ksh, Gsh = as_tensor(C), as_tensor(Kq), as_tensor(Gq), as_tensor(Ksh), as_tensor(Gsh)

        # HS lower bound with shale as matrix
        K_sat = (C / (Ksh + 4 * Gsh / 3) + (1 - C) / (Kq + 4 * Gsh / 3 + EPS)) ** (-1) - 4 * Gsh / 3

        Zsh = Gsh / 6 * (9 * Ksh + 8 * Gsh) / (Ksh + 2 * Gsh + EPS)
        G_sat = (C / (Gsh + Zsh) + (1 - C) / (Gq + Zsh + EPS)) ** (-1) - Zsh

        return K_sat, G_sat


@register(name="ShalySand", category="granular")
class ShalySand(GranularModel):
    """
    Shaly sand model using HS lower bound.

    Models elastic properties of sand with dispersed shale laminae
    using the Hashin-Shtrikman lower bound.

    References:
        Marion, D., Nur, A., Yin, H., & Han, D. (1992). Compressional velocity
            and porosity in sand-clay mixtures. Geophysics, 57(4), 554-563.

    Args:
        None (stateless model)

    Example:
        >>> shaly = ShalySand()
        >>> K_sat, G_sat = shaly(
        ...     phi_s=0.35, C=0.1,
        ...     Kss=20.0, Gss=15.0,   # Sand frame moduli
        ...     Kcc=15.0, Gcc=8.0     # Shale moduli
        ... )
    """

    def __init__(self):
        super().__init__()

    def forward(
            self,
            phi_s: Tensor,
            C: Tensor,
            Kss: Tensor,
            Gss: Tensor,
            Kcc: Tensor,
            Gcc: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute saturated moduli for shaly sand.

        Args:
            phi_s: Sand porosity (fraction)
            C: Shale volume in pore space (C/phi_s = shale fraction)
            Kss: Sand frame bulk modulus (GPa)
            Gss: Sand frame shear modulus (GPa)
            Kcc: Shale bulk modulus (GPa)
            Gcc: Shale shear modulus (GPa)

        Returns:
            Tuple of (K_sat, G_sat) in GPa
        """
        phi_s, C = as_tensor(phi_s), as_tensor(C)
        Kss, Gss, Kcc, Gcc = as_tensor(Kss), as_tensor(Gss), as_tensor(Kcc), as_tensor(Gcc)

        # HS lower bound
        K_sat = ((1 - C / phi_s) / (Kss + 4 * Gss / 3) + (C / phi_s) / (Kcc + 4 * Gss / 3 + EPS)) ** (-1) - 4 * Gss / 3

        Zss = Gss / 6 * (9 * Kss + 8 * Gss) / (Kss + 2 * Gss + EPS)
        G_sat = ((1 - C / phi_s) / (Gss + Zss) + (C / phi_s) / (Gcc + Zss + EPS)) ** (-1) - Zss

        return K_sat, G_sat


# =============================================================================
# Additional Granular Models
# =============================================================================

@register(name="Digby", category="granular")
class Digby(GranularModel):
    """
    Digby (1981) bonded sphere pack model.

    Computes effective moduli for a random packing of identical
    elastic spheres with initially bonded contacts.

    References:
        Digby, P.J. (1981). The effective elastic moduli of porous granular
            rocks. Journal of Applied Mechanics, 48(4), 803-808.

    Args:
        None (stateless model)
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "Digby bonded contact model")

    def forward(
            self,
            K0: Tensor,
            G0: Tensor,
            phi: Tensor,
            Cn: Tensor,
            sigma: Tensor,
            a_R: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute effective moduli.

        Args:
            K0: Grain bulk modulus (GPa)
            G0: Grain shear modulus (GPa)
            phi: Porosity (fraction)
            Cn: Coordination number
            sigma: Effective stress (MPa)
            a_R: Ratio of bonded area radius to grain radius

        Returns:
            Tuple of (Keff, Geff) in GPa
        """
        K0, G0 = as_tensor(K0), as_tensor(G0)
        phi, Cn = as_tensor(phi), as_tensor(Cn)
        sigma, a_R = as_tensor(sigma), as_tensor(a_R)

        sigma_gpa = sigma / 1000.0  # MPa to GPa
        nu = (3.0 * K0 - 2.0 * G0) / (6.0 * K0 + 2.0 * G0 + EPS)

        # Solve cubic: x^3 + (3/2)*a_R^2*x - 3*pi*(1-nu)*sigma/(2*Cn*(1-phi)*G0) = 0
        # Using Newton iteration
        rhs = 3.0 * PI * (1.0 - nu) * sigma_gpa / (2.0 * Cn * (1.0 - phi) * G0 + EPS)
        # Initial guess (use the dominant term approximation)
        x = rhs ** (1.0 / 3.0)

        for _ in range(20):
            fx = x ** 3 + 1.5 * a_R ** 2 * x - rhs
            dfx = 3.0 * x ** 2 + 1.5 * a_R ** 2 + EPS
            x = x - fx / dfx
            x = torch.clamp(x, min=EPS)

        b_R = torch.sqrt(x ** 2 + a_R ** 2)
        Sn_R = 4.0 * G0 * b_R / (1.0 - nu + EPS)
        St_R = 8.0 * G0 * a_R / (2.0 - nu + EPS)
        Keff = Cn * (1.0 - phi) * Sn_R / (12.0 * PI)
        Geff = Cn * (1.0 - phi) * (Sn_R + 1.5 * St_R) / (20.0 * PI)
        return Keff, Geff


@register(name="ConstantCement", category="granular")
class ConstantCement(GranularModel):
    """
    Constant cement model (Avseth et al. 2000).

    Computes dry elastic moduli at a reference cemented porosity,
    then interpolates to target porosity using HS bounds.

    References:
        Avseth, P., Dvorkin, J., Mavko, G., & Rykkje, J. (2000). Rock physics
            diagnostic of North Sea sands: Link between microstructure and
            seismic properties. Geophysical Research Letters, 27(17), 2761-2764.

    Args:
        None (stateless model)
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "Constant cement model")

    def forward(
            self,
            phi_b: Tensor,
            K0: Tensor,
            G0: Tensor,
            Kc: Tensor,
            Gc: Tensor,
            phi: Tensor,
            phi_c: Tensor,
            Cn: Tensor,
            scheme: int = 1
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute dry elastic moduli.

        Args:
            phi_b: Adjusted high-porosity end member
            K0: Grain bulk modulus (GPa)
            G0: Grain shear modulus (GPa)
            Kc: Cement bulk modulus (GPa)
            Gc: Cement shear modulus (GPa)
            phi: Target porosity (fraction)
            phi_c: Critical porosity
            Cn: Coordination number
            scheme: Cement deposition scheme (1=contacts, 2=surfaces)

        Returns:
            Tuple of (K_dry, G_dry) in GPa
        """
        phi_b = as_tensor(phi_b)
        K0, G0 = as_tensor(K0), as_tensor(G0)
        Kc, Gc = as_tensor(Kc), as_tensor(Gc)
        phi, phi_c, Cn = as_tensor(phi), as_tensor(phi_c), as_tensor(Cn)

        # Get moduli at reference porosity using contact cement
        nu_0 = (3.0 * K0 - 2.0 * G0) / (6.0 * K0 + 2.0 * G0 + EPS)
        nu_c = (3.0 * Kc - 2.0 * Gc) / (6.0 * Kc + 2.0 * Gc + EPS)

        if scheme == 1:
            alpha = 2.0 * ((phi_c - phi_b) / (3.0 * Cn * (1.0 - phi_c) + EPS)) ** 0.25
        else:
            alpha = ((2.0 * (phi_c - phi_b)) / (3.0 * (1.0 - phi_c) + EPS)) ** 0.5

        LambdaN = (2.0 * Gc * (1.0 - nu_0) * (1.0 - nu_c)) / (PI * G0 * (1.0 - 2.0 * nu_c) + EPS)
        N1 = -0.024153 * LambdaN ** (-1.3646)
        N2 = 0.20405 * LambdaN ** (-0.89008)
        N3 = 0.00024649 * LambdaN ** (-1.9864)
        Sn = N1 * alpha ** 2 + N2 * alpha + N3

        LambdaT = Gc / (PI * G0 + EPS)
        T1 = -1e-2 * (2.26 * nu_0 ** 2 + 2.07 * nu_0 + 2.3) * LambdaT ** (0.079 * nu_0 ** 2 + 0.1754 * nu_0 - 1.342)
        T2 = (0.0573 * nu_0 ** 2 + 0.0937 * nu_0 + 0.202) * LambdaT ** (0.0274 * nu_0 ** 2 + 0.0529 * nu_0 - 0.8765)
        T3 = 1e-4 * (9.654 * nu_0 ** 2 + 4.945 * nu_0 + 3.1) * LambdaT ** (0.01867 * nu_0 ** 2 + 0.4011 * nu_0 - 1.8186)
        St = T1 * alpha ** 2 + T2 * alpha + T3

        K_b = 1.0 / 6.0 * Cn * (1.0 - phi_c) * (Kc + 4.0 / 3.0 * Gc) * Sn
        G_b = 3.0 / 5.0 * K_b + 3.0 / 20.0 * Cn * (1.0 - phi_c) * Gc * St

        # Interpolate from reference to target porosity (HS lower bound)
        T = phi / (phi_b + EPS)
        Z = G_b / 6.0 * (9.0 * K_b + 8.0 * G_b) / (K_b + 2.0 * G_b + EPS)
        K_dry = (T / (K_b + 4.0 * G_b / 3.0 + EPS) + (1.0 - T) / (K0 + 4.0 * G_b / 3.0 + EPS)) ** (-1) - 4.0 * G_b / 3.0
        G_dry = (T / (G_b + Z + EPS) + (1.0 - T) / (G0 + Z + EPS)) ** (-1) - Z

        return K_dry, G_dry


@register(name="VPCM", category="granular")
class VPCM(GranularModel):
    """
    Varying Patchiness Cement Model (Yu et al. 2023).

    Mixes soft and stiff sand end members with a stress-dependent
    diluting parameter to model varying degrees of cementation.

    References:
        Avseth, P., Skjei, N., & Mavko, G. (2016). Rock-physics modeling of
            stress sensitivity and 4D time shifts in patchy cemented sandstones.
            The Leading Edge, 35(10), 868-878.

    Args:
        None (stateless model)
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "Varying patchiness cement model")

    def forward(
            self,
            alpha_d: Tensor,
            f: Tensor,
            sigma: Tensor,
            K0: Tensor,
            G0: Tensor,
            phi: Tensor,
            phi_c: Tensor,
            v_cem: Tensor,
            v_ci: Tensor,
            Kc: Tensor,
            Gc: Tensor,
            Cn: Tensor,
            scheme: int = 1,
            f_hm: float = 0.5
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute effective moduli with varying patchiness.

        Args:
            alpha_d: Diluting parameter (0=stiff, 1=soft)
            f: Volume fraction of cemented rock
            sigma: Effective stress (MPa)
            K0: Grain bulk modulus (GPa)
            G0: Grain shear modulus (GPa)
            phi: Porosity
            phi_c: Critical porosity
            v_cem: Cement fraction
            v_ci: Cement threshold for increasing cement model
            Kc: Cement bulk modulus (GPa)
            Gc: Cement shear modulus (GPa)
            Cn: Coordination number
            scheme: Cement deposition scheme (1 or 2)
            f_hm: Hertz-Mindlin slip factor

        Returns:
            Tuple of (K_DRY, G_DRY) in GPa
        """
        alpha_d = as_tensor(alpha_d)
        f, sigma = as_tensor(f), as_tensor(sigma)
        K0, G0 = as_tensor(K0), as_tensor(G0)
        phi, phi_c = as_tensor(phi), as_tensor(phi_c)
        v_cem, v_ci = as_tensor(v_cem), as_tensor(v_ci)
        Kc, Gc, Cn = as_tensor(Kc), as_tensor(Gc), as_tensor(Cn)

        # Hertz-Mindlin unconsolidated end member
        sigma_gpa = sigma / 1000.0
        nu = (3.0 * K0 - 2.0 * G0) / (6.0 * K0 + 2.0 * G0 + EPS)
        Kunc = (sigma_gpa * (Cn ** 2 * (1.0 - phi_c) ** 2 * G0 ** 2) / (18.0 * PI ** 2 * (1.0 - nu) ** 2 + EPS)) ** (1.0 / 3.0)
        Gunc = ((2.0 + 3.0 * f_hm - nu * (1.0 + 3.0 * f_hm)) / (5.0 * (2.0 - nu) + EPS)) * \
               ((sigma_gpa * (3.0 * Cn ** 2 * (1.0 - phi_c) ** 2 * G0 ** 2) / (2.0 * PI ** 2 * (1.0 - nu) ** 2 + EPS))) ** (1.0 / 3.0)

        # Cemented end member (contact cement)
        nu_0 = nu
        nu_c = (3.0 * Kc - 2.0 * Gc) / (6.0 * Kc + 2.0 * Gc + EPS)
        phi_cem = phi_c - v_cem

        if scheme == 1:
            alpha_cc = 2.0 * ((phi_c - phi_cem) / (3.0 * Cn * (1.0 - phi_c) + EPS)) ** 0.25
        else:
            alpha_cc = ((2.0 * (phi_c - phi_cem)) / (3.0 * (1.0 - phi_c) + EPS)) ** 0.5

        LambdaN = (2.0 * Gc * (1.0 - nu_0) * (1.0 - nu_c)) / (PI * G0 * (1.0 - 2.0 * nu_c) + EPS)
        N1 = -0.024153 * LambdaN ** (-1.3646)
        N2 = 0.20405 * LambdaN ** (-0.89008)
        N3 = 0.00024649 * LambdaN ** (-1.9864)
        Sn = N1 * alpha_cc ** 2 + N2 * alpha_cc + N3

        LambdaT = Gc / (PI * G0 + EPS)
        T1 = -1e-2 * (2.26 * nu_0 ** 2 + 2.07 * nu_0 + 2.3) * LambdaT ** (0.079 * nu_0 ** 2 + 0.1754 * nu_0 - 1.342)
        T2 = (0.0573 * nu_0 ** 2 + 0.0937 * nu_0 + 0.202) * LambdaT ** (0.0274 * nu_0 ** 2 + 0.0529 * nu_0 - 0.8765)
        T3 = 1e-4 * (9.654 * nu_0 ** 2 + 4.945 * nu_0 + 3.1) * LambdaT ** (0.01867 * nu_0 ** 2 + 0.4011 * nu_0 - 1.8186)
        St = T1 * alpha_cc ** 2 + T2 * alpha_cc + T3

        Kcem = 1.0 / 6.0 * Cn * (1.0 - phi_c) * (Kc + 4.0 / 3.0 * Gc) * Sn
        Gcem = 3.0 / 5.0 * Kcem + 3.0 / 20.0 * Cn * (1.0 - phi_c) * Gc * St

        # Stiff mixing (HS upper)
        Kp_stiff = Kcem + (1.0 - f) / ((Kunc - Kcem) ** (-1) + f * (Kcem + 4.0 * Gcem / 3.0) ** (-1) + EPS)
        Temp_stiff = (Kcem + 2.0 * Gcem) / (5.0 * Gcem * (Kcem + 4.0 * Gcem / 3.0) + EPS)
        Gp_stiff = Gcem + (1.0 - f) / ((Gunc - Gcem) ** (-1) + 2.0 * f * Temp_stiff + EPS)

        # Soft mixing (HS lower)
        Kp_soft = Kunc + f / ((Kcem - Kunc) ** (-1) + (1.0 - f) * (Kunc + 4.0 * Gunc / 3.0) ** (-1) + EPS)
        Temp_soft = (Kunc + 2.0 * Gunc) / (5.0 * Gunc * (Kunc + 4.0 * Gunc / 3.0) + EPS)
        Gp_soft = Gunc + f / ((Gcem - Gunc) ** (-1) + 2.0 * (1.0 - f) * Temp_soft + EPS)

        # Binary mixing with diluting parameter
        Kp = Kp_stiff * (1.0 - alpha_d) + Kp_soft * alpha_d
        Gp = Gp_stiff * (1.0 - alpha_d) + Gp_soft * alpha_d

        # Second HS interpolation to mineral point
        K_DRY = -4.0 / 3.0 * Gp + ((phi / phi_c) / (Kp + 4.0 / 3.0 * Gp + EPS) + (1.0 - phi / phi_c) / (K0 + 4.0 / 3.0 * Gp + EPS)) ** (-1)
        tmp = Gp / 6.0 * (9.0 * Kp + 8.0 * Gp) / (Kp + 2.0 * Gp + EPS)
        G_DRY = -tmp + ((phi / phi_c) / (Gp + tmp + EPS) + (1.0 - phi / phi_c) / (G0 + tmp + EPS)) ** (-1)

        return K_DRY, G_DRY


@register(name="Diluting", category="granular")
class Diluting(GranularModel):
    """
    Stress-dependent diluting parameter for VPCM.

    Computes the diluting parameter alpha that controls the
    balance between stiff and soft mixing in VPCM:

        alpha = k * (1 - sigma/sigma0)^m

    References:
        Marion, D., Nur, A., Yin, H., & Han, D. (1992). Compressional velocity
            and porosity in sand-clay mixtures. Geophysics, 57(4), 554-563.

    Args:
        None (stateless model)
    """

    def __init__(self):
        super().__init__()
        self.set_metadata("description", "Stress-dependent diluting parameter")

    def forward(
            self,
            k: Tensor,
            sigma0: Tensor,
            sigma: Tensor,
            m: Tensor
    ) -> Tensor:
        """
        Compute diluting parameter.

        Args:
            k: Cement crushing factor (k<=1: no crumbling, k>1: crumbling)
            sigma0: Reference stress (e.g., max effective stress)
            sigma: Effective stress
            m: Curvature parameter

        Returns:
            Stress-dependent diluting parameter alpha
        """
        k, sigma0 = as_tensor(k), as_tensor(sigma0)
        sigma, m = as_tensor(sigma), as_tensor(m)
        ratio = 1.0 - sigma / (sigma0 + EPS)
        ratio = torch.clamp(ratio, min=0.0)
        alpha = k * ratio ** m
        return alpha