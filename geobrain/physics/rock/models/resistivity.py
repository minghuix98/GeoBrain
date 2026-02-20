#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Electromagnetic property modeling module.

This module provides implementations for calculating electromagnetic
properties of rock formations, including electrical resistivity.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Union

EPS = 1e-10


class ArchieResistivity(nn.Module):
    """
    Formation resistivity using Archie's law.

    Calculates true formation resistivity for clean (non-shaly) formations:

        R_t = a · R_w · φ^(-m) · S_w^(-n)

    where:
        - R_t: True formation resistivity (ohm·m)
        - R_w: Formation water resistivity (ohm·m)
        - φ: Porosity (fraction)
        - S_w: Water saturation (fraction)
        - a: Tortuosity factor (typically 0.62-1.0)
        - m: Cementation exponent (typically 1.8-2.2)
        - n: Saturation exponent (typically 1.8-2.2)

    Typical parameter values by lithology:
        - Sandstone: a=0.62-1.0, m=1.8-2.0
        - Carbonate: a=1.0, m=2.0-2.5
        - Unconsolidated sand: a=0.62, m=2.15 (Humble formula)

    Args:
        Rw: Water resistivity (ohm·m). Default: 0.17
        a: Tortuosity factor. Default: 1.0

    Attributes:
        Rw: Water resistivity
        a: Tortuosity factor

    Example:
        Basic usage:
        >>> archie = ArchieResistivity(Rw=0.05, a=1.0)
        >>> Rt = archie(poro=0.25, Sw=0.8, m=2.0, n=2.0)
        >>> print(f"Formation resistivity: {Rt.item():.2f} ohm·m")

        Batch computation:
        >>> import torch
        >>> poro = torch.linspace(0.1, 0.35, 50)
        >>> Sw = torch.ones(50) * 0.8
        >>> Rt = archie(poro, Sw, m=2.0, n=2.0)

        With autodiff for inversion:
        >>> poro = torch.tensor([0.25], requires_grad=True)
        >>> Rt = archie(poro, Sw=0.8, m=2.0, n=2.0)
        >>> Rt.backward()
        >>> print(f"dRt/dporo = {poro.grad.item():.2f}")

    Note:
        This model assumes clean (clay-free) formations. For shaly sands,
        consider using modified Archie equations (Simandoux, Indonesia, etc.)
        which account for clay conductivity.
    """

    def __init__(self, Rw: float = 0.17, a: float = 1.0):
        super().__init__()
        self.Rw = Rw
        self.a = a

    def forward(
            self,
            poro: Tensor,
            Sw: Tensor,
            m: Union[float, Tensor],
            n: Union[float, Tensor]
    ) -> Tensor:
        """
        Calculate formation resistivity using Archie's law.

        Args:
            poro: Porosity (fraction, 0-1)
            Sw: Water saturation (fraction, 0-1)
            m: Cementation exponent (typically 1.8-2.2)
            n: Saturation exponent (typically 1.8-2.2)

        Returns:
            Formation resistivity (ohm·m)

        Note:
            Small epsilon (EPS) is added to porosity and saturation
            to prevent numerical instability at extreme values.
        """
        # Ensure tensor format
        if not isinstance(poro, Tensor):
            poro = torch.tensor(poro)
        if not isinstance(Sw, Tensor):
            Sw = torch.tensor(Sw)

        # Archie's law: R_t = a * R_w * φ^(-m) * S_w^(-n)
        Rt = self.a * self.Rw * torch.pow(poro + EPS, -m) * torch.pow(Sw + EPS, -n)

        return Rt

    def formation_factor(self, poro: Tensor, m: Union[float, Tensor]) -> Tensor:
        """
        Calculate formation factor (F = a / φ^m).

        The formation factor relates formation resistivity to water resistivity:
            R_0 = F * R_w (for 100% water saturation)

        Args:
            poro: Porosity (fraction)
            m: Cementation exponent

        Returns:
            Formation factor (dimensionless)
        """
        if not isinstance(poro, Tensor):
            poro = torch.tensor(poro)
        return self.a * torch.pow(poro + EPS, -m)

    def resistivity_index(self, Sw: Tensor, n: Union[float, Tensor]) -> Tensor:
        """
        Calculate resistivity index (I = S_w^(-n)).

        The resistivity index relates true resistivity to water-saturated resistivity:
            R_t = I * R_0

        Args:
            Sw: Water saturation (fraction)
            n: Saturation exponent

        Returns:
            Resistivity index (dimensionless)
        """
        if not isinstance(Sw, Tensor):
            Sw = torch.tensor(Sw)
        return torch.pow(Sw + EPS, -n)

    def water_saturation(
            self,
            Rt: Tensor,
            poro: Tensor,
            m: Union[float, Tensor],
            n: Union[float, Tensor]
    ) -> Tensor:
        """
        Inverse Archie: calculate water saturation from resistivity.

        Rearranging Archie's equation:
            S_w = (a * R_w / (R_t * φ^m))^(1/n)

        Args:
            Rt: True formation resistivity (ohm·m)
            poro: Porosity (fraction)
            m: Cementation exponent
            n: Saturation exponent

        Returns:
            Water saturation (fraction, clamped to 0-1)

        Example:
            >>> Sw = archie.water_saturation(Rt=10.0, poro=0.25, m=2.0, n=2.0)
        """
        if not isinstance(Rt, Tensor):
            Rt = torch.tensor(Rt)
        if not isinstance(poro, Tensor):
            poro = torch.tensor(poro)

        Sw = torch.pow(
            self.a * self.Rw / (Rt * torch.pow(poro + EPS, m) + EPS),
            1.0 / n
        )
        return torch.clamp(Sw, min=0.0, max=1.0)

    def __repr__(self) -> str:
        return f"ArchieResistivity(Rw={self.Rw}, a={self.a})"