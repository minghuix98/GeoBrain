#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified rock physics workflow.

This module provides a high-level interface for computing seismic
velocities from rock composition and conditions, integrating mineral
mixing, dry rock modeling, and fluid substitution.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
from torch import Tensor
from typing import Tuple, Optional, Dict, Any, Union

from .core import CompositeModel, as_tensor, EPS, ModelRegistry
from .config import RockPhysicsConfig, get_preset, PRESETS
from .data import get_mineral, get_fluid, mix_minerals_vrh
from .utils import v_from_moduli, validate_porosity
from .core import defaults


class RockPhysicsWorkflow(CompositeModel):
    """
    Unified rock physics workflow.

    Provides an end-to-end pipeline for computing seismic velocities
    from rock composition and conditions:

        Input (φ, Sw, P) → Mineral Mixing → Dry Rock → Fluid Sub → Output (Vp, Vs, ρ)

    Supports multiple dry rock models (SoftSand, StiffSand, DEM, CriticalPorosity)
    and fluid mixing methods (Wood, Brie).

    Args:
        config: RockPhysicsConfig or dictionary with configuration

    Attributes:
        config: The RockPhysicsConfig instance
        _dry_rock: Dry rock model component
        _fluid_sub: Fluid substitution component
        _fluid_mix: Fluid mixing component

    Example:
        From preset:
        >>> workflow = RockPhysicsWorkflow.from_preset('shaly_sand')
        >>> Vp, Vs, rho = workflow(phi=0.2, Sw=0.8, P=30.0)
        >>> print(f"Vp = {Vp.item():.0f} m/s")

        Custom configuration:
        >>> config = RockPhysicsConfig(
        ...     minerals={'quartz': 0.8, 'clay': 0.2},
        ...     fluids={'water': 0.7, 'gas': 0.3},
        ...     models={'dry_rock': 'StiffSand', 'fluid_mix': 'Brie'}
        ... )
        >>> workflow = RockPhysicsWorkflow(config)
        >>> Vp, Vs, rho = workflow(phi=0.15)

        Batch computation with autodiff:
        >>> phi = torch.linspace(0.05, 0.35, 100, requires_grad=True)
        >>> Vp, Vs, rho = workflow(phi=phi)
        >>> loss = Vp.mean()
        >>> loss.backward()
    """

    def __init__(self, config: Union[RockPhysicsConfig, Dict[str, Any]]):
        super().__init__()

        if isinstance(config, dict):
            config = RockPhysicsConfig.from_dict(config)
        self.config = config
        self._setup_components()

    def _setup_components(self) -> None:
        """Initialize model components based on configuration."""
        # Dry rock model
        dry_rock_name = self.config.models.get('dry_rock', 'SoftSand')
        dry_rock_params = self._get_model_params(dry_rock_name)
        self._dry_rock = ModelRegistry.create_model(dry_rock_name, **dry_rock_params)

        # Fluid substitution model
        fluid_sub_name = self.config.models.get('fluid_sub', 'Gassmann')
        self._fluid_sub = ModelRegistry.create_model(fluid_sub_name)

        # Fluid mixing model
        fluid_mix_name = self.config.models.get('fluid_mix', 'Wood')
        self._fluid_mix = ModelRegistry.create_model(fluid_mix_name)

    def _get_model_params(self, model_name: str) -> Dict[str, Any]:
        """Get model-specific constructor parameters from config."""
        params = {}
        if model_name == 'CriticalPorosity':
            params['phi_c'] = self.config.parameters.get('phi_c', defaults.PHI_C)
        elif model_name == 'DEM':
            if 'n_steps' in self.config.parameters:
                params['n_steps'] = self.config.parameters['n_steps']
        elif model_name in ('ContactCement', 'ContactCementFull', 'MUHS', 'PCM'):
            if 'scheme' in self.config.parameters:
                params['scheme'] = self.config.parameters['scheme']
        elif model_name == 'Brie':
            params['e'] = self.config.parameters.get('e', defaults.BRIE_E)
        return params

    def _compute_fluid(self, Sw: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Compute effective fluid properties for N fluids.

        Uses Wood mixing for any number of fluids. For two fluids,
        Brie mixing is used if configured.

        Args:
            Sw: Water saturation (optional override for first fluid fraction)

        Returns:
            Tuple of (K_fluid, rho_fluid)
        """
        names = list(self.config.fluids.keys())

        # Single fluid case
        if len(names) == 1:
            fl = get_fluid(names[0])
            return as_tensor(fl.K), as_tensor(fl.rho)

        # Two-fluid case: may use Brie
        if len(names) == 2:
            fl1, fl2 = get_fluid(names[0]), get_fluid(names[1])
            if Sw is None:
                Sw = as_tensor(self.config.fluids.get(names[0], 0.5))
            return self._fluid_mix.compute_fluid_mix(
                fl1.K, fl2.K, fl1.rho, fl2.rho, Sw
            )

        # N-fluid case: always use Wood (Reuss average)
        K_vals = []
        rho_vals = []
        f_vals = []
        for name in names:
            fl = get_fluid(name)
            K_vals.append(fl.K)
            rho_vals.append(fl.rho)
            f_vals.append(self.config.fluids[name])

        K = as_tensor(K_vals)
        rho = as_tensor(rho_vals)
        f = as_tensor(f_vals)

        # Override first fluid fraction with Sw if provided
        if Sw is not None:
            Sw = as_tensor(Sw)
            remaining = 1.0 - Sw
            f_rest_sum = f[1:].sum()
            if f_rest_sum > 0:
                f = torch.cat([Sw.unsqueeze(0), f[1:] / f_rest_sum * remaining])
            else:
                f = torch.cat([Sw.unsqueeze(0), torch.zeros(len(names) - 1)])

        K_eff = 1.0 / torch.sum(f / (K + 1e-10))
        rho_eff = torch.sum(f * rho)
        return K_eff, rho_eff

    def _compute_minerals(self) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute effective mineral properties using configured mixer.

        Supports 'VRH' (default), 'Voigt', and 'Reuss' mixing methods.

        Returns:
            Tuple of (K0, G0, rho0): mineral moduli (GPa) and density (g/cm3)
        """
        mixer = self.config.models.get('mineral_mixer', 'VRH')

        names = list(self.config.minerals.keys())
        fractions = list(self.config.minerals.values())

        K_list, G_list, rho_list = [], [], []
        for name in names:
            m = get_mineral(name)
            K_list.append(m.K)
            G_list.append(m.G)
            rho_list.append(m.rho)

        K = as_tensor(K_list)
        G = as_tensor(G_list)
        rho = as_tensor(rho_list)
        f = as_tensor(fractions)

        rho0 = torch.sum(f * rho)

        if mixer == 'VRH':
            K0 = 0.5 * (torch.sum(f * K) + 1.0 / torch.sum(f / (K + 1e-10)))
            G0 = 0.5 * (torch.sum(f * G) + 1.0 / torch.sum(f / (G + 1e-10)))
        elif mixer == 'Voigt':
            K0 = torch.sum(f * K)
            G0 = torch.sum(f * G)
        elif mixer == 'Reuss':
            K0 = 1.0 / torch.sum(f / (K + 1e-10))
            G0 = 1.0 / torch.sum(f / (G + 1e-10))
        else:
            raise ValueError(f"Unknown mineral mixer: {mixer}. Use 'VRH', 'Voigt', or 'Reuss'.")

        return K0, G0, rho0

    def forward(
            self,
            phi: Union[float, Tensor],
            Sw: Optional[Union[float, Tensor]] = None,
            P: Optional[float] = None,
            **kwargs
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute seismic velocities for given porosity and saturation.

        Args:
            phi: Porosity (fraction, 0-1). Can be scalar or tensor for batch.
            Sw: Water saturation (fraction, 0-1). If None, uses config default.
            P: Effective pressure (MPa). If None, uses config default.
            **kwargs: Additional parameters passed to dry rock model.

        Returns:
            Tuple of (Vp, Vs, rho_bulk):
                - Vp: P-wave velocity (m/s)
                - Vs: S-wave velocity (m/s)
                - rho_bulk: Bulk density (g/cm³)

        Example:
            Single point:
            >>> Vp, Vs, rho = workflow(phi=0.2)

            Batch computation:
            >>> phi = torch.linspace(0.05, 0.35, 100)
            >>> Vp, Vs, rho = workflow(phi=phi)
        """
        phi = as_tensor(phi)
        phi = validate_porosity(phi, clamp=True)

        # Get parameters (fallback to centralized defaults)
        phi_c = self.config.parameters.get('phi_c', defaults.PHI_C)
        Cn = self.config.parameters.get('Cn', defaults.CN)
        P = P or self.config.parameters.get('P', defaults.P)

        if Sw is not None:
            Sw = as_tensor(Sw)
            Sw = validate_porosity(Sw, name="Sw", clamp=True)

        # Mineral mixing
        K0, G0, rho0 = self._compute_minerals()

        # Dry rock moduli
        K_dry, G_dry = self._dry_rock.compute_dry_rock(
            K0, G0, phi, phi_c=phi_c, Cn=Cn, P=P, **kwargs
        )

        # Fluid properties
        K_fl, rho_fl = self._compute_fluid(Sw)

        # Gassmann fluid substitution
        K_sat, G_sat = self._fluid_sub(K_dry, G_dry, K0, K_fl, phi)

        # Bulk density
        rho_bulk = (1 - phi) * rho0 + phi * rho_fl

        # Velocities
        Vp, Vs = v_from_moduli(K_sat, G_sat, rho_bulk)

        return Vp, Vs, rho_bulk

    @classmethod
    def from_preset(cls, name: str, **overrides) -> 'RockPhysicsWorkflow':
        """
        Create workflow from a preset configuration.

        Args:
            name: Preset name (e.g., 'shaly_sand', 'gas_sand', 'limestone')
            **overrides: Parameter overrides to apply to the preset

        Returns:
            Configured RockPhysicsWorkflow instance

        Example:
            >>> workflow = RockPhysicsWorkflow.from_preset('gas_sand', P=40.0)
            >>> workflow = RockPhysicsWorkflow.from_preset(
            ...     'shaly_sand',
            ...     minerals={'quartz': 0.6, 'clay': 0.4}
            ... )
        """
        config = get_preset(name)
        if overrides:
            config = config.with_overrides(**overrides)
        return cls(config)

    @staticmethod
    def list_presets() -> list:
        """
        List available preset names.

        Returns:
            List of preset names

        Example:
            >>> RockPhysicsWorkflow.list_presets()
            ['cemented_sand', 'chalk', 'clean_sand', ...]
        """
        return list(PRESETS.keys())

    def __repr__(self) -> str:
        minerals = ', '.join(f"{k}:{v:.0%}" for k, v in self.config.minerals.items())
        return (
            f"RockPhysicsWorkflow("
            f"minerals=[{minerals}], "
            f"model={self.config.models.get('dry_rock', 'SoftSand')})"
        )