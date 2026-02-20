"""
Elastic velocity models for wave simulation.

Implements isotropic elastic model with P-wave velocity, S-wave velocity,
and density parameters. Computes Lamé parameters for elastic wave equation.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Union

from ..core import AbstractModel, GridConfig, BoundaryConfig
from ..utils import numpy2tensor


class IsotropicElasticModel(AbstractModel):
    """
    Isotropic elastic velocity model.
    
    Manages P-wave velocity (vp), S-wave velocity (vs), and density (rho)
    for elastic wave equation simulation. Automatically computes Lamé
    parameters (λ, μ) from velocities.
    
    Args:
        grid: Grid configuration.
        boundary: Boundary condition configuration.
        vp: P-wave velocity array (m/s) with shape (nz, nx).
        vs: S-wave velocity array (m/s) with shape (nz, nx).
        rho: Density array (kg/m³) with shape (nz, nx).
        vp_bound: P-velocity bounds (min, max). Default: None.
        vs_bound: S-velocity bounds (min, max). Default: None.
        rho_bound: Density bounds (min, max). Default: None.
        vp_grad: Enable gradient for vp. Default: False.
        vs_grad: Enable gradient for vs. Default: False.
        rho_grad: Enable gradient for rho. Default: False.
        auto_update_rho: Update rho from vp using Gardner's relation.
            Default: True.
        water_layer_mask: Boolean mask for water layer. Default: None.
        device: Computation device. Default: 'cpu'.
        dtype: Data type. Default: torch.float32.

    Attributes:
        mu: Shear modulus μ = vs² × ρ (Pa).
        lamu: P-wave modulus λ + 2μ = vp² × ρ (Pa).
        lam: First Lamé parameter λ = (λ + 2μ) - 2μ (Pa).
        b: Buoyancy 1/ρ (m³/kg).

    Example:
        >>> grid = GridConfig(nx=200, nz=100, dx=10, dz=10)
        >>> boundary = BoundaryConfig(type='pml', n_layers=20)
        >>> model = IsotropicElasticModel(
        ...     grid=grid,
        ...     boundary=boundary,
        ...     vp=vp_array,
        ...     vs=vs_array,
        ...     rho=rho_array,
        ... )
        >>> model.forward()  # Computes Lamé parameters
        >>> print(model.mu.shape)  # (100, 200)

    Note:
        For fluid regions (water), set vs=0 and the model degenerates
        to acoustic behavior in those regions.
    """

    def __init__(
        self,
        grid: GridConfig,
        boundary: BoundaryConfig,
        vp: Union[np.ndarray, torch.Tensor],
        vs: Union[np.ndarray, torch.Tensor],
        rho: Union[np.ndarray, torch.Tensor],
        vp_bound: Optional[Tuple[float, float]] = None,
        vs_bound: Optional[Tuple[float, float]] = None,
        rho_bound: Optional[Tuple[float, float]] = None,
        vp_grad: bool = False,
        vs_grad: bool = False,
        rho_grad: bool = False,
        auto_update_rho: bool = True,
        water_layer_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(grid, boundary, device, dtype)

        # Parameter names
        self.pars = ['vp', 'vs', 'rho']

        # Gradient flags
        self.requires_grad['vp'] = vp_grad
        self.requires_grad['vs'] = vs_grad
        self.requires_grad['rho'] = rho_grad
        self.auto_update_rho = auto_update_rho

        # Initialize parameters
        self.vp = nn.Parameter(
            numpy2tensor(vp, dtype).to(device),
            requires_grad=vp_grad
        )
        self.vs = nn.Parameter(
            numpy2tensor(vs, dtype).to(device),
            requires_grad=vs_grad
        )
        self.rho = nn.Parameter(
            numpy2tensor(rho, dtype).to(device),
            requires_grad=rho_grad
        )

        # Set bounds
        self.lower_bound['vp'] = vp_bound[0] if vp_bound else None
        self.upper_bound['vp'] = vp_bound[1] if vp_bound else None
        self.lower_bound['vs'] = vs_bound[0] if vs_bound else None
        self.upper_bound['vs'] = vs_bound[1] if vs_bound else None
        self.lower_bound['rho'] = rho_bound[0] if rho_bound else None
        self.upper_bound['rho'] = rho_bound[1] if rho_bound else None

        # Water mask
        if water_layer_mask is not None:
            self.water_layer_mask = numpy2tensor(
                water_layer_mask, torch.bool
            ).to(device)
        else:
            self.water_layer_mask = None

        # Derived Lamé parameters (computed in forward)
        self.mu: Optional[torch.Tensor] = None
        self.lamu: Optional[torch.Tensor] = None
        self.lam: Optional[torch.Tensor] = None
        self.b: Optional[torch.Tensor] = None

        # Validate
        self._validate()

    def _validate(self) -> None:
        """Validate model configuration."""
        for par in self.pars:
            model = getattr(self, par)
            if model.shape != self.shape:
                raise ValueError(
                    f"Shape mismatch for {par}: expected {self.shape}, "
                    f"got {tuple(model.shape)}"
                )

    def _compute_lame(self) -> None:
        """
        Compute Lamé parameters from velocities and density.

        Relations:
            μ (shear modulus) = vs² × ρ
            λ + 2μ (P-wave modulus) = vp² × ρ
            λ (first Lamé parameter) = (λ + 2μ) - 2μ
            b (buoyancy) = 1/ρ
        """
        self.mu = self.vs ** 2 * self.rho
        self.lamu = self.vp ** 2 * self.rho
        self.lam = self.lamu - 2 * self.mu
        self.b = 1.0 / self.rho

    def forward(self) -> None:
        """
        Prepare model for wave propagation.

        Performs:
            1. Clip parameters to bounds
            2. Compute Lamé parameters from velocities
        """
        self.clip_params()
        self._compute_lame()

    def get_poisson_ratio(self) -> np.ndarray:
        """
        Compute Poisson's ratio from velocities.

        The formula is:

            ν = (vp² - 2vs²) / (2(vp² - vs²))

        Returns:
            Poisson's ratio array with shape (nz, nx).
        """
        vp = self.vp.detach().cpu().numpy()
        vs = self.vs.detach().cpu().numpy()

        vp2 = vp ** 2
        vs2 = vs ** 2

        # Avoid division by zero
        denom = 2 * (vp2 - vs2)
        denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)

        nu = (vp2 - 2 * vs2) / denom
        return nu

    def get_vp_vs_ratio(self) -> np.ndarray:
        """
        Compute Vp/Vs ratio.

        Returns:
            Vp/Vs ratio array with shape (nz, nx).

        Note:
            For fluid regions where vs ≈ 0, returns a large value
            to avoid division by zero.
        """
        vp = self.vp.detach().cpu().numpy()
        vs = self.vs.detach().cpu().numpy()

        # Avoid division by zero for fluid regions
        vs_safe = np.where(vs < 1e-6, 1e-6, vs)
        return vp / vs_safe