"""
Acoustic velocity model for wave simulation.

Implements the acoustic approximation where only P-wave velocity
and density are considered, suitable for marine seismic modeling.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Union

from ..core import AbstractModel, GridConfig, BoundaryConfig
from ..utils import numpy2tensor


class AcousticModel(AbstractModel):
    """
    Acoustic velocity model.
    
    Manages P-wave velocity (vp) and density (rho) parameters for
    acoustic wave equation simulation. Supports automatic density
    updates using Gardner's empirical relation.
    
    Args:
        grid: Grid configuration.
        boundary: Boundary condition configuration.
        vp: P-wave velocity array (m/s) with shape (nz, nx).
        rho: Density array (kg/m³) with shape (nz, nx).
        vp_bound: Velocity bounds (min, max). Default: None.
        rho_bound: Density bounds (min, max). Default: None.
        vp_grad: Enable gradient for vp. Default: False.
        rho_grad: Enable gradient for rho. Default: False.
        auto_update_rho: Update rho from vp using Gardner's relation.
            Only applied when rho_grad=False. Default: True.
        water_layer_mask: Boolean mask for water layer where rho
            should not be updated. Default: None.
        device: Computation device. Default: 'cpu'.
        dtype: Data type. Default: torch.float32.
    
    Example:
        Basic usage:
        >>> grid = GridConfig(nx=200, nz=100, dx=10, dz=10)
        >>> boundary = BoundaryConfig(type='pml', n_layers=20)
        >>> model = AcousticModel(
        ...     grid=grid,
        ...     boundary=boundary,
        ...     vp=vp_array,
        ...     rho=rho_array,
        ...     vp_grad=True,  # Enable gradient for FWI
        ... )
        >>> model.forward()  # Prepare for simulation
        
        With bounds for inversion:
        >>> model = AcousticModel(
        ...     grid=grid,
        ...     boundary=boundary,
        ...     vp=vp_array,
        ...     rho=rho_array,
        ...     vp_bound=(1500, 5000),
        ...     vp_grad=True,
        ... )
    
    Note:
        Gardner's relation: ρ = 310 × vp^0.25
    """

    def __init__(
        self,
        grid: GridConfig,
        boundary: BoundaryConfig,
        vp: Union[np.ndarray, torch.Tensor],
        rho: Union[np.ndarray, torch.Tensor],
        vp_bound: Optional[Tuple[float, float]] = None,
        rho_bound: Optional[Tuple[float, float]] = None,
        vp_grad: bool = False,
        rho_grad: bool = False,
        auto_update_rho: bool = True,
        water_layer_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(grid, boundary, device, dtype)

        # Parameter names
        self.pars = ['vp', 'rho']

        # Gradient flags
        self.vp_grad = vp_grad
        self.rho_grad = rho_grad
        self.requires_grad['vp'] = vp_grad
        self.requires_grad['rho'] = rho_grad

        # Auto-update settings
        self.auto_update_rho = auto_update_rho

        # Store original copies for reset
        self._vp_init = vp.copy() if isinstance(vp, np.ndarray) else vp.clone()
        self._rho_init = rho.copy() if isinstance(rho, np.ndarray) else rho.clone()

        # Initialize parameters
        self._init_parameters(vp, rho)

        # Set bounds
        self.lower_bound['vp'] = vp_bound[0] if vp_bound else None
        self.upper_bound['vp'] = vp_bound[1] if vp_bound else None
        self.lower_bound['rho'] = rho_bound[0] if rho_bound else None
        self.upper_bound['rho'] = rho_bound[1] if rho_bound else None

        # Water layer mask
        if water_layer_mask is not None:
            self.water_layer_mask = numpy2tensor(
                water_layer_mask, torch.bool
            ).to(device)
        else:
            self.water_layer_mask = None

        # Validate
        self._validate()

    def _init_parameters(
        self,
        vp: Union[np.ndarray, torch.Tensor],
        rho: Union[np.ndarray, torch.Tensor],
    ) -> None:
        """Initialize model parameters as nn.Parameter tensors."""
        vp_tensor = numpy2tensor(vp, self.dtype).to(self.device)
        rho_tensor = numpy2tensor(rho, self.dtype).to(self.device)

        self.vp = nn.Parameter(vp_tensor, requires_grad=self.vp_grad)
        self.rho = nn.Parameter(rho_tensor, requires_grad=self.rho_grad)

    def _validate(self) -> None:
        """Validate model configuration."""
        # Check bounds
        for par in self.pars:
            lb = self.lower_bound.get(par)
            ub = self.upper_bound.get(par)
            if lb is not None and ub is not None and lb >= ub:
                raise ValueError(
                    f"Invalid bounds for {par}: lower={lb} >= upper={ub}"
                )

        # Check dimensions
        for par in self.pars:
            model = getattr(self, par)
            if model.shape != self.shape:
                raise ValueError(
                    f"Shape mismatch for {par}: expected {self.shape}, "
                    f"got {tuple(model.shape)}"
                )

    def set_rho_empirical(self) -> None:
        """
        Update density from velocity using Gardner's relation.

        Gardner's relation: ρ = 310 × vp^0.25

        Water layer regions (if masked) retain their original density.
        """
        with torch.no_grad():
            vp_np = self.vp.detach().cpu().numpy()
            rho_np = self.rho.detach().cpu().numpy()

            # Gardner's relation
            rho_emp = 310.0 * np.power(vp_np, 0.25)

            # Preserve water layer density
            if self.water_layer_mask is not None:
                mask = self.water_layer_mask.cpu().numpy()
                rho_emp[mask] = rho_np[mask]

            # Update parameter
            self.rho = nn.Parameter(
                numpy2tensor(rho_emp, self.dtype).to(self.device),
                requires_grad=self.rho_grad
            )

    def forward(self) -> None:
        """
        Prepare model for wave propagation.

        Performs:
            1. Auto-update density if enabled and rho_grad=False
            2. Clip parameters to bounds
        """
        # Auto-update density
        if self.auto_update_rho and not self.rho_grad:
            self.set_rho_empirical()

        # Clip to bounds
        self.clip_params()

    def reset(self) -> None:
        """Reset model parameters to initial values."""
        self._init_parameters(self._vp_init, self._rho_init)