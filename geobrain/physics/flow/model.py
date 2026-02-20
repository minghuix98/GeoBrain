"""
Reservoir model module for reservoir simulation.

Provides ReservoirModel class,
serving as the central data container for reservoir simulation.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union

from .constants import DEVICE, DTYPE
from .grid import CartGrid, ConnList
from .rock import Rock
from .pvt import PVT, PVTTable, PVTAnalytic
from .relperm import RelPerm, RelPermTable, RelPermCorey
from .fluid import OilWaterFluid
from .well import Well, WellGroup


# =============================================================================
# Reservoir Model
# =============================================================================

class ReservoirModel(nn.Module):
    """
    Reservoir model for oil-water simulation.

    Central data container that holds grid, rock, fluid, and well
    information. Follows ADFWI-style design with method chaining.

    Args:
        nx: Number of cells in x-direction.
        ny: Number of cells in y-direction.
        nz: Number of cells in z-direction.
        device: Torch device.
        dtype: Torch dtype.

    Attributes:
        grid: Cartesian grid.
        rock: Rock properties.
        fluid: Oil-water fluid system.
        wells: Well group.

    Example:
        >>> model = ReservoirModel(nx=30, ny=15, nz=1)
        >>> model.set_grid(dx=100.0, dy=100.0, dz=20.0)
        >>> model.set_rock(perm=100.0, poro=0.2)
        >>> model.set_pvt(pvt_o=pvt_oil, pvt_w=pvt_water)
        >>> model.set_relperm(relperm=relperm_model)
        >>> model.add_well(producer)
        >>> model.initialize(po=4000.0, sw=0.2)
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        nz: int = 1,
        device: torch.device = DEVICE,
        dtype: torch.dtype = DTYPE
    ):
        super().__init__()

        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.nc = nx * ny * nz
        self.device = device
        self.dtype = dtype

        # Components
        self.grid: Optional[CartGrid] = None
        self.rock: Optional[Rock] = None
        self.fluid: Optional[OilWaterFluid] = None
        self.wells: WellGroup = WellGroup(device, dtype)

        # PVT and RelPerm models
        self._pvt_o: Optional[PVT] = None
        self._pvt_w: Optional[PVT] = None
        self._relperm: Optional[RelPerm] = None

        # Fluid densities
        self._rho_o_sc: float = 45.0
        self._rho_w_sc: float = 64.0

        # Initial conditions
        self._po_init: Optional[torch.Tensor] = None
        self._sw_init: Optional[torch.Tensor] = None

        self._initialized: bool = False

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def is_initialized(self) -> bool:
        """Check if model is initialized."""
        return self._initialized

    @property
    def connlist(self) -> ConnList:
        """Connection list."""
        if self.grid is None:
            raise ValueError("Grid not set")
        return self.grid.connlist

    @property
    def nconn(self) -> int:
        """Number of connections."""
        return self.connlist.nconn

    # -------------------------------------------------------------------------
    # Grid Configuration
    # -------------------------------------------------------------------------

    def set_grid(
        self,
        dx: Union[float, torch.Tensor],
        dy: Union[float, torch.Tensor],
        dz: Union[float, torch.Tensor],
        top_depth: float = 0.0
    ) -> 'ReservoirModel':
        """
        Set grid dimensions.

        Args:
            dx: Cell size in x-direction (ft).
            dy: Cell size in y-direction (ft).
            dz: Cell size in z-direction (ft).
            top_depth: Depth to reservoir top (ft).

        Returns:
            Self for method chaining.
        """
        self.grid = CartGrid(self.nx, self.ny, self.nz, self.device, self.dtype)
        self.grid.set_dimensions(dx, dy, dz, top_depth)
        return self

    # -------------------------------------------------------------------------
    # Rock Configuration
    # -------------------------------------------------------------------------

    def set_rock(
        self,
        perm: Union[float, torch.Tensor],
        poro: Union[float, torch.Tensor],
        perm_y: Optional[Union[float, torch.Tensor]] = None,
        perm_z: Optional[Union[float, torch.Tensor]] = None,
        comp: float = 0.0,
        pref: float = 0.0
    ) -> 'ReservoirModel':
        """
        Set rock properties.

        Args:
            perm: Permeability in x-direction (md).
            poro: Porosity (fraction).
            perm_y: Permeability in y-direction (md).
            perm_z: Permeability in z-direction (md).
            comp: Rock compressibility (1/psi).
            pref: Reference pressure (psi).

        Returns:
            Self for method chaining.
        """
        self.rock = Rock(self.nc, self.device, self.dtype)
        self.rock.set_permeability(perm, perm_y, perm_z)
        self.rock.set_porosity(poro)
        if comp > 0:
            self.rock.set_compressibility(comp, pref)

        if self.grid is not None:
            self.grid.build_connections(
                self.rock.perm_x,
                self.rock.perm_y,
                self.rock.perm_z
            )

        return self

    # -------------------------------------------------------------------------
    # PVT Configuration
    # -------------------------------------------------------------------------

    def set_pvt(
        self,
        pvt_o: Optional[PVT] = None,
        pvt_w: Optional[PVT] = None,
        pvdo: Optional[List[List[float]]] = None,
        pvtw: Optional[List[float]] = None,
        rho_o_sc: float = 45.0,
        rho_w_sc: float = 64.0
    ) -> 'ReservoirModel':
        """
        Set PVT models.

        Args:
            pvt_o: Oil PVT model.
            pvt_w: Water PVT model.
            pvdo: Oil PVT data [[p, Bo, μo], ...].
            pvtw: Water PVT data [pref, Bw, cw, μw, cμ].
            rho_o_sc: Oil density at SC (lb/ft³).
            rho_w_sc: Water density at SC (lb/ft³).

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If oil or water PVT is not provided.
        """
        if pvt_o is not None:
            self._pvt_o = pvt_o
        elif pvdo is not None:
            self._pvt_o = PVTTable.from_data(pvdo, self.device, self.dtype)
        else:
            raise ValueError("Oil PVT required")

        if pvt_w is not None:
            self._pvt_w = pvt_w
        elif pvtw is not None:
            self._pvt_w = PVTAnalytic.from_data(pvtw, self.device, self.dtype)
        else:
            raise ValueError("Water PVT required")

        self._rho_o_sc = rho_o_sc
        self._rho_w_sc = rho_w_sc

        return self

    # -------------------------------------------------------------------------
    # Relative Permeability Configuration
    # -------------------------------------------------------------------------

    def set_relperm(
        self,
        relperm: Optional[RelPerm] = None,
        swof: Optional[List[List[float]]] = None,
        corey: Optional[Dict] = None
    ) -> 'ReservoirModel':
        """
        Set relative permeability model.

        Args:
            relperm: RelPerm model.
            swof: SWOF data [[Sw, Krw, Kro, Pc], ...].
            corey: Corey parameters {swc, sor, nw, no, ...}.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If no relperm model is provided.
        """
        if relperm is not None:
            self._relperm = relperm
        elif swof is not None:
            self._relperm = RelPermTable.from_data(swof, self.device, self.dtype)
        elif corey is not None:
            self._relperm = RelPermCorey(
                swc=corey.get('swc', 0.2),
                sor=corey.get('sor', 0.2),
                krw_max=corey.get('krw_max', 1.0),
                kro_max=corey.get('kro_max', 1.0),
                nw=corey.get('nw', 2.0),
                no=corey.get('no', 2.0),
                device=self.device,
                dtype=self.dtype
            )
        else:
            raise ValueError("RelPerm required")

        return self

    # -------------------------------------------------------------------------
    # Well Configuration
    # -------------------------------------------------------------------------

    def add_well(self, well: Well) -> 'ReservoirModel':
        """
        Add a well.

        Args:
            well: Well to add.

        Returns:
            Self for method chaining.
        """
        self.wells.add_well(well)
        return self

    def get_well(self, name: str) -> Well:
        """
        Get well by name.

        Args:
            name: Well name.

        Returns:
            Well instance.
        """
        return self.wells.get_well(name)

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def initialize(
        self,
        po: Union[float, torch.Tensor],
        sw: Union[float, torch.Tensor]
    ) -> 'ReservoirModel':
        """
        Initialize the model.

        Args:
            po: Initial oil pressure (psi).
            sw: Initial water saturation.

        Returns:
            Self for method chaining.
        """
        self._validate_components()

        # Always build/rebuild connections to ensure consistency
        # regardless of set_grid() / set_rock() call order
        self.grid.build_connections(
            self.rock.perm_x,
            self.rock.perm_y,
            self.rock.perm_z
        )

        self._po_init = self._to_tensor(po)
        self._sw_init = self._to_tensor(sw)

        self.fluid = OilWaterFluid(
            nc=self.nc,
            nconn=self.nconn,
            rho_o=self._rho_o_sc,
            rho_w=self._rho_w_sc,
            pvt_o=self._pvt_o,
            pvt_w=self._pvt_w,
            relperm=self._relperm,
            device=self.device,
            dtype=self.dtype
        )

        self.fluid.initialize(self._po_init, self._sw_init)
        self.fluid.compute_properties(self.connlist)

        self._initialized = True
        return self

    def reset(self) -> 'ReservoirModel':
        """
        Reset to initial conditions.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If model not initialized.
        """
        if not self._initialized:
            raise ValueError("Model not initialized")

        self.fluid.initialize(self._po_init, self._sw_init)
        self.fluid.compute_properties(self.connlist)

        for well in self.wells.wells.values():
            well.reset_history()

        return self

    # -------------------------------------------------------------------------
    # State Access
    # -------------------------------------------------------------------------

    def get_state(self) -> Dict[str, torch.Tensor]:
        """
        Get current state.

        Returns:
            Dictionary with keys: po, sw, so, bo, bw, kro, krw.

        Raises:
            ValueError: If model not initialized.
        """
        if not self._initialized:
            raise ValueError("Model not initialized")

        return {
            'po': self.fluid.oil.p.clone(),
            'sw': self.fluid.water.s.clone(),
            'so': self.fluid.oil.s.clone(),
            'bo': self.fluid.oil.b.clone(),
            'bw': self.fluid.water.b.clone(),
            'kro': self.fluid.oil.kr.clone(),
            'krw': self.fluid.water.kr.clone()
        }

    def set_state(self, po: torch.Tensor, sw: torch.Tensor) -> None:
        """
        Set current state.

        Args:
            po: Oil pressure (psi), shape (nc,).
            sw: Water saturation, shape (nc,).

        Raises:
            ValueError: If model not initialized.
        """
        if not self._initialized:
            raise ValueError("Model not initialized")

        self.fluid.set_state(po, sw)
        self.fluid.compute_properties(self.connlist)

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def _validate_components(self) -> None:
        """Validate all required components are set."""
        missing = []
        if self.grid is None:
            missing.append("grid (call set_grid())")
        if self.rock is None:
            missing.append("rock (call set_rock())")
        if self._pvt_o is None or self._pvt_w is None:
            missing.append("pvt (call set_pvt())")
        if self._relperm is None:
            missing.append("relperm (call set_relperm())")
        if missing:
            raise ValueError(
                f"Model incomplete. Missing: {', '.join(missing)}"
            )

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_dict(cls, config: Dict) -> 'ReservoirModel':
        """
        Create model from config dictionary.

        Args:
            config: Configuration with grid, rock, pvt, relperm, wells, initial.

        Returns:
            Initialized ReservoirModel.
        """
        grid_cfg = config['grid']
        model = cls(
            nx=grid_cfg['nx'],
            ny=grid_cfg['ny'],
            nz=grid_cfg.get('nz', 1)
        )

        model.set_grid(
            dx=grid_cfg['dx'],
            dy=grid_cfg['dy'],
            dz=grid_cfg['dz'],
            top_depth=grid_cfg.get('top_depth', 0.0)
        )

        rock_cfg = config['rock']
        model.set_rock(
            perm=rock_cfg['perm'],
            poro=rock_cfg['poro'],
            perm_y=rock_cfg.get('perm_y'),
            perm_z=rock_cfg.get('perm_z'),
            comp=rock_cfg.get('comp', 0.0),
            pref=rock_cfg.get('pref', 0.0)
        )

        pvt_cfg = config['pvt']
        model.set_pvt(
            pvdo=pvt_cfg.get('pvdo'),
            pvtw=pvt_cfg.get('pvtw'),
            rho_o_sc=pvt_cfg.get('rho_o_sc', 45.0),
            rho_w_sc=pvt_cfg.get('rho_w_sc', 64.0)
        )

        relperm_cfg = config['relperm']
        if 'swof' in relperm_cfg:
            model.set_relperm(swof=relperm_cfg['swof'])
        elif 'corey' in relperm_cfg:
            model.set_relperm(corey=relperm_cfg['corey'])

        if 'wells' in config:
            for well_cfg in config['wells']:
                well = Well(
                    name=well_cfg['name'],
                    well_type=well_cfg.get('type', 'PROD'),
                    device=model.device,
                    dtype=model.dtype
                )
                for perf in well_cfg.get('perforations', []):
                    well.add_perforation(
                        cell_idx=perf['cell_idx'],
                        wi=perf['wi'],
                        depth=perf.get('depth', 0.0)
                    )
                if 'control' in well_cfg:
                    ctrl = well_cfg['control']
                    well.set_control(
                        control_type=ctrl['type'],
                        target=ctrl['target'],
                        phase=ctrl.get('phase', 'LIQ'),
                        bhp_limit=ctrl.get('bhp_limit')
                    )
                model.add_well(well)

        init_cfg = config['initial']
        model.initialize(po=init_cfg['po'], sw=init_cfg['sw'])

        return model

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _to_tensor(self, value: Union[float, torch.Tensor]) -> torch.Tensor:
        """Convert scalar to tensor."""
        if isinstance(value, (int, float)):
            return torch.full(
                (self.nc,), float(value),
                device=self.device, dtype=self.dtype
            )
        return value.to(device=self.device, dtype=self.dtype)

    def forward(self) -> Dict[str, torch.Tensor]:
        """
        Forward pass returns current state.

        Returns:
            Current state dictionary.
        """
        return self.get_state()

    def __call__(self) -> Dict[str, torch.Tensor]:
        """
        Shortcut for get_state() - enables model() syntax.

        Returns:
            Current state dictionary with keys: po, sw, so, bo, bw, kro, krw.

        Raises:
            ValueError: If model not initialized.
        """
        return self.get_state()

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not initialized"
        n_wells = self.wells.nwell if self.wells else 0
        return (
            f"ReservoirModel(nx={self.nx}, ny={self.ny}, nz={self.nz}, "
            f"nc={self.nc}, wells={n_wells}, {status})"
        )
