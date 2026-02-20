"""
Grid module for reservoir simulation.

Provides Cartesian grid and connection list classes for
structured reservoir simulation domains.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Tuple
from dataclasses import dataclass

from .constants import DEVICE, DTYPE, ALPHA


# =============================================================================
# Connection List
# =============================================================================

@dataclass
class ConnList:
    """
    Connection list for cell-to-cell transmissibilities.

    Stores connectivity information and precomputed transmissibilities
    for efficient flux calculations.

    Attributes:
        left: Left cell indices for each connection.
        right: Right cell indices for each connection.
        trans: Transmissibility for each connection (md·ft).
        delta_depth: Depth difference (right - left) for gravity (ft).
    """
    left: torch.Tensor
    right: torch.Tensor
    trans: torch.Tensor
    delta_depth: torch.Tensor

    @property
    def nconn(self) -> int:
        """Number of connections."""
        return self.left.shape[0]

    def to(self, device: torch.device) -> 'ConnList':
        """
        Move all tensors to specified device.

        Args:
            device: Target torch device.

        Returns:
            ConnList with tensors on new device.
        """
        return ConnList(
            left=self.left.to(device),
            right=self.right.to(device),
            trans=self.trans.to(device),
            delta_depth=self.delta_depth.to(device)
        )


# =============================================================================
# Cartesian Grid
# =============================================================================

class CartGrid(nn.Module):
    """
    Cartesian grid for structured reservoir simulation.

    Supports uniform and non-uniform grids with automatic
    transmissibility calculation.

    Args:
        nx: Number of cells in x-direction.
        ny: Number of cells in y-direction.
        nz: Number of cells in z-direction.
        device: Torch device.
        dtype: Torch dtype.

    Example:
        >>> grid = CartGrid(30, 15, 1)
        >>> grid.set_dimensions(dx=100.0, dy=100.0, dz=20.0)
        >>> connlist = grid.build_connections(perm_x, perm_y, perm_z)
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

        # Cell dimensions (can be scalar or per-cell)
        self._dx: Optional[torch.Tensor] = None
        self._dy: Optional[torch.Tensor] = None
        self._dz: Optional[torch.Tensor] = None

        # Cell volumes and depths
        self._volume: Optional[torch.Tensor] = None
        self._depth: Optional[torch.Tensor] = None

        # Connection list (built on demand)
        self._connlist: Optional[ConnList] = None

        # Top depth
        self._top_depth: float = 0.0

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def dx(self) -> torch.Tensor:
        """Cell dimensions in x-direction (ft)."""
        if self._dx is None:
            raise ValueError("Grid dimensions not set. Call set_dimensions first.")
        return self._dx

    @property
    def dy(self) -> torch.Tensor:
        """Cell dimensions in y-direction (ft)."""
        if self._dy is None:
            raise ValueError("Grid dimensions not set. Call set_dimensions first.")
        return self._dy

    @property
    def dz(self) -> torch.Tensor:
        """Cell dimensions in z-direction (ft)."""
        if self._dz is None:
            raise ValueError("Grid dimensions not set. Call set_dimensions first.")
        return self._dz

    @property
    def volume(self) -> torch.Tensor:
        """Cell volumes (ft³)."""
        if self._volume is None:
            self._compute_volumes()
        return self._volume

    @property
    def depth(self) -> torch.Tensor:
        """Cell center depths (ft)."""
        if self._depth is None:
            self._compute_depths()
        return self._depth

    @property
    def connlist(self) -> ConnList:
        """Connection list (requires permeability to be set)."""
        if self._connlist is None:
            raise ValueError("Connection list not built. Call build_connections first.")
        return self._connlist

    # -------------------------------------------------------------------------
    # Configuration Methods
    # -------------------------------------------------------------------------

    def set_dimensions(
        self,
        dx: Union[float, torch.Tensor],
        dy: Union[float, torch.Tensor],
        dz: Union[float, torch.Tensor],
        top_depth: float = 0.0
    ) -> 'CartGrid':
        """
        Set cell dimensions.

        Args:
            dx: Cell size in x-direction (ft), scalar or (nx,) tensor.
            dy: Cell size in y-direction (ft), scalar or (ny,) tensor.
            dz: Cell size in z-direction (ft), scalar or (nz,) tensor.
            top_depth: Depth to top of reservoir (ft).

        Returns:
            Self for method chaining.
        """
        self._dx = self._expand_dim(dx, self.nx)
        self._dy = self._expand_dim(dy, self.ny)
        self._dz = self._expand_dim(dz, self.nz)
        self._top_depth = top_depth

        # Reset derived quantities
        self._volume = None
        self._depth = None
        self._connlist = None

        return self

    def build_connections(
        self,
        perm_x: torch.Tensor,
        perm_y: Optional[torch.Tensor] = None,
        perm_z: Optional[torch.Tensor] = None
    ) -> ConnList:
        """
        Build connection list with transmissibilities.

        Uses harmonic averaging for permeability at cell interfaces.
        Includes ALPHA constant in transmissibility calculation.

        Args:
            perm_x: Permeability in x-direction (md), shape (nc,).
            perm_y: Permeability in y-direction (md), defaults to perm_x.
            perm_z: Permeability in z-direction (md), defaults to perm_x.

        Returns:
            ConnList with precomputed transmissibilities.
        """
        if perm_y is None:
            perm_y = perm_x
        if perm_z is None:
            perm_z = perm_x

        # Reshape permeabilities to 3D for indexing
        kx = perm_x.reshape(self.nz, self.ny, self.nx)
        ky = perm_y.reshape(self.nz, self.ny, self.nx)
        kz = perm_z.reshape(self.nz, self.ny, self.nx)

        # Build connections in each direction
        conn_x = self._build_x_connections(kx)
        conn_y = self._build_y_connections(ky)
        conn_z = self._build_z_connections(kz)

        # Concatenate all connections
        self._connlist = ConnList(
            left=torch.cat([conn_x[0], conn_y[0], conn_z[0]]),
            right=torch.cat([conn_x[1], conn_y[1], conn_z[1]]),
            trans=torch.cat([conn_x[2], conn_y[2], conn_z[2]]),
            delta_depth=torch.cat([conn_x[3], conn_y[3], conn_z[3]])
        )

        return self._connlist

    # -------------------------------------------------------------------------
    # Index Conversion Methods
    # -------------------------------------------------------------------------

    def ijk_to_global(self, i: int, j: int, k: int) -> int:
        """
        Convert (i,j,k) indices to global cell index.

        Args:
            i: Index in x-direction (0-based).
            j: Index in y-direction (0-based).
            k: Index in z-direction (0-based).

        Returns:
            Global cell index.
        """
        return i + j * self.nx + k * self.nx * self.ny

    def global_to_ijk(self, idx: int) -> Tuple[int, int, int]:
        """
        Convert global cell index to (i,j,k) indices.

        Args:
            idx: Global cell index.

        Returns:
            Tuple of (i, j, k) indices (0-based).
        """
        k = idx // (self.nx * self.ny)
        j = (idx % (self.nx * self.ny)) // self.nx
        i = idx % self.nx
        return i, j, k

    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------

    def _expand_dim(self, value: Union[float, torch.Tensor], n: int) -> torch.Tensor:
        """Expand scalar to tensor if needed."""
        if isinstance(value, (int, float)):
            return torch.full((n,), float(value), device=self.device, dtype=self.dtype)
        return value.to(device=self.device, dtype=self.dtype)

    def _compute_volumes(self) -> None:
        """Compute cell volumes."""
        dx_grid = self._dx.repeat(self.ny * self.nz)
        dy_grid = self._dy.repeat_interleave(self.nx).repeat(self.nz)
        dz_grid = self._dz.repeat_interleave(self.nx * self.ny)

        self._volume = dx_grid * dy_grid * dz_grid

    def _compute_depths(self) -> None:
        """Compute cell center depths."""
        self._depth = torch.zeros(self.nc, device=self.device, dtype=self.dtype)

        idx = 0
        cumulative_depth = self._top_depth
        for k in range(self.nz):
            cell_depth = cumulative_depth + 0.5 * self._dz[k]
            for j in range(self.ny):
                for i in range(self.nx):
                    self._depth[idx] = cell_depth
                    idx += 1
            cumulative_depth += self._dz[k]

    def _build_x_connections(self, kx: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Build x-direction connections (vectorized)."""
        if self.nx <= 1:
            empty = torch.tensor([], device=self.device, dtype=torch.long)
            empty_f = torch.tensor([], device=self.device, dtype=self.dtype)
            return empty, empty, empty_f, empty_f

        # Create index arrays
        k_idx = torch.arange(self.nz, device=self.device)
        j_idx = torch.arange(self.ny, device=self.device)
        i_idx = torch.arange(self.nx - 1, device=self.device)

        # Meshgrid for vectorized indexing
        kk, jj, ii = torch.meshgrid(k_idx, j_idx, i_idx, indexing='ij')
        kk, jj, ii = kk.flatten(), jj.flatten(), ii.flatten()

        # Global indices
        left = ii + jj * self.nx + kk * self.nx * self.ny
        right = left + 1

        # Get permeabilities
        k_left = kx[kk, jj, ii]
        k_right = kx[kk, jj, ii + 1]

        dx_left = self._dx[ii]
        dx_right = self._dx[ii + 1]

        # Cross-sectional area
        area = self._dy[jj] * self._dz[kk]

        # Transmissibility with ALPHA constant
        # T = 2 * ALPHA * k_L * k_R * A / (k_L * dx_R + k_R * dx_L)
        trans = 2.0 * ALPHA * k_left * k_right * area / (
            k_left * dx_right + k_right * dx_left + 1e-30
        )

        # No depth difference in x-direction
        nconn = (self.nx - 1) * self.ny * self.nz
        delta_depth = torch.zeros(nconn, device=self.device, dtype=self.dtype)

        return left, right, trans, delta_depth

    def _build_y_connections(self, ky: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Build y-direction connections (vectorized)."""
        if self.ny <= 1:
            empty = torch.tensor([], device=self.device, dtype=torch.long)
            empty_f = torch.tensor([], device=self.device, dtype=self.dtype)
            return empty, empty, empty_f, empty_f

        # Create index arrays
        k_idx = torch.arange(self.nz, device=self.device)
        j_idx = torch.arange(self.ny - 1, device=self.device)
        i_idx = torch.arange(self.nx, device=self.device)

        # Meshgrid for vectorized indexing
        kk, jj, ii = torch.meshgrid(k_idx, j_idx, i_idx, indexing='ij')
        kk, jj, ii = kk.flatten(), jj.flatten(), ii.flatten()

        # Global indices
        left = ii + jj * self.nx + kk * self.nx * self.ny
        right = left + self.nx

        # Transmissibility
        k_left = ky[kk, jj, ii]
        k_right = ky[kk, jj + 1, ii]

        dy_left = self._dy[jj]
        dy_right = self._dy[jj + 1]

        area = self._dx[ii] * self._dz[kk]

        # Include ALPHA constant
        trans = 2.0 * ALPHA * k_left * k_right * area / (
            k_left * dy_right + k_right * dy_left + 1e-30
        )

        # No depth difference in y-direction
        nconn = self.nx * (self.ny - 1) * self.nz
        delta_depth = torch.zeros(nconn, device=self.device, dtype=self.dtype)

        return left, right, trans, delta_depth

    def _build_z_connections(self, kz: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Build z-direction connections (vectorized)."""
        if self.nz <= 1:
            empty = torch.tensor([], device=self.device, dtype=torch.long)
            empty_f = torch.tensor([], device=self.device, dtype=self.dtype)
            return empty, empty, empty_f, empty_f

        # Create index arrays
        k_idx = torch.arange(self.nz - 1, device=self.device)
        j_idx = torch.arange(self.ny, device=self.device)
        i_idx = torch.arange(self.nx, device=self.device)

        # Meshgrid for vectorized indexing
        kk, jj, ii = torch.meshgrid(k_idx, j_idx, i_idx, indexing='ij')
        kk, jj, ii = kk.flatten(), jj.flatten(), ii.flatten()

        # Global indices
        left = ii + jj * self.nx + kk * self.nx * self.ny
        right = left + self.nx * self.ny

        # Transmissibility
        k_left = kz[kk, jj, ii]
        k_right = kz[kk + 1, jj, ii]

        dz_left = self._dz[kk]
        dz_right = self._dz[kk + 1]

        area = self._dx[ii] * self._dy[jj]

        # Include ALPHA constant
        trans = 2.0 * ALPHA * k_left * k_right * area / (
            k_left * dz_right + k_right * dz_left + 1e-30
        )

        # Depth difference for gravity
        delta_depth = 0.5 * (dz_left + dz_right)

        return left, right, trans, delta_depth

    def forward(self) -> None:
        """Forward pass (placeholder for nn.Module compatibility)."""
        pass

    def __repr__(self) -> str:
        has_conn = self._connlist is not None
        nconn = self._connlist.nconn if has_conn else 0
        return (
            f"CartGrid(nx={self.nx}, ny={self.ny}, nz={self.nz}, "
            f"nc={self.nc}, nconn={nconn})"
        )
