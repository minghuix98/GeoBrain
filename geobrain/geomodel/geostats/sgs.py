"""
Sequential Gaussian Simulation (SGS).

Conditional geostatistical simulation that honors well/sample data.
Uses kriging to estimate local mean and variance at each grid node,
then draws from the conditional distribution.

Algorithm:
    1. Assign conditioning data to nearest grid nodes
    2. Define a random path through all grid nodes
    3. For each node along the path:
       a. Search nearby conditioning data and previously simulated nodes
       b. Build and solve the kriging system (SK or OK)
       c. Draw from N(kriging_mean, kriging_variance)
    4. Repeat for each realization

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import numpy as np
import torch
from torch import Tensor
from typing import Optional

from ..base import GeostatSimulator
from ..config import SimulationConfig
from ..registry import register_model
from .variogram import VariogramModel

# Sentinel value for unsimulated nodes
UNEST = -999.0


@register_model(
    'sgs',
    category='geostat',
    implemented=True,
    description='Sequential Gaussian Simulation (conditional, honors well data)'
)
class SGSSimulator(GeostatSimulator):
    """
    Sequential Gaussian Simulation (SGS) simulator.

    Generates conditional or unconditional Gaussian random fields
    using sequential simulation with kriging.

    Args:
        correlation_model: Variogram model type ('spherical', 'exponential', 'gaussian').
        conditioning_data: Values at known locations.
        conditioning_locations: Coordinates of known points (n_points, ndim).
        transform: Optional output transform function.
        kriging_type: 'sk' (Simple Kriging) or 'ok' (Ordinary Kriging).
        sk_mean: Mean for Simple Kriging (default 0.0).
        max_conditioning: Max number of conditioning data to use (default 12).
        max_previously_simulated: Max previously simulated nodes to use (default 12).
        search_radius: Search radius for nearby data (default: None, auto from config).
        min_conditioning: Min conditioning data required (default 0).
        nugget: Nugget effect (default 0.0).

    Example:
        >>> from geobrain.geomodel import Simulator, SimulationConfig
        >>> config = SimulationConfig(shape=(64, 64), lh=10, lv=10, seed=42)
        >>> sim = Simulator.create('sgs', kriging_type='ok')
        >>> field = sim.simulate(config)

        >>> # With conditioning data
        >>> import torch
        >>> cond_data = torch.tensor([0.5, -0.3, 1.2])
        >>> cond_locs = torch.tensor([[10., 10.], [30., 30.], [50., 50.]])
        >>> sim = Simulator.create('sgs').set_conditioning(cond_data, cond_locs)
        >>> field = sim.simulate(config)
    """

    def __init__(
        self,
        correlation_model: str = "spherical",
        conditioning_data: Optional[Tensor] = None,
        conditioning_locations: Optional[Tensor] = None,
        transform=None,
        kriging_type: str = "sk",
        sk_mean: float = 0.0,
        max_conditioning: int = 12,
        max_previously_simulated: int = 12,
        search_radius: Optional[float] = None,
        min_conditioning: int = 0,
        nugget: float = 0.0,
    ):
        super().__init__(
            correlation_model=correlation_model,
            conditioning_data=conditioning_data,
            conditioning_locations=conditioning_locations,
            transform=transform,
        )
        self.kriging_type = kriging_type.lower()
        self.sk_mean = sk_mean
        self.max_conditioning = max_conditioning
        self.max_previously_simulated = max_previously_simulated
        self.search_radius = search_radius
        self.min_conditioning = min_conditioning
        self.nugget = nugget

        if self.kriging_type not in ("sk", "ok"):
            raise ValueError(
                f"kriging_type must be 'sk' or 'ok', got '{self.kriging_type}'"
            )

    @property
    def supports_conditioning(self) -> bool:
        """SGS supports conditioning data."""
        return True

    def _simulate(self, config: SimulationConfig) -> Tensor:
        """
        Run SGS simulation.

        Args:
            config: Simulation configuration.

        Returns:
            Simulated field(s). Shape is config.shape if n_realizations=1,
            otherwise (n_realizations, *config.shape).
        """
        # Build variogram model from config
        vario = VariogramModel.from_config(config, self.correlation_model)
        vario.nugget = self.nugget
        sill = vario.sill

        # Grid setup
        shape = config.shape
        ndim = len(shape)
        ncells = int(np.prod(shape))

        # Generate grid coordinates (cell centers, unit spacing, origin=0)
        if ndim == 3:
            nx, ny, nz = shape
            gx = np.arange(nx, dtype=np.float64)
            gy = np.arange(ny, dtype=np.float64)
            gz = np.arange(nz, dtype=np.float64)
            GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing='ij')
            grid_x = GX.ravel()
            grid_y = GY.ravel()
            grid_z = GZ.ravel()
        else:
            nx, ny = shape
            gx = np.arange(nx, dtype=np.float64)
            gy = np.arange(ny, dtype=np.float64)
            GX, GY = np.meshgrid(gx, gy, indexing='ij')
            grid_x = GX.ravel()
            grid_y = GY.ravel()
            grid_z = np.zeros(ncells, dtype=np.float64)

        # Search radius: default to max range * 2
        search_radius = self.search_radius
        if search_radius is None:
            max_range = max(config.lh, config.lv)
            search_radius = max_range * 2.0

        # Process conditioning data
        cond_x, cond_y, cond_z, cond_v = self._prepare_conditioning(ndim)
        n_cond = len(cond_v) if cond_v is not None else 0

        # Build cKDTree for conditioning data search
        cond_tree = None
        if n_cond > 0:
            from scipy.spatial import cKDTree
            if ndim == 3:
                cond_coords = np.column_stack([cond_x, cond_y, cond_z])
            else:
                cond_coords = np.column_stack([cond_x, cond_y])
            cond_tree = cKDTree(cond_coords)

        # Assign conditioning data to nearest grid nodes
        cond_node_indices = None
        if n_cond > 0:
            cond_node_indices = self._assign_to_nodes(
                cond_x, cond_y, cond_z, shape, ndim
            )

        # RNG setup
        seed = config.seed if config.seed is not None else None
        rng = np.random.default_rng(seed)

        # Simulate realizations
        all_fields = np.zeros((config.n_realizations, ncells), dtype=np.float64)

        for ireal in range(config.n_realizations):
            sim = self._simulate_one_realization(
                grid_x, grid_y, grid_z,
                ncells, shape, ndim,
                cond_x, cond_y, cond_z, cond_v, n_cond,
                cond_tree, cond_node_indices,
                vario, sill, search_radius,
                rng,
            )
            all_fields[ireal] = sim

        # Apply mean and std transformation
        mean_val = config.mean
        std_val = config.std
        if hasattr(mean_val, 'item'):
            mean_val = mean_val.item()
        if hasattr(std_val, 'item'):
            std_val = std_val.item()

        all_fields = float(mean_val) + float(std_val) * all_fields

        # Reshape to grid
        if config.n_realizations == 1:
            result = all_fields[0].reshape(shape)
        else:
            result = all_fields.reshape(config.n_realizations, *shape)

        return torch.from_numpy(result).to(
            device=config.torch_device, dtype=config.torch_dtype
        )

    def _prepare_conditioning(self, ndim):
        """Extract conditioning data as numpy arrays."""
        if self.conditioning_data is None:
            return None, None, None, None

        cond_v = self.conditioning_data.detach().cpu().numpy().astype(np.float64)
        cond_locs = self.conditioning_locations.detach().cpu().numpy().astype(np.float64)

        if ndim == 3:
            cond_x = cond_locs[:, 0]
            cond_y = cond_locs[:, 1]
            cond_z = cond_locs[:, 2] if cond_locs.shape[1] > 2 else np.zeros(len(cond_v))
        else:
            cond_x = cond_locs[:, 0]
            cond_y = cond_locs[:, 1]
            cond_z = np.zeros(len(cond_v))

        return cond_x, cond_y, cond_z, cond_v

    def _assign_to_nodes(self, cond_x, cond_y, cond_z, shape, ndim):
        """Assign conditioning data to nearest grid nodes (GSLIB style)."""
        n_cond = len(cond_x)
        node_indices = np.zeros(n_cond, dtype=np.int64)

        if ndim == 3:
            nx, ny, nz = shape
            for i in range(n_cond):
                ix = int(np.clip(np.round(cond_x[i]), 0, nx - 1))
                iy = int(np.clip(np.round(cond_y[i]), 0, ny - 1))
                iz = int(np.clip(np.round(cond_z[i]), 0, nz - 1))
                node_indices[i] = ix * ny * nz + iy * nz + iz
        else:
            nx, ny = shape
            for i in range(n_cond):
                ix = int(np.clip(np.round(cond_x[i]), 0, nx - 1))
                iy = int(np.clip(np.round(cond_y[i]), 0, ny - 1))
                node_indices[i] = ix * ny + iy

        return node_indices

    def _simulate_one_realization(
        self,
        grid_x, grid_y, grid_z,
        ncells, shape, ndim,
        cond_x, cond_y, cond_z, cond_v, n_cond,
        cond_tree, cond_node_indices,
        vario, sill, search_radius,
        rng,
    ):
        """Simulate a single realization using SGS."""
        sim = np.full(ncells, UNEST, dtype=np.float64)

        # Assign conditioning data to grid
        if n_cond > 0:
            for i in range(n_cond):
                sim[cond_node_indices[i]] = cond_v[i]

        # Random path (skip nodes with conditioning data)
        path = rng.permutation(ncells)

        # Track simulated node locations for neighbor search
        sim_indices = []  # indices of simulated nodes (excluding conditioning)
        if n_cond > 0:
            # Conditioning nodes are already simulated
            for idx in cond_node_indices:
                sim_indices.append(idx)

        for node_idx in path:
            # Skip already simulated nodes (conditioning data)
            if sim[node_idx] != UNEST:
                continue

            x0 = grid_x[node_idx]
            y0 = grid_y[node_idx]
            z0 = grid_z[node_idx]

            # --- Collect nearby data ---
            # 1. Search conditioning data
            near_x = []
            near_y = []
            near_z = []
            near_v = []

            if n_cond > 0 and cond_tree is not None:
                if ndim == 3:
                    query_pt = [x0, y0, z0]
                else:
                    query_pt = [x0, y0]

                dists, idxs = cond_tree.query(
                    query_pt,
                    k=min(self.max_conditioning, n_cond),
                    distance_upper_bound=search_radius,
                )
                # cKDTree returns inf for missing neighbors
                if np.isscalar(dists):
                    dists = np.array([dists])
                    idxs = np.array([idxs])

                valid = dists < np.inf
                for j in np.where(valid)[0]:
                    ci = idxs[j]
                    near_x.append(cond_x[ci])
                    near_y.append(cond_y[ci])
                    near_z.append(cond_z[ci])
                    near_v.append(cond_v[ci])

            # 2. Search previously simulated nodes
            if len(sim_indices) > 0:
                sim_idx_arr = np.array(sim_indices)
                sx = grid_x[sim_idx_arr]
                sy = grid_y[sim_idx_arr]
                sz = grid_z[sim_idx_arr]

                # Vectorized distance computation
                dx = sx - x0
                dy = sy - y0
                dz = sz - z0
                dists_sq = dx * dx + dy * dy + dz * dz
                r2 = search_radius * search_radius

                within = dists_sq < r2
                if np.any(within):
                    candidates = np.where(within)[0]
                    # Sort by distance and take closest
                    sorted_idx = candidates[np.argsort(dists_sq[candidates])]
                    n_take = min(self.max_previously_simulated, len(sorted_idx))
                    for j in range(n_take):
                        si = sim_idx_arr[sorted_idx[j]]
                        near_x.append(grid_x[si])
                        near_y.append(grid_y[si])
                        near_z.append(grid_z[si])
                        near_v.append(sim[si])

            n_nearby = len(near_v)

            # --- Kriging ---
            if n_nearby == 0:
                # No data nearby: draw from marginal
                cmean = self.sk_mean if self.kriging_type == "sk" else 0.0
                cstdev = np.sqrt(max(sill, 0.0))
            else:
                near_x_arr = np.array(near_x)
                near_y_arr = np.array(near_y)
                near_z_arr = np.array(near_z)
                near_v_arr = np.array(near_v)

                cmean, cstdev = self._krige(
                    near_x_arr, near_y_arr, near_z_arr, near_v_arr,
                    x0, y0, z0,
                    vario, sill, n_nearby,
                )

            # Draw from conditional distribution
            sim[node_idx] = rng.normal() * cstdev + cmean
            sim_indices.append(node_idx)

        # Restore exact conditioning data values
        if n_cond > 0:
            for i in range(n_cond):
                sim[cond_node_indices[i]] = cond_v[i]

        return sim

    def _krige(
        self,
        data_x, data_y, data_z, data_v,
        x0, y0, z0,
        vario, sill, n_data,
    ):
        """
        Solve kriging system for a single estimation point.

        Returns (cmean, cstdev) - kriging mean and standard deviation.
        """
        from scipy.linalg import solve
        import warnings as _warnings

        use_ok = self.kriging_type == "ok" and n_data >= 4

        # Build covariance matrix between data points
        cov_dd = vario.covariance_matrix_np(data_x, data_y, data_z)

        # Add nugget to diagonal (small epsilon for numerical stability)
        nugget_val = max(vario.nugget, sill * 1e-8)
        np.fill_diagonal(cov_dd, cov_dd.diagonal() + nugget_val)

        # Build covariance vector between data and estimation point
        cov_d0 = vario.covariance_vector_np(data_x, data_y, data_z, x0, y0, z0)

        if use_ok:
            # Ordinary Kriging: add Lagrange constraint
            neq = n_data + 1
            a_mat = np.ones((neq, neq), dtype=np.float64)
            a_mat[:n_data, :n_data] = cov_dd
            a_mat[n_data, n_data] = 0.0

            rhs = np.ones(neq, dtype=np.float64)
            rhs[:n_data] = cov_d0
        else:
            # Simple Kriging
            a_mat = cov_dd
            rhs = cov_d0

        # Solve kriging system
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", category=RuntimeWarning)
                _warnings.filterwarnings("ignore", message=".*ill-conditioned.*")
                w = solve(a_mat, rhs, assume_a='sym', check_finite=False)
            ising = 0
        except Exception:
            # Singular: fall back to marginal
            cmean = self.sk_mean if self.kriging_type == "sk" else 0.0
            return cmean, np.sqrt(max(sill, 0.0))

        # Compute kriging estimate and variance
        if use_ok:
            weights = w[:n_data]
            lagrange = w[n_data]
            cmean = float(np.dot(weights, data_v))
            cvar = sill - float(np.dot(weights, cov_d0)) - lagrange
        else:
            weights = w
            cmean = self.sk_mean + float(
                np.dot(weights, data_v - self.sk_mean)
            )
            cvar = sill - float(np.dot(weights, cov_d0))

        # Clamp variance
        cstdev = np.sqrt(max(cvar, 0.0))

        return cmean, cstdev
