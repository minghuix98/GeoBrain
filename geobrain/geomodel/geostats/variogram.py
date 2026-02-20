"""
Variogram model and covariance calculation for geostatistical simulation.

Provides variogram structures and covariance computation used by SGS
and other kriging-based methods.
    - Spherical:    C(h) = cc * (1 - 1.5*hr + 0.5*hr^3),  hr < 1
    - Exponential:  C(h) = cc * exp(-3*h/a)
    - Gaussian:     C(h) = cc * exp(-3*(h/a)^2)

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


# Variogram model type codes
MODEL_TYPES = {
    "spherical": 1,
    "exponential": 2,
    "gaussian": 3,
}


@dataclass
class VariogramStructure:
    """
    A single nested variogram structure.

    Args:
        model: Model type ('spherical', 'exponential', or 'gaussian').
        contribution: Sill contribution (variance) of this structure.
        range_h: Horizontal range (same in x and y directions).
        range_v: Vertical range (z direction). Defaults to range_h.
    """

    model: str
    contribution: float
    range_h: float
    range_v: Optional[float] = None

    def __post_init__(self):
        if self.model not in MODEL_TYPES:
            raise ValueError(
                f"Unknown model type: '{self.model}'. "
                f"Supported: {list(MODEL_TYPES.keys())}"
            )
        if self.range_v is None:
            self.range_v = self.range_h

    @property
    def model_code(self) -> int:
        """GSLIB integer code for this model type."""
        return MODEL_TYPES[self.model]


class VariogramModel:
    """
    Variogram model with nugget effect and nested structures.

    Computes covariance values for use in kriging systems.

    Args:
        nugget: Nugget effect (discontinuity at origin).
        structures: List of VariogramStructure objects.

    Example:
        >>> model = VariogramModel(
        ...     nugget=0.0,
        ...     structures=[VariogramStructure('spherical', 1.0, 20.0, 5.0)]
        ... )
        >>> cov = model.covariance_3d(5.0, 0.0, 0.0)
    """

    def __init__(
        self,
        nugget: float = 0.0,
        structures: Optional[List[VariogramStructure]] = None,
    ):
        self.nugget = nugget
        self.structures = structures or []

    @classmethod
    def from_config(cls, config, correlation_model: str = "spherical") -> "VariogramModel":
        """
        Create a VariogramModel from a SimulationConfig.

        Maps config parameters:
            - lh -> range_h
            - lv -> range_v
            - std^2 -> sill (contribution)

        Args:
            config: SimulationConfig instance.
            correlation_model: Variogram model type.

        Returns:
            VariogramModel instance.
        """
        std = config.std
        if hasattr(std, 'item'):
            std = std.item()
        contribution = float(std) ** 2

        structure = VariogramStructure(
            model=correlation_model,
            contribution=contribution,
            range_h=config.lh,
            range_v=config.lv,
        )
        return cls(nugget=0.0, structures=[structure])

    @property
    def sill(self) -> float:
        """Total sill (nugget + sum of contributions)."""
        return self.nugget + sum(s.contribution for s in self.structures)

    def covariance_3d(self, dx: float, dy: float, dz: float) -> float:
        """
        Compute 3D anisotropic covariance for a lag vector (dx, dy, dz).

        Uses axis-aligned anisotropy: horizontal range for x,y and
        vertical range for z.

        Args:
            dx: Lag in x direction.
            dy: Lag in y direction.
            dz: Lag in z direction.

        Returns:
            Covariance value C(dx, dy, dz).
        """
        cov = 0.0
        for s in self.structures:
            # Anisotropic distance: scale by range
            rh = s.range_h
            rv = s.range_v
            h = np.sqrt(
                (dx / rh) ** 2 + (dy / rh) ** 2 + (dz / rv) ** 2
            ) if rh > 0 and rv > 0 else 0.0

            if h < 1e-10:
                cov += s.contribution
                continue

            if s.model == "spherical":
                if h < 1.0:
                    cov += s.contribution * (1.0 - 1.5 * h + 0.5 * h ** 3)
            elif s.model == "exponential":
                cov += s.contribution * np.exp(-3.0 * h)
            elif s.model == "gaussian":
                cov += s.contribution * np.exp(-3.0 * h ** 2)

        return cov

    def covariance_distance(self, h: float) -> float:
        """
        Compute isotropic covariance at normalized distance h.

        Args:
            h: Normalized distance (already divided by range).

        Returns:
            Covariance value.
        """
        cov = 0.0
        for s in self.structures:
            if h < 1e-10:
                cov += s.contribution
                continue

            if s.model == "spherical":
                if h < 1.0:
                    cov += s.contribution * (1.0 - 1.5 * h + 0.5 * h ** 3)
            elif s.model == "exponential":
                cov += s.contribution * np.exp(-3.0 * h)
            elif s.model == "gaussian":
                cov += s.contribution * np.exp(-3.0 * h ** 2)

        return cov

    def covariance_matrix_np(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        """
        Build covariance matrix for a set of points.

        Args:
            x: X coordinates (n,).
            y: Y coordinates (n,).
            z: Z coordinates (n,).

        Returns:
            Covariance matrix (n, n).
        """
        n = len(x)
        cov_matrix = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            cov_matrix[i, i] = self.sill
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dz = z[i] - z[j]
                c = self.covariance_3d(dx, dy, dz)
                cov_matrix[i, j] = c
                cov_matrix[j, i] = c

        return cov_matrix

    def covariance_vector_np(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        z_data: np.ndarray,
        x0: float,
        y0: float,
        z0: float,
    ) -> np.ndarray:
        """
        Build covariance vector between data points and a target point.

        Args:
            x_data, y_data, z_data: Data point coordinates (n,).
            x0, y0, z0: Target point coordinates.

        Returns:
            Covariance vector (n,).
        """
        n = len(x_data)
        cov_vec = np.zeros(n, dtype=np.float64)

        for i in range(n):
            dx = x_data[i] - x0
            dy = y_data[i] - y0
            dz = z_data[i] - z0
            cov_vec[i] = self.covariance_3d(dx, dy, dz)

        return cov_vec
