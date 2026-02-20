"""
AVO (Amplitude Variation with Offset) reflectivity models.

Provides various methods to compute P-P reflection coefficients
as a function of incidence angle for seismic AVO analysis.

Available models:
    - Zoeppritz: Exact solution (handles post-critical angles)
    - AkiRichards: Linearized 4-term approximation
    - Shuey: 3-term approximation with AVO attributes

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

from torch import Tensor
from typing import Union

from .base import (
    AVOModel,
    as_tensor,
    normal_incidence_rc,
    vectorize_angles,
)
from .zoeppritz import Zoeppritz
from .approximations import AkiRichards, Shuey


def compute_reflectivity(
    vp1, vs1, rho1,
    vp2, vs2, rho2,
    theta,
    method: str = 'shuey',
) -> Tensor:
    """
    Compute reflection coefficients using specified method.

    Convenience function that creates the appropriate model
    and computes reflectivity in one call.

    Args:
        vp1: P-wave velocity of upper layer (m/s).
        vs1: S-wave velocity of upper layer (m/s).
        rho1: Density of upper layer (kg/m³).
        vp2: P-wave velocity of lower layer (m/s).
        vs2: S-wave velocity of lower layer (m/s).
        rho2: Density of lower layer (kg/m³).
        theta: Incidence angles (degrees).
        method: Computation method. Default: 'shuey'.
            Options:
                - 'shuey': Shuey 3-term approximation
                - 'aki' or 'aki_richards': Aki-Richards linearization
                - 'zoeppritz' or 'exact': Full Zoeppritz solution

    Returns:
        Reflection coefficients with shape [n_angles, *input_shape].

    Raises:
        ValueError: If unknown method is specified.

    Example:
        >>> rc = compute_reflectivity(
        ...     vp1=2000, vs1=1000, rho1=2.0,
        ...     vp2=2500, vs2=1250, rho2=2.2,
        ...     theta=[0, 15, 30, 45],
        ...     method='shuey'
        ... )
    """
    method = method.lower()

    if method == 'shuey':
        model = Shuey()
    elif method in ('aki', 'aki_richards', 'ar'):
        model = AkiRichards()
    elif method in ('zoeppritz', 'exact'):
        model = Zoeppritz()
    else:
        raise ValueError(
            f"Unknown method: '{method}'. "
            f"Choose from: 'shuey', 'aki', 'zoeppritz'"
        )

    return model(vp1, vs1, rho1, vp2, vs2, rho2, theta)


__all__ = [
    # Base
    'AVOModel',
    'as_tensor',
    'normal_incidence_rc',
    # Models
    'Zoeppritz',
    'AkiRichards',
    'Shuey',
    # Convenience
    'compute_reflectivity',
]