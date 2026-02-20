"""
Category-specific base classes for rock physics models.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

from .base import ComponentModel


class EffectiveModel(ComponentModel):
    """
    Base class for effective medium models.

    Examples: Voigt/Reuss bounds, Hashin-Shtrikman, DEM, Self-Consistent
    """
    pass


class FluidModel(ComponentModel):
    """
    Base class for fluid-related models.

    Examples: Gassmann, Wood's equation, Brie, Batzle-Wang
    """

    def compute_fluid_mix(self, fl1_K, fl2_K, fl1_rho, fl2_rho, Sw, **params):
        """
        Compute effective fluid properties for two-fluid mixing.

        Standard interface for RockPhysicsWorkflow integration. Fluid
        mixing models (Wood, Brie, etc.) should override this to map
        workflow parameters to their native forward() signature.

        Args:
            fl1_K: Fluid 1 bulk modulus (GPa).
            fl2_K: Fluid 2 bulk modulus (GPa).
            fl1_rho: Fluid 1 density (g/cm³).
            fl2_rho: Fluid 2 density (g/cm³).
            Sw: Volume fraction of fluid 1 (e.g., water saturation).
            **params: Additional parameters.

        Returns:
            Tuple of (K_eff, rho_eff).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement compute_fluid_mix(). "
            "Override this method for workflow integration."
        )


class GranularModel(ComponentModel):
    """
    Base class for granular medium models.

    Examples: Hertz-Mindlin, Soft/Stiff sand, Contact cement
    """
    pass


class EmpiricalModel(ComponentModel):
    """
    Base class for empirical models.

    Examples: Han, Castagna, Gardner, compaction trends
    """
    pass


class AnisotropyModel(ComponentModel):
    """
    Base class for anisotropy models.

    Examples: Thomsen parameters, Backus averaging, Bond transform
    """
    pass


class AVOModel(ComponentModel):
    """
    Base class for AVO (Amplitude Variation with Offset) models.

    Examples: Zoeppritz, Aki-Richards, Shuey approximation
    """
    pass


class PermeabilityModel(ComponentModel):
    """
    Base class for permeability models.

    Examples: Kozeny-Carman, Owolabi, Panda-Lake
    """
    pass
