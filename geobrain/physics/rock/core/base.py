"""
Base model classes for rock physics.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all rock physics models.

    Inherits from both nn.Module (for PyTorch integration) and ABC
    (for interface enforcement).

    Features:
        - Full PyTorch autodiff support
        - Metadata storage for model documentation
        - Parameter range tracking for validation
        - GPU acceleration via standard PyTorch .to() method
    """

    def __init__(self):
        super().__init__()
        self._metadata: Dict[str, Any] = {}
        self._param_ranges: Dict[str, Tuple[float, float]] = {}

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """Forward computation of the model."""
        pass

    def set_metadata(self, key: str, value: Any) -> None:
        """Store metadata about the model."""
        self._metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Retrieve metadata about the model."""
        return self._metadata.get(key, default)

    def set_parameter_range(self, name: str, range_tuple: Tuple[float, float]) -> None:
        """Define valid range for a parameter."""
        self._param_ranges[name] = range_tuple

    def get_parameter_range(self, name: str) -> Optional[Tuple[float, float]]:
        """Get the valid range for a parameter."""
        return self._param_ranges.get(name)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ComponentModel(BaseModel):
    """
    Base class for single-function component models.

    Component models implement a single rock physics equation or
    transformation (e.g., Voigt average, Gassmann equation).
    """

    def compute_dry_rock(self, K0, G0, phi, **params):
        """
        Compute dry rock moduli from mineral moduli and porosity.

        Standard interface for RockPhysicsWorkflow integration. Models
        used as dry rock components should override this to map workflow
        parameters to their native forward() signature.

        Args:
            K0: Mineral bulk modulus (GPa).
            G0: Mineral shear modulus (GPa).
            phi: Porosity (fraction).
            **params: Additional physical parameters (phi_c, Cn, P, etc.).

        Returns:
            Tuple of (K_dry, G_dry) in GPa.
        """
        return self.forward(K0, G0, phi)


class CompositeModel(BaseModel):
    """
    Base class for composite models combining multiple components.

    Composite models orchestrate multiple component models to implement
    complex rock physics workflows.
    """

    def __init__(self):
        super().__init__()
        self._components: Dict[str, ComponentModel] = {}

    def add_component(self, name: str, component: ComponentModel) -> None:
        """Register a component model."""
        self._components[name] = component
        setattr(self, name, component)

    def get_component(self, name: str) -> Optional[ComponentModel]:
        """Retrieve a registered component by name."""
        return self._components.get(name)

    def list_components(self) -> List[str]:
        """List all registered component names."""
        return list(self._components.keys())
