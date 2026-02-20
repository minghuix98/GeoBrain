"""
Factory integration for the implicit geological modeling module.

Registers ImplicitSimulator with the geomodel Simulator factory so it
can be created via ``Simulator.create('implicit', ...)``.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
"""

import torch
from typing import Callable, Dict, List, Optional, Union

from ..base import BaseSimulator
from ..registry import register_model
from .data import ImplicitModelConfig, SeriesDefinition, FaultDefinition
from .model import ImplicitModel


@register_model(
    'implicit',
    category='implicit',
    implemented=True,
    description='Differentiable implicit geological modeling via Universal Cokriging',
)
class ImplicitSimulator(BaseSimulator):
    """Simulator wrapper for ImplicitModel.

    Integrates the differentiable implicit geological model into the
    GeoBrain Simulator factory pattern.

    Args:
        series: List of SeriesDefinition objects (geological layers).
        faults: Optional list of FaultDefinition objects.
        soft: Use differentiable soft classification (default True).
        temperature: Sigmoid sharpness for soft classification.
        transform: Optional output transform applied to the block tensor.

    Example:
        >>> from geobrain.geomodel import Simulator
        >>> sim = Simulator.create('implicit', series=[series_def])
        >>> result = sim.simulate(config)
    """

    def __init__(
        self,
        series: List[SeriesDefinition],
        faults: Optional[List[FaultDefinition]] = None,
        soft: bool = True,
        temperature: float = 50.0,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(transform=transform)
        self.series = series
        self.faults = faults
        self.soft = soft
        self.temperature = temperature
        self._model: Optional[ImplicitModel] = None

    def _simulate(
        self, config: ImplicitModelConfig,
    ) -> Dict[str, torch.Tensor]:
        """Run implicit geological model simulation.

        Args:
            config: ImplicitModelConfig with extent, resolution, kernel, etc.

        Returns:
            Dict with 'block', 'scalar_fields', and 'grid' tensors.
        """
        self._model = ImplicitModel(
            config=config,
            series=self.series,
            faults=self.faults,
        )
        return self._model(soft=self.soft, temperature=self.temperature)

    def _get_default_config(self):
        return ImplicitModelConfig()

    @property
    def is_differentiable(self) -> bool:
        return True

    @property
    def supports_conditioning(self) -> bool:
        return True

    @property
    def model(self) -> Optional[ImplicitModel]:
        """Access the underlying ImplicitModel after simulation."""
        return self._model
