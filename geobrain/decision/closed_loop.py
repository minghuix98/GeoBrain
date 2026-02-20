"""
Closed-loop reservoir management.

Orchestrates the cycle:
    observe → update ensemble → decide → (repeat)

Keeps the ensemble state across cycles and records full history,
while leaving each component (simulator, updater, decision rule)
pluggable.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import time
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


# =============================================================================
# Step result
# =============================================================================

@dataclass
class ClosedLoopStep:
    """
    Record of a single closed-loop cycle.

    Attributes:
        step: Cycle index (0-based).
        ensemble: Updated ensemble after assimilation [n_samples, dim].
        observed: Observation data used in this cycle.
        decision: Decision returned by the decision function.
        decision_info: Extra information from the decision function.
        elapsed: Wall-clock time for this cycle (seconds).
    """
    step: int
    ensemble: torch.Tensor
    observed: torch.Tensor
    decision: Any
    decision_info: Dict[str, Any]
    elapsed: float


# =============================================================================
# Manager
# =============================================================================

class ClosedLoopManager:
    """
    Stateful manager for closed-loop reservoir management.

    Each call to :meth:`step` performs one cycle:

    1. **Update posterior** — ``posterior.set_data(observed)``
    2. **Update ensemble** — warm-start SVGD from the current ensemble
    3. **Decide** — call ``decision_fn(ensemble) -> (decision, info)``

    The manager keeps the running ensemble and a full history of every
    cycle, so the user can inspect or visualize the evolution of
    beliefs and decisions.

    Args:
        ensemble: Initial ensemble of shape ``[n_samples, dim]``.
        posterior: A :class:`~geobrain.bayes.Posterior` whose
            ``set_data()`` method will be called with each new
            observation.
        svgd: A configured :class:`~geobrain.bayes.SVGD` instance.
        decision_fn: ``(ensemble) -> (decision, info_dict)``.
            If the callable returns a non-tuple, it is wrapped as
            ``(value, {})``.
        n_update_steps: Default number of SVGD steps per cycle.

    Example:
        >>> from geobrain import SVGD, Posterior
        >>> from geobrain.decision import ClosedLoopManager
        >>>
        >>> manager = ClosedLoopManager(
        ...     ensemble=initial_samples,
        ...     posterior=posterior,
        ...     svgd=svgd,
        ...     decision_fn=my_decision,
        ...     n_update_steps=200,
        ... )
        >>>
        >>> # Cycle 1: new production data arrives
        >>> step1 = manager.step(obs_day90)
        >>> print(step1.decision)
        >>>
        >>> # Cycle 2
        >>> step2 = manager.step(obs_day180)
        >>> print(manager.summary())
    """

    def __init__(
        self,
        ensemble: torch.Tensor,
        posterior,
        svgd,
        decision_fn: Callable[[torch.Tensor], Any],
        n_update_steps: int = 200,
    ):
        self.ensemble = ensemble.detach().clone()
        self.posterior = posterior
        self.svgd = svgd
        self.decision_fn = decision_fn
        self.n_update_steps = n_update_steps

        self.history: List[ClosedLoopStep] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def step(
        self,
        observed: torch.Tensor,
        n_update_steps: Optional[int] = None,
        verbose: bool = True,
    ) -> ClosedLoopStep:
        """
        Run one closed-loop cycle.

        Args:
            observed: New observation data (tensor).
            n_update_steps: SVGD steps for this cycle.  Falls back to
                the default set in the constructor.
            verbose: Print progress.

        Returns:
            A :class:`ClosedLoopStep` recording the cycle.
        """
        t0 = time.time()
        cycle = len(self.history)
        n_steps = n_update_steps or self.n_update_steps

        if verbose:
            logger.info(f"Closed-loop cycle {cycle}: updating ensemble "
                        f"({n_steps} SVGD steps) ...")

        # 1. Feed new data to posterior
        self.posterior.set_data(observed)

        # 2. Warm-start SVGD from current ensemble
        n_samples = self.ensemble.shape[0]
        result = self.svgd.run(
            n_samples=n_samples,
            n_steps=n_steps,
            initial_samples=self.ensemble,
            target=self.posterior,
            verbose=False,
        )
        self.ensemble = result.samples.detach().clone()

        # 3. Decision
        raw = self.decision_fn(self.ensemble)
        if isinstance(raw, tuple) and len(raw) == 2:
            decision, info = raw
        else:
            decision, info = raw, {}

        elapsed = time.time() - t0

        if verbose:
            logger.info(f"Cycle {cycle} done ({elapsed:.1f}s): "
                        f"decision = {decision}")

        record = ClosedLoopStep(
            step=cycle,
            ensemble=self.ensemble.clone(),
            observed=observed,
            decision=decision,
            decision_info=info,
            elapsed=elapsed,
        )
        self.history.append(record)
        return record

    def run(
        self,
        observations: List[torch.Tensor],
        n_update_steps: Optional[int] = None,
        verbose: bool = True,
    ) -> List[ClosedLoopStep]:
        """
        Run multiple cycles with a pre-defined observation sequence.

        Convenience wrapper around :meth:`step` for synthetic
        experiments where all observations are known upfront.

        Args:
            observations: List of observation tensors, one per cycle.
            n_update_steps: SVGD steps per cycle (overrides default).
            verbose: Print progress.

        Returns:
            List of :class:`ClosedLoopStep` records.
        """
        steps = []
        for obs in observations:
            s = self.step(obs, n_update_steps=n_update_steps, verbose=verbose)
            steps.append(s)
        return steps

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    @property
    def decisions(self) -> List[Any]:
        """List of decisions made so far."""
        return [s.decision for s in self.history]

    @property
    def n_cycles(self) -> int:
        """Number of completed cycles."""
        return len(self.history)

    def summary(self) -> str:
        """Return a human-readable summary of the full history."""
        lines = [
            "=== Closed-Loop History ===",
            f"Cycles completed : {self.n_cycles}",
            f"Ensemble shape   : {list(self.ensemble.shape)}",
        ]
        for s in self.history:
            lines.append(
                f"  Cycle {s.step}: decision={s.decision}  "
                f"({s.elapsed:.1f}s)"
            )
        return "\n".join(lines)
