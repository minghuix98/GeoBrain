"""
Value of Information (VOI) analysis.

Computes the Value of Perfect Information (VOPI) — the upper bound on what
additional data is worth before a decision must be made:

    VOI = E_m[max_d obj(d, m)] - max_d E_m[obj(d, m)]

Only requires a decision-model objective matrix [n_decisions, n_samples].

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
# Result Container
# =============================================================================

@dataclass
class VOIResult:
    """
    Result of Value of Information analysis.

    Attributes:
        voi: Value of (perfect) information.
        prior_optimal_value: Expected value of the best decision under
            current information (max_d E_m[obj]).
        prior_optimal_decision: The decision achieving prior_optimal_value.
        preposterior_value: Expected value with perfect information
            (E_m[max_d obj]).
        per_sample_best_values: Best objective per sample [n_samples].
        per_sample_best_decisions: Decision index achieving best per sample.
        decision_values: Full objective matrix [n_decisions, n_samples].
        total_time: Wall-clock time (seconds).
        config: Configuration dict.
    """
    voi: float
    prior_optimal_value: float
    prior_optimal_decision: Any
    preposterior_value: float
    per_sample_best_values: torch.Tensor
    per_sample_best_decisions: List[Any]
    decision_values: torch.Tensor
    total_time: float
    config: Dict[str, Any]

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            "=== Value of Information Result ===",
            f"VOI (perfect info)      : {self.voi:.4f}",
            f"Prior optimal value     : {self.prior_optimal_value:.4f}",
            f"Prior optimal decision  : {self.prior_optimal_decision}",
            f"Preposterior value      : {self.preposterior_value:.4f}",
            f"Decisions evaluated     : {self.decision_values.shape[0]}",
            f"Ensemble size           : {self.decision_values.shape[1]}",
            f"Total time              : {self.total_time:.2f} s",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary (numpy arrays)."""
        return {
            'voi': self.voi,
            'prior_optimal_value': self.prior_optimal_value,
            'prior_optimal_decision': self.prior_optimal_decision,
            'preposterior_value': self.preposterior_value,
            'per_sample_best_values': self.per_sample_best_values.cpu().numpy(),
            'per_sample_best_decisions': self.per_sample_best_decisions,
            'decision_values': self.decision_values.cpu().numpy(),
            'total_time': self.total_time,
            'config': self.config,
        }


# =============================================================================
# VOI Analyzer
# =============================================================================

class ValueOfInformation:
    """
    Value of Perfect Information (VOPI) calculator.

    Args:
        objective_fn: Callable ``(ensemble, decisions) -> Tensor`` that returns
            an objective matrix of shape ``[n_decisions, n_samples]``.
        decisions: List of decision alternatives (any hashable type).
        device: Torch device.

    Example:
        >>> voi_calc = ValueOfInformation(
        ...     objective_fn=my_obj_fn,
        ...     decisions=['A', 'B', 'C'],
        ... )
        >>> result = voi_calc.compute(ensemble)
        >>> print(result.summary())
    """

    def __init__(
        self,
        objective_fn: Callable[[torch.Tensor, List], torch.Tensor],
        decisions: List[Any],
        device: Optional[torch.device] = None,
    ):
        if len(decisions) < 2:
            raise ValueError("Need at least 2 decisions for VOI analysis")
        self.objective_fn = objective_fn
        self.decisions = decisions
        self.device = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        ensemble: torch.Tensor,
        data_cost: float = 0.0,
        verbose: bool = True,
    ) -> VOIResult:
        """
        Compute the Value of Perfect Information.

        Args:
            ensemble: Posterior samples, shape ``[n_samples, dim]``.
            data_cost: Cost of acquiring perfect information.  The net VOI
                is ``max(0, VOI_raw - data_cost)``.
            verbose: Print progress.

        Returns:
            VOIResult with VOI value and supporting diagnostics.
        """
        t_start = time.time()

        if verbose:
            logger.info(
                f"Computing VOI: {len(self.decisions)} decisions, "
                f"{ensemble.shape[0]} samples"
            )

        # Build objective matrix [n_decisions, n_samples]
        obj_matrix = self.objective_fn(ensemble, self.decisions)

        if obj_matrix.shape != (len(self.decisions), ensemble.shape[0]):
            raise ValueError(
                f"objective_fn must return shape [{len(self.decisions)}, "
                f"{ensemble.shape[0]}], got {list(obj_matrix.shape)}"
            )

        if self.device is not None:
            obj_matrix = obj_matrix.to(self.device)

        # --- Prior optimal: max_d E_m[obj(d, m)] ---
        expected = obj_matrix.mean(dim=1)            # [n_decisions]
        prior_best_idx = int(expected.argmax())
        prior_value = float(expected[prior_best_idx])
        prior_decision = self.decisions[prior_best_idx]

        # --- Preposterior: E_m[max_d obj(d, m)] ---
        best_per_sample_vals, best_per_sample_idx = obj_matrix.max(dim=0)
        preposterior_value = float(best_per_sample_vals.mean())
        per_sample_decisions = [
            self.decisions[int(idx)] for idx in best_per_sample_idx
        ]

        # --- VOI ---
        voi_raw = preposterior_value - prior_value
        voi = max(0.0, voi_raw - data_cost)

        elapsed = time.time() - t_start

        if verbose:
            logger.info(
                f"VOI = {voi:.4f}  "
                f"(preposterior={preposterior_value:.4f}, "
                f"prior_optimal={prior_value:.4f}, "
                f"data_cost={data_cost:.4f})"
            )

        return VOIResult(
            voi=voi,
            prior_optimal_value=prior_value,
            prior_optimal_decision=prior_decision,
            preposterior_value=preposterior_value,
            per_sample_best_values=best_per_sample_vals,
            per_sample_best_decisions=per_sample_decisions,
            decision_values=obj_matrix,
            total_time=elapsed,
            config={
                'n_decisions': len(self.decisions),
                'n_samples': ensemble.shape[0],
                'data_cost': data_cost,
            },
        )

