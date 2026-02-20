"""
State management for inversion process.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import torch
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class InversionState:
    """
    Tracks the current state of an inversion process.
    
    This is a mutable container used during optimization to track
    progress, loss history, and best model found.
    
    Attributes:
        epoch: Current epoch number.
        max_epochs: Maximum number of epochs.
        current_loss: Loss value at current epoch.
        best_loss: Best loss value seen so far.
        best_epoch: Epoch at which best loss was achieved.
        current_model: Current model parameters.
        best_model: Model parameters at best loss.
        loss_history: List of all loss values.
        grad_norm: Gradient norm at current epoch.
        converged: Whether optimization has converged.
    """

    # Progress
    epoch: int = 0
    max_epochs: int = 100

    # Loss tracking
    current_loss: float = float('inf')
    best_loss: float = float('inf')
    best_epoch: int = 0

    # Model tracking
    current_model: Optional[torch.Tensor] = None
    best_model: Optional[torch.Tensor] = None

    # History
    loss_history: List[float] = field(default_factory=list)

    # Timing
    start_time: Optional[float] = None
    total_time: float = 0.0

    # Gradients
    grad_norm: Optional[float] = None

    # Status
    converged: bool = False

    def update_loss(self, loss: float) -> None:
        """
        Update loss and track best model.
        
        Args:
            loss: Current loss value.
        """
        self.current_loss = loss
        self.loss_history.append(loss)

        if loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = self.epoch
            if self.current_model is not None:
                self.best_model = self.current_model.detach().clone()

    def start_epoch(self) -> None:
        """Mark the start of an epoch."""
        if self.start_time is None:
            self.start_time = time.time()

    def end_epoch(self) -> None:
        """Mark the end of an epoch and update timing."""
        if self.start_time is not None:
            self.total_time = time.time() - self.start_time

    @property
    def improvement_ratio(self) -> float:
        """Relative improvement from initial loss."""
        if len(self.loss_history) < 2 or self.loss_history[0] == 0:
            return 0.0
        return (self.loss_history[0] - self.current_loss) / self.loss_history[0]


@dataclass
class InversionResult:
    """
    Results of a completed inversion.
    
    This is an immutable container returned after optimization completes,
    containing the final model, predictions, and optimization history.
    
    Attributes:
        model: Final model parameters.
        final_loss: Loss at the end of optimization.
        predictions: Model predictions for the observed data.
        best_model: Model with lowest loss during optimization.
        best_loss: Lowest loss achieved.
        best_epoch: Epoch at which best loss was achieved.
        loss_history: Complete loss history.
        n_epochs: Total number of epochs run.
        converged: Whether optimization converged.
        total_time: Total optimization time in seconds.
        config: Configuration used for optimization.
    
    Example:
        >>> result = inverter.run(data, max_epochs=100)
        >>> print(result.summary())
        >>> final_velocity = result.model
    """

    # Core results
    model: torch.Tensor
    final_loss: float
    predictions: Optional[torch.Tensor] = None

    # Best model tracking
    best_model: Optional[torch.Tensor] = None
    best_loss: float = float('inf')
    best_epoch: int = 0

    # History
    loss_history: List[float] = field(default_factory=list)
    
    # Metadata
    n_epochs: int = 0
    converged: bool = False
    total_time: float = 0.0
    config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.best_model is None:
            self.best_model = self.model
        if self.best_loss == float('inf'):
            self.best_loss = self.final_loss

    @classmethod
    def from_state(
        cls, 
        state: InversionState, 
        final_model: torch.Tensor,
        predictions: Optional[torch.Tensor] = None
    ) -> 'InversionResult':
        """
        Create result from optimization state.
        
        Args:
            state: Final optimization state.
            final_model: Final model parameters.
            predictions: Model predictions.
            
        Returns:
            InversionResult instance.
        """
        return cls(
            model=final_model,
            final_loss=state.current_loss,
            predictions=predictions,
            best_model=state.best_model if state.best_model is not None else final_model,
            best_loss=state.best_loss,
            best_epoch=state.best_epoch,
            loss_history=state.loss_history.copy(),
            n_epochs=state.epoch + 1,
            converged=state.converged,
            total_time=state.total_time
        )

    def summary(self) -> str:
        """
        Generate a text summary of results.
        
        Returns:
            Formatted summary string.
        """
        lines = [
            "=" * 50,
            "Inversion Result",
            "=" * 50,
            f"Final loss: {self.final_loss:.6e}",
            f"Best loss: {self.best_loss:.6e} (epoch {self.best_epoch})",
        ]
        
        if self.loss_history:
            improvement = (1 - self.best_loss / self.loss_history[0]) * 100
            lines.append(f"Improvement: {improvement:.1f}%")
        
        lines.extend([
            f"Epochs: {self.n_epochs}",
            f"Time: {self.total_time:.2f}s",
            f"Converged: {self.converged}",
            "=" * 50,
        ])
        
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary with all result data.
        """
        return {
            'model': self.model.cpu().numpy() if self.model is not None else None,
            'final_loss': self.final_loss,
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'loss_history': self.loss_history,
            'n_epochs': self.n_epochs,
            'converged': self.converged,
            'total_time': self.total_time,
            'config': self.config,
        }

    def __repr__(self) -> str:
        return (
            f"InversionResult(final_loss={self.final_loss:.6e}, "
            f"best_loss={self.best_loss:.6e}, n_epochs={self.n_epochs}, "
            f"converged={self.converged})"
        )
