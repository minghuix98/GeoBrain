"""
Main inverter implementation for deterministic optimization.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import logging
import torch
import torch.nn as nn
import numpy as np
import time
from typing import Callable, List, Optional, Union, Dict, Any

logger = logging.getLogger(__name__)

from .state import InversionState, InversionResult
from .parameterization import (
    Parameterization,
    ExplicitParameterization,
    LatentParameterization,
    NetworkParameterization
)
from .losses import mse_loss, get_loss_fn
from .regularizers import get_regularizer


class Inverter:
    """
    Unified inverter supporting multiple parameterization strategies.
    
    Provides a flexible framework for solving inverse problems using
    gradient-based optimization. Supports three parameterization strategies:
    
    1. **Explicit**: Direct optimization of model parameters
    2. **Latent**: Optimize latent code with fixed decoder
    3. **Network**: Optimize network weights with fixed input (Deep Image Prior)
    
    Args:
        parameterization: Parameterization strategy defining how model
                         parameters are represented and constrained.
        forward_fn: Forward model function mapping parameters to predictions.
        device: Computation device.
    
    Example:
        Basic usage with explicit parameterization:
        
        >>> inverter = Inverter.create(
        ...     initial_model=velocity_model,
        ...     forward_fn=acoustic_forward
        ... )
        >>> result = inverter.run(observed_seismic, max_epochs=100)
        >>> inverted_velocity = result.model
        
        With regularization and constraints:

        >>> from geobrain.optim import bound_constraint, l2_regularizer
        >>> inverter = Inverter.create(
        ...     initial_model=velocity_model,
        ...     forward_fn=forward,
        ...     regularizer=l2_regularizer,
        ...     constraints=bound_constraint(1500, 6000)
        ... )
        >>> result = inverter.run(data, regularization_weight=0.01)
        
        Latent space optimization:
        
        >>> inverter = Inverter.create_latent(
        ...     decoder=pretrained_decoder,
        ...     initial_latent=z0,
        ...     forward_fn=forward
        ... )
        
        Deep Image Prior:
        
        >>> inverter = Inverter.create_network(
        ...     network=unet,
        ...     fixed_input=noise,
        ...     forward_fn=forward
        ... )
    """

    def __init__(
        self,
        parameterization: Parameterization,
        forward_fn: Callable[[torch.Tensor], torch.Tensor],
        device: Optional[torch.device] = None
    ):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            device = torch.device(device)

        self.device = device
        self.parameterization = parameterization.to(self.device)
        self.forward_fn = forward_fn
        self.state = InversionState()

    def run(
        self,
        observed_data: torch.Tensor,
        max_epochs: int = 100,
        *,
        # Optimizer settings
        optimizer: str = 'adam',
        lr: float = 0.01,
        weight_decay: float = 0.0,
        # Loss settings
        loss_fn: Union[str, Callable] = 'mse',
        regularization_weight: float = 0.0,
        # Convergence settings
        tolerance: float = 1e-6,
        patience: int = 20,
        min_improvement: float = 1e-4,
        # Gradient settings
        grad_clip_norm: Optional[float] = None,
        # Output settings
        verbose: bool = True,
        print_every: int = 10,
        track_history: bool = True,
    ) -> InversionResult:
        """
        Run the inversion optimization.
        
        Args:
            observed_data: Target observations to fit.
            max_epochs: Maximum number of epochs.
            optimizer: Optimizer name ('adam', 'sgd', 'lbfgs', 'adamw',
                'adam+lbfgs'). 'adam+lbfgs' (or 'hybrid') runs Adam for
                global search then L-BFGS for local refinement.
            lr: Learning rate.
            weight_decay: L2 weight decay.
            loss_fn: Loss function or name ('mse', 'nrmse', 'l1', 'huber').
            regularization_weight: Weight for regularization term.
            tolerance: Convergence tolerance for loss variance.
            patience: Number of epochs to check for convergence.
            min_improvement: Minimum relative improvement for convergence.
            grad_clip_norm: Maximum gradient norm for clipping.
            verbose: Whether to print progress.
            print_every: Print frequency.
            track_history: Whether to track full loss history.
            
        Returns:
            InversionResult with optimized model and diagnostics.
        """
        # Dispatch to custom L-BFGS if requested
        if optimizer.lower() == 'lbfgs':
            return self._run_lbfgs(
                observed_data, max_epochs,
                loss_fn=loss_fn,
                regularization_weight=regularization_weight,
                tolerance=tolerance,
                verbose=verbose,
                print_every=print_every,
                track_history=track_history,
            )

        # Hybrid Adam → L-BFGS: Adam for global search, L-BFGS for refinement
        if optimizer.lower() in ('adam+lbfgs', 'hybrid'):
            return self._run_hybrid(
                observed_data, max_epochs,
                lr=lr,
                weight_decay=weight_decay,
                loss_fn=loss_fn,
                regularization_weight=regularization_weight,
                tolerance=tolerance,
                verbose=verbose,
                print_every=print_every,
                track_history=track_history,
            )

        # Setup
        observed_data = observed_data.to(self.device)
        opt = self._create_optimizer(optimizer, lr, weight_decay)
        loss_fn = self._resolve_loss_fn(loss_fn)
        
        # Initialize state
        self.state = InversionState(max_epochs=max_epochs)
        model_history: List[torch.Tensor] = []
        
        # Timing
        start_time = time.time()
        
        if verbose:
            logger.info("Starting optimization: %d epochs, lr=%s", max_epochs, lr)
            logger.info("-" * 50)

        # Main optimization loop
        for epoch in range(max_epochs):
            self.state.epoch = epoch
            self.state.start_epoch()
            
            opt.zero_grad()

            # Forward pass
            model, reg_term = self.parameterization.forward()
            predictions = self.forward_fn(model)

            # Compute loss
            data_loss = loss_fn(predictions, observed_data)
            total_loss = data_loss

            if reg_term is not None and regularization_weight > 0:
                total_loss = total_loss + regularization_weight * reg_term

            # Backward pass
            total_loss.backward()

            # Check for NaN/Inf in loss
            if not torch.isfinite(total_loss):
                raise RuntimeError(
                    f"Loss became NaN/Inf at epoch {epoch}: {total_loss.item()}"
                )

            # Check for NaN/Inf in gradients
            has_invalid_grad = False
            for p in self.parameterization.optimizable_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    has_invalid_grad = True
                    break

            if has_invalid_grad:
                raise RuntimeError(
                    f"Gradients contain NaN/Inf at epoch {epoch}. "
                    "Consider reducing learning rate or checking forward model."
                )

            # Gradient clipping
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.parameterization.optimizable_parameters(),
                    grad_clip_norm
                )

            # Optimizer step
            opt.step()

            # Update state
            self.state.current_model = self.get_model()
            self.state.update_loss(total_loss.item())
            self.state.grad_norm = self._compute_gradient_norm()
            self.state.end_epoch()

            # Track history
            if track_history:
                model_history.append(self.state.current_model.cpu().clone())

            # Print progress
            if verbose and (epoch % print_every == 0 or epoch == max_epochs - 1):
                self._print_progress(epoch, max_epochs, total_loss.item())

            # Check convergence
            if self._check_convergence(tolerance, patience, min_improvement):
                self.state.converged = True
                if verbose:
                    logger.info("Converged at epoch %d", epoch + 1)
                break

        # Final timing
        self.state.total_time = time.time() - start_time

        # Final predictions
        with torch.no_grad():
            final_predictions = self.predict()

        # Create result
        result = InversionResult.from_state(
            self.state,
            final_model=self.get_model(),
            predictions=final_predictions
        )
        result.config = {
            'max_epochs': max_epochs,
            'optimizer': optimizer,
            'lr': lr,
            'regularization_weight': regularization_weight,
            'tolerance': tolerance,
            'patience': patience,
        }

        if verbose:
            logger.info("-" * 50)
            logger.info("Completed in %.2fs", self.state.total_time)

        return result

    def predict(self, model: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate predictions from model.
        
        Args:
            model: Model parameters. If None, uses current model.
            
        Returns:
            Predicted observations.
        """
        if model is None:
            model, _ = self.parameterization.forward()
        else:
            model = model.to(self.device)
        return self.forward_fn(model)

    def get_model(self) -> torch.Tensor:
        """
        Get current model parameters.
        
        Returns:
            Current model tensor (detached).
        """
        model, _ = self.parameterization.forward()
        return model.detach()

    def reset(self) -> None:
        """Reset parameterization to initial state."""
        self.parameterization.reset()
        self.state = InversionState()

    def __call__(
        self,
        observed_data: torch.Tensor,
        max_epochs: int = 100,
        **kwargs
    ) -> InversionResult:
        """
        Shortcut for fit() - enables inverter(data, max_epochs) syntax.

        Args:
            observed_data: Target observations to fit.
            max_epochs: Maximum number of epochs.
            **kwargs: Additional arguments passed to fit().

        Returns:
            InversionResult with optimized model and diagnostics.

        Example:
            >>> inverter = Inverter.create(initial_model=model, forward_fn=forward)
            >>> result = inverter(observed_data, max_epochs=100)
        """
        return self.run(observed_data, max_epochs, **kwargs)

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def from_problem(
        cls,
        problem,
        initial_model: torch.Tensor,
        **kwargs,
    ) -> 'Inverter':
        """
        Create inverter from an InverseProblem definition.

        Provides a convenient entry point when the problem is already
        defined as an :class:`~geobrain.inverse.InverseProblem`.

        Args:
            problem: InverseProblem instance.
            initial_model: Initial model parameters.
            **kwargs: Passed to :meth:`create`
                     (regularizer, transform, constraints, device).

        Returns:
            Configured Inverter instance.

        Example:
            >>> from geobrain import InverseProblem
            >>> problem = InverseProblem(forward_fn=fwd, observed=data)
            >>> inverter = Inverter.from_problem(problem, initial_model=v0)
            >>> result = inverter.run(problem.observed, max_epochs=100)
        """
        return cls.create(
            initial_model=initial_model,
            forward_fn=problem.forward_fn,
            **kwargs,
        )

    @classmethod
    def create(
        cls,
        initial_model: torch.Tensor,
        forward_fn: Callable,
        regularizer: Optional[Union[str, Callable]] = None,
        transform: Optional[Callable] = None,
        constraints: Optional[Callable] = None,
        device: Optional[torch.device] = None
    ) -> 'Inverter':
        """
        Create inverter with explicit parameterization.
        
        This is the most common setup for traditional inverse problems.
        
        Args:
            initial_model: Initial model parameters.
            forward_fn: Forward model function.
            regularizer: Regularization function or name ('l1', 'l2').
            transform: Differentiable transform for soft constraints.
            constraints: Hard constraint (projection) function.
            device: Computation device.
            
        Returns:
            Configured Inverter instance.
        
        Example:
            >>> inverter = Inverter.create(
            ...     initial_model=velocity,
            ...     forward_fn=forward,
            ...     regularizer='l2',
            ...     constraints=bound_constraint(1500, 6000)
            ... )
        """
        if isinstance(regularizer, str):
            regularizer = get_regularizer(regularizer)

        param = ExplicitParameterization(
            initial_model=initial_model,
            regularizer=regularizer,
            transform=transform,
            constraints=constraints,
            device=device
        )
        return cls(param, forward_fn, device)

    @classmethod
    def create_latent(
        cls,
        decoder: nn.Module,
        initial_latent: torch.Tensor,
        forward_fn: Callable,
        regularizer: Optional[Union[str, Callable]] = None,
        latent_constraints: Optional[Callable] = None,
        device: Optional[torch.device] = None
    ) -> 'Inverter':
        """
        Create inverter with latent code optimization.
        
        Optimize in a learned latent space defined by a pre-trained decoder.
        
        Args:
            decoder: Pre-trained decoder network.
            initial_latent: Initial latent code.
            forward_fn: Forward model function.
            regularizer: Regularization on latent code.
            latent_constraints: Constraints on latent code.
            device: Computation device.
            
        Returns:
            Configured Inverter instance.
        
        Example:
            >>> inverter = Inverter.create_latent(
            ...     decoder=vae.decoder,
            ...     initial_latent=torch.zeros(1, 128),
            ...     forward_fn=forward
            ... )
        """
        if isinstance(regularizer, str):
            regularizer = get_regularizer(regularizer)

        param = LatentParameterization(
            decoder=decoder,
            initial_latent=initial_latent,
            regularizer=regularizer,
            latent_constraints=latent_constraints,
            device=device
        )
        return cls(param, forward_fn, device)

    @classmethod
    def create_network(
        cls,
        network: nn.Module,
        fixed_input: torch.Tensor,
        forward_fn: Callable,
        regularizer: Optional[Union[str, Callable]] = None,
        output_transform: Optional[Callable] = None,
        device: Optional[torch.device] = None
    ) -> 'Inverter':
        """
        Create inverter with network weight optimization (Deep Image Prior).
        
        The model is generated by a neural network with fixed random input.
        Network architecture provides implicit regularization.
        
        Args:
            network: Neural network to optimize.
            fixed_input: Fixed input tensor.
            forward_fn: Forward model function.
            regularizer: Regularization on network weights.
            output_transform: Transform applied to network output.
            device: Computation device.
            
        Returns:
            Configured Inverter instance.
        
        Example:
            >>> inverter = Inverter.create_network(
            ...     network=UNet(),
            ...     fixed_input=torch.randn(1, 1, 64, 64),
            ...     forward_fn=forward
            ... )
        """
        if isinstance(regularizer, str):
            regularizer = get_regularizer(regularizer)

        param = NetworkParameterization(
            network=network,
            fixed_input=fixed_input,
            regularizer=regularizer,
            output_transform=output_transform,
            device=device
        )
        return cls(param, forward_fn, device)

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _run_hybrid(
        self,
        observed_data: torch.Tensor,
        max_epochs: int,
        *,
        lr: float = 0.01,
        weight_decay: float = 0.0,
        loss_fn: Union[str, Callable] = 'mse',
        regularization_weight: float = 0.0,
        tolerance: float = 1e-6,
        verbose: bool = True,
        print_every: int = 10,
        track_history: bool = True,
        adam_fraction: float = 0.6,
    ) -> InversionResult:
        """Run hybrid Adam → L-BFGS optimization.

        Phase 1 (Adam): Navigates the non-convex landscape to find a
        good basin of attraction.
        Phase 2 (L-BFGS): Refines the solution with quasi-Newton steps
        for fast local convergence.

        Args:
            adam_fraction: Fraction of max_epochs for the Adam phase.
        """
        adam_epochs = max(1, int(max_epochs * adam_fraction))
        lbfgs_epochs = max_epochs - adam_epochs

        if verbose:
            logger.info("Hybrid optimizer: Adam(%d) + L-BFGS(%d)",
                        adam_epochs, lbfgs_epochs)

        # Phase 1: Adam
        result_adam = self.run(
            observed_data, adam_epochs,
            optimizer='adam', lr=lr, weight_decay=weight_decay,
            loss_fn=loss_fn,
            regularization_weight=regularization_weight,
            tolerance=tolerance, patience=adam_epochs,
            verbose=verbose, print_every=print_every,
            track_history=track_history,
        )

        if verbose:
            logger.info("Phase 1 (Adam) done: loss=%.6e", result_adam.final_loss)
            logger.info("Starting Phase 2 (L-BFGS)...")

        # Phase 2: L-BFGS (starts from where Adam left off)
        result_lbfgs = self._run_lbfgs(
            observed_data, lbfgs_epochs,
            loss_fn=loss_fn,
            regularization_weight=regularization_weight,
            tolerance=tolerance,
            verbose=verbose, print_every=print_every,
            track_history=track_history,
        )

        # Merge histories
        merged_history = result_adam.loss_history + result_lbfgs.loss_history
        best_loss = min(merged_history)
        best_epoch = merged_history.index(best_loss)

        # Build combined result
        total_time = result_adam.total_time + result_lbfgs.total_time
        final_model = result_lbfgs.model
        best_model = (result_lbfgs.best_model
                      if result_lbfgs.best_loss <= result_adam.best_loss
                      else result_adam.best_model)

        return InversionResult(
            model=final_model,
            final_loss=result_lbfgs.final_loss,
            predictions=result_lbfgs.predictions,
            best_model=best_model,
            best_loss=best_loss,
            best_epoch=best_epoch,
            loss_history=merged_history,
            n_epochs=adam_epochs + result_lbfgs.n_epochs,
            converged=result_lbfgs.converged,
            total_time=total_time,
            config={
                'max_epochs': max_epochs,
                'optimizer': 'adam+lbfgs',
                'adam_epochs': adam_epochs,
                'lbfgs_epochs': lbfgs_epochs,
                'lr': lr,
                'regularization_weight': regularization_weight,
                'tolerance': tolerance,
            },
        )

    def _run_lbfgs(
        self,
        observed_data: torch.Tensor,
        max_epochs: int,
        *,
        loss_fn: Union[str, Callable] = 'mse',
        regularization_weight: float = 0.0,
        tolerance: float = 1e-6,
        verbose: bool = True,
        print_every: int = 10,
        track_history: bool = True,
    ) -> InversionResult:
        """Run optimization using the custom L-BFGS optimizer."""
        from .lbfgs import LBFGS

        observed_data = observed_data.to(self.device)
        loss_fn = self._resolve_loss_fn(loss_fn)
        param = self.parameterization

        # Flatten all optimizable parameters into a single vector
        opt_params = param.optimizable_parameters()
        shapes = [p.shape for p in opt_params]
        x0 = torch.cat([p.data.flatten() for p in opt_params])

        # Extract box bounds from constraints if available
        bounds = None
        if hasattr(param, 'constraints') and param.constraints is not None:
            c = param.constraints
            if hasattr(c, 'min_val') and hasattr(c, 'max_val'):
                n = x0.numel()
                lo = c.min_val if c.min_val is not None else -1e30
                hi = c.max_val if c.max_val is not None else 1e30
                bounds = torch.stack([
                    torch.full((n,), lo, dtype=x0.dtype, device=self.device),
                    torch.full((n,), hi, dtype=x0.dtype, device=self.device),
                ], dim=1)

        def fgrad_fn(x):
            # Write flat vector back into parameters
            offset = 0
            for p, s in zip(opt_params, shapes):
                numel = p.numel()
                p.data.copy_(x[offset:offset + numel].view(s))
                offset += numel

            # Zero gradients
            for p in opt_params:
                if p.grad is not None:
                    p.grad.zero_()

            # Forward pass through parameterization
            model, reg_term = param.forward()
            predictions = self.forward_fn(model)

            # Loss
            total_loss = loss_fn(predictions, observed_data)
            if reg_term is not None and regularization_weight > 0:
                total_loss = total_loss + regularization_weight * reg_term

            total_loss.backward()

            grad = torch.cat([p.grad.flatten() for p in opt_params])
            return total_loss.item(), grad.clone()

        lbfgs = LBFGS(
            memory=10,
            max_iter=max_epochs,
            tol_grad=tolerance,
            bounds=bounds,
            track_history=track_history,
        )

        lbfgs_result = lbfgs.minimize(
            fgrad_fn, x0, verbose=verbose, print_every=print_every,
        )

        # Write solution back into parameters
        offset = 0
        with torch.no_grad():
            for p, s in zip(opt_params, shapes):
                numel = p.numel()
                p.data.copy_(lbfgs_result.x[offset:offset + numel].view(s))
                offset += numel

        # Build InversionResult
        with torch.no_grad():
            final_model = self.get_model()
            final_predictions = self.predict()

        self.state = InversionState(max_epochs=max_epochs)
        self.state.loss_history = lbfgs_result.f_history
        self.state.current_loss = lbfgs_result.f_history[-1]
        self.state.best_loss = min(lbfgs_result.f_history)
        self.state.best_epoch = lbfgs_result.f_history.index(self.state.best_loss)
        self.state.epoch = lbfgs_result.n_iter - 1
        self.state.converged = lbfgs_result.converged
        self.state.total_time = lbfgs_result.total_time
        self.state.current_model = final_model

        # Recover best model from history if it differs from final
        if (lbfgs_result.x_history is not None
                and self.state.best_epoch < len(lbfgs_result.x_history)):
            best_x = lbfgs_result.x_history[self.state.best_epoch].to(self.device)
            with torch.no_grad():
                off = 0
                for p, s in zip(opt_params, shapes):
                    numel = p.numel()
                    p.data.copy_(best_x[off:off + numel].view(s))
                    off += numel
                self.state.best_model = self.get_model()
                # Restore final model parameters
                off = 0
                for p, s in zip(opt_params, shapes):
                    numel = p.numel()
                    p.data.copy_(lbfgs_result.x[off:off + numel].view(s))
                    off += numel
        else:
            self.state.best_model = final_model

        result = InversionResult.from_state(
            self.state, final_model=final_model, predictions=final_predictions,
        )
        result.config = {
            'max_epochs': max_epochs,
            'optimizer': 'lbfgs',
            'regularization_weight': regularization_weight,
            'tolerance': tolerance,
        }
        return result

    def _create_optimizer(
        self,
        name: str,
        lr: float,
        weight_decay: float
    ) -> torch.optim.Optimizer:
        """Create optimizer from name."""
        optimizers = {
            'sgd': lambda p: torch.optim.SGD(p, lr=lr, weight_decay=weight_decay, momentum=0.9),
            'adam': lambda p: torch.optim.Adam(p, lr=lr, weight_decay=weight_decay),
            'adamw': lambda p: torch.optim.AdamW(p, lr=lr, weight_decay=weight_decay),
            'adagrad': lambda p: torch.optim.Adagrad(p, lr=lr, weight_decay=weight_decay),
            'rmsprop': lambda p: torch.optim.RMSprop(p, lr=lr, weight_decay=weight_decay),
        }

        name = name.lower()
        if name not in optimizers:
            raise ValueError(f"Unknown optimizer: {name}. Available: {list(optimizers.keys())}")

        return optimizers[name](self.parameterization.optimizable_parameters())

    def _resolve_loss_fn(self, loss_fn: Union[str, Callable]) -> Callable:
        """Resolve loss function from string or callable."""
        if callable(loss_fn):
            return loss_fn
        return get_loss_fn(loss_fn)

    def _compute_gradient_norm(self) -> float:
        """Compute total gradient norm."""
        total_norm = 0.0
        for p in self.parameterization.optimizable_parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def _check_convergence(
        self, 
        tolerance: float, 
        patience: int, 
        min_improvement: float
    ) -> bool:
        """Check if optimization has converged."""
        if len(self.state.loss_history) < patience:
            return False

        recent = self.state.loss_history[-patience:]
        loss_std = np.std(recent)

        if patience > 1:
            old_loss = self.state.loss_history[-patience]
            new_loss = self.state.loss_history[-1]

            if old_loss > 0:
                improvement = abs(old_loss - new_loss) / abs(old_loss)
                if improvement < min_improvement and loss_std < tolerance:
                    return True

        return False

    def _print_progress(self, epoch: int, max_epochs: int, loss: float) -> None:
        """Print optimization progress."""
        grad_norm = self.state.grad_norm or 0.0
        msg = (
            f"[{epoch + 1:4d}/{max_epochs}] "
            f"Loss: {loss:.6e} | Best: {self.state.best_loss:.6e} | "
            f"Grad: {grad_norm:.3e}"
        )
        logger.info(msg)

    def to(self, device: Union[str, torch.device]) -> 'Inverter':
        """
        Move inverter to specified device.
        
        Args:
            device: Target device.
            
        Returns:
            Self for method chaining.
        """
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.parameterization.to(device)
        return self

    def __repr__(self) -> str:
        return (
            f"Inverter(parameterization={self.parameterization.__class__.__name__}, "
            f"device={self.device})"
        )
