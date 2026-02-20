"""
Assessment metrics for model evaluation.

Provides common metrics for comparing true and inverted
velocity models in full waveform inversion.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.1.0
"""

import numpy as np


def mse(true_v: np.ndarray, inv_v: np.ndarray) -> float:
    """
    Mean squared error.
    
    Computes the sum of squared differences between true and
    inverted models.

    Args:
        true_v: True model with shape (nz, nx).
        inv_v: Inverted/estimated model with shape (nz, nx).

    Returns:
        Sum of squared differences (float).

    Example:
        >>> error = mse(true_velocity, inverted_velocity)
    """
    return float(np.sum((true_v - inv_v) ** 2))


def rmse(true_v: np.ndarray, inv_v: np.ndarray) -> np.ndarray:
    """
    Root mean squared error.

    Computes RMSE between true and inverted models. Supports
    batch comparison when inv_v has an extra leading dimension.

    Args:
        true_v: True model with shape (nz, nx).
        inv_v: Inverted model with shape (nz, nx) or (n_models, nz, nx).

    Returns:
        RMSE value (scalar) or array of RMSE values (n_models,).

    Example:
        >>> # Single model
        >>> error = rmse(true_v, inv_v)
        >>>
        >>> # Multiple models
        >>> errors = rmse(true_v, inv_v_batch)  # shape: (n_models,)
    """
    if len(true_v.shape) != len(inv_v.shape):
        true_v = true_v[np.newaxis, :, :]
        return np.sqrt(np.sum((true_v - inv_v) ** 2, axis=(1, 2)))
    return np.sqrt(np.sum((true_v - inv_v) ** 2))


def mape(true_v: np.ndarray, inv_v: np.ndarray) -> np.ndarray:
    """
    Mean absolute percentage error.

    Computes MAPE as a percentage. Supports batch comparison
    when inv_v has an extra leading dimension.

    Args:
        true_v: True model with shape (nz, nx).
        inv_v: Inverted model with shape (nz, nx) or (n_models, nz, nx).

    Returns:
        MAPE in percent (scalar or array).

    Example:
        >>> error_pct = mape(true_v, inv_v)
        >>> print(f"MAPE: {error_pct:.2f}%")
    """
    if len(true_v.shape) != len(inv_v.shape):
        nz, nx = true_v.shape
        true_v = true_v[np.newaxis, :, :]
        return 100 / (nx * nz) * np.sum(
            np.abs(inv_v - true_v) / true_v, axis=(1, 2)
        )

    nz, nx = true_v.shape[-2:]
    return 100 / (nx * nz) * np.sum(np.abs(inv_v - true_v) / true_v)


def snr(true_v: np.ndarray, inv_v: np.ndarray) -> np.ndarray:
    """
    Signal-to-noise ratio in dB.

    Computes SNR where the true model is the signal and the
    difference is the noise:

        SNR = 10 × log₁₀(Σtrue² / Σ(true - inv)²)

    Supports batch comparison when inv_v has an extra leading dimension.

    Args:
        true_v: True model (signal) with shape (nz, nx).
        inv_v: Inverted model with shape (nz, nx) or (n_models, nz, nx).

    Returns:
        SNR in dB (scalar or array).

    Example:
        >>> snr_db = snr(true_v, inv_v)
        >>> print(f"SNR: {snr_db:.2f} dB")
    """
    if len(true_v.shape) != len(inv_v.shape):
        snr_values = []
        for i in range(inv_v.shape[0]):
            signal = np.sum(true_v ** 2)
            noise = np.sum((true_v - inv_v[i]) ** 2)
            snr_values.append(10 * np.log10(signal / noise))
        return np.array(snr_values)

    signal = np.sum(true_v ** 2)
    noise = np.sum((true_v - inv_v) ** 2)
    return 10 * np.log10(signal / noise)