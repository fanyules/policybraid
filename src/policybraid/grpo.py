from __future__ import annotations

import numpy as np


def group_advantages(rewards: list[float], epsilon: float) -> np.ndarray:
    values = np.asarray(rewards, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("rewards must be a nonempty vector")
    if not np.isfinite(values).all():
        raise ValueError("rewards must be finite")
    if epsilon <= 0:
        raise ValueError("advantage epsilon must be positive")
    centered = values - values.mean()
    return centered / np.sqrt(np.mean(centered * centered) + epsilon)


def gradient_distance(left: np.ndarray, right: np.ndarray, epsilon: float) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("gradient vectors must have the same one-dimensional shape")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("gradient vectors must be finite")
    if epsilon <= 0:
        raise ValueError("distance epsilon must be positive")
    return float(np.linalg.norm(left - right) / (np.linalg.norm(left) + epsilon))


def quantile_higher(values: list[float], probability: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("quantile values must be a finite nonempty vector")
    if not 0 <= probability <= 1:
        raise ValueError("probability must lie in [0,1]")
    return float(np.quantile(array, probability, method="higher"))

