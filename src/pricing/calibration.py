from __future__ import annotations
import numpy as np

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))

def apply_logit_shift(p: np.ndarray, shift: float) -> np.ndarray:
    """p' = sigmoid(logit(p) + shift)"""
    return _sigmoid(_logit(p) + float(shift))

def estimate_logit_shift(p: np.ndarray, target_mean: float) -> float:
    """
    Find shift so mean(apply_logit_shift(p, shift)) ~= target_mean.
    Monotone => bisection.
    """
    target_mean = float(target_mean)
    if not (0.0 < target_mean < 1.0):
        raise ValueError("target_mean must be in (0,1)")

    p = np.asarray(p, dtype=float)
    lo, hi = -20.0, 20.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        m = float(apply_logit_shift(p, mid).mean())
        if m < target_mean:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
