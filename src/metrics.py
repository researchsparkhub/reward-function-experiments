"""Shared comparison metrics used across E1-E4's analysis."""
from __future__ import annotations

import numpy as np


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1); p = p / p.sum()
    q = np.clip(np.asarray(q, dtype=float), 1e-9, 1); q = q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * np.log(a / b))
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def cosine_sim(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    denom = np.linalg.norm(p) * np.linalg.norm(q)
    return float(np.dot(p, q) / denom) if denom > 0 else 0.0


def rubric_mae(a: dict, b: dict, fields: list[str]) -> float:
    return float(np.mean([abs(a[k] - b[k]) for k in fields]))


def agreement_rate(p: np.ndarray, q: np.ndarray) -> float:
    """1.0 if both distributions pick the same argmax action."""
    return float(np.argmax(p) == np.argmax(q))


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (robust at small/extreme
    n, unlike the normal approximation)."""
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1 + z ** 2 / n
    centre = phat + z ** 2 / (2 * n)
    margin = z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2))
    lo = (centre - margin) / denom
    hi = (centre + margin) / denom
    return (float(max(0.0, lo)), float(min(1.0, hi)))


def mean_ci(values: list[float], z: float = 1.96) -> tuple[float, float, float]:
    """(mean, lo, hi) 95% CI for the mean via the normal approximation
    (reasonable once n gtrsim 30, which every experiment here meets)."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return (0.0, 0.0, 0.0)
    m = float(arr.mean())
    se = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return (m, m - z * se, m + z * se)
