"""
metrics/significance.py
=======================
Statistical significance testing, multiple-testing correction, and utility
normalisation functions.

All functions are extracted **verbatim** from ``segmentation_common.py``.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp, norm


# ---------------------------------------------------------------------------
# Hypothesis tests
# ---------------------------------------------------------------------------

def two_proportion_ztest_pvalue(x1: int, n1: int, x2: int, n2: int) -> float:
    """Two-sided z-test for the equality of two proportions."""
    if n1 == 0 or n2 == 0:
        return np.nan
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    denom = p_pool * (1 - p_pool) * (1 / n1 + 1 / n2)
    if denom <= 0:
        return np.nan
    z = (p2 - p1) / np.sqrt(denom)
    return float(2 * (1 - norm.cdf(abs(z))))


def ks_score_shift_pvalue(dev_scores, mon_scores) -> float:
    """Two-sample KS test p-value for score-distribution shift."""
    if len(dev_scores) < 2 or len(mon_scores) < 2:
        return np.nan
    try:
        return float(ks_2samp(dev_scores, mon_scores).pvalue)
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Multiple-testing correction
# ---------------------------------------------------------------------------

def adjust_p_values_bh(p_values) -> np.ndarray:
    """Benjamini-Hochberg FDR correction.  Shared so every technique that runs
    many simultaneous hypothesis tests corrects the same way."""
    p = np.asarray(p_values, dtype=float)
    valid = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    if valid.sum() == 0:
        return q
    pv = p[valid]
    n = len(pv)
    order = np.argsort(pv)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    qv = np.minimum.accumulate((pv[order] * n / ranks[order])[::-1])[::-1]
    qv_full = np.empty(n)
    qv_full[order] = np.minimum(qv, 1.0)
    q[valid] = qv_full
    return q


# ---------------------------------------------------------------------------
# Confidence & normalisation helpers
# ---------------------------------------------------------------------------

def drift_confidence(p_br: float, p_ks: float) -> float:
    """Combines bad-rate and score-shift p-values into a single [0, 1]
    confidence score."""
    p_br = 1.0 if np.isnan(p_br) else float(p_br)
    p_ks = 1.0 if np.isnan(p_ks) else float(p_ks)
    return float(max(0.0, min(1.0, 1.0 - min(p_br, p_ks))))


def min_max_normalize(values) -> np.ndarray:
    """Min-max normalisation to [0, 1].  Returns zeros when range is
    degenerate."""
    arr = np.asarray(values, dtype=float)
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)
