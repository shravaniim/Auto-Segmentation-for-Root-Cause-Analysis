"""
metrics/performance_metrics.py
==============================
Core model-performance metrics: AUC, Gini, KS.

All functions are extracted **verbatim** from ``segmentation_common.py`` so that
numerical output is identical.  Feature-binning's local ``safe_auc`` and
``calculate_ks`` duplicates are retired in favour of these shared versions.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def calculate_auc_gini(y_true, y_score) -> tuple[float, float]:
    """Return ``(auc, gini)`` for a binary classification.  Returns
    ``(nan, nan)`` if the target is constant or the calculation fails."""
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    try:
        auc = roc_auc_score(y_true, y_score)
        return auc, 2.0 * auc - 1.0
    except Exception:
        return np.nan, np.nan


def safe_auc(y_true, y_score) -> float:
    """Convenience wrapper returning only AUC (no Gini).  Replaces the
    duplicate ``safe_auc`` that lived in ``feature_binning_segmentation.py``."""
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return np.nan


def calculate_ks(y_true, y_score) -> float:
    """Kolmogorov-Smirnov statistic for rank-ordering quality.

    Logic is identical to the original ``segmentation_common.calculate_ks``
    *and* the local copy that lived in ``feature_binning_segmentation.py``."""
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        idx = np.argsort(-y_score)
        y_true_sorted = y_true[idx]
        total_bads = np.sum(y_true_sorted == 1)
        total_goods = np.sum(y_true_sorted == 0)
        if total_bads == 0 or total_goods == 0:
            return np.nan
        cum_bads = np.cumsum(y_true_sorted == 1) / total_bads
        cum_goods = np.cumsum(y_true_sorted == 0) / total_goods
        return float(np.max(np.abs(cum_bads - cum_goods)))
    except Exception:
        return np.nan


def calculate_calibration_drift(
    dev_actual: float, dev_expected: float,
    mon_actual: float, mon_expected: float
) -> float:
    """
    Calculates the signed shift in the Actual/Expected (A/E) ratio
    between development and monitoring datasets (monitoring - development).
    Positive = A/E ratio rose in monitoring; negative = it fell.
    """
    dev_ae = dev_actual / max(dev_expected, 1e-6)
    mon_ae = mon_actual / max(mon_expected, 1e-6)
    return float(mon_ae - dev_ae)

