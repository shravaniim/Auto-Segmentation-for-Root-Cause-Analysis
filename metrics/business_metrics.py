"""
metrics/business_metrics.py
===========================
Business-impact and composite scoring metrics: Business Impact Score,
Segment Impact Score (SIS), Drift Impact Score (DIS).

All functions are extracted **verbatim** from ``segmentation_common.py``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def calculate_business_impact(
    pct_mon: float,
    mon_weight_pct: float,
    psi: float = 0.0,
    delta_gini: float = 0.0,
) -> float:
    """Composite business impact: population share × exposure share ×
    (1 + PSI + Gini penalty)."""
    psi_val = max(0.0, psi) if not np.isnan(psi) else 0.0
    gini_penalty = max(0.0, -delta_gini) if not np.isnan(delta_gini) else 0.0
    return float((pct_mon * mon_weight_pct) * (1.0 + psi_val + gini_penalty))


def calculate_confidence(mon_count: float) -> float:
    """Sample-size confidence: sqrt(n/100), capped at 1.0.

    Matches the baseline (segment_discovery/segment_root_cause_analysis.py)
    convention: small monitoring-side samples get down-weighted.
    """
    if mon_count is None or np.isnan(mon_count):
        return 0.0
    return float(min(1.0, np.sqrt(max(0.0, mon_count) / 100.0)))


def calculate_sis(
    auc_drop: float,
    exposure_factor: float,
    pct_mon: float,
    confidence: float,
) -> float:
    """Segment Impact Score = AUC drop x exposure share x population share
    x sample-size confidence. Matches the baseline formula exactly."""
    auc_drop_val = max(0.0, auc_drop) if not np.isnan(auc_drop) else 0.0
    return float(auc_drop_val * exposure_factor * pct_mon * confidence)


def calculate_dis(
    population_drift: float,
    auc_drop: float,
    exposure_factor: float,
    total_psi: float,
) -> float:
    """Drift Impact Score = population drift x AUC drop x exposure share x
    (1 + feature-drift PSI), floored at 0. Matches the baseline formula."""
    auc_drop_val = max(0.0, auc_drop) if not np.isnan(auc_drop) else 0.0
    psi_val = max(0.0, total_psi) if not np.isnan(total_psi) else 0.0
    dis = population_drift * auc_drop_val * exposure_factor * (1.0 + psi_val)
    return float(max(0.0, dis))


def calculate_root_cause_score(
    norm_sis: float,
    norm_dis: float,
    norm_total_psi: float,
    norm_total_shap_shift: float,
    weights: Optional[dict] = None,
) -> float:
    """Root Cause Score = weighted sum of min-max-normalized SIS, DIS,
    feature-drift PSI, and SHAP shift, normalized across one technique's
    own candidate pool. Matches the baseline weights (0.35/0.30/0.20/0.15).
    """
    w = {"sis": 0.35, "dis": 0.30, "psi": 0.20, "shap": 0.15}
    if weights:
        w.update(weights)
    return float(
        w["sis"] * norm_sis
        + w["dis"] * norm_dis
        + w["psi"] * norm_total_psi
        + w["shap"] * norm_total_shap_shift
    )

