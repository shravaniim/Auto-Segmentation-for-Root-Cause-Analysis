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


def calculate_sis(
    psi: float,
    delta_gini: float,
    delta_ks: float,
    delta_br: float,
    pct_mon: float,
    mon_weight_pct: float,
    weights: Optional[dict] = None,
) -> dict:
    """Segment Impact Score — weighted composite of drift and performance
    components.  Returns a dict with ``raw``, ``business_impact``,
    ``gini_drop``, ``ks_drop``, ``br_shift``."""
    default_weights = {
        "psi": 0.25,
        "business_impact": 0.25,
        "gini_drop": 0.20,
        "ks_drop": 0.15,
        "br_shift": 0.15,
    }
    if weights:
        default_weights.update(weights)

    psi_val = max(0.0, psi) if not np.isnan(psi) else 0.0
    business_impact = calculate_business_impact(pct_mon, mon_weight_pct, psi, delta_gini)
    gini_drop = max(0.0, -delta_gini) if not np.isnan(delta_gini) else 0.0
    ks_drop = max(0.0, -delta_ks) if not np.isnan(delta_ks) else 0.0
    # delta_br = mon_br - dev_br; only a POSITIVE shift (more defaults in
    # monitoring) is deterioration. A negative shift is bad-rate
    # improvement and must not score as severity.
    br_shift = max(0.0, delta_br) if not np.isnan(delta_br) else 0.0

    raw = (
        default_weights["psi"] * psi_val
        + default_weights["business_impact"] * business_impact
        + default_weights["gini_drop"] * gini_drop
        + default_weights["ks_drop"] * ks_drop
        + default_weights["br_shift"] * br_shift
    )
    return {
        "raw": float(raw),
        "business_impact": float(business_impact),
        "gini_drop": float(gini_drop),
        "ks_drop": float(ks_drop),
        "br_shift": float(br_shift),
    }


def calculate_dis(
    psi: float, delta_gini: float, delta_ks: float, delta_br: float
) -> float:
    """Drift Impact Score — simple additive composite.

    All performance/bad-rate terms are one-sided (deterioration only):
    delta = monitoring - development, so a positive delta_br (more
    defaults) or negative delta_gini/delta_ks (worse discrimination)
    increases DIS. Improvements score 0, not a symmetric penalty.
    """
    psi_val = max(0.0, psi) if not np.isnan(psi) else 0.0
    ks_drop = max(0.0, -delta_ks) if not np.isnan(delta_ks) else 0.0
    br_shift = max(0.0, delta_br) if not np.isnan(delta_br) else 0.0
    return float(psi_val + max(0.0, -delta_gini) + ks_drop + br_shift)


def calculate_root_cause_score(
    shap_shift_magnitude: float,
    segment_impact_score: float
) -> float:
    """
    Computes the Root Cause Score for a segment based on the magnitude of the
    primary feature's SHAP shift multiplied by the segment's overall business impact.
    """
    if np.isnan(shap_shift_magnitude) or np.isnan(segment_impact_score):
        return 0.0
    return float(shap_shift_magnitude * segment_impact_score)

