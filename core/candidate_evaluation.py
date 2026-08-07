"""
core/candidate_evaluation.py
============================
Shared candidate-segment evaluation pipeline.  Computes performance metrics
(AUC, Gini, KS), drift metrics (PSI, bad-rate shift), statistical significance
(z-test, KS-test), business metrics (SIS, DIS), and Root Cause analysis via
feature-level PSI within segment (no SHAP required).

This module centralises the evaluation logic that was previously copy-pasted
across autoslicer, drift_tree, and gradient_boosting technique files.  The
numerical computations are **identical** — only the code organisation changed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from metrics.performance_metrics import calculate_auc_gini, calculate_ks, calculate_calibration_drift
from metrics.drift_metrics import calculate_psi, compute_psi_for_feature, build_decile_bins
from metrics.significance import (
    two_proportion_ztest_pvalue,
    ks_score_shift_pvalue,
)


def _compute_feature_drift_within_segment(
    dev_sub: pd.DataFrame,
    mon_sub: pd.DataFrame,
    feature_cols: list[str],
) -> dict:
    """
    Computes feature-level PSI for every numeric/categorical feature within
    the segment (dev segment rows vs mon segment rows).

    Returns a dict with:
        - top_drift_feature: feature with highest PSI
        - top_drift_psi: the PSI value for that feature
        - feature_drift_details: list of {feature, psi} for all features
    """
    if not feature_cols or len(dev_sub) == 0 or len(mon_sub) == 0:
        return {"top_drift_feature": None, "top_drift_psi": 0.0, "feature_drift_details": []}

    details = []
    for col in feature_cols:
        if col not in dev_sub.columns or col not in mon_sub.columns:
            continue
        try:
            dev_series = dev_sub[col].dropna()
            mon_series = mon_sub[col].dropna()
            if len(dev_series) < 5 or len(mon_series) < 5:
                continue

            if pd.api.types.is_numeric_dtype(dev_series):
                # Build bins from combined distribution (dev+mon) to ensure fair comparison
                combined = pd.concat([dev_series, mon_series])
                bins = build_decile_bins(combined)
                psi = compute_psi_for_feature(dev_series, mon_series, bins)
            else:
                # Categorical: compute PSI on category frequencies
                all_cats = set(dev_series.unique()) | set(mon_series.unique())
                dev_counts = dev_series.value_counts()
                mon_counts = mon_series.value_counts()
                psi = 0.0
                for cat in all_cats:
                    d_pct = (dev_counts.get(cat, 0) + 1e-6) / (len(dev_series) + 1e-6)
                    m_pct = (mon_counts.get(cat, 0) + 1e-6) / (len(mon_series) + 1e-6)
                    psi += (m_pct - d_pct) * np.log(m_pct / d_pct)
                psi = max(0.0, float(psi))

            details.append({"feature": col, "psi": psi})
        except Exception:
            continue

    if not details:
        return {"top_drift_feature": None, "top_drift_psi": 0.0, "feature_drift_details": []}

    top = max(details, key=lambda x: x["psi"])
    return {
        "top_drift_feature": top["feature"],
        "top_drift_psi": top["psi"],
        "feature_drift_details": details,
    }


def evaluate_segment(
    dev_df: pd.DataFrame,
    mon_df: pd.DataFrame,
    dev_mask: np.ndarray,
    mon_mask: np.ndarray,
    target_col: str,
    score_col: str,
    weight_col: Optional[str],
    total_dev_n: int,
    total_mon_n: int,
    total_dev_weight: float,
    total_mon_weight: float,
    shap_cols: Optional[list[str]] = None,
    feature_cols: Optional[list[str]] = None,
) -> Optional[dict]:
    """Compute all performance / drift / significance / root-cause metrics for one segment.

    Returns ``None`` if the masks are empty (caller should pre-filter on
    support/event counts before calling this).  Otherwise returns a flat dict
    of all metrics ready for DataFrame conversion.

    Root Cause is computed via feature-level PSI within the segment
    (dev segment rows vs mon segment rows), so no SHAP data is required.
    If ``feature_cols`` is provided, all those features are checked.
    """
    dev_sub = dev_df[dev_mask]
    mon_sub = mon_df[mon_mask]
    n_dev, n_mon = int(dev_mask.sum()), int(mon_mask.sum())
    if n_dev == 0 or n_mon == 0:
        return None

    pct_dev = n_dev / total_dev_n
    pct_mon = n_mon / total_mon_n

    # --- PSI (population share) ---
    psi = calculate_psi(pct_dev, pct_mon)

    # --- Bad-rate and z-test ---
    x_dev = int(dev_sub[target_col].sum())
    x_mon = int(mon_sub[target_col].sum())
    br_dev = float(dev_sub[target_col].mean())
    br_mon = float(mon_sub[target_col].mean())
    delta_br = br_mon - br_dev
    br_pvalue = two_proportion_ztest_pvalue(x_dev, n_dev, x_mon, n_mon)

    # --- AUC / Gini ---
    auc_dev, gini_dev = calculate_auc_gini(dev_sub[target_col], dev_sub[score_col])
    auc_mon, gini_mon = calculate_auc_gini(mon_sub[target_col], mon_sub[score_col])
    delta_auc = (
        auc_mon - auc_dev
        if not (np.isnan(auc_dev) or np.isnan(auc_mon))
        else np.nan
    )
    delta_gini = (
        gini_mon - gini_dev
        if not (np.isnan(gini_dev) or np.isnan(gini_mon))
        else np.nan
    )

    # --- Calibration Drift (A/E ratio shift) ---
    dev_actual = float(dev_sub[target_col].sum())
    dev_expected = float(dev_sub[score_col].sum())
    mon_actual = float(mon_sub[target_col].sum())
    mon_expected = float(mon_sub[score_col].sum())
    calibration_drift = calculate_calibration_drift(
        dev_actual, dev_expected, mon_actual, mon_expected
    )

    # --- KS ---
    ks_dev = calculate_ks(dev_sub[target_col], dev_sub[score_col])
    ks_mon = calculate_ks(mon_sub[target_col], mon_sub[score_col])
    delta_ks = (
        ks_mon - ks_dev
        if not (np.isnan(ks_dev) or np.isnan(ks_mon))
        else np.nan
    )

    # --- Score-distribution shift ---
    score_shift_pvalue = ks_score_shift_pvalue(
        dev_sub[score_col].values, mon_sub[score_col].values
    )

    # --- Weight / exposure ---
    w_dev = float(dev_sub[weight_col].sum()) if weight_col and weight_col in dev_sub.columns else 0.0
    w_mon = float(mon_sub[weight_col].sum()) if weight_col and weight_col in mon_sub.columns else 0.0
    dev_weight_pct = w_dev / total_dev_weight if total_dev_weight > 0 else 0.0
    mon_weight_pct = w_mon / total_mon_weight if total_mon_weight > 0 else 0.0
    exposure_drift = mon_weight_pct - dev_weight_pct

    # --- Root Cause: Feature-Level PSI within segment ---
    # Determine feature columns to analyse (exclude target, score, weight, id-like cols)
    if feature_cols is None:
        _exclude = {target_col, score_col}
        if weight_col:
            _exclude.add(weight_col)
        feature_cols = [
            c for c in dev_df.columns
            if c not in _exclude
            and not c.lower().startswith("shap_")
            and dev_df[c].dtype != object or dev_df[c].nunique() <= 50
        ]

    rc = _compute_feature_drift_within_segment(dev_sub, mon_sub, feature_cols)
    top_drift_feature = rc["top_drift_feature"]
    top_drift_psi = rc["top_drift_psi"]

    # Root Cause Score = feature PSI × population impact (Gini drop component)
    gini_drop = max(0.0, -delta_gini) if not np.isnan(delta_gini) else 0.0
    root_cause_score = round(top_drift_psi * (1.0 + gini_drop) * pct_mon, 6)

    return {
        "n_dev": n_dev,
        "n_mon": n_mon,
        "pct_dev": pct_dev,
        "pct_mon": pct_mon,
        "x_dev": x_dev,
        "x_mon": x_mon,
        "psi": psi,
        "br_dev": br_dev,
        "br_mon": br_mon,
        "delta_br": delta_br,
        "br_pvalue": br_pvalue,
        "auc_dev": auc_dev,
        "auc_mon": auc_mon,
        "delta_auc": delta_auc,
        "gini_dev": gini_dev,
        "gini_mon": gini_mon,
        "delta_gini": delta_gini,
        "ks_dev": ks_dev,
        "ks_mon": ks_mon,
        "delta_ks": delta_ks,
        "score_shift_pvalue": score_shift_pvalue,
        "weight_dev": w_dev,
        "weight_mon": w_mon,
        "dev_weight_pct": dev_weight_pct,
        "mon_weight_pct": mon_weight_pct,
        "exposure_drift": exposure_drift,
        "calibration_drift": calibration_drift,
        # Root Cause (feature-level drift within segment)
        "top_drift_feature": top_drift_feature,
        "top_drift_psi": top_drift_psi,
        "root_cause_score": root_cause_score,
        # Keep SHAP fields for backward compatibility (will be None if no SHAP data)
        "top_shap_shift_feature": top_drift_feature,
        "top_shap_shift_psi": top_drift_psi,
    }
