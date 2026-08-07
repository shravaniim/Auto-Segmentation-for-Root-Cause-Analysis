"""
metrics/drift_metrics.py
========================
Population-level drift metrics: PSI (Population Stability Index), feature-level
PSI, PSI interpretation, and SHAP-shift detection.

All functions are extracted **verbatim** from ``segmentation_common.py``.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


def calculate_psi(dev_pct: float, mon_pct: float, eps: float = 1e-6) -> float:
    """Single-bin PSI contribution (Population Stability Index)."""
    dev_pct_adj = max(dev_pct, eps)
    mon_pct_adj = max(mon_pct, eps)
    return (mon_pct_adj - dev_pct_adj) * np.log(mon_pct_adj / dev_pct_adj)


def build_decile_bins(series: pd.Series) -> np.ndarray:
    """Build 10-quantile bin edges for a numeric series."""
    clean = series.dropna().to_numpy(dtype=float)
    if clean.size == 0:
        return np.array([-np.inf, np.inf], dtype=float)

    quantiles = np.quantile(clean, np.linspace(0.0, 1.0, 11))
    unique_edges = np.unique(quantiles)
    if unique_edges.size < 2:
        return np.array([-np.inf, np.inf], dtype=float)

    edges = unique_edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def compute_psi_for_feature(
    dev_series: pd.Series, mon_series: pd.Series, bins: np.ndarray
) -> float:
    """Full-feature PSI using pre-computed bin edges."""
    dev_clean = dev_series.dropna()
    mon_clean = mon_series.dropna()
    if dev_clean.empty or mon_clean.empty:
        return 0.0

    dev_binned = pd.cut(dev_clean, bins=bins, include_lowest=True, duplicates="drop")
    mon_binned = pd.cut(mon_clean, bins=bins, include_lowest=True, duplicates="drop")

    categories = dev_binned.cat.categories.union(mon_binned.cat.categories)
    dev_counts = dev_binned.value_counts(sort=False).reindex(categories, fill_value=0)
    mon_counts = mon_binned.value_counts(sort=False).reindex(categories, fill_value=0)

    dev_pct = (dev_counts.to_numpy(dtype=float) + 1e-6) / (
        dev_counts.sum() + 1e-6 * len(categories)
    )
    mon_pct = (mon_counts.to_numpy(dtype=float) + 1e-6) / (
        mon_counts.sum() + 1e-6 * len(categories)
    )
    psi = np.sum((mon_pct - dev_pct) * np.log(mon_pct / dev_pct))
    return float(max(0.0, psi))


def interpret_psi(value: float) -> str:
    """Human-readable PSI interpretation band."""
    if value < 0.1:
        return "Stable"
    if value <= 0.25:
        return "Moderate Drift"
    return "Significant Drift"


def detect_shap_shift(
    dev_df: pd.DataFrame, mon_df: pd.DataFrame, shap_cols: list[str]
) -> dict:
    """Measures SHAP-value PSI across columns; returns the top-shifting feature."""
    if not shap_cols:
        return {"top_shift_feature": None, "top_shift_psi": 0.0, "details": []}

    combined = pd.concat(
        [dev_df[shap_cols], mon_df[shap_cols]], ignore_index=True
    ).stack()
    bins = build_decile_bins(combined)

    details = []
    for col in shap_cols:
        psi = compute_psi_for_feature(dev_df[col], mon_df[col], bins)
        dev_mean = float(dev_df[col].mean()) if len(dev_df[col]) > 0 else np.nan
        mon_mean = float(mon_df[col].mean()) if len(mon_df[col]) > 0 else np.nan
        delta = (
            float(mon_mean - dev_mean)
            if not (np.isnan(dev_mean) or np.isnan(mon_mean))
            else np.nan
        )
        details.append(
            {
                "feature": col,
                "dev_mean": dev_mean,
                "mon_mean": mon_mean,
                "delta": delta,
                "psi": psi,
                "interpretation": interpret_psi(psi),
            }
        )

    top = max(details, key=lambda row: row["psi"])
    return {
        "top_shift_feature": top["feature"],
        "top_shift_psi": top["psi"],
        "details": details,
    }
