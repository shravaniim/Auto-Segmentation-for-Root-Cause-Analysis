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


def compute_categorical_psi(dev_series: pd.Series, mon_series: pd.Series) -> float:
    """Frequency-based PSI: one term per distinct value, instead of continuous
    quantile bins. Same methodology already used for categorical features in
    core/candidate_evaluation.py's per-feature drift loop, factored out here
    so it can be reused for low-cardinality/near-discrete numeric columns
    (e.g. SHAP-proxy columns that only take 2-3 distinct values within a
    segment) -- decile bins are unstable for those and can blow up to
    non-standard magnitudes when a value present in one period is absent
    from the other.
    """
    dev_clean = dev_series.dropna()
    mon_clean = mon_series.dropna()
    if dev_clean.empty or mon_clean.empty:
        return 0.0

    all_cats = set(dev_clean.unique()) | set(mon_clean.unique())
    dev_counts = dev_clean.value_counts()
    mon_counts = mon_clean.value_counts()

    eps = 1e-6
    dev_denominator = len(dev_clean) + eps
    mon_denominator = len(mon_clean) + eps

    psi = 0.0
    for category in all_cats:
        dev_pct = (dev_counts.get(category, 0) + eps) / dev_denominator
        mon_pct = (mon_counts.get(category, 0) + eps) / mon_denominator
        psi += (mon_pct - dev_pct) * np.log(mon_pct / dev_pct)

    return float(max(0.0, psi))


def detect_shap_shift(
    dev_df: pd.DataFrame,
    mon_df: pd.DataFrame,
    shap_cols: list[str],
    categorical_max_card: int = 20,
) -> dict:
    """Measures SHAP-value PSI across columns; returns the top-shifting feature.

    Each column gets its own bins, built from that column's own dev+mon
    values -- not a set of bins pooled across every shap_cols column. Pooling
    collapses distinct columns onto the same bin edges whenever their scales
    differ, which previously produced identical, inflated PSI values across
    unrelated columns (e.g. three different SHAP columns all reporting the
    exact same PSI). Columns with few distinct values (<= categorical_max_card,
    matching SchemaConfig's default) use frequency-based categorical PSI
    instead of quantile bins, since decile bins are unstable/misleading for
    near-discrete values.
    """
    if not shap_cols:
        return {"top_shift_feature": None, "top_shift_psi": 0.0, "details": []}

    details = []
    for col in shap_cols:
        dev_series = dev_df[col]
        mon_series = mon_df[col]
        combined = pd.concat([dev_series, mon_series], ignore_index=True).dropna()

        if combined.nunique() <= categorical_max_card:
            psi = compute_categorical_psi(dev_series, mon_series)
        else:
            bins = build_decile_bins(combined)
            psi = compute_psi_for_feature(dev_series, mon_series, bins)

        dev_mean = float(dev_series.mean()) if len(dev_series) > 0 else np.nan
        mon_mean = float(mon_series.mean()) if len(mon_series) > 0 else np.nan
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
