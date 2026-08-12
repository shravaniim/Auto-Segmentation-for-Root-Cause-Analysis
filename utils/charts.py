"""
utils/charts.py
================
Heatmap, bubble chart, and waterfall chart for cross-technique segment
insights (task-flow step 2.15 / 4.6). Matplotlib only — no new dependency.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def segment_metric_heatmap(df: pd.DataFrame, label_col: str = "Segment_Definition"):
    """Segments (rows) x drift/performance metrics (columns)."""
    metric_cols = [c for c in ["PSI", "Delta_Gini", "Delta_KS", "Delta_BR", "Root_Cause_Score"] if c in df.columns]
    if df.empty or not metric_cols:
        return None

    data = df[metric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    labels = df[label_col].astype(str).str.slice(0, 40) if label_col in df.columns else df.index.astype(str)

    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(df))))
    norm_data = (data - data.min()) / (data.max() - data.min() + 1e-9)
    im = ax.imshow(norm_data.values, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(metric_cols)))
    ax.set_xticklabels(metric_cols, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data.values[i, j]:.3f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, label="Normalized severity")
    ax.set_title("Segment x Metric Heatmap")
    fig.tight_layout()
    return fig


def segment_bubble_chart(df: pd.DataFrame):
    """Population % (x) vs Gini drop (y), bubble size = exposure %, color = technique."""
    needed = ["Mon_Pct", "Delta_Gini"]
    if df.empty or not all(c in df.columns for c in needed):
        return None

    d = df.copy()
    d["_gini_drop"] = pd.to_numeric(d["Delta_Gini"], errors="coerce").apply(lambda x: max(0.0, -x) if pd.notna(x) else 0.0)
    exposure_col = "Mon_Exposure_Pct" if "Mon_Exposure_Pct" in d.columns else None
    d["_exposure"] = pd.to_numeric(d[exposure_col], errors="coerce").fillna(0.02) if exposure_col else 0.05
    sizes = 200 + 3000 * d["_exposure"].clip(lower=0)

    fig, ax = plt.subplots(figsize=(7, 5))
    techniques = d["Technique"].astype(str).unique() if "Technique" in d.columns else ["All"]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(techniques), 1)))
    for tech, color in zip(techniques, colors):
        mask = (d["Technique"] == tech) if "Technique" in d.columns else np.ones(len(d), dtype=bool)
        ax.scatter(
            pd.to_numeric(d.loc[mask, "Mon_Pct"], errors="coerce"),
            d.loc[mask, "_gini_drop"],
            s=sizes[mask], alpha=0.6, label=str(tech), color=color, edgecolors="black",
        )
    ax.set_xlabel("Monitoring Population %")
    ax.set_ylabel("Gini Drop (positive = worse)")
    ax.set_title("Segment Impact Bubble Chart (bubble size = exposure %)")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    return fig


def sis_waterfall_chart(segment_row: pd.Series, weights: dict | None = None):
    """Breaks SIS_Raw down into its weighted components for one segment."""
    weights = weights or {"psi": 0.25, "business_impact": 0.25, "gini_drop": 0.20, "ks_drop": 0.15, "br_shift": 0.15}

    def _get(col, default=0.0):
        return float(segment_row[col]) if col in segment_row and pd.notna(segment_row[col]) else default

    psi = max(0.0, _get("PSI"))
    gini_drop = max(0.0, -_get("Delta_Gini"))
    ks_drop = max(0.0, -_get("Delta_KS"))
    br_shift = max(0.0, _get("Delta_BR"))
    business_impact = _get("Business_Impact_Score")

    components = {
        "PSI": weights["psi"] * psi,
        "Business Impact": weights["business_impact"] * business_impact,
        "Gini Drop": weights["gini_drop"] * gini_drop,
        "KS Drop": weights["ks_drop"] * ks_drop,
        "Bad-Rate Shift": weights["br_shift"] * br_shift,
    }

    labels = list(components.keys()) + ["SIS_Raw"]
    values = list(components.values())
    cumulative = np.cumsum(values)
    total = cumulative[-1] if len(cumulative) else 0.0

    fig, ax = plt.subplots(figsize=(7, 4))
    bottoms = [0] + list(cumulative[:-1])
    ax.bar(labels[:-1], values, bottom=bottoms, color="#d62728")
    ax.bar(labels[-1], total, color="#2ca02c")
    for i, v in enumerate(values):
        ax.text(i, bottoms[i] + v / 2, f"{v:.3f}", ha="center", va="center", fontsize=8)
    ax.text(len(labels) - 1, total / 2, f"{total:.3f}", ha="center", va="center", fontsize=8)
    ax.set_ylabel("Contribution to SIS_Raw")
    seg_name = segment_row.get("Segment_Definition", "Segment")
    ax.set_title(f"SIS Waterfall — {seg_name}")
    fig.tight_layout()
    return fig
