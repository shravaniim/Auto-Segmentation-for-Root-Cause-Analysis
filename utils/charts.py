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


def segment_metric_time_series_chart(
    df: pd.DataFrame,
    metric_col: str,
    segment_label: str = "",
    forecast_point: float | None = None,
):
    """One segment's raw metric value across months (x = Period, y = metric_col).
    df must already be filtered to a single (Technique, Segment_Definition)
    and sorted by Period_Index -- see core.multi_period_analysis.build_segment_time_series.

    forecast_point, if given, plots one extra dashed-line point after the
    last actual month -- the early-warning forecast's predicted next value
    (see core.multi_period_analysis.forecast_segment_scores). Only applies
    to Root_Cause_Score, since that's what the forecast is fit on."""
    if df.empty or metric_col not in df.columns or "Period" not in df.columns:
        return None

    fig, ax = plt.subplots(figsize=(7, 3.5))
    x_labels = df["Period"].astype(str).tolist()
    y_values = pd.to_numeric(df[metric_col], errors="coerce").tolist()
    ax.plot(x_labels, y_values, marker="o", color="#1f77b4", label="Actual")

    if forecast_point is not None and metric_col == "Root_Cause_Score" and y_values:
        ax.plot(
            [x_labels[-1], "Next month (forecast)"],
            [y_values[-1], forecast_point],
            marker="o", linestyle="--", color="#d62728", label="Forecast",
        )
        ax.legend(fontsize=8)

    ax.set_xlabel("Month")
    ax.set_ylabel(metric_col)
    ax.set_title(f"{metric_col} over time" + (f" — {segment_label}" if segment_label else ""))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def segment_all_metrics_chart(df: pd.DataFrame, segment_label: str = ""):
    """All of a segment's key 0-1-scale metrics on one chart, across every
    month analyzed -- Mon_Pct, Mon_AUC, Mon_BR, Root_Cause_Score. df must
    already be reindexed against the *full* period list (see
    core.multi_period_analysis.reindex_segment_time_series), so a month the
    segment didn't rank in its technique's worst-N shows as a real gap in
    each line, not a skipped step that implies continuity there wasn't."""
    metric_cols = [c for c in ["Mon_Pct", "Mon_AUC", "Mon_BR", "Root_Cause_Score"] if c in df.columns]
    if df.empty or not metric_cols or "Period" not in df.columns:
        return None

    colors = {"Mon_Pct": "#1f77b4", "Mon_AUC": "#2ca02c", "Mon_BR": "#d62728", "Root_Cause_Score": "#9467bd"}
    labels = {
        "Mon_Pct": "Population %", "Mon_AUC": "AUC", "Mon_BR": "Bad Rate",
        "Root_Cause_Score": "Root Cause Score",
    }

    fig, ax = plt.subplots(figsize=(9, 4.2))
    x_labels = df["Period"].astype(str).tolist()
    any_data = False
    for col in metric_cols:
        y = pd.to_numeric(df[col], errors="coerce").tolist()
        if all(v != v for v in y):  # all-NaN column, e.g. never appeared
            continue
        any_data = True
        ax.plot(x_labels, y, marker="o", label=labels.get(col, col), color=colors.get(col))
    if not any_data:
        return None

    ax.set_xlabel("Month")
    ax.set_ylabel("Value (0-1 scale)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("All Key Metrics Over Time" + (f" — {segment_label}" if segment_label else ""))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    return fig


def early_warning_urgency_chart(forecast_df: pd.DataFrame, top_n: int = 10):
    """Horizontal bar chart of the most urgent Early Warning segments --
    shortest bar (fewest months to breach) at the top. Bars are solid for
    Medium (3+ points) confidence and hatched for Low (2 points), so the
    least-certain forecasts are visually distinguishable at a glance, not
    just readable from a text column.

    forecast_df is core.multi_period_analysis.forecast_segment_scores'
    output; only Early_Warning=True rows are plotted."""
    if forecast_df.empty or "Early_Warning" not in forecast_df.columns:
        return None

    flagged = forecast_df[forecast_df["Early_Warning"]].sort_values("Months_To_Breach")
    if flagged.empty:
        return None

    flagged = flagged.head(top_n)
    labels = [
        f"{r.Technique}: {r.Segment_Definition}"[:60]
        for r in flagged.itertuples()
    ]
    months = flagged["Months_To_Breach"].tolist()
    hatches = ["" if "Medium" in c else "//" for c in flagged["Confidence"]]

    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.5 * len(flagged))))
    y_pos = range(len(flagged))
    bars = ax.barh(y_pos, months, color="#f97316", edgecolor="#7c2d12")
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()  # most urgent (fewest months) at the top
    ax.set_xlabel("Months Until Projected to Cross Alert Threshold")
    ax.set_title("Early Warning — Most Urgent Segments")
    for bar, m in zip(bars, months):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2, str(m),
                va="center", fontsize=8)
    # Legend explaining the hatch pattern, since color alone doesn't show confidence.
    solid_patch = plt.Rectangle((0, 0), 1, 1, facecolor="#f97316", edgecolor="#7c2d12", label="Medium confidence (3+ months of data)")
    hatch_patch = plt.Rectangle((0, 0), 1, 1, facecolor="#f97316", edgecolor="#7c2d12", hatch="//", label="Low confidence (2 months of data)")
    ax.legend(handles=[solid_patch, hatch_patch], fontsize=7, loc="lower right")
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
