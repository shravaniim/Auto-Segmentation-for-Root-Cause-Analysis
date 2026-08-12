"""
core/multi_period_analysis.py
==============================
Step 3 of the task flow: run segment analysis across multiple scoring-date
snapshots and identify segments that recur as worst performers.
"""

from __future__ import annotations

import pandas as pd


def track_recurring_worst_segments(
    period_results: dict[str, pd.DataFrame],
    n_worst: int = 10,
) -> pd.DataFrame:
    """
    period_results: {period_label: segments_df} for each scoring date,
    each segments_df already ranked with a Severity_Score column and a
    Segment_Definition column (e.g. one snapshot per monitoring month).

    Returns segments ranked by how many periods they appeared in the
    worst-N list, tie-broken by mean Severity_Score across appearances.
    """
    rows = []
    for period, df in period_results.items():
        if df.empty or "Severity_Score" not in df.columns:
            continue
        worst = df.nlargest(n_worst, "Severity_Score")
        for _, r in worst.iterrows():
            rows.append({
                "Segment_Definition": r.get("Segment_Definition", "Unknown"),
                "Technique": r.get("Technique", "Unknown"),
                "Period": period,
                "Severity_Score": r.get("Severity_Score", 0.0),
            })

    if not rows:
        return pd.DataFrame()

    long_df = pd.DataFrame(rows)
    summary = (
        long_df.groupby(["Segment_Definition", "Technique"])
        .agg(
            Periods_Appeared=("Period", "nunique"),
            Total_Periods=("Period", lambda s: long_df["Period"].nunique()),
            Avg_Severity_Score=("Severity_Score", "mean"),
        )
        .reset_index()
    )
    summary["Recurrence_Pct"] = (
        summary["Periods_Appeared"] / summary["Total_Periods"] * 100
    ).round(1)
    return summary.sort_values(
        ["Periods_Appeared", "Avg_Severity_Score"], ascending=False
    ).reset_index(drop=True)
