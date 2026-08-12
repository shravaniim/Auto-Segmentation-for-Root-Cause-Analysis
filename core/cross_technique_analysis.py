"""
core/cross_technique_analysis.py
=================================
Step 4 of the task flow: aggregate the worst segments across all
segmentation techniques into one normalized, ranked top-10.
"""

from __future__ import annotations

import pandas as pd


def build_cross_technique_top10(
    combined_segments_df: pd.DataFrame,
    per_technique_n: int = 10,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    4.2 Take the N worst segments per technique (by Severity_Score).
    4.3 Normalize Root_Cause_Score across that pool.
    4.4 Return the top_n worst overall, ranked by the normalized score.
    """
    if combined_segments_df.empty or "Technique" not in combined_segments_df.columns:
        return pd.DataFrame()

    rank_col = next(
        (c for c in ["Severity_Score", "Business_Impact_Score"] if c in combined_segments_df.columns),
        None,
    )
    if rank_col is None:
        return pd.DataFrame()

    worst_per_technique = pd.concat(
        [
            g.nlargest(per_technique_n, rank_col)
            for _, g in combined_segments_df.groupby("Technique")
        ],
        ignore_index=True,
    )

    if "Root_Cause_Score" not in worst_per_technique.columns:
        worst_per_technique["Root_Cause_Score"] = 0.0
    rc = worst_per_technique["Root_Cause_Score"].fillna(0.0)
    rc_min, rc_max = rc.min(), rc.max()
    worst_per_technique["Normalized_Root_Cause_Score"] = (
        (rc - rc_min) / (rc_max - rc_min) if rc_max - rc_min > 1e-9 else 0.0
    )

    top10 = (
        worst_per_technique.sort_values("Normalized_Root_Cause_Score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    top10.insert(0, "Overall_Rank", top10.index + 1)
    return top10
