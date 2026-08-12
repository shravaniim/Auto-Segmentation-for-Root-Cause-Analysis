"""
core/cross_technique_analysis.py
=================================
Step 4 of the task flow: aggregate the worst segments across all
segmentation techniques into one normalized, ranked top-10.
"""

from __future__ import annotations

import pandas as pd


def _population_fingerprint(row: pd.Series) -> tuple:
    """Proxy for Jaccard mask-overlap once row-level masks are no longer
    available (they're dropped after each technique's own dedup). Two
    candidates from different techniques that land on identical
    Dev_Count/Mon_Count/PSI/Delta_BR/Delta_Gini are, in practice, the same
    underlying population discovered independently.
    """
    def _r(col):
        val = row.get(col)
        return round(float(val), 6) if pd.notna(val) else None

    dev_count, mon_count = row.get("Dev_Count"), row.get("Mon_Count")
    if pd.isna(dev_count) or pd.isna(mon_count):
        # Not enough signal to safely treat this as a duplicate of anything.
        return (row.name,)

    return (dev_count, mon_count, _r("PSI"), _r("Delta_BR"), _r("Delta_Gini"))


def _merge_cross_technique_duplicates(pool: pd.DataFrame, rank_col: str) -> pd.DataFrame:
    """Consolidate rows that are the same population discovered by more than
    one technique into a single row, keeping the highest-rank_col instance
    and listing every technique that found it."""
    pool = pool.copy()
    pool["_fp"] = pool.apply(_population_fingerprint, axis=1)

    kept_rows = []
    for _, group in pool.groupby("_fp", sort=False):
        techniques = sorted(group["Technique"].astype(str).unique())
        best = group.sort_values(rank_col, ascending=False).iloc[0].copy()
        best["Discovered_By"] = ", ".join(techniques)
        kept_rows.append(best)

    merged = pd.DataFrame(kept_rows).drop(columns=["_fp"]).reset_index(drop=True)
    return merged


def build_cross_technique_top10(
    combined_segments_df: pd.DataFrame,
    per_technique_n: int = 10,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    4.2 Take the N worst segments per technique (by Severity_Score).
    Consolidate segments independently discovered by multiple techniques
    into one row (Discovered_By lists them) before ranking.
    4.3 Normalize Root_Cause_Score across that pool (true min-max).
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

    worst_per_technique = _merge_cross_technique_duplicates(worst_per_technique, rank_col)

    if "Root_Cause_Score" not in worst_per_technique.columns:
        worst_per_technique["Root_Cause_Score"] = 0.0
    rc = worst_per_technique["Root_Cause_Score"].fillna(0.0)
    rc_min, rc_max = rc.min(), rc.max()
    # True min-max: (value - min) / (max - min). Note this pool's minimum is
    # commonly 0.0 (a candidate can legitimately score Root_Cause_Score=0),
    # in which case the formula algebraically reduces to value/max -- that
    # is a property of this data, not a different (max-relative) formula.
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
