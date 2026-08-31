"""
core/multi_period_analysis.py
==============================
Step 3 of the task flow: run segment analysis across multiple scoring-date
snapshots and identify segments that recur as worst performers.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from metrics.business_metrics import calculate_root_cause_score
from metrics.significance import min_max_normalize
from models.config import TrendAnalysisConfig


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


def _build_period_appearance_long_df(
    period_results: dict[str, pd.DataFrame],
    n_worst: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Long format: one row per (technique, segment, period) it appeared in
    that technique's own worst-n_worst (by Severity_Score) for that period.
    Carries the *entire* original row (not just the trend inputs) so both
    compute_trend_metrics (chart-compatible summary) and
    build_segment_time_series (raw month-by-month metrics) can be built
    from the same appearance data without recomputing it twice.

    Returns (long_df, periods) where periods is the period labels in the
    order period_results was given (relies on dict insertion order).
    """
    periods = list(period_results.keys())
    rows = []
    for period_idx, period in enumerate(periods):
        df = period_results[period]
        needed = {"Technique", "Segment_Definition", "Severity_Score"}
        if df.empty or not needed.issubset(df.columns):
            continue
        for tech, tdf in df.groupby("Technique"):
            worst = tdf.nlargest(n_worst, "Severity_Score")
            for _, r in worst.iterrows():
                row = r.to_dict()
                row["Period"] = period
                row["Period_Index"] = period_idx
                row.setdefault("SIS_Raw", 0.0)
                row.setdefault("DIS_Raw", 0.0)
                row.setdefault("mean_feature_psi", 0.0)
                row.setdefault("mean_shap_shift", 0.0)
                rows.append(row)

    return pd.DataFrame(rows), periods


def build_segment_time_series(
    period_results: dict[str, pd.DataFrame],
    cfg: TrendAnalysisConfig | None = None,
) -> pd.DataFrame:
    """
    Add-on to the Trend Analysis Summary table (which only keeps each
    segment's *latest* month's raw numbers plus the derived trend score):
    this returns the full month-by-month raw metrics for every matched
    segment, one row per (Technique, Segment_Definition, Period), so a UI
    can let someone pick a segment and see how Dev/Mon population %, AUC,
    KS, bad rate, SIS, DIS and Root_Cause_Score actually moved release to
    release -- not just the single trend-boosted summary number.

    Uses the exact same "appeared in that technique's own worst-N" matching
    as compute_trend_metrics, so the segments covered here are identical to
    the ones in the summary table.
    """
    cfg = cfg or TrendAnalysisConfig()
    long_df, periods = _build_period_appearance_long_df(period_results, cfg.n_worst)
    if long_df.empty:
        return pd.DataFrame()

    metric_cols = [
        "Dev_Pct", "Mon_Pct", "Dev_AUC", "Mon_AUC", "Dev_KS", "Mon_KS",
        "Dev_BR", "Mon_BR", "SIS_Raw", "DIS_Raw", "Root_Cause_Score",
    ]
    keep_cols = ["Technique", "Segment_Definition", "Period", "Period_Index"] + [
        c for c in metric_cols if c in long_df.columns
    ]
    out = long_df[keep_cols].sort_values(["Technique", "Segment_Definition", "Period_Index"])
    return out.reset_index(drop=True)


def reindex_segment_time_series(seg_ts: pd.DataFrame, all_periods: list[str]) -> pd.DataFrame:
    """Reindex one segment's sparse time-series rows (from
    build_segment_time_series, already filtered to a single Technique +
    Segment_Definition) against the *full* list of months analyzed --
    inserting a blank row (NaN metrics) for any month the segment didn't
    rank in that technique's worst-N.

    Without this, a segment that appeared in months 1 and 4 only would
    plot as two adjacent points on a 2-tick axis, visually implying a
    single smooth step from month 1 to month 4 -- this makes the real gap
    (months 2-3: not in the worst-N that period) show up as an actual
    break in the line/table instead, matching what Consistency_Score is
    measuring.
    """
    if seg_ts.empty or not all_periods:
        return seg_ts

    full = pd.DataFrame({"Period": all_periods, "Period_Index": range(len(all_periods))})
    merge_cols = [c for c in seg_ts.columns if c not in ("Period", "Period_Index")]
    out = full.merge(seg_ts[["Period"] + merge_cols], on="Period", how="left")

    # Technique/Segment_Definition are constant for a single segment --
    # carry them into the blank rows too, rather than leaving those NaN.
    for col in ("Technique", "Segment_Definition"):
        if col in out.columns:
            out[col] = out[col].ffill().bfill()

    return out


def compute_trend_metrics(
    period_results: dict[str, pd.DataFrame],
    cfg: TrendAnalysisConfig | None = None,
) -> pd.DataFrame:
    """
    Trend-analysis version of the recurrence tracking above: for each
    technique, matches segments across periods by their (already
    canonicalized) Segment_Definition text, computes Frequency /
    Recency_Factor / Consistency_Score -> Trend_Impact_Score, folds that
    into SIS (SIS_Trend), and re-derives Root_Cause_Score_Trend /
    Severity_Score_Trend using the same formula as the single-period
    Root_Cause_Score (metrics.business_metrics.calculate_root_cause_score),
    just with SIS_Trend substituted for SIS_Raw and renormalized across
    each technique's matched-segment set.

    period_results: {period_label: combined_segments_df}, in chronological
    order (relies on dict insertion order) -- one entry per monitoring
    month, each the combined_segments_df returned by
    compare_segmentation_techniques.benchmark_all_techniques(). Must
    contain Technique, Segment_Definition, Severity_Score, SIS_Raw,
    DIS_Raw, mean_feature_psi, mean_shap_shift.

    "Appeared" means the segment ranked in that technique's own top
    cfg.n_worst (by Severity_Score) for that period -- same definition
    track_recurring_worst_segments uses above.

    Returns one row per (Technique, Segment_Definition) that appeared in
    at least one period, ranked by Severity_Score_Trend within its
    technique.
    """
    cfg = cfg or TrendAnalysisConfig()
    long_df, periods = _build_period_appearance_long_df(period_results, cfg.n_worst)
    total_periods = len(periods)
    if total_periods == 0 or long_df.empty:
        return pd.DataFrame()

    summary_rows = []
    for (tech, seg), grp in long_df.groupby(["Technique", "Segment_Definition"]):
        appeared_idx = sorted(grp["Period_Index"].unique().tolist())
        frequency = len(appeared_idx) / total_periods

        last_idx = appeared_idx[-1]
        periods_since_last = (total_periods - 1) - last_idx
        recency_factor = 0.5 ** (periods_since_last / cfg.recency_half_life_periods)

        # Longest run of *consecutive* period indices this segment appeared
        # in -- distinct from Frequency (total count regardless of pattern).
        longest_run = 1
        current_run = 1
        for i in range(1, len(appeared_idx)):
            if appeared_idx[i] == appeared_idx[i - 1] + 1:
                current_run += 1
                longest_run = max(longest_run, current_run)
            else:
                current_run = 1
        consistency_score = longest_run / total_periods

        tis = (
            cfg.w_frequency * frequency
            + cfg.w_recency * recency_factor
            + cfg.w_consistency * consistency_score
        )

        # Base row: this segment's most recent appearance, in full -- keeps
        # every original column (PSI, Delta_Gini, Mon_Pct, Business_Impact_Score,
        # Root_Cause_Feature, feature_drift_details, etc.) so the trend
        # summary is chart-compatible, not just a handful of trend inputs.
        latest = grp.loc[grp["Period_Index"].idxmax()].to_dict()
        sis_raw = float(latest.get("SIS_Raw", 0.0) or 0.0)
        sis_trend = sis_raw * (1.0 + cfg.trend_weight * tis)

        latest.update({
            "Technique": tech,
            "Segment_Definition": seg,
            "Periods_Appeared": len(appeared_idx),
            "Total_Periods": total_periods,
            "Appeared_In": ", ".join(periods[i] for i in appeared_idx),
            "Frequency": round(frequency, 4),
            "Recency_Factor": round(recency_factor, 4),
            "Consistency_Score": round(consistency_score, 4),
            "Trend_Impact_Score": round(tis, 4),
            "SIS_Raw": sis_raw,
            "SIS_Trend": sis_trend,
            "DIS_Raw": float(latest.get("DIS_Raw", 0.0) or 0.0),
            "mean_feature_psi": float(latest.get("mean_feature_psi", 0.0) or 0.0),
            "mean_shap_shift": float(latest.get("mean_shap_shift", 0.0) or 0.0),
        })
        latest.pop("Period", None)
        latest.pop("Period_Index", None)
        summary_rows.append(latest)

    result = pd.DataFrame(summary_rows)

    # Root_Cause_Score_Trend / Severity_Score_Trend: same formula/weights as
    # the single-period Root_Cause_Score, SIS_Trend substituted for SIS_Raw,
    # renormalized within each technique's matched-segment set.
    out_parts = []
    for tech, grp in result.groupby("Technique"):
        grp = grp.copy()
        norm_sis = min_max_normalize(grp["SIS_Trend"].values)
        norm_dis = min_max_normalize(grp["DIS_Raw"].values)
        norm_psi = min_max_normalize(grp["mean_feature_psi"].values)
        norm_shap = min_max_normalize(grp["mean_shap_shift"].values)
        grp["Root_Cause_Score_Trend"] = [
            calculate_root_cause_score(norm_sis[i], norm_dis[i], norm_psi[i], norm_shap[i])
            for i in range(len(grp))
        ]
        grp["Severity_Score_Trend"] = grp["Root_Cause_Score_Trend"] * 20.0
        out_parts.append(grp)

    result = pd.concat(out_parts, ignore_index=True)
    return result.sort_values(
        ["Technique", "Severity_Score_Trend"], ascending=[True, False]
    ).reset_index(drop=True)


def forecast_segment_scores(
    time_series_df: pd.DataFrame,
    cfg: TrendAnalysisConfig | None = None,
) -> pd.DataFrame:
    """
    Early-warning forecast: for each segment with >= 2 months of history in
    time_series_df (from build_segment_time_series), fits a simple
    least-squares line through (Period_Index, Root_Cause_Score) and
    projects one month forward. This is a plain linear extrapolation, not a
    statistical model -- it's meant to flag "worth watching," not to be a
    precise prediction.

    A segment is flagged (Early_Warning=True) when its trend is worsening
    (positive slope) and, projected forward, is on track to cross
    cfg.forecast_alert_threshold within cfg.forecast_horizon_months.

    Returns one row per forecastable segment: Technique, Segment_Definition,
    Periods_Used, Current_Root_Cause_Score, Predicted_Next_Root_Cause_Score,
    Trend_Slope, Months_To_Breach, Confidence, Early_Warning.
    """
    cfg = cfg or TrendAnalysisConfig()
    if time_series_df.empty or "Root_Cause_Score" not in time_series_df.columns:
        return pd.DataFrame()

    rows = []
    for (tech, seg), grp in time_series_df.groupby(["Technique", "Segment_Definition"]):
        grp = grp.sort_values("Period_Index")
        n = len(grp)
        if n < 2:
            continue

        x = grp["Period_Index"].to_numpy(dtype=float)
        y = pd.to_numeric(grp["Root_Cause_Score"], errors="coerce").to_numpy(dtype=float)
        if np.isnan(y).any():
            continue

        slope, intercept = np.polyfit(x, y, deg=1)
        last_idx = x.max()
        current_score = float(y[-1])
        predicted_next = float(np.clip(slope * (last_idx + 1) + intercept, 0.0, 1.0))

        months_to_breach = None
        early_warning = False
        if slope > 0 and current_score < cfg.forecast_alert_threshold:
            raw_breach_idx = (cfg.forecast_alert_threshold - intercept) / slope
            months_out = math.ceil(raw_breach_idx - last_idx)
            if 0 < months_out <= cfg.forecast_horizon_months:
                months_to_breach = months_out
                early_warning = True

        last_period = grp["Period"].iloc[-1] if "Period" in grp.columns else None

        rows.append({
            "Technique": tech,
            "Segment_Definition": seg,
            "Periods_Used": n,
            "Last_Appeared_Period": last_period,
            "Current_Root_Cause_Score": round(current_score, 4),
            "Predicted_Next_Root_Cause_Score": round(predicted_next, 4),
            "Trend_Slope": round(float(slope), 5),
            "Months_To_Breach": months_to_breach,
            "Confidence": "Low (2 points)" if n == 2 else "Medium (3+ points)",
            "Early_Warning": early_warning,
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    return result.sort_values(
        ["Early_Warning", "Months_To_Breach"], ascending=[False, True], na_position="last"
    ).reset_index(drop=True)
