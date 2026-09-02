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
from scipy import stats as scipy_stats

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
    can let someone pick a segment and see how Dev/Mon population %,
    exposure, AUC, KS, bad rate, SIS/DIS, SHAP shift and Root_Cause_Score
    actually moved release to release -- not just the single trend-boosted
    summary number.

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
        "Dev_BR", "Mon_BR", "Delta_AUC", "Delta_Gini", "Delta_KS", "Delta_BR", "PSI",
        "SIS_Raw", "DIS_Raw", "DIS_Symmetric", "Root_Cause_Score",
        "Root_Cause_Feature", "Root_Cause_PSI",
        "Dev_EAD", "Mon_EAD", "Dev_Exposure_Pct", "Mon_Exposure_Pct", "Exposure_Drift",
        "Top_SHAP_Feature", "Top_SHAP_PSI", "mean_shap_shift",
        "Business_Impact_Score", "Calibration_Drift",
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
        # SIS_Trend is kept as its own informational column (SIS on its own,
        # boosted by persistence) -- but it is NOT what feeds Root_Cause_Score_Trend
        # below, to avoid double-counting the persistence boost (see there).
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

    # Root_Cause_Score_Trend / Severity_Score_Trend: blend the same four
    # RAW (unboosted) components the single-period Root_Cause_Score uses --
    # SIS_Raw, not SIS_Trend -- then apply the persistence boost once, to
    # the whole blended score, not just to the 35%-weighted SIS slice of it.
    #
    # Previously the boost only touched SIS_Trend before blending, so
    # persistence could only ever move 35% of the final score -- a segment
    # that recurred every month (Trend_Impact_Score=1.0, the max boost)
    # could still be outranked by a technique's single most-severe one-off
    # finding, since the other 65% of the score (DIS/PSI/SHAP) was never
    # persistence-aware at all. Boosting the whole blend makes a perfectly
    # recurring segment's score move by the full trend_weight, not 35% of it.
    # Clipped at 1.0 since the boost can now push the blend above the
    # normalized [0,1] range that Severity_Score_Trend's 0-20 scale assumes.
    out_parts = []
    for tech, grp in result.groupby("Technique"):
        grp = grp.copy()
        norm_sis = min_max_normalize(grp["SIS_Raw"].values)
        norm_dis = min_max_normalize(grp["DIS_Raw"].values)
        norm_psi = min_max_normalize(grp["mean_feature_psi"].values)
        norm_shap = min_max_normalize(grp["mean_shap_shift"].values)
        tis_values = grp["Trend_Impact_Score"].to_numpy(dtype=float)
        base_scores = [
            calculate_root_cause_score(norm_sis[i], norm_dis[i], norm_psi[i], norm_shap[i])
            for i in range(len(grp))
        ]
        grp["Root_Cause_Score_Trend"] = [
            min(1.0, base_scores[i] * (1.0 + cfg.trend_weight * tis_values[i]))
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
    time_series_df (from build_segment_time_series), fits an ordinary
    least-squares linear regression of Root_Cause_Score on Period_Index
    (scipy.stats.linregress) and projects one period forward.

    Reports the regression statistics alongside the point forecast, not just
    the point forecast itself: R_Squared (goodness of fit), Trend_P_Value
    (is the slope statistically distinguishable from zero), and a 95%
    prediction interval on the forecast (Predicted_Next_Lower95/Upper95) --
    so "this segment is trending worse" is a claim backed by a confidence
    interval, not just an extrapolated line. With only 2 data points there
    are 0 residual degrees of freedom, so no valid p-value or prediction
    interval can be computed -- those segments still get a point forecast
    (the regression line through 2 points is well-defined), just flagged
    "Low (2 points)" confidence with R_Squared/Trend_P_Value/interval left
    as NaN rather than reporting a false sense of precision.

    A segment is flagged (Early_Warning=True) when its trend is worsening
    (positive slope) and, projected forward, is on track to cross
    cfg.forecast_alert_threshold within cfg.forecast_horizon_months -- same
    flagging rule as before, so no previously-flagged segment silently
    disappears now that more statistics are reported alongside it.

    Returns one row per forecastable segment: Technique, Segment_Definition,
    Periods_Used, Current_Root_Cause_Score, Predicted_Next_Root_Cause_Score,
    Predicted_Next_Lower95, Predicted_Next_Upper95, Trend_Slope, R_Squared,
    Trend_P_Value, Months_To_Breach, Confidence, Early_Warning.
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

        fit = scipy_stats.linregress(x, y)
        slope, intercept = fit.slope, fit.intercept
        last_idx = x.max()
        next_x = last_idx + 1
        current_score = float(y[-1])
        predicted_next = float(np.clip(slope * next_x + intercept, 0.0, 1.0))

        # 95% prediction interval for the forecast point -- only defined
        # with >= 1 residual degree of freedom (n >= 3). With exactly 2
        # points the line passes through both exactly (r_value = +-1),
        # leaving no residual variance to estimate an interval from.
        r_squared = float(fit.rvalue ** 2)
        p_value = float(fit.pvalue)
        lower95 = upper95 = np.nan
        if n >= 3:
            residuals = y - (slope * x + intercept)
            dof = n - 2
            resid_std_err = math.sqrt(np.sum(residuals ** 2) / dof)
            mean_x = float(np.mean(x))
            sxx = float(np.sum((x - mean_x) ** 2))
            if sxx > 0:
                se_pred = resid_std_err * math.sqrt(1.0 + 1.0 / n + (next_x - mean_x) ** 2 / sxx)
                t_crit = scipy_stats.t.ppf(0.975, df=dof)
                margin = t_crit * se_pred
                lower95 = float(np.clip(predicted_next - margin, 0.0, 1.0))
                upper95 = float(np.clip(predicted_next + margin, 0.0, 1.0))
        else:
            r_squared = np.nan
            p_value = np.nan

        months_to_breach = None
        early_warning = False
        if slope > 0 and current_score < cfg.forecast_alert_threshold:
            raw_breach_idx = (cfg.forecast_alert_threshold - intercept) / slope
            # Round before ceil(): linregress's slope/intercept differ from
            # np.polyfit's in the last few bits of floating-point precision
            # (different underlying algorithms for the same OLS fit), which
            # can push a case that lands exactly on an integer month boundary
            # a hair over it (e.g. 1.0000000000000004), flipping ceil() to
            # the next month up for what's really a whole-number answer.
            months_out = math.ceil(round(raw_breach_idx - last_idx, 6))
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
            "Predicted_Next_Lower95": round(lower95, 4) if not np.isnan(lower95) else None,
            "Predicted_Next_Upper95": round(upper95, 4) if not np.isnan(upper95) else None,
            "Trend_Slope": round(float(slope), 5),
            "R_Squared": round(r_squared, 4) if not np.isnan(r_squared) else None,
            "Trend_P_Value": round(p_value, 4) if not np.isnan(p_value) else None,
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
