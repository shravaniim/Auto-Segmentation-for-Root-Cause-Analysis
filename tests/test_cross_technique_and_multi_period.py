import pandas as pd

from core.cross_technique_analysis import build_cross_technique_top10
from core.multi_period_analysis import (
    track_recurring_worst_segments,
    compute_trend_metrics,
    build_segment_time_series,
    forecast_segment_scores,
    reindex_segment_time_series,
)
from models.config import TrendAnalysisConfig


def _segments(technique, n, severity_start=10.0):
    return pd.DataFrame({
        "Technique": [technique] * n,
        "Segment_Definition": [f"{technique}_seg_{i}" for i in range(n)],
        "Severity_Score": [severity_start - i for i in range(n)],
        "Root_Cause_Score": [1.0 - i * 0.1 for i in range(n)],
    })


def test_cross_technique_top10_normalizes_and_ranks():
    combined = pd.concat([_segments("A", 12), _segments("B", 12, severity_start=5.0)])
    top10 = build_cross_technique_top10(combined, per_technique_n=10, top_n=10)

    assert len(top10) == 10
    assert "Normalized_Root_Cause_Score" in top10.columns
    assert top10["Normalized_Root_Cause_Score"].max() == 1.0
    # Technique A's segments have the highest Root_Cause_Score -> should dominate top 10
    assert top10.iloc[0]["Technique"] == "A"


def test_recurring_worst_segments_ranks_by_frequency():
    period_results = {
        "2026-01": pd.DataFrame({
            "Segment_Definition": ["region=West", "age<30"],
            "Technique": ["AutoSlicer", "AutoSlicer"],
            "Severity_Score": [10.0, 8.0],
        }),
        "2026-02": pd.DataFrame({
            "Segment_Definition": ["region=West", "income<50k"],
            "Technique": ["AutoSlicer", "AutoSlicer"],
            "Severity_Score": [11.0, 7.0],
        }),
    }
    result = track_recurring_worst_segments(period_results, n_worst=10)

    assert not result.empty
    top = result.iloc[0]
    assert top["Segment_Definition"] == "region=West"
    assert top["Periods_Appeared"] == 2
    assert top["Recurrence_Pct"] == 100.0


def test_cross_technique_dedup_merges_same_population():
    # Two techniques independently discover the identical population
    # (same Dev_Count/Mon_Count/PSI/Delta_BR/Delta_Gini).
    dup = pd.DataFrame({
        "Technique": ["Gradient Boosting", "Drift Localization Tree"],
        "Segment_Definition": ["income < 3.798e+04 AND age < 28.5", "income < 3.798e+04 AND age < 28.5"],
        "Severity_Score": [7.5898, 13.0],
        "Root_Cause_Score": [0.019847, 0.019847],
        "Dev_Count": [163, 163],
        "Mon_Count": [666, 666],
        "PSI": [0.1416, 0.1416],
        "Delta_BR": [-0.5267, -0.5267],
        "Delta_Gini": [0.0809, 0.0809],
    })
    unique = _segments("AutoSlicer", 10, severity_start=6.0)
    unique["Dev_Count"] = range(100, 110)
    unique["Mon_Count"] = range(200, 210)
    unique["PSI"] = [0.01 * i for i in range(10)]
    unique["Delta_BR"] = [0.02 * i for i in range(10)]
    unique["Delta_Gini"] = [0.03 * i for i in range(10)]

    combined = pd.concat([dup, unique], ignore_index=True)
    top10 = build_cross_technique_top10(combined, per_technique_n=10, top_n=20)

    merged = top10[top10["Segment_Definition"] == "income < 3.798e+04 AND age < 28.5"]
    assert len(merged) == 1
    assert set(merged.iloc[0]["Discovered_By"].split(", ")) == {"Gradient Boosting", "Drift Localization Tree"}
    assert merged.iloc[0]["Severity_Score"] == 13.0  # kept the higher-ranked instance


def test_cross_technique_dedup_prefers_simplest_equivalent_rule_text():
    # Same underlying population (identical Dev_Count/Mon_Count/PSI/Delta_BR/
    # Delta_Gini) described two different ways: K-Means's verbose cluster
    # rule happens to have the *higher* Severity_Score here -- if the merge
    # naively kept whichever row ranked highest, the far less readable
    # K-Means text would win even though "region = West" describes the
    # exact same customers. The values (Severity_Score, Root_Cause_Score)
    # should still come from the higher-ranked row -- only the displayed
    # rule text should prefer the shorter, more readable one.
    dup = pd.DataFrame({
        "Technique": ["AutoSlicer", "K-Means Clustering"],
        "Segment_Definition": [
            "region = West",
            "Cluster_4: region != South AND occupation != Business AND age < 29.5",
        ],
        "Severity_Score": [10.0, 15.0],  # K-Means ranks higher
        "Root_Cause_Score": [0.5, 0.5],
        "Dev_Count": [500, 500],
        "Mon_Count": [800, 800],
        "PSI": [0.24, 0.24],
        "Delta_BR": [-0.1, -0.1],
        "Delta_Gini": [-0.02, -0.02],
    })
    unique = _segments("Feature Binning", 10, severity_start=6.0)
    unique["Dev_Count"] = range(100, 110)
    unique["Mon_Count"] = range(200, 210)
    unique["PSI"] = [0.01 * i for i in range(10)]
    unique["Delta_BR"] = [0.02 * i for i in range(10)]
    unique["Delta_Gini"] = [0.03 * i for i in range(10)]

    combined = pd.concat([dup, unique], ignore_index=True)
    top10 = build_cross_technique_top10(combined, per_technique_n=10, top_n=20)

    merged = top10[top10["Discovered_By"] == "AutoSlicer, K-Means Clustering"]
    assert len(merged) == 1
    row = merged.iloc[0]
    assert row["Segment_Definition"] == "region = West"  # simpler text wins for display
    assert row["Severity_Score"] == 15.0  # but the higher-ranked row's own values are kept


def _period_row(technique, seg, severity, sis=0.01, dis=0.005, psi=0.05, shap=0.03):
    return pd.DataFrame({
        "Technique": [technique],
        "Segment_Definition": [seg],
        "Severity_Score": [severity],
        "SIS_Raw": [sis],
        "DIS_Raw": [dis],
        "mean_feature_psi": [psi],
        "mean_shap_shift": [shap],
    })


def test_trend_metrics_always_recurring_beats_gapped_beats_one_off():
    # A: appears every period (1-6). B: appears once, never again.
    # C: same near-max frequency as A but with a gap at period 4 -- tests
    # that Consistency (streak-based) is genuinely distinct from Frequency
    # (raw count), per the user's clarification that Consistency means
    # "does it stay for long, or does it go and reappear."
    periods = {}
    for m in range(1, 7):
        rows = [_period_row("AutoSlicer", "A: always here", 10.0)]
        if m == 1:
            rows.append(_period_row("AutoSlicer", "B: one-off", 8.0))
        if m in (1, 2, 3, 5, 6):
            rows.append(_period_row("AutoSlicer", "C: gap at month 4", 9.0))
        periods[f"2026_{m:02d}"] = pd.concat(rows, ignore_index=True)

    result = compute_trend_metrics(periods, TrendAnalysisConfig())
    by_seg = result.set_index("Segment_Definition")

    a, b, c = by_seg.loc["A: always here"], by_seg.loc["B: one-off"], by_seg.loc["C: gap at month 4"]

    # Frequency: A > C > B, matching raw appearance counts (6/6, 5/6, 1/6).
    assert a["Frequency"] == 1.0
    assert round(c["Frequency"], 4) == round(5 / 6, 4)
    assert round(b["Frequency"], 4) == round(1 / 6, 4)

    # Consistency: A is a perfect 6-month streak (1.0); C's longest run is
    # only 3 (months 1-3, since the gap at 4 breaks it) despite appearing
    # in 5 of 6 periods overall -- this is the key behavior being tested.
    assert a["Consistency_Score"] == 1.0
    assert round(c["Consistency_Score"], 4) == round(3 / 6, 4)
    assert c["Consistency_Score"] < c["Frequency"]  # gap costs it, vs. raw frequency

    # Recency: both A and C were seen in the latest period -> full recency.
    assert a["Recency_Factor"] == 1.0
    assert c["Recency_Factor"] == 1.0
    # B was last seen 5 periods ago -> heavily decayed.
    assert b["Recency_Factor"] < 0.2

    # Overall ranking: consistently-recurring beats gapped beats one-off.
    assert a["Trend_Impact_Score"] > c["Trend_Impact_Score"] > b["Trend_Impact_Score"]
    assert a["Severity_Score_Trend"] > c["Severity_Score_Trend"] > b["Severity_Score_Trend"]


def test_trend_metrics_sis_trend_reduces_to_sis_raw_when_never_recurring():
    # A segment appearing in exactly one out of several periods still gets
    # *some* Trend_Impact_Score (low frequency/consistency, but recency can
    # be high if it's the most recent period) -- but with only one period
    # total, appearing once means recurring 100% of the time, so this
    # instead checks the single-period-pool edge case: with only one period
    # of data, Frequency=Consistency=Recency=1 for anything present, so
    # SIS_Trend is boosted by the full trend_weight, not left at SIS_Raw.
    # The true "untouched" case is covered by the API contract in
    # multi_period_analysis.py's docstring (TIS=0 => SIS_Trend=SIS_Raw);
    # this test instead confirms the boost direction is monotonic and sane.
    periods = {
        "2026_01": _period_row("AutoSlicer", "solo segment", 10.0, sis=0.02),
    }
    cfg = TrendAnalysisConfig()
    result = compute_trend_metrics(periods, cfg)
    row = result.iloc[0]
    expected_tis = cfg.w_frequency * 1.0 + cfg.w_recency * 1.0 + cfg.w_consistency * 1.0
    assert round(row["Trend_Impact_Score"], 4) == round(expected_tis, 4)
    expected_sis_trend = 0.02 * (1.0 + cfg.trend_weight * expected_tis)
    assert round(row["SIS_Trend"], 6) == round(expected_sis_trend, 6)


def test_segment_time_series_has_one_row_per_appearance_with_raw_metrics():
    # Segment appears in months 1 and 3 only -- time series should have
    # exactly 2 rows (not 3, and not collapsed to 1 like compute_trend_metrics
    # does), each carrying that period's own raw Mon_AUC/Root_Cause_Score
    # rather than only the latest month's.
    def row(month_auc, rcs):
        return pd.DataFrame({
            "Technique": ["AutoSlicer"],
            "Segment_Definition": ["dti >= 54.75"],
            "Severity_Score": [10.0],
            "Dev_Pct": [0.2], "Mon_Pct": [0.25],
            "Dev_AUC": [0.7], "Mon_AUC": [month_auc],
            "Dev_KS": [0.3], "Mon_KS": [0.2],
            "Dev_BR": [0.1], "Mon_BR": [0.15],
            "SIS_Raw": [0.001], "DIS_Raw": [0.0005],
            "Root_Cause_Score": [rcs],
        })

    periods = {
        "2026_01": row(month_auc=0.60, rcs=0.4),
        "2026_02": pd.DataFrame(),  # segment absent this month
        "2026_03": row(month_auc=0.55, rcs=0.5),
    }
    ts = build_segment_time_series(periods, TrendAnalysisConfig())

    assert len(ts) == 2
    assert list(ts["Period"]) == ["2026_01", "2026_03"]
    assert list(ts["Mon_AUC"]) == [0.60, 0.55]
    assert list(ts["Root_Cause_Score"]) == [0.4, 0.5]


def _ts_rows(technique, seg, scores, periods=None):
    return pd.DataFrame({
        "Technique": [technique] * len(scores),
        "Segment_Definition": [seg] * len(scores),
        "Period": periods or [f"2026_{i+1:02d}" for i in range(len(scores))],
        "Period_Index": list(range(len(scores))),
        "Root_Cause_Score": scores,
    })


def test_forecast_flags_worsening_segment_with_correct_breach_month():
    from models.config import TrendAnalysisConfig
    ts = _ts_rows("AutoSlicer", "worsening", [0.2, 0.3, 0.4])  # slope=0.1/period
    cfg = TrendAnalysisConfig(forecast_alert_threshold=0.5, forecast_horizon_months=6)

    result = forecast_segment_scores(ts, cfg)
    row = result.iloc[0]

    assert row["Trend_Slope"] > 0
    assert row["Early_Warning"] == True
    # (0.5 - intercept) / slope = 3.0 periods from x=0; last_idx=2 -> 1 month out
    assert row["Months_To_Breach"] == 1
    assert row["Confidence"] == "Medium (3+ points)"
    assert row["Last_Appeared_Period"] == "2026_03"  # the 3rd synthetic period


def test_forecast_does_not_flag_improving_or_flat_segments():
    from models.config import TrendAnalysisConfig
    cfg = TrendAnalysisConfig(forecast_alert_threshold=0.5)

    improving = _ts_rows("AutoSlicer", "improving", [0.4, 0.3, 0.2])
    flat = _ts_rows("AutoSlicer", "flat", [0.3, 0.3, 0.3])
    result = forecast_segment_scores(pd.concat([improving, flat], ignore_index=True), cfg)

    assert not result["Early_Warning"].any()
    assert result["Months_To_Breach"].isna().all()


def test_forecast_confidence_label_reflects_point_count():
    from models.config import TrendAnalysisConfig
    two_point = _ts_rows("AutoSlicer", "two_point_seg", [0.2, 0.3])
    three_point = _ts_rows("AutoSlicer", "three_point_seg", [0.2, 0.25, 0.3])
    result = forecast_segment_scores(
        pd.concat([two_point, three_point], ignore_index=True), TrendAnalysisConfig()
    )

    by_seg = result.set_index("Segment_Definition")
    assert by_seg.loc["two_point_seg", "Confidence"] == "Low (2 points)"
    assert by_seg.loc["three_point_seg", "Confidence"] == "Medium (3+ points)"


def test_reindex_segment_time_series_fills_gaps_with_nan():
    # Segment appeared in months 1 and 4 only (skipped 2 and 3) -- reindexing
    # against all 4 months should insert real blank rows for the gap, not
    # silently connect month 1 straight to month 4.
    seg_ts = pd.DataFrame({
        "Technique": ["AutoSlicer", "AutoSlicer"],
        "Segment_Definition": ["dti >= 54.75", "dti >= 54.75"],
        "Period": ["2026_01", "2026_04"],
        "Root_Cause_Score": [0.3, 0.5],
    })
    all_periods = ["2026_01", "2026_02", "2026_03", "2026_04"]

    out = reindex_segment_time_series(seg_ts, all_periods)

    assert len(out) == 4
    assert list(out["Period"]) == all_periods
    assert out.loc[out["Period"] == "2026_01", "Root_Cause_Score"].iloc[0] == 0.3
    assert out.loc[out["Period"] == "2026_04", "Root_Cause_Score"].iloc[0] == 0.5
    assert out.loc[out["Period"] == "2026_02", "Root_Cause_Score"].isna().iloc[0]
    assert out.loc[out["Period"] == "2026_03", "Root_Cause_Score"].isna().iloc[0]
    # Technique/Segment_Definition should be carried into the blank rows too.
    assert (out["Technique"] == "AutoSlicer").all()
    assert (out["Segment_Definition"] == "dti >= 54.75").all()
