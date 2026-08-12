import pandas as pd

from core.cross_technique_analysis import build_cross_technique_top10
from core.multi_period_analysis import track_recurring_worst_segments


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
