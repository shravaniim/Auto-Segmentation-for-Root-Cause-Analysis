"""
Regression tests for the correctness audit fixes:
  - bad-rate shift must be one-sided (deterioration only), not abs()
  - KS shift in DIS must be one-sided, matching Gini's convention
  - Feature Binning's exposure column must map into the cross-technique summary
  - Feature Binning must enforce max_segment_pct like every other technique
  - parameter_optimization must surface which params it selected
  - Gini_Drop in the executive summary must be positive-when-worse
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.candidate_ranking import compute_severity_scores
from core.parameter_optimization import optimize_parameters
from core.segment_insights import generate_executive_summary
from compare_segmentation_techniques import standardize_columns
from metrics.business_metrics import calculate_sis, calculate_dis
from techniques.feature_binning import point_estimate_metrics


# ---------------------------------------------------------------------------
# Bad-rate / KS sign convention (delta = monitoring - development)
# ---------------------------------------------------------------------------

def test_bad_rate_improvement_does_not_inflate_sis():
    improved = calculate_sis(0.05, 0.0, 0.0, -0.5, 0.1, 0.1)   # bad rate fell
    neutral = calculate_sis(0.05, 0.0, 0.0, 0.0, 0.1, 0.1)
    deteriorated = calculate_sis(0.05, 0.0, 0.0, 0.5, 0.1, 0.1)  # bad rate rose

    assert improved["br_shift"] == 0.0
    assert improved["raw"] == neutral["raw"]
    assert deteriorated["br_shift"] == 0.5
    assert deteriorated["raw"] > improved["raw"]


def test_bad_rate_and_ks_improvement_do_not_inflate_dis():
    assert calculate_dis(0.0, 0.0, 0.0, -0.5) == 0.0   # bad rate improved
    assert calculate_dis(0.0, 0.0, 0.0, 0.5) == 0.5     # bad rate deteriorated
    assert calculate_dis(0.0, 0.0, 0.5, 0.0) == 0.0     # KS improved (positive delta)
    assert calculate_dis(0.0, 0.0, -0.5, 0.0) == 0.5    # KS deteriorated (negative delta)


def _evaluated_row(delta_br):
    return {
        "psi": 0.24, "delta_gini": -0.02, "delta_ks": 0.03, "delta_br": delta_br,
        "pct_dev": 0.08, "pct_mon": 0.27, "mon_weight_pct": 0.4,
        "gini_dev": 0.0, "gini_mon": -0.02, "ks_dev": 0.05, "ks_mon": 0.08,
        "br_dev": 0.72, "br_mon": 0.72 + delta_br,
        "br_pvalue": 0.0001, "score_shift_pvalue": 0.5,
    }


def test_bad_rate_improvement_does_not_win_severity_ranking():
    # Mirrors the observed "region = West" case: a large bad-rate *fall*
    # must not outrank a genuine (smaller) bad-rate deterioration.
    improved = _evaluated_row(delta_br=-0.566)
    deteriorated = _evaluated_row(delta_br=0.10)

    records = compute_severity_scores([improved, deteriorated])
    imp_rec = next(r for r in records if r["delta_br"] < 0)
    det_rec = next(r for r in records if r["delta_br"] > 0)

    assert det_rec["Severity_Score"] > imp_rec["Severity_Score"]


# ---------------------------------------------------------------------------
# Feature Binning exposure column mapping
# ---------------------------------------------------------------------------

def test_feature_binning_exposure_pct_is_mapped_to_shared_column():
    df = pd.DataFrame({"Exposure_Pct": [0.42], "Segment_Definition": ["age < 30"]})
    out = standardize_columns(df, "Feature Binning")
    assert out["Mon_Exposure_Pct"].iloc[0] == 0.42


# ---------------------------------------------------------------------------
# Feature Binning max_segment_pct cap
# ---------------------------------------------------------------------------

def _make_df(n, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "x": rng.uniform(0, 1, n),
        "target": rng.integers(0, 2, n),
        "score": rng.uniform(0, 1, n),
    })


def test_point_estimate_metrics_rejects_oversized_segment():
    dev_df, mon_df = _make_df(200, 1), _make_df(200, 2)
    bin_def = {"feature": "x", "type": "numeric", "label": "all", "lower": -np.inf, "upper": np.inf}

    result = point_estimate_metrics(
        bin_def, dev_df, mon_df, len(dev_df), len(mon_df),
        "target", "score", None, min_bin_pct=0.01, max_segment_pct=0.35,
    )
    assert result is None  # 100% of population exceeds the 35% cap


def test_point_estimate_metrics_accepts_segment_within_cap():
    dev_df, mon_df = _make_df(200, 1), _make_df(200, 2)
    bin_def = {"feature": "x", "type": "numeric", "label": "low", "lower": -np.inf, "upper": 0.2}

    result = point_estimate_metrics(
        bin_def, dev_df, mon_df, len(dev_df), len(mon_df),
        "target", "score", None, min_bin_pct=0.01, max_segment_pct=0.35,
    )
    assert result is not None
    assert result["dev_pct"] <= 0.35
    assert result["mon_pct"] <= 0.35


# ---------------------------------------------------------------------------
# Parameter-optimization transparency
# ---------------------------------------------------------------------------

@dataclass
class _FakeConfig:
    param_grid: dict = field(default_factory=dict)
    x: int = 0


class _FakeResult:
    def __init__(self, x):
        self.segments_df = pd.DataFrame({"Business_Impact_Score": [x]})
        self.extra = {}


def test_optimizer_surfaces_selected_params():
    cfg = _FakeConfig(param_grid={"x": [1, 5, 2]})

    def technique_func(dev_df, mon_df, config):
        return _FakeResult(config.x)

    result = optimize_parameters(technique_func, cfg, pd.DataFrame(), pd.DataFrame())

    assert result.extra["selected_params"] == {"x": 5}
    assert result.extra["params_evaluated"] == 3


# ---------------------------------------------------------------------------
# Executive summary Gini_Drop sign
# ---------------------------------------------------------------------------

def test_executive_summary_gini_drop_is_positive_when_worse():
    summary_df = pd.DataFrame({
        "Severity_Rank": [1], "Technique": ["AutoSlicer"], "Overall_Score_100": [80.0],
        "Max_Gini_Drop": [0.26], "Max_PSI": [0.24], "Calibration_Drift": [0.1],
    })
    segments_df = pd.DataFrame({
        "Segment_Definition": ["region = West"], "Technique": ["AutoSlicer"],
        "Severity_Score": [13.19], "Delta_Gini": [-0.26],
        "Root_Cause_Feature": ["age"],
    })

    exec_df = generate_executive_summary(summary_df, segments_df, None)
    row = exec_df[(exec_df["Section"] == "TOP WORST SEGMENTS") & (exec_df["Key"] == "  -> Gini Drop")]

    assert row.iloc[0]["Value"] == "0.2600"


# ---------------------------------------------------------------------------
# Feature Binning cross-technique mapping (Calibration_Drift, Severity_Score)
# ---------------------------------------------------------------------------

def test_feature_binning_calibration_and_severity_are_mapped_to_shared_columns():
    df = pd.DataFrame({
        "Delta_Calibration_Intercept": [-2.355],
        "Overall_Score": [0.223],
        "Segment_Definition": ["region = West"],
    })
    out = standardize_columns(df, "Feature Binning")
    assert out["Calibration_Drift"].iloc[0] == -2.355
    assert out["Severity_Score"].iloc[0] == 0.223


def test_feature_binning_config_has_dedup_threshold():
    from models.config import FeatureBinningConfig
    assert FeatureBinningConfig().overlap_jaccard_threshold == 0.70
