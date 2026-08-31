"""
models/config.py
===============

Centralized configuration dataclasses for all segmentation techniques.

All configurable thresholds, weights, and parameters are defined here instead
of being scattered as magic numbers across technique scripts.

Every existing default value is preserved exactly so that output parity with
the original codebase is maintained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ============================================================================
# Schema Detection Configuration
# ============================================================================

@dataclass
class SchemaConfig:
    """
    Controls how column roles such as target, score, weight, ID and
    exclusions are automatically detected from the development dataframe.
    """

    target_col: Optional[str] = None

    score_col: Optional[str] = None

    weight_col: Optional[str] = None

    id_cols: Optional[list] = None

    # Columns that must never be used as features.
    exclude_cols: Optional[list] = None

    # Force selected columns to categorical.
    categorical_cols: Optional[list] = None

    # Force selected columns to numeric.
    numeric_cols: Optional[list] = None

    # Numeric columns with cardinality <= this value may be treated
    # as categorical unless explicitly forced numeric.
    categorical_max_card: int = 20

    # Columns with uniqueness ratio above this threshold are considered
    # identifier-like.
    id_uniqueness_ratio: float = 0.98


# ============================================================================
# Severity Scoring Weights
# ============================================================================

@dataclass
class SeverityWeights:
    """
    Weights for the composite severity score.

    The weights should sum to 1.0 for interpretability, although the
    implementation does not enforce this.
    """

    w_psi: float = 0.25

    w_business_impact: float = 0.25

    w_gini_drop: float = 0.20

    w_ks_drop: float = 0.15

    w_br_shift: float = 0.15


# ============================================================================
# Candidate Filtering Thresholds
# ============================================================================

@dataclass
class FilteringThresholds:
    """
    Controls which candidate segments pass minimum-quality gates.
    """

    # Fraction of rows a segment must contain.
    min_support: Optional[float] = None

    # Absolute minimum number of rows per segment.
    min_abs_count: Optional[int] = None

    # Minimum number of events required in each segment.
    min_events_per_slice: int = 30

    # Significance threshold after multiple-testing adjustment.
    significance_alpha: float = 0.05

    # Reject segments broader than this population percentage.
    max_segment_pct: float = 0.35


# ============================================================================
# AutoSlicer Configuration
# ============================================================================

@dataclass
class SlicerConfig:
    """
    Configuration for the AutoSlicer beam-search technique.
    """

    schema: SchemaConfig = field(
        default_factory=SchemaConfig
    )

    param_grid: Optional[dict] = None

    # ------------------------------------------------------------------------
    # Search space
    # ------------------------------------------------------------------------

    max_combo_depth: int = 6

    depth_improvement_tolerance: float = 0.05

    beam_width: int = 30

    min_beam_width: int = 8

    beam_retain_ratio: float = 0.5

    top_n: int = 10

    numeric_bins: int = 4

    # ------------------------------------------------------------------------
    # Support / significance thresholds
    # ------------------------------------------------------------------------

    min_support: Optional[float] = None

    min_abs_count: Optional[int] = None

    min_events_per_slice: int = 30

    significance_alpha: float = 0.05

    max_segment_pct: float = 0.35

    # ------------------------------------------------------------------------
    # Severity weighting
    # ------------------------------------------------------------------------

    w_psi: float = 0.25

    w_business_impact: float = 0.25

    w_gini_drop: float = 0.20

    w_ks_drop: float = 0.15

    w_br_shift: float = 0.15

    # ------------------------------------------------------------------------
    # De-duplication
    # ------------------------------------------------------------------------

    overlap_jaccard_threshold: float = 0.70

    # ------------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------------

    random_state: int = 42


# ============================================================================
# Drift Localization Tree Configuration
# ============================================================================

@dataclass
class DLTConfig:
    """
    Configuration for the Drift Localization Tree technique.
    """

    schema: SchemaConfig = field(
        default_factory=SchemaConfig
    )

    param_grid: Optional[dict] = None

    # ------------------------------------------------------------------------
    # Tree parameters
    # ------------------------------------------------------------------------

    max_depth: int = 3

    min_samples_leaf: float = 0.05

    random_state: int = 42

    # ------------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------------

    top_n: int = 10

    # ------------------------------------------------------------------------
    # Support / significance thresholds
    # ------------------------------------------------------------------------

    min_support: Optional[float] = None

    min_abs_count: Optional[int] = None

    min_events_per_slice: int = 30

    significance_alpha: float = 0.05

    max_segment_pct: float = 0.35

    # ------------------------------------------------------------------------
    # Severity weighting
    # ------------------------------------------------------------------------

    w_psi: float = 0.25

    w_business_impact: float = 0.25

    w_gini_drop: float = 0.20

    w_ks_drop: float = 0.15

    w_br_shift: float = 0.15


# ============================================================================
# Gradient Boosting Configuration
# ============================================================================

@dataclass
class GBConfig:
    """
    Configuration for the Gradient Boosting segmentation technique.
    """

    schema: SchemaConfig = field(
        default_factory=SchemaConfig
    )

    param_grid: Optional[dict] = None

    # ------------------------------------------------------------------------
    # Gradient Boosting parameters
    # ------------------------------------------------------------------------

    n_estimators: int = 100

    max_depth: int = 3

    learning_rate: float = 0.1

    min_samples_leaf: float = 0.05

    random_state: int = 42

    # ------------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------------

    top_n: int = 10

    # ------------------------------------------------------------------------
    # Drift scanning
    # ------------------------------------------------------------------------

    drift_quantiles: tuple = (
        0.70,
        0.85,
        0.95,
    )

    max_trees_scanned: Optional[int] = None

    # ------------------------------------------------------------------------
    # Support / significance thresholds
    # ------------------------------------------------------------------------

    min_support: Optional[float] = None

    min_abs_count: Optional[int] = None

    min_events_per_slice: int = 30

    significance_alpha: float = 0.05

    max_segment_pct: float = 0.35

    # ------------------------------------------------------------------------
    # Severity weighting
    # ------------------------------------------------------------------------

    w_psi: float = 0.25

    w_business_impact: float = 0.25

    w_gini_drop: float = 0.20

    w_ks_drop: float = 0.15

    w_br_shift: float = 0.15

    # ------------------------------------------------------------------------
    # De-duplication
    # ------------------------------------------------------------------------

    overlap_jaccard_threshold: float = 0.70


# ============================================================================
# K-Means Configuration
# ============================================================================

@dataclass
class KMeansConfig:
    """
    Configuration for the K-Means clustering technique.

    The K-Means implementation expects both clustering parameters and
    candidate-evaluation/output parameters. These are all centralized here.
    """

    schema: SchemaConfig = field(
        default_factory=SchemaConfig
    )

    param_grid: Optional[dict] = None

    # ------------------------------------------------------------------------
    # K selection
    # ------------------------------------------------------------------------

    k_range: range = range(2, 13)

    # Minimum population percentage required for a cluster.
    min_cluster_pct: float = 0.03

    # Reproducibility.
    seed: int = 42

    # Maximum depth used when converting clusters into explainable rules.
    max_tree_depth: int = 4

    # ------------------------------------------------------------------------
    # Drift-aware clustering
    # ------------------------------------------------------------------------

    drift_aware: bool = True

    drift_weight: float = 0.4

    # ------------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------------

    # Number of final candidate segments returned.
    #
    # Required by techniques/kmeans.py.
    top_n: int = 10

    # ------------------------------------------------------------------------
    # Support / significance thresholds
    # ------------------------------------------------------------------------

    min_support: Optional[float] = None

    min_abs_count: Optional[int] = None

    # Required by K-Means candidate filtering/evaluation.
    min_events_per_slice: int = 30

    significance_alpha: float = 0.05

    max_segment_pct: float = 0.35

    # ------------------------------------------------------------------------
    # Severity weighting
    # ------------------------------------------------------------------------

    w_psi: float = 0.25

    w_business_impact: float = 0.25

    w_gini_drop: float = 0.20

    w_ks_drop: float = 0.15

    w_br_shift: float = 0.15


# ============================================================================
# Feature Binning Configuration
# ============================================================================

@dataclass
class FeatureBinningConfig:
    """
    Configuration for the Feature Binning segmentation technique.
    """

    schema: SchemaConfig = field(
        default_factory=SchemaConfig
    )

    param_grid: Optional[dict] = None

    # ------------------------------------------------------------------------
    # Feature / bin limits
    # ------------------------------------------------------------------------

    max_features: int = 30

    max_bins: int = 8

    min_bin_pct: float = 0.01

    # Reject bins broader than this population percentage (dev or mon side).
    # Matches the same cap enforced by core/candidate_filtering.py for the
    # other four techniques.
    max_segment_pct: float = 0.35

    # Jaccard-overlap threshold for candidate_deduplication.deduplicate_by_jaccard,
    # matching the default used by AutoSlicer/Gradient Boosting.
    overlap_jaccard_threshold: float = 0.70

    # ------------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------------

    top_n: int = 20

    bootstrap_top_n: int = 40

    # ------------------------------------------------------------------------
    # Statistical settings
    # ------------------------------------------------------------------------

    n_iter: int = 50

    alpha: float = 0.10

    # ------------------------------------------------------------------------
    # Interactions
    # ------------------------------------------------------------------------

    include_interactions: bool = True

    max_interaction_features: int = 8

    max_interaction_pairs: int = 15


# ============================================================================
# Trend Analysis Configuration
# ============================================================================

@dataclass
class TrendAnalysisConfig:
    """
    Configuration for multi-period trend analysis (core/multi_period_analysis.py).

    Segments are rediscovered independently for each dev-vs-monN period, then
    matched across periods by their (already-canonicalized) Segment_Definition
    text within one technique's pool.
    """

    # How many of a technique's worst segments (by Severity_Score) count as
    # "appearing" in a given period, for Frequency/Recency/Consistency.
    n_worst: int = 10

    # Recency_Factor = 0.5 ** (periods_since_last_appearance / recency_half_life_periods)
    recency_half_life_periods: float = 2.0

    # Trend_Impact_Score = w_frequency*Frequency + w_recency*Recency_Factor + w_consistency*Consistency_Score
    # Must sum to 1.0.
    w_frequency: float = 0.4

    w_recency: float = 0.3

    w_consistency: float = 0.3

    # SIS_Trend = SIS_Raw * (1 + trend_weight * Trend_Impact_Score)
    trend_weight: float = 0.5

    # Early-warning forecast (core.multi_period_analysis.forecast_segment_scores):
    # a segment is flagged once its projected Root_Cause_Score is on track to
    # cross this threshold within forecast_horizon_months. Not a validated
    # industry cutoff -- same transparency caveat as the TIS weights above.
    forecast_alert_threshold: float = 0.5

    forecast_horizon_months: int = 6