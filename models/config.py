"""
models/config.py
================
Centralized configuration dataclasses for all segmentation techniques.

All configurable thresholds, weights, and parameters are defined here instead of
being scattered as magic numbers across technique scripts.  Every existing default
value is preserved exactly so that output parity with the original codebase is
maintained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Schema detection configuration
# ---------------------------------------------------------------------------

@dataclass
class SchemaConfig:
    """Controls how column roles (target, score, weight, id, exclusions) are
    auto-detected from the development dataframe."""
    target_col: Optional[str] = None
    score_col: Optional[str] = None
    weight_col: Optional[str] = None
    id_cols: Optional[list] = None
    exclude_cols: Optional[list] = None       # columns that must never be used as features
    categorical_cols: Optional[list] = None   # force certain cols categorical
    numeric_cols: Optional[list] = None       # force certain cols numeric
    categorical_max_card: int = 20
    id_uniqueness_ratio: float = 0.98


# ---------------------------------------------------------------------------
# Severity scoring weights — shared across techniques
# ---------------------------------------------------------------------------

@dataclass
class SeverityWeights:
    """Weights for the composite severity score.  Must sum to 1.0 for
    interpretability, though the code does not enforce this."""
    w_psi: float = 0.25
    w_business_impact: float = 0.25
    w_gini_drop: float = 0.20
    w_ks_drop: float = 0.15
    w_br_shift: float = 0.15


# ---------------------------------------------------------------------------
# Candidate filtering thresholds — shared across techniques
# ---------------------------------------------------------------------------

@dataclass
class FilteringThresholds:
    """Controls which candidate segments pass minimum-quality gates."""
    min_support: Optional[float] = None       # fraction of rows a segment must contain
    min_abs_count: Optional[int] = None       # absolute floor on rows per segment
    min_events_per_slice: int = 30            # enforced directly on each segment's event count
    significance_alpha: float = 0.05          # BH-adjusted p-value cut
    max_segment_pct: float = 0.35             # reject segments broader than this


# ---------------------------------------------------------------------------
# Technique-specific configurations
# ---------------------------------------------------------------------------

@dataclass
class SlicerConfig:
    """Configuration for the AutoSlicer beam-search technique."""
    schema: SchemaConfig = field(default_factory=SchemaConfig)
    param_grid: Optional[dict] = None

    # --- search space ---
    max_combo_depth: int = 6
    depth_improvement_tolerance: float = 0.05
    beam_width: int = 30
    min_beam_width: int = 8
    beam_retain_ratio: float = 0.5
    top_n: int = 10
    numeric_bins: int = 4

    # --- support / significance thresholds ---
    min_support: Optional[float] = None
    min_abs_count: Optional[int] = None
    min_events_per_slice: int = 30
    significance_alpha: float = 0.05

    # --- severity weighting ---
    w_psi: float = 0.25
    w_business_impact: float = 0.25
    w_gini_drop: float = 0.20
    w_ks_drop: float = 0.15
    w_br_shift: float = 0.15

    # --- de-duplication ---
    overlap_jaccard_threshold: float = 0.70
    max_segment_pct: float = 0.35

    random_state: int = 42


@dataclass
class DLTConfig:
    """Configuration for the Drift Localization Tree technique."""
    schema: SchemaConfig = field(default_factory=SchemaConfig)
    param_grid: Optional[dict] = None

    max_depth: int = 3
    min_samples_leaf: float = 0.05
    random_state: int = 42

    top_n: int = 10

    min_support: Optional[float] = None
    min_abs_count: Optional[int] = None
    min_events_per_slice: int = 30
    significance_alpha: float = 0.05
    max_segment_pct: float = 0.35

    w_psi: float = 0.25
    w_business_impact: float = 0.25
    w_gini_drop: float = 0.20
    w_ks_drop: float = 0.15
    w_br_shift: float = 0.15


@dataclass
class GBConfig:
    """Configuration for the Gradient Boosting segmentation technique."""
    schema: SchemaConfig = field(default_factory=SchemaConfig)
    param_grid: Optional[dict] = None

    n_estimators: int = 100
    max_depth: int = 3
    learning_rate: float = 0.1
    min_samples_leaf: float = 0.05
    random_state: int = 42

    top_n: int = 10
    drift_quantiles: tuple = (0.70, 0.85, 0.95)
    max_trees_scanned: Optional[int] = None

    min_support: Optional[float] = None
    min_abs_count: Optional[int] = None
    min_events_per_slice: int = 30
    significance_alpha: float = 0.05
    max_segment_pct: float = 0.35

    w_psi: float = 0.25
    w_business_impact: float = 0.25
    w_gini_drop: float = 0.20
    w_ks_drop: float = 0.15
    w_br_shift: float = 0.15

    overlap_jaccard_threshold: float = 0.70


@dataclass
class KMeansConfig:
    """Configuration for the K-Means clustering technique."""
    schema: SchemaConfig = field(default_factory=SchemaConfig)
    param_grid: Optional[dict] = None
    k_range: range = range(2, 13)
    min_cluster_pct: float = 0.03
    seed: int = 42
    max_tree_depth: int = 4
    drift_aware: bool = True
    drift_weight: float = 0.4


@dataclass
class FeatureBinningConfig:
    """Configuration for the Feature Binning technique."""
    schema: SchemaConfig = field(default_factory=SchemaConfig)
    param_grid: Optional[dict] = None
    max_features: int = 30
    max_bins: int = 8
    min_bin_pct: float = 0.01
    top_n: int = 20
    bootstrap_top_n: int = 40
    n_iter: int = 50
    alpha: float = 0.10
    include_interactions: bool = True
    max_interaction_features: int = 8
    max_interaction_pairs: int = 15
