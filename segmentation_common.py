"""
segmentation_common.py
======================
Backwards-compatibility facade for shared statistical and schema utilities.

All implementation logic has been refactored into focused sub-modules:
  - utils.schema_detection
  - metrics.performance_metrics
  - metrics.drift_metrics
  - metrics.business_metrics
  - metrics.significance
  - utils.logging_config

This module re-exports all legacy symbols so existing code and tests continue
to function without modification.
"""

from __future__ import annotations

from models.config import SchemaConfig
from utils.logging_config import get_logger, configure_logging
from utils.schema_detection import (
    detect_schema,
    auto_detect_shap_columns,
)
from metrics.performance_metrics import (
    calculate_auc_gini,
    calculate_ks,
    safe_auc,
)
from metrics.drift_metrics import (
    calculate_psi,
    build_decile_bins,
    compute_psi_for_feature,
    interpret_psi,
    detect_shap_shift,
)
from metrics.business_metrics import (
    calculate_business_impact,
    calculate_sis,
    calculate_dis,
)
from metrics.significance import (
    two_proportion_ztest_pvalue,
    ks_score_shift_pvalue,
    adjust_p_values_bh,
    drift_confidence,
    min_max_normalize,
)

__all__ = [
    "SchemaConfig",
    "get_logger",
    "configure_logging",
    "detect_schema",
    "auto_detect_shap_columns",
    "calculate_auc_gini",
    "calculate_ks",
    "safe_auc",
    "calculate_psi",
    "build_decile_bins",
    "compute_psi_for_feature",
    "interpret_psi",
    "detect_shap_shift",
    "calculate_business_impact",
    "calculate_sis",
    "calculate_dis",
    "two_proportion_ztest_pvalue",
    "ks_score_shift_pvalue",
    "adjust_p_values_bh",
    "drift_confidence",
    "min_max_normalize",
]
