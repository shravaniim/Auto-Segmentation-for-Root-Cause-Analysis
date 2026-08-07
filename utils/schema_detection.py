"""
utils/schema_detection.py
=========================
Single source of truth for automatic schema detection (column roles).

Extracted from the original ``segmentation_common.py`` without any logic
changes.  Every segmentation technique imports column-role detection from
this module so the decision of "which column is the target / score / weight /
ID" is made in exactly ONE place.

Data files are expected to live in:
  data/development_data_NEW.csv
  data/monitoring_2026_01.csv
  data/development_data_5000_shap.csv
  data/monitoring_data_5000_shap.csv
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from models.config import SchemaConfig
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Resolve data/ directory relative to project root (two levels up from utils/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _PROJECT_ROOT / "data"


def get_data_path(filename: str) -> Path:
    """Return the canonical path inside data/ for the given filename.

    If *filename* is already an absolute path, it is returned as-is so
    callers that pass full paths are unaffected.
    """
    p = Path(filename)
    if p.is_absolute():
        return p
    return DATA_DIR / filename


# ---------------------------------------------------------------------------
# Name-matching patterns (unchanged from original)
# ---------------------------------------------------------------------------

_TARGET_PATTERNS = re.compile(
    r"(target|default|bad_flag|is_bad|label|outcome|event)", re.I
)
_SCORE_PATTERNS_TIERED = [
    re.compile(r"(predicted_pd|pred_pd|model_score|pd_hat|proba|probability)", re.I),
    re.compile(r"\bpd\b", re.I),
    re.compile(r"(prob|risk_score|model_output)", re.I),
    re.compile(r"score", re.I),  # last resort
]
_WEIGHT_PATTERNS = re.compile(r"(ead|exposure|balance|weight|amount)", re.I)
_ID_PATTERNS = re.compile(
    r"(^id$|_id$|^id_|customer_?id|account_?id|application_?id|uuid)", re.I
)
_TIME_PATTERNS = re.compile(
    r"(month|period|date|vintage|snapshot|as_of|year)", re.I
)

_DEFAULT_EXCLUDED_COLUMNS = {
    "predicted_pd",
    "actual_default",
    "monitoring_month",
    "ead",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _auto_detect_role(df: pd.DataFrame, pattern, dtype_filter=None) -> Optional[str]:
    """Return the first column whose name matches *pattern* (and optionally
    passes *dtype_filter*)."""
    for col in df.columns:
        if pattern.search(col):
            if dtype_filter is None or dtype_filter(df[col]):
                return col
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_schema(dev_df: pd.DataFrame, cfg: SchemaConfig) -> dict:
    """Auto-detects column roles from *dev_df* and returns a dict with:
    ``target_col``, ``score_col``, ``weight_col``, ``id_cols``, ``time_cols``,
    ``excluded_cols``, ``categorical_cols``, ``numeric_cols``.

    Logic is identical to the original ``segmentation_common.detect_schema``.
    """
    cols = list(dev_df.columns)
    n = len(dev_df)

    # --- target ---
    target_col = cfg.target_col or _auto_detect_role(
        dev_df, _TARGET_PATTERNS, lambda s: s.dropna().nunique() == 2
    )
    if target_col is None:
        for c in cols:
            vals = pd.Series(dev_df[c]).dropna().unique()
            if set(np.unique(vals)).issubset({0, 1}):
                target_col = c
                break
    if target_col is None:
        raise ValueError(
            "Could not auto-detect a binary target column. "
            "Pass SchemaConfig(target_col=...)."
        )

    # --- score (tiered) ---
    score_col = cfg.score_col
    if score_col is None:
        for pattern in _SCORE_PATTERNS_TIERED:
            hit = _auto_detect_role(
                dev_df, pattern, lambda s: pd.api.types.is_numeric_dtype(s)
            )
            if hit is not None and hit != target_col:
                score_col = hit
                break
    if score_col is None:
        prob_like = [
            c for c in cols
            if c != target_col
            and pd.api.types.is_numeric_dtype(dev_df[c])
            and dev_df[c].nunique() > cfg.categorical_max_card
            and dev_df[c].min() >= 0
            and dev_df[c].max() <= 1
        ]
        if prob_like:
            score_col = prob_like[0]
    if score_col is None:
        numeric_candidates = [
            c for c in cols
            if c != target_col
            and pd.api.types.is_numeric_dtype(dev_df[c])
            and dev_df[c].nunique() > cfg.categorical_max_card
        ]
        if numeric_candidates:
            score_col = numeric_candidates[0]
    if score_col is None:
        raise ValueError(
            "Could not auto-detect a model score column. "
            "Pass SchemaConfig(score_col=...)."
        )

    # --- weight / exposure ---
    weight_col = cfg.weight_col or _auto_detect_role(
        dev_df, _WEIGHT_PATTERNS, lambda s: pd.api.types.is_numeric_dtype(s)
    )

    # --- id columns ---
    id_cols = set(cfg.id_cols or [])
    for c in cols:
        if _ID_PATTERNS.search(c):
            id_cols.add(c)
        elif dev_df[c].nunique() / max(n, 1) > cfg.id_uniqueness_ratio:
            id_cols.add(c)

    # --- time / vintage columns ---
    time_cols = {c for c in cols if _TIME_PATTERNS.search(c)}

    # --- explicit exclusions ---
    explicit_excl = set(cfg.exclude_cols or []) | _DEFAULT_EXCLUDED_COLUMNS

    non_feature_cols = id_cols | time_cols | explicit_excl | {target_col, score_col}
    if weight_col:
        non_feature_cols.add(weight_col)

    feature_cols = [c for c in cols if c not in non_feature_cols]

    forced_cat = set(cfg.categorical_cols or [])
    forced_num = set(cfg.numeric_cols or [])
    cat_cols, num_cols = [], []
    for c in feature_cols:
        if c in forced_cat:
            cat_cols.append(c)
        elif c in forced_num:
            num_cols.append(c)
        elif not pd.api.types.is_numeric_dtype(dev_df[c]):
            cat_cols.append(c)
        elif dev_df[c].nunique() <= cfg.categorical_max_card:
            cat_cols.append(c)
        else:
            num_cols.append(c)

    logger.info(
        "Schema detected — target=%r  score=%r  weight=%r  "
        "numeric=%s  categorical=%s",
        target_col, score_col, weight_col, num_cols, cat_cols,
    )

    return {
        "target_col": target_col,
        "score_col": score_col,
        "weight_col": weight_col,
        "id_cols": sorted(id_cols),
        "time_cols": sorted(time_cols),
        "excluded_cols": sorted(non_feature_cols),
        "categorical_cols": cat_cols,
        "numeric_cols": num_cols,
    }


def auto_detect_shap_columns(df: pd.DataFrame) -> list[str]:
    """Return column names that look like SHAP values."""
    return [c for c in df.columns if re.search(r"\bshap[_\-]", c, re.I)]
