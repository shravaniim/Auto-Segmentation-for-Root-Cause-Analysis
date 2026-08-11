"""
utils/schema_detection.py
=========================

Single source of truth for automatic schema detection.

This module determines column roles once and exposes them to all
segmentation techniques.

Detected roles
--------------

    target_col
        Binary outcome / target.

    score_col
        Model prediction / PD / probability / score.

    weight_col
        Exposure / EAD / balance / weight.

    id_cols
        Customer/account/application/transaction identifiers.

    time_cols
        Month/date/vintage/snapshot columns.

    shap_cols
        SHAP-value columns.

    categorical_cols
        Legitimate categorical model/business features.

    numeric_cols
        Legitimate numeric model/business features.

    feature_cols
        All legitimate features suitable for segmentation/RCA.

Important
---------

Identifiers, target, model score, exposure, time/vintage and SHAP columns
are NOT returned as normal model features.

This prevents columns such as:

    customer_id
    account_id
    application_id
    transaction_id

from incorrectly appearing as root-cause features.

IMPORTANT DESIGN RULE
---------------------

schema["feature_cols"] is the AUTHORITATIVE list of legitimate features.

All segmentation techniques should pass:

    feature_cols=schema["feature_cols"]

to evaluate_segment().

The candidate_evaluation layer performs a second defensive sanitization
before Root Cause Analysis.
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


# ============================================================================
# PROJECT PATHS
# ============================================================================

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = _PROJECT_ROOT / "data"


def get_data_path(filename: str) -> Path:
    """
    Return the canonical path inside data/ for the given filename.

    Absolute paths are returned unchanged.
    """

    path = Path(filename)

    if path.is_absolute():
        return path

    return DATA_DIR / filename


# ============================================================================
# COLUMN NAME PATTERNS
# ============================================================================

# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------

_TARGET_PATTERNS = re.compile(
    r"(^target$|target_|_target$|"
    r"^default$|default_|_default$|"
    r"bad_flag|is_bad|"
    r"^label$|label_|_label$|"
    r"^outcome$|outcome_|_outcome$|"
    r"^event$|event_|_event$)",
    re.I,
)


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------

_SCORE_PATTERNS_TIERED = [

    # Strongest indicators
    re.compile(
        r"^(predicted_pd|pred_pd|model_score|pd_hat|"
        r"prediction|predicted_probability|predicted_prob)$",
        re.I,
    ),

    # PD
    re.compile(
        r"(^pd$|_pd$|^pd_)",
        re.I,
    ),

    # Probability
    re.compile(
        r"(probability|prob_|_prob$|proba|"
        r"risk_score|model_output)",
        re.I,
    ),

    # Generic score
    re.compile(
        r"(^score$|score_|_score$)",
        re.I,
    ),
]


# ---------------------------------------------------------------------------
# Weight / exposure
# ---------------------------------------------------------------------------

_WEIGHT_PATTERNS = re.compile(
    r"(^ead$|_ead$|^ead_|"
    r"exposure|"
    r"balance|"
    r"^weight$|_weight$|^weight_|"
    r"amount)",
    re.I,
)


# ---------------------------------------------------------------------------
# Identifier
# ---------------------------------------------------------------------------

_ID_PATTERNS = re.compile(
    r"(^id$|"
    r"_id$|"
    r"^id_|"
    r"customer_?id|"
    r"account_?id|"
    r"application_?id|"
    r"loan_?id|"
    r"transaction_?id|"
    r"contract_?id|"
    r"policy_?id|"
    r"member_?id|"
    r"client_?id|"
    r"party_?id|"
    r"reference_?id|"
    r"uuid|"
    r"identifier)",
    re.I,
)


# ---------------------------------------------------------------------------
# Possible identifier names WITHOUT "_id"
# ---------------------------------------------------------------------------
#
# These are intentionally more conservative than the uniqueness detector.
# A column called "customer_number" is very likely an identifier, whereas
# a high-cardinality column such as "income" is not automatically an ID.
#

_ID_NAME_PATTERNS = re.compile(
    r"(customer_?number|"
    r"account_?number|"
    r"application_?number|"
    r"transaction_?number|"
    r"loan_?number|"
    r"contract_?number|"
    r"policy_?number|"
    r"member_?number|"
    r"client_?number|"
    r"reference_?number|"
    r"customer_?key|"
    r"account_?key|"
    r"application_?key|"
    r"transaction_?key)",
    re.I,
)


# ---------------------------------------------------------------------------
# Time / vintage
# ---------------------------------------------------------------------------

_TIME_PATTERNS = re.compile(
    r"(^month$|_month$|^month_|"
    r"^period$|_period$|^period_|"
    r"date|"
    r"vintage|"
    r"snapshot|"
    r"as_of|"
    r"timestamp|"
    r"^year$|_year$|^year_)",
    re.I,
)


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

_SHAP_PATTERNS = re.compile(
    r"^shap[_\-]",
    re.I,
)


# ============================================================================
# EXPLICITLY EXCLUDED COLUMNS
# ============================================================================

_DEFAULT_EXCLUDED_COLUMNS = {
    "predicted_pd",
    "actual_default",
    "monitoring_month",
    "ead",
}


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _auto_detect_role(
    df: pd.DataFrame,
    pattern,
    dtype_filter=None,
) -> Optional[str]:
    """
    Return the first column whose name matches pattern and optionally
    satisfies dtype_filter.
    """

    for col in df.columns:

        if not pattern.search(str(col)):
            continue

        if dtype_filter is None:
            return col

        try:
            if dtype_filter(df[col]):
                return col
        except Exception:
            continue

    return None


# ============================================================================
# ID DETECTION
# ============================================================================

def _detect_id_columns(
    df: pd.DataFrame,
    cfg: SchemaConfig,
) -> set[str]:
    """
    Detect identifier columns.

    Detection uses:

        1. Explicit cfg.id_cols
        2. Strong ID-name patterns
        3. Conservative high-uniqueness detection

    IMPORTANT
    ---------

    We do NOT automatically classify every high-cardinality numeric
    variable as an ID.

    This prevents legitimate features such as:

        income
        balance
        utilization
        age
        score-like business variables

    from being removed merely because they have many unique values.
    """

    id_cols = set(
        cfg.id_cols or []
    )

    n = len(df)

    if n == 0:
        return id_cols

    for col in df.columns:

        col_str = str(col)

        # ---------------------------------------------------------------
        # Strong name-based detection
        # ---------------------------------------------------------------

        if _ID_PATTERNS.search(col_str):
            id_cols.add(col)
            continue

        if _ID_NAME_PATTERNS.search(col_str):
            id_cols.add(col)
            continue

        # ---------------------------------------------------------------
        # Conservative uniqueness detection
        # ---------------------------------------------------------------
        #
        # Only use uniqueness as supporting evidence.
        #
        # A high-cardinality NUMERIC variable is not automatically an ID.
        #
        # Stronger candidates:
        #   - object/string columns
        #   - integer-like identifier columns
        #
        # We intentionally avoid automatically removing continuous floats.
        # ---------------------------------------------------------------

        try:

            series = df[col]

            uniqueness_ratio = (
                series.nunique(
                    dropna=False
                )
                / max(n, 1)
            )

        except Exception:
            continue

        if uniqueness_ratio <= cfg.id_uniqueness_ratio:
            continue

        # ---------------------------------------------------------------
        # String/object high-cardinality columns
        # ---------------------------------------------------------------

        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            id_cols.add(col)
            continue

        # ---------------------------------------------------------------
        # Integer-like high-cardinality columns
        # ---------------------------------------------------------------

        if pd.api.types.is_integer_dtype(series):

            # An integer column with nearly one unique value per row is a
            # reasonable identifier candidate.
            if uniqueness_ratio >= 0.98:
                id_cols.add(col)
                continue

    return id_cols


# ============================================================================
# TIME DETECTION
# ============================================================================

def _detect_time_columns(
    df: pd.DataFrame,
) -> set[str]:
    """
    Detect time/vintage/snapshot columns by name.
    """

    return {
        col
        for col in df.columns
        if _TIME_PATTERNS.search(
            str(col)
        )
    }


# ============================================================================
# SHAP DETECTION
# ============================================================================

def _detect_shap_columns(
    df: pd.DataFrame,
) -> list[str]:
    """
    Detect SHAP-value columns.
    """

    return [
        col
        for col in df.columns
        if _SHAP_PATTERNS.search(
            str(col)
        )
    ]


# ============================================================================
# FEATURE CLASSIFICATION
# ============================================================================

def _classify_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    cfg: SchemaConfig,
) -> tuple[list[str], list[str]]:
    """
    Classify legitimate feature columns into categorical and numeric.

    Explicit configuration takes precedence over automatic inference.
    """

    forced_cat = set(
        cfg.categorical_cols or []
    )

    forced_num = set(
        cfg.numeric_cols or []
    )

    cat_cols: list[str] = []
    num_cols: list[str] = []

    for col in feature_cols:

        if col not in df.columns:
            continue

        # ---------------------------------------------------------------
        # Explicit categorical override
        # ---------------------------------------------------------------

        if col in forced_cat:

            cat_cols.append(col)

            continue

        # ---------------------------------------------------------------
        # Explicit numeric override
        # ---------------------------------------------------------------

        if col in forced_num:

            num_cols.append(col)

            continue

        series = df[col]

        # ---------------------------------------------------------------
        # Non-numeric -> categorical
        # ---------------------------------------------------------------

        if not pd.api.types.is_numeric_dtype(
            series
        ):

            cat_cols.append(col)

            continue

        # ---------------------------------------------------------------
        # Low-cardinality numeric -> categorical
        # ---------------------------------------------------------------

        try:

            cardinality = (
                series.nunique(
                    dropna=True
                )
            )

        except Exception:

            cardinality = 0

        if (
            cardinality
            <= cfg.categorical_max_card
        ):

            cat_cols.append(col)

        else:

            num_cols.append(col)

    return cat_cols, num_cols


# ============================================================================
# PUBLIC API
# ============================================================================

def detect_schema(
    dev_df: pd.DataFrame,
    cfg: SchemaConfig,
) -> dict:
    """
    Automatically detect column roles from the development dataframe.

    Returns
    -------

    {
        "target_col": str,
        "score_col": str,
        "weight_col": str | None,

        "id_cols": list[str],
        "time_cols": list[str],
        "shap_cols": list[str],

        "excluded_cols": list[str],

        "feature_cols": list[str],

        "categorical_cols": list[str],
        "numeric_cols": list[str],
    }

    IMPORTANT
    ---------

    feature_cols is the authoritative list of legitimate model/business
    features and should be used by all segmentation techniques for
    feature-level Root Cause Analysis.
    """

    if not isinstance(
        dev_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "dev_df must be a pandas DataFrame."
        )

    if dev_df.empty:
        raise ValueError(
            "Development dataframe is empty."
        )

    cols = list(
        dev_df.columns
    )

    # =========================================================================
    # TARGET
    # =========================================================================

    target_col = (
        cfg.target_col
        or _auto_detect_role(
            dev_df,
            _TARGET_PATTERNS,
            lambda s: (
                s.dropna().nunique()
                == 2
            ),
        )
    )

    # -------------------------------------------------------------------------
    # Binary 0/1 fallback
    # -------------------------------------------------------------------------

    if target_col is None:

        for col in cols:

            try:

                values = (
                    pd.Series(
                        dev_df[col]
                    )
                    .dropna()
                    .unique()
                )

                if (
                    len(values) == 2
                    and set(
                        np.asarray(values)
                    ).issubset({0, 1})
                ):

                    target_col = col

                    break

            except Exception:
                continue

    if target_col is None:

        raise ValueError(
            "Could not auto-detect a binary target column. "
            "Pass SchemaConfig(target_col=...)."
        )

    if target_col not in dev_df.columns:

        raise ValueError(
            f"Configured target column "
            f"'{target_col}' does not exist."
        )

    # =========================================================================
    # MODEL SCORE
    # =========================================================================

    score_col = cfg.score_col

    if score_col is None:

        for pattern in _SCORE_PATTERNS_TIERED:

            hit = _auto_detect_role(
                dev_df,
                pattern,
                lambda s: (
                    pd.api.types.is_numeric_dtype(
                        s
                    )
                ),
            )

            if (
                hit is not None
                and hit != target_col
            ):

                score_col = hit

                break

    # -------------------------------------------------------------------------
    # Probability-like fallback
    # -------------------------------------------------------------------------

    if score_col is None:

        prob_like = []

        for col in cols:

            if col == target_col:
                continue

            series = dev_df[col]

            if not pd.api.types.is_numeric_dtype(
                series
            ):
                continue

            if (
                series.nunique(
                    dropna=True
                )
                <= cfg.categorical_max_card
            ):
                continue

            try:

                min_value = series.min()
                max_value = series.max()

            except Exception:
                continue

            if (
                min_value >= 0
                and max_value <= 1
            ):

                prob_like.append(
                    col
                )

        if prob_like:

            score_col = prob_like[0]

    # -------------------------------------------------------------------------
    # Numeric high-cardinality fallback
    # -------------------------------------------------------------------------

    if score_col is None:

        numeric_candidates = []

        for col in cols:

            if col == target_col:
                continue

            series = dev_df[col]

            if not pd.api.types.is_numeric_dtype(
                series
            ):
                continue

            if (
                series.nunique(
                    dropna=True
                )
                > cfg.categorical_max_card
            ):

                numeric_candidates.append(
                    col
                )

        if numeric_candidates:

            score_col = (
                numeric_candidates[0]
            )

    if score_col is None:

        raise ValueError(
            "Could not auto-detect a model score column. "
            "Pass SchemaConfig(score_col=...)."
        )

    if score_col not in dev_df.columns:

        raise ValueError(
            f"Configured score column "
            f"'{score_col}' does not exist."
        )

    # =========================================================================
    # WEIGHT / EXPOSURE
    # =========================================================================

    weight_col = (
        cfg.weight_col
        or _auto_detect_role(
            dev_df,
            _WEIGHT_PATTERNS,
            lambda s: (
                pd.api.types.is_numeric_dtype(
                    s
                )
            ),
        )
    )

    if (
        weight_col is not None
        and weight_col not in dev_df.columns
    ):

        raise ValueError(
            f"Configured weight column "
            f"'{weight_col}' does not exist."
        )

    # =========================================================================
    # IDENTIFIERS
    # =========================================================================

    id_cols = _detect_id_columns(
        dev_df,
        cfg,
    )

    # =========================================================================
    # TIME / VINTAGE
    # =========================================================================

    time_cols = _detect_time_columns(
        dev_df
    )

    # =========================================================================
    # SHAP
    # =========================================================================

    shap_cols = _detect_shap_columns(
        dev_df
    )

    # =========================================================================
    # EXPLICIT EXCLUSIONS
    # =========================================================================

    explicit_exclusions = set(
        cfg.exclude_cols or []
    )

    explicit_exclusions.update(
        _DEFAULT_EXCLUDED_COLUMNS
    )

    # =========================================================================
    # ALL NON-FEATURE COLUMNS
    # =========================================================================

    non_feature_cols = set()

    non_feature_cols.update(
        id_cols
    )

    non_feature_cols.update(
        time_cols
    )

    non_feature_cols.update(
        shap_cols
    )

    non_feature_cols.update(
        explicit_exclusions
    )

    non_feature_cols.add(
        target_col
    )

    non_feature_cols.add(
        score_col
    )

    if weight_col:

        non_feature_cols.add(
            weight_col
        )

    # Only keep columns actually present in the dataframe.
    non_feature_cols = {
        col
        for col in non_feature_cols
        if col in dev_df.columns
    }

    # =========================================================================
    # AUTHORITATIVE FEATURE LIST
    # =========================================================================
    #
    # This is the most important part of the module.
    #
    # Only these columns are allowed to become segmentation/RCA features.
    #
    # customer_id -> excluded
    # account_id  -> excluded
    # target      -> excluded
    # score       -> excluded
    # EAD         -> excluded
    # vintage     -> excluded
    # SHAP        -> excluded
    #
    # income, age, occupation, region, etc. -> retained
    # =========================================================================

    feature_cols = [
        col
        for col in cols
        if col not in non_feature_cols
    ]

    # =========================================================================
    # CLASSIFY LEGITIMATE FEATURES
    # =========================================================================

    categorical_cols, numeric_cols = (
        _classify_features(
            dev_df,
            feature_cols,
            cfg,
        )
    )

    # =========================================================================
    # FINAL SAFETY CHECK
    # =========================================================================
    #
    # Defensive validation ensures that no excluded role leaks into
    # feature_cols due to configuration or future code changes.
    # =========================================================================

    forbidden_in_features = set(
        non_feature_cols
    )

    leaked_features = [
        col
        for col in feature_cols
        if col in forbidden_in_features
    ]

    if leaked_features:

        raise RuntimeError(
            "Schema detection safety check failed. "
            "Excluded columns appeared in feature_cols: "
            f"{leaked_features}"
        )

    # =========================================================================
    # FINAL FEATURE VALIDATION
    # =========================================================================

    categorical_cols = [
        col
        for col in categorical_cols
        if col in feature_cols
    ]

    numeric_cols = [
        col
        for col in numeric_cols
        if col in feature_cols
    ]

    # =========================================================================
    # LOGGING
    # =========================================================================

    logger.info(
        "============================================================"
    )

    logger.info(
        "Automatic Schema Detection"
    )

    logger.info(
        "============================================================"
    )

    logger.info(
        "target_col      = %r",
        target_col,
    )

    logger.info(
        "score_col       = %r",
        score_col,
    )

    logger.info(
        "weight_col      = %r",
        weight_col,
    )

    logger.info(
        "id_cols         = %s",
        sorted(
            id_cols
        ),
    )

    logger.info(
        "time_cols       = %s",
        sorted(
            time_cols
        ),
    )

    logger.info(
        "shap_cols       = %s",
        sorted(
            shap_cols
        ),
    )

    logger.info(
        "excluded_cols   = %s",
        sorted(
            non_feature_cols
        ),
    )

    logger.info(
        "numeric_cols    = %s",
        numeric_cols,
    )

    logger.info(
        "categorical_cols = %s",
        categorical_cols,
    )

    logger.info(
        "feature_cols    = %s",
        feature_cols,
    )

    logger.info(
        "============================================================"
    )

    # =========================================================================
    # RETURN AUTHORITATIVE SCHEMA
    # =========================================================================

    return {
        "target_col": target_col,

        "score_col": score_col,

        "weight_col": weight_col,

        "id_cols": sorted(
            id_cols
        ),

        "time_cols": sorted(
            time_cols
        ),

        "shap_cols": sorted(
            shap_cols
        ),

        "excluded_cols": sorted(
            non_feature_cols
        ),

        # IMPORTANT:
        # This is the authoritative list for segmentation and RCA.
        "feature_cols": feature_cols,

        "categorical_cols": categorical_cols,

        "numeric_cols": numeric_cols,
    }


# ============================================================================
# SHAP PUBLIC HELPER
# ============================================================================

def auto_detect_shap_columns(
    df: pd.DataFrame,
) -> list[str]:
    """
    Return column names that look like SHAP values.

    Examples detected:

        shap_age
        shap_income
        shap-region
        shap_region
    """

    return [
        col
        for col in df.columns
        if _SHAP_PATTERNS.search(
            str(col)
        )
    ]