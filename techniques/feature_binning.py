"""
techniques/feature_binning.py
=============================

Multi-Feature Binning Segmentation Technique.

Discovers univariate and pairwise interaction bins where model performance
deteriorates, applying bootstrap resampling and FDR correction.

Important schema rules
----------------------

Feature selection is controlled by utils.schema_detection.

Only:

    schema["feature_cols"]

are considered legitimate segmentation features.

The following are excluded from feature analysis:

    - customer_id / account_id / application_id
    - target
    - model score
    - EAD / exposure
    - time / vintage columns
    - SHAP columns

This prevents identifier columns such as customer_id from being selected
as segmentation or root-cause features.
"""

from __future__ import annotations

import time
from typing import Optional, Union

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from models.config import FeatureBinningConfig
from models.result import TechniqueResult
from techniques.base import BaseSegmentationTechnique

from utils.logging_config import get_logger
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
    interpret_psi,
    detect_shap_shift,
)

from metrics.business_metrics import (
    calculate_sis,
    calculate_dis,
)

from metrics.significance import (
    adjust_p_values_bh,
    min_max_normalize,
)


logger = get_logger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

MISSING_LABEL = "__MISSING__"

RANDOM_STATE = 42

# These are internal names used AFTER target/score/EAD are renamed.
#
# The authoritative exclusion of customer_id, target, score, EAD, time
# columns, etc. comes from schema_detection.py.
IGNORE_FEATURES = {
    "target",
    "score",
    "ead",
    "is_monitoring",
    "leaf_id",
    "customer_id",
}


# ============================================================================
# CALIBRATION / ECE
# ============================================================================

def safe_ece(
    df,
    score_col="score",
    target_col="target",
    n_bins=10,
):
    """
    Calculate Expected Calibration Error safely.

    Returns NaN when the dataframe is empty or calibration cannot be
    calculated.
    """

    if len(df) == 0:
        return np.nan

    scores = np.clip(
        df[score_col].astype(float),
        1e-6,
        1 - 1e-6,
    )

    try:
        bins = min(
            n_bins,
            len(scores),
        )

        if bins < 1:
            return np.nan

        df2 = df.copy()

        df2["score_bin"] = pd.qcut(
            scores,
            q=bins,
            duplicates="drop",
        )

        grouped = (
            df2
            .groupby(
                "score_bin",
                observed=True,
            )
            .agg(
                p_pred=(
                    score_col,
                    "mean",
                ),
                p_obs=(
                    target_col,
                    "mean",
                ),
                n=(
                    target_col,
                    "size",
                ),
            )
        )

        if grouped.empty:
            return np.nan

        weights = (
            grouped["n"]
            / grouped["n"].sum()
        )

        return float(
            (
                weights
                * np.abs(
                    grouped["p_pred"]
                    - grouped["p_obs"]
                )
            ).sum()
        )

    except Exception:
        return np.nan


def fast_ece(
    scores,
    targets,
    edges,
):
    """
    Fast ECE calculation using predefined score bins.
    """

    if len(scores) == 0:
        return np.nan

    scores = np.asarray(
        scores,
        dtype=float,
    )

    targets = np.asarray(
        targets,
        dtype=float,
    )

    bin_idx = np.clip(
        np.digitize(
            scores,
            edges,
        ) - 1,
        0,
        len(edges) - 2,
    )

    total = 0.0
    n = len(scores)

    for b in np.unique(
        bin_idx
    ):

        mask = (
            bin_idx == b
        )

        if not mask.any():
            continue

        w = (
            mask.sum()
            / n
        )

        total += (
            w
            * abs(
                scores[mask].mean()
                - targets[mask].mean()
            )
        )

    return float(total)


# ============================================================================
# LOGISTIC REGRESSION / CALIBRATION
# ============================================================================

def _make_unregularized_logreg():
    """
    Create an approximately unregularized logistic regression model.
    """

    return LogisticRegression(
        C=1e9,
        solver="lbfgs",
        max_iter=200,
    )


def compute_calibration_stats(
    df,
    score_col="score",
    target_col="target",
    n_bins=10,
):
    """
    Calculate calibration intercept, slope and ECE.
    """

    result = {
        "intercept": np.nan,
        "slope": np.nan,
        "ece": np.nan,
    }

    if len(df) == 0:
        return result

    try:

        X = (
            np.clip(
                df[score_col]
                .astype(float),
                1e-6,
                1 - 1e-6,
            )
            .values
            .reshape(-1, 1)
        )

        y = (
            df[target_col]
            .astype(int)
            .values
        )

        # If there is only one target class, logistic regression cannot
        # be fitted.
        if len(
            np.unique(y)
        ) < 2:

            result["ece"] = safe_ece(
                df,
                score_col,
                target_col,
                n_bins,
            )

            return result

        lr = (
            _make_unregularized_logreg()
        )

        lr.fit(
            X,
            y,
        )

        result["intercept"] = float(
            lr.intercept_[0]
        )

        result["slope"] = float(
            lr.coef_[0][0]
        )

    except Exception:
        pass

    result["ece"] = safe_ece(
        df,
        score_col,
        target_col,
        n_bins,
    )

    return result


# ============================================================================
# FEATURE SELECTION
# ============================================================================

def get_candidate_features(
    df,
    allowed_features=None,
    excluded_cols=None,
):
    """
    Return the legitimate segmentation features.

    Parameters
    ----------
    df:
        DataFrame containing the data.

    allowed_features:
        Authoritative feature list from schema_detection.

        When supplied, ONLY these columns are considered.

    excluded_cols:
        Additional explicit exclusions.

    Notes
    -----
    This function intentionally gives priority to allowed_features.

    This is important because using:

        all df columns - a small IGNORE_FEATURES set

    is unsafe. New metadata columns could accidentally enter the analysis.

    The preferred flow is therefore:

        schema["feature_cols"]
                ↓
        get_candidate_features()
    """

    excluded_cols = set(
        excluded_cols or []
    )

    # ----------------------------------------------------------------------
    # Authoritative schema-controlled feature list
    # ----------------------------------------------------------------------

    if allowed_features is not None:

        candidates = [
            col
            for col in allowed_features
            if col in df.columns
        ]

    # ----------------------------------------------------------------------
    # Backward-compatible fallback
    # ----------------------------------------------------------------------

    else:

        candidates = [
            col
            for col in df.columns
            if col not in IGNORE_FEATURES
        ]

    final_features = []

    for col in candidates:

        if col in excluded_cols:
            continue

        # Defensive protection against SHAP columns.
        if str(col).lower().startswith(
            "shap_"
        ):
            continue

        # Defensive ID protection.
        #
        # schema_detection should already remove these, but this prevents
        # accidental leakage if the function is called independently.
        col_lower = str(col).lower()

        if (
            col_lower == "customer_id"
            or col_lower.endswith("_id")
            or col_lower.startswith("id_")
            or "customer_id" in col_lower
            or "account_id" in col_lower
            or "application_id" in col_lower
        ):
            continue

        final_features.append(
            col
        )

    return final_features


def feature_strength(
    df,
    feature,
):
    """
    Estimate how strongly a feature separates target outcomes.

    Numeric:
        absolute distance of AUC from 0.5.

    Categorical:
        target bad-rate encoding followed by AUC.
    """

    if feature not in df.columns:
        return 0.0

    try:

        if pd.api.types.is_numeric_dtype(
            df[feature]
        ):

            vals = df[
                feature
            ]

            mask = (
                vals.notna()
            )

            if mask.sum() < 2:
                return 0.0

            auc = safe_auc(
                df.loc[
                    mask,
                    "target",
                ],
                vals[mask],
            )

            if np.isnan(auc):
                return 0.0

            return abs(
                auc - 0.5
            )

        values = (
            df[feature]
            .astype(str)
        )

        # Correctly preserve missing values.
        values = values.where(
            df[feature].notna(),
            MISSING_LABEL,
        )

        agg = (
            df.assign(
                _feature_value=values
            )
            .groupby(
                "_feature_value"
            )["target"]
            .mean()
        )

        encoded = values.map(
            agg
        )

        auc = safe_auc(
            df["target"],
            encoded,
        )

        if np.isnan(auc):
            return 0.0

        return abs(
            auc - 0.5
        )

    except Exception:
        return 0.0


def select_top_features(
    df,
    features,
    max_features=30,
):
    """
    Select the strongest candidate features to control the binning search
    space.
    """

    scores = []

    for feature in features:

        try:

            scores.append(
                (
                    feature,
                    feature_strength(
                        df,
                        feature,
                    ),
                )
            )

        except Exception:

            scores.append(
                (
                    feature,
                    0.0,
                )
            )

    scores.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        feature
        for feature, _
        in scores[
            :max_features
        ]
    ]


# ============================================================================
# NUMERIC BINNING
# ============================================================================

def build_numeric_bins(
    df,
    feature,
    max_bins=8,
    min_bin_pct=0.01,
):
    """
    Build supervised numeric bins using a decision tree.
    """

    values = df[
        feature
    ]

    non_missing = (
        values.notna()
    )

    bins = []

    # ----------------------------------------------------------------------
    # Missing-value bin
    # ----------------------------------------------------------------------

    n_missing = int(
        (~non_missing).sum()
    )

    if (
        len(df) > 0
        and n_missing / len(df)
        >= min_bin_pct
    ):

        bins.append(
            {
                "feature": feature,
                "type": "missing",
                "label": (
                    f"{feature} is Missing"
                ),
            }
        )

    # ----------------------------------------------------------------------
    # Non-missing observations
    # ----------------------------------------------------------------------

    X = (
        values[non_missing]
        .values
        .reshape(-1, 1)
    )

    y = (
        df.loc[
            non_missing,
            "target",
        ]
        .values
    )

    if len(X) == 0:
        return bins

    min_samples_leaf = max(
        int(
            len(df)
            * min_bin_pct
        ),
        20,
    )

    if (
        len(X)
        < 2 * min_samples_leaf
    ):
        return bins

    # ----------------------------------------------------------------------
    # Supervised tree
    # ----------------------------------------------------------------------

    clf = DecisionTreeClassifier(
        max_leaf_nodes=max(
            2,
            max_bins,
        ),
        min_samples_leaf=min_samples_leaf,
        random_state=RANDOM_STATE,
    )

    try:
        clf.fit(
            X,
            y,
        )
    except Exception:
        return bins

    intervals = []

    tree = clf.tree_

    def recurse(
        node,
        lower,
        upper,
    ):

        if (
            tree.children_left[node]
            == tree.children_right[node]
        ):

            intervals.append(
                (
                    lower,
                    upper,
                )
            )

            return

        threshold = (
            tree.threshold[node]
        )

        recurse(
            tree.children_left[node],
            lower,
            min(
                upper,
                threshold,
            ),
        )

        recurse(
            tree.children_right[node],
            max(
                lower,
                threshold,
            ),
            upper,
        )

    recurse(
        0,
        -np.inf,
        np.inf,
    )

    # ----------------------------------------------------------------------
    # Convert intervals into predicates
    # ----------------------------------------------------------------------

    for (
        lower,
        upper,
    ) in intervals:

        if (
            lower == -np.inf
            and upper == np.inf
        ):

            label = (
                f"{feature} "
                f"(all non-missing values)"
            )

        elif lower == -np.inf:

            label = (
                f"{feature} <= "
                f"{round(upper, 3)}"
            )

        elif upper == np.inf:

            label = (
                f"{feature} > "
                f"{round(lower, 3)}"
            )

        else:

            label = (
                f"{round(lower, 3)} < "
                f"{feature} <= "
                f"{round(upper, 3)}"
            )

        bins.append(
            {
                "feature": feature,
                "type": "numeric",
                "lower": lower,
                "upper": upper,
                "label": label,
            }
        )

    return bins


# ============================================================================
# CATEGORICAL BINNING
# ============================================================================

def build_categorical_bins(
    df,
    feature,
    max_bins=8,
    min_bin_pct=0.01,
):
    """
    Build categorical bins based on target bad-rate ordering.
    """

    # Correct missing handling:
    # do NOT call astype(str) before checking isna().
    values = (
        df[feature]
        .where(
            df[feature].notna(),
            MISSING_LABEL,
        )
        .astype(str)
    )

    counts = (
        df.assign(
            _v=values
        )
        .groupby(
            "_v"
        )["target"]
        .agg(
            [
                "size",
                "mean",
            ]
        )
        .rename(
            columns={
                "size": "count",
                "mean": "bad_rate",
            }
        )
    )

    counts = counts.sort_values(
        "bad_rate"
    )

    total = len(df)

    if total == 0:
        return []

    counts["pct"] = (
        counts["count"]
        / total
    )

    # ----------------------------------------------------------------------
    # Low-cardinality categorical variable
    # ----------------------------------------------------------------------

    if len(counts) <= max_bins:

        result = []

        for cat in counts.index:

            if (
                counts.loc[
                    cat,
                    "pct",
                ]
                < min_bin_pct
            ):
                continue

            if cat == MISSING_LABEL:

                result.append(
                    {
                        "feature": feature,
                        "type": "missing",
                        "categories": [
                            cat
                        ],
                        "label": (
                            f"{feature} "
                            f"is Missing"
                        ),
                    }
                )

            else:

                result.append(
                    {
                        "feature": feature,
                        "type": "categorical",
                        "categories": [
                            cat
                        ],
                        "label": (
                            f"{feature} = "
                            f"{cat}"
                        ),
                    }
                )

        return result

    # ----------------------------------------------------------------------
    # High-cardinality categorical variable
    # ----------------------------------------------------------------------

    try:

        counts["group"] = pd.qcut(
            counts["bad_rate"]
            .rank(
                method="first"
            ),
            q=max_bins,
            labels=False,
            duplicates="drop",
        )

    except Exception:
        return []

    bins = []

    for (
        group_id,
        group_rows,
    ) in counts.groupby(
        "group"
    ):

        cats = list(
            group_rows.index
        )

        # Remove groups whose overall population is too small.
        group_pct = (
            group_rows["count"].sum()
            / total
        )

        if (
            group_pct
            < min_bin_pct
        ):
            continue

        if cats == [
            MISSING_LABEL
        ]:

            label = (
                f"{feature} "
                f"is Missing"
            )

        else:

            label = (
                f"{feature} IN "
                f"({', '.join(map(str, cats))})"
            )

        bins.append(
            {
                "feature": feature,
                "type": "categorical",
                "categories": cats,
                "label": label,
            }
        )

    return bins


# ============================================================================
# MASK CREATION
# ============================================================================

def _single_condition_mask(
    df,
    cond,
):
    """
    Evaluate a single bin condition.
    """

    feat = cond[
        "feature"
    ]

    if feat not in df.columns:

        return np.zeros(
            len(df),
            dtype=bool,
        )

    # ----------------------------------------------------------------------
    # Missing condition
    # ----------------------------------------------------------------------

    if cond["type"] == "missing":

        return (
            df[feat]
            .isna()
            .to_numpy()
        )

    # ----------------------------------------------------------------------
    # Numeric condition
    # ----------------------------------------------------------------------

    if cond["type"] == "numeric":

        mask = (
            df[feat]
            .notna()
            .to_numpy()
            .copy()
        )

        if (
            cond["lower"]
            != -np.inf
        ):

            mask &= (
                df[feat]
                > cond["lower"]
            ).values

        if (
            cond["upper"]
            != np.inf
        ):

            mask &= (
                df[feat]
                <= cond["upper"]
            ).values

        return mask

    # ----------------------------------------------------------------------
    # Categorical condition
    # ----------------------------------------------------------------------

    cats = cond[
        "categories"
    ]

    values = (
        df[feat]
        .where(
            df[feat].notna(),
            MISSING_LABEL,
        )
        .astype(str)
    )

    return (
        values
        .isin(cats)
        .values
    )


def bin_mask(
    df,
    bin_def,
):
    """
    Evaluate a complete bin definition.

    Supports:

        - single feature bins
        - interaction bins
    """

    if (
        bin_def["type"]
        == "interaction"
    ):

        mask = np.ones(
            len(df),
            dtype=bool,
        )

        for cond in (
            bin_def["conditions"]
        ):

            mask &= (
                _single_condition_mask(
                    df,
                    cond,
                )
            )

        return mask

    return _single_condition_mask(
        df,
        bin_def,
    )


# ============================================================================
# POINT ESTIMATE METRICS
# ============================================================================

def point_estimate_metrics(
    bin_def,
    dev_df,
    mon_df,
    total_dev,
    total_mon,
    target_col,
    score_col,
    weight_col,
    min_bin_pct=0.01,
    max_segment_pct=0.35,
):
    """
    Calculate point-estimate performance, drift and business metrics for
    one candidate bin.

    IMPORTANT DELTA CONVENTION
    --------------------------

    This module historically defines:

        delta_auc  = Dev_AUC  - Mon_AUC
        delta_gini = Dev_Gini - Mon_Gini
        delta_ks   = Dev_KS   - Mon_KS

    Therefore:

        Positive delta = deterioration.

    This convention is preserved for compatibility with the existing
    Feature Binning ranking logic.
    """

    dev_mask = bin_mask(
        dev_df,
        bin_def,
    )

    mon_mask = bin_mask(
        mon_df,
        bin_def,
    )

    n_dev = int(
        dev_mask.sum()
    )

    n_mon = int(
        mon_mask.sum()
    )

    min_dev_count = max(
        30,
        int(
            total_dev
            * min_bin_pct
        ),
    )

    min_mon_count = max(
        30,
        int(
            total_mon
            * min_bin_pct
        ),
    )

    if (
        n_dev < min_dev_count
        or n_mon < min_mon_count
    ):
        return None

    dev_sub = dev_df.loc[
        dev_mask
    ]

    mon_sub = mon_df.loc[
        mon_mask
    ]

    # ----------------------------------------------------------------------
    # Performance
    # ----------------------------------------------------------------------

    auc_dev, gini_dev = (
        calculate_auc_gini(
            dev_sub[target_col],
            dev_sub[score_col],
        )
    )

    auc_mon, gini_mon = (
        calculate_auc_gini(
            mon_sub[target_col],
            mon_sub[score_col],
        )
    )

    ks_dev = calculate_ks(
        dev_sub[target_col],
        dev_sub[score_col],
    )

    ks_mon = calculate_ks(
        mon_sub[target_col],
        mon_sub[score_col],
    )

    # ----------------------------------------------------------------------
    # Bad rate
    # ----------------------------------------------------------------------

    br_dev = float(
        dev_sub[
            target_col
        ].mean()
    )

    br_mon = float(
        mon_sub[
            target_col
        ].mean()
    )

    # ----------------------------------------------------------------------
    # Calibration
    # ----------------------------------------------------------------------

    cal_dev = (
        compute_calibration_stats(
            dev_sub,
            score_col=score_col,
            target_col=target_col,
        )
    )

    cal_mon = (
        compute_calibration_stats(
            mon_sub,
            score_col=score_col,
            target_col=target_col,
        )
    )

    # ----------------------------------------------------------------------
    # Exposure
    # ----------------------------------------------------------------------

    if (
        weight_col
        and weight_col
        in dev_sub.columns
    ):

        ead_dev = float(
            dev_sub[
                weight_col
            ].sum()
        )

    else:

        ead_dev = np.nan

    if (
        weight_col
        and weight_col
        in mon_sub.columns
    ):

        ead_mon = float(
            mon_sub[
                weight_col
            ].sum()
        )

    else:

        ead_mon = np.nan

    if (
        weight_col
        and weight_col
        in mon_df.columns
    ):

        total_ead_mon = float(
            mon_df[
                weight_col
            ].sum()
        )

    else:

        total_ead_mon = np.nan

    if (
        not np.isnan(ead_mon)
        and total_ead_mon > 0
    ):

        exposure_pct = (
            ead_mon
            / total_ead_mon
        )

    else:

        exposure_pct = 0.0

    # ----------------------------------------------------------------------
    # Segment population-share PSI
    # ----------------------------------------------------------------------

    dev_pct = (
        n_dev / total_dev
        if total_dev
        else 0.0
    )

    mon_pct = (
        n_mon / total_mon
        if total_mon
        else 0.0
    )

    psi = calculate_psi(
        dev_pct,
        mon_pct,
    )

    # ----------------------------------------------------------------------
    # Segment size cap (both dev and mon side), matching
    # core/candidate_filtering.py's max_segment_pct gate for the other
    # techniques.
    # ----------------------------------------------------------------------

    if (
        dev_pct > max_segment_pct
        or mon_pct > max_segment_pct
    ):
        return None

    # ----------------------------------------------------------------------
    # Return metrics
    # ----------------------------------------------------------------------

    return {
        "feature": bin_def[
            "feature"
        ],

        "segment": bin_def[
            "label"
        ],

        "bin_type": bin_def[
            "type"
        ],

        "dev_count": n_dev,

        "mon_count": n_mon,

        "dev_pct": dev_pct,

        "mon_pct": mon_pct,

        "psi": psi,

        "psi_interpretation": (
            interpret_psi(psi)
        ),

        # AUC
        "auc_dev": auc_dev,

        "auc_mon": auc_mon,

        "delta_auc": (
            auc_dev - auc_mon
            if not (
                np.isnan(auc_dev)
                or np.isnan(auc_mon)
            )
            else np.nan
        ),

        # Gini
        "gini_dev": gini_dev,

        "gini_mon": gini_mon,

        "delta_gini": (
            gini_dev - gini_mon
            if not (
                np.isnan(gini_dev)
                or np.isnan(gini_mon)
            )
            else np.nan
        ),

        # KS
        "ks_dev": ks_dev,

        "ks_mon": ks_mon,

        "delta_ks": (
            ks_dev - ks_mon
            if not (
                np.isnan(ks_dev)
                or np.isnan(ks_mon)
            )
            else np.nan
        ),

        # Bad rate
        "br_dev": br_dev,

        "br_mon": br_mon,

        "delta_br": (
            br_mon - br_dev
        ),

        # Calibration intercept
        "cal_dev_intercept": (
            cal_dev["intercept"]
        ),

        "cal_mon_intercept": (
            cal_mon["intercept"]
        ),

        "delta_intercept": (
            cal_mon["intercept"]
            - cal_dev["intercept"]
            if not (
                np.isnan(
                    cal_dev["intercept"]
                )
                or np.isnan(
                    cal_mon["intercept"]
                )
            )
            else np.nan
        ),

        # Calibration slope
        "cal_dev_slope": (
            cal_dev["slope"]
        ),

        "cal_mon_slope": (
            cal_mon["slope"]
        ),

        "delta_slope": (
            cal_mon["slope"]
            - cal_dev["slope"]
            if not (
                np.isnan(
                    cal_dev["slope"]
                )
                or np.isnan(
                    cal_mon["slope"]
                )
            )
            else np.nan
        ),

        # ECE
        "ece_dev": cal_dev[
            "ece"
        ],

        "ece_mon": cal_mon[
            "ece"
        ],

        "delta_ece": (
            cal_mon["ece"]
            - cal_dev["ece"]
            if not (
                np.isnan(
                    cal_dev["ece"]
                )
                or np.isnan(
                    cal_mon["ece"]
                )
            )
            else np.nan
        ),

        # Exposure
        "ead_dev": ead_dev,

        "ead_mon": ead_mon,

        "exposure_pct": exposure_pct,

        # Monitoring defaults
        "default_count": int(
            mon_sub[
                target_col
            ].sum()
        ),

        # Internal masks
        "_dev_mask": dev_mask,

        "_mon_mask": mon_mask,
    }


# ============================================================================
# BOOTSTRAP SIGNIFICANCE
# ============================================================================

def bootstrap_signed_difference(
    dev_sub,
    mon_sub,
    func,
    n_iter=50,
):
    """
    Bootstrap the signed monitoring-development difference.

    Returns:

        lower CI
        upper CI
        mean point estimate
        two-sided p-value
    """

    if (
        len(dev_sub) < 30
        or len(mon_sub) < 30
    ):

        return (
            np.nan,
            np.nan,
            np.nan,
            1.0,
        )

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    diffs = []

    for _ in range(
        n_iter
    ):

        dev_sample = (
            dev_sub.sample(
                n=len(dev_sub),
                replace=True,
                random_state=(
                    rng.randint(
                        1_000_000
                    )
                ),
            )
        )

        mon_sample = (
            mon_sub.sample(
                n=len(mon_sub),
                replace=True,
                random_state=(
                    rng.randint(
                        1_000_000
                    )
                ),
            )
        )

        try:

            val = func(
                dev_sample,
                mon_sample,
            )

            if not np.isnan(
                val
            ):

                diffs.append(
                    val
                )

        except Exception:
            continue

    if not diffs:

        return (
            np.nan,
            np.nan,
            np.nan,
            1.0,
        )

    diffs = np.asarray(
        diffs,
        dtype=float,
    )

    lower = float(
        np.percentile(
            diffs,
            2.5,
        )
    )

    upper = float(
        np.percentile(
            diffs,
            97.5,
        )
    )

    point = float(
        np.mean(diffs)
    )

    p_le = np.mean(
        diffs <= 0.0
    )

    p_ge = np.mean(
        diffs >= 0.0
    )

    p_two_sided = float(
        min(
            1.0,
            2.0
            * min(
                p_le,
                p_ge,
            ),
        )
    )

    return (
        lower,
        upper,
        point,
        p_two_sided,
    )


# ============================================================================
# SIGNIFICANCE
# ============================================================================

def add_significance(
    row,
    dev_df,
    mon_df,
    n_iter=50,
):
    """
    Add bootstrap confidence intervals and p-values to a candidate row.
    """

    dev_sub = dev_df.loc[
        row["_dev_mask"]
    ]

    mon_sub = mon_df.loc[
        row["_mon_mask"]
    ]

    # ----------------------------------------------------------------------
    # Difference functions
    # ----------------------------------------------------------------------

    def gini_diff(
        dv,
        mv,
    ):

        return (
            2
            * safe_auc(
                mv["target"],
                mv["score"],
            )
            - 1
        ) - (
            2
            * safe_auc(
                dv["target"],
                dv["score"],
            )
            - 1
        )

    def ks_diff(
        dv,
        mv,
    ):

        return (
            calculate_ks(
                mv["target"],
                mv["score"],
            )
            - calculate_ks(
                dv["target"],
                dv["score"],
            )
        )

    def auc_diff(
        dv,
        mv,
    ):

        return (
            safe_auc(
                mv["target"],
                mv["score"],
            )
            - safe_auc(
                dv["target"],
                dv["score"],
            )
        )

    def br_diff(
        dv,
        mv,
    ):

        return (
            float(
                mv["target"].mean()
            )
            - float(
                dv["target"].mean()
            )
        )

    # ----------------------------------------------------------------------
    # Common score bins for ECE
    # ----------------------------------------------------------------------

    all_scores = pd.concat(
        [
            dev_sub["score"],
            mon_sub["score"],
        ]
    )

    all_scores = np.clip(
        all_scores.astype(float),
        1e-6,
        1 - 1e-6,
    )

    try:

        edges = np.quantile(
            all_scores,
            np.linspace(
                0,
                1,
                11,
            ),
        )

        edges[0] = -np.inf
        edges[-1] = np.inf

    except Exception:

        edges = np.array(
            [
                -np.inf,
                np.inf,
            ]
        )

    def ece_diff(
        dv,
        mv,
    ):

        return (
            fast_ece(
                mv["score"].values,
                mv["target"].values,
                edges,
            )
            - fast_ece(
                dv["score"].values,
                dv["target"].values,
                edges,
            )
        )

    # ----------------------------------------------------------------------
    # Bootstrap all metrics
    # ----------------------------------------------------------------------

    functions = [
        (
            "gini",
            gini_diff,
        ),
        (
            "ks",
            ks_diff,
        ),
        (
            "auc",
            auc_diff,
        ),
        (
            "br",
            br_diff,
        ),
        (
            "ece",
            ece_diff,
        ),
    ]

    for (
        name,
        fn,
    ) in functions:

        (
            lo,
            hi,
            point,
            p,
        ) = bootstrap_signed_difference(
            dev_sub,
            mon_sub,
            fn,
            n_iter=n_iter,
        )

        row[
            f"{name}_ci_low"
        ] = lo

        row[
            f"{name}_ci_high"
        ] = hi

        row[
            f"{name}_p"
        ] = p

    return row


# ============================================================================
# SEGMENT SCORING
# ============================================================================

def score_segments(
    segments,
    apply_bh=True,
    alpha=0.10,
):
    """
    Score candidate segments using deterioration, confidence and business
    impact.
    """

    df = pd.DataFrame(
        segments
    )

    if df.empty:
        return df

    # ----------------------------------------------------------------------
    # Absolute shifts
    # ----------------------------------------------------------------------

    df["abs_delta_br"] = (
        np.abs(
            df["delta_br"]
        )
    )

    df["abs_delta_ece"] = (
        np.abs(
            df["delta_ece"]
        )
    )

    # ----------------------------------------------------------------------
    # Normalized deterioration components
    # ----------------------------------------------------------------------

    df["gini_score"] = (
        min_max_normalize(
            df["delta_gini"]
        )
    )

    df["ks_score"] = (
        min_max_normalize(
            df["delta_ks"]
        )
    )

    df["auc_score"] = (
        min_max_normalize(
            df["delta_auc"]
        )
    )

    df["br_score"] = (
        min_max_normalize(
            df["abs_delta_br"]
        )
    )

    df["ece_score"] = (
        min_max_normalize(
            df["abs_delta_ece"]
        )
    )

    deterioration_components = [
        "gini_score",
        "ks_score",
        "auc_score",
        "br_score",
        "ece_score",
    ]

    df["deterioration_score"] = (
        df[
            deterioration_components
        ]
        .mean(
            axis=1
        )
    )

    # ----------------------------------------------------------------------
    # Multiple-testing correction
    # ----------------------------------------------------------------------

    p_cols = [
        "gini_p",
        "ks_p",
        "auc_p",
        "br_p",
        "ece_p",
    ]

    if apply_bh:

        for c in p_cols:

            if c not in df.columns:
                continue

            try:

                df[
                    f"{c}_adj"
                ] = adjust_p_values_bh(
                    df[c].values
                )

            except Exception:

                df[
                    f"{c}_adj"
                ] = df[c]

        p_adj_cols = [
            f"{c}_adj"
            for c in p_cols
            if f"{c}_adj"
            in df.columns
        ]

        if p_adj_cols:

            df["min_p_adj"] = (
                df[
                    p_adj_cols
                ]
                .min(
                    axis=1,
                    skipna=True,
                )
            )

            def confidence_from_row(
                row
            ):

                values = (
                    row[
                        p_adj_cols
                    ]
                    .fillna(1.0)
                    .astype(float)
                )

                values = np.clip(
                    values,
                    0.0,
                    1.0,
                )

                return float(
                    np.nanmean(
                        1.0 - values
                    )
                )

            df[
                "confidence_score"
            ] = df.apply(
                confidence_from_row,
                axis=1,
            )

            df[
                "significant"
            ] = (
                df["min_p_adj"]
                <= alpha
            )

        else:

            df[
                "min_p_adj"
            ] = 1.0

            df[
                "confidence_score"
            ] = 0.5

            df[
                "significant"
            ] = False

    else:

        df[
            "confidence_score"
        ] = 0.5

        df[
            "significant"
        ] = False

    # ----------------------------------------------------------------------
    # PSI interpretation
    # ----------------------------------------------------------------------

    if "psi" in df.columns:

        df[
            "psi_interpretation"
        ] = df[
            "psi"
        ].apply(
            interpret_psi
        )

    else:

        df[
            "psi_interpretation"
        ] = "Stable"

    # ----------------------------------------------------------------------
    # Business metrics
    # ----------------------------------------------------------------------

    def calculate_sis_safe(
        row
    ):

        try:

            if np.isnan(
                row["psi"]
            ):
                return np.nan

            # calculate_sis expects delta = monitoring - development
            # (shared convention). This module's own delta_gini/delta_ks
            # are dev - mon (see point_estimate_metrics docstring), so
            # negate them at this boundary only.
            return calculate_sis(
                row["psi"],
                -row["delta_gini"],
                -row["delta_ks"],
                row["delta_br"],
                row["mon_pct"],
                row.get(
                    "exposure_pct",
                    0.0,
                ),
            )[
                "raw"
            ]

        except Exception:
            return np.nan

    def calculate_sis_business_safe(
        row
    ):

        try:

            if np.isnan(
                row["psi"]
            ):
                return np.nan

            return calculate_sis(
                row["psi"],
                -row["delta_gini"],
                -row["delta_ks"],
                row["delta_br"],
                row["mon_pct"],
                row.get(
                    "exposure_pct",
                    0.0,
                ),
            )[
                "business_impact"
            ]

        except Exception:
            return np.nan

    def calculate_dis_safe(
        row
    ):

        try:

            if np.isnan(
                row["psi"]
            ):
                return np.nan

            return calculate_dis(
                row["psi"],
                -row["delta_gini"],
                -row["delta_ks"],
                row["delta_br"],
            )

        except Exception:
            return np.nan

    df[
        "sis_raw"
    ] = df.apply(
        calculate_sis_safe,
        axis=1,
    )

    df[
        "sis_business_impact"
    ] = df.apply(
        calculate_sis_business_safe,
        axis=1,
    )

    df[
        "dis_raw"
    ] = df.apply(
        calculate_dis_safe,
        axis=1,
    )

    # ----------------------------------------------------------------------
    # Business impact
    # ----------------------------------------------------------------------

    if "exposure_pct" in df.columns:

        df[
            "business_impact_score"
        ] = (
            df["exposure_pct"]
            .fillna(0.0)
            .clip(
                0.0,
                1.0,
            )
        )

    else:

        df[
            "business_impact_score"
        ] = 0.0

    # ----------------------------------------------------------------------
    # Final score
    # ----------------------------------------------------------------------

    df[
        "final_score"
    ] = (
        df[
            "deterioration_score"
        ]
        * df[
            "confidence_score"
        ]
        * (
            0.5
            + 0.5
            * df[
                "business_impact_score"
            ]
        )
    )

    # ----------------------------------------------------------------------
    # Rank
    # ----------------------------------------------------------------------

    df[
        "rank"
    ] = (
        df[
            "final_score"
        ]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    return (
        df.sort_values(
            [
                "final_score",
                "deterioration_score",
            ],
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================================
# FEATURE BINNING TECHNIQUE
# ============================================================================

class FeatureBinningTechnique(
    BaseSegmentationTechnique
):
    """
    Feature Binning Technique.

    Discovers feature bins where monitoring behavior differs from
    development behavior.
    """

    @property
    def name(self) -> str:
        return "Feature Binning"

    def run(
        self,
        dev_data: Union[
            str,
            pd.DataFrame,
        ],
        mon_data: Union[
            str,
            pd.DataFrame,
        ],
        config: Optional[
            FeatureBinningConfig
        ] = None,
    ) -> TechniqueResult:

        cfg = (
            config
            or FeatureBinningConfig()
        )

        start_time = (
            time.perf_counter()
        )

        # ==================================================================
        # LOAD DATA
        # ==================================================================

        dev_df = (
            pd.read_csv(dev_data)
            if isinstance(
                dev_data,
                str,
            )
            else dev_data.copy()
        )

        mon_df = (
            pd.read_csv(mon_data)
            if isinstance(
                mon_data,
                str,
            )
            else mon_data.copy()
        )

        # ==================================================================
        # SCHEMA DETECTION
        # ==================================================================

        schema = detect_schema(
            dev_df,
            cfg.schema,
        )

        target_col = schema[
            "target_col"
        ]

        score_col = schema[
            "score_col"
        ]

        weight_col = schema[
            "weight_col"
        ]

        # IMPORTANT:
        #
        # This is now the authoritative feature list.
        #
        # Do NOT use:
        #
        #     all columns - IGNORE_FEATURES
        #
        # because that could allow an unrecognized metadata/ID column to
        # enter the analysis.
        feature_cols = schema[
            "feature_cols"
        ]

        total_dev = len(
            dev_df
        )

        total_mon = len(
            mon_df
        )

        # ==================================================================
        # RENAME INTERNAL ANALYSIS COLUMNS
        # ==================================================================
        #
        # The rest of this module historically expects:
        #
        #     target
        #     score
        #     ead
        #
        # We preserve that internal convention.
        #
        # The original schema names remain available in "schema".
        # ==================================================================

        dev_df = dev_df.rename(
            columns={
                target_col: "target",
                score_col: "score",
            }
        )

        mon_df = mon_df.rename(
            columns={
                target_col: "target",
                score_col: "score",
            }
        )

        if weight_col:

            dev_df = dev_df.rename(
                columns={
                    weight_col: "ead"
                }
            )

            mon_df = mon_df.rename(
                columns={
                    weight_col: "ead"
                }
            )

        # ==================================================================
        # SHAP DETECTION
        # ==================================================================
        #
        # SHAP columns are used only for SHAP-shift diagnostics.
        #
        # They are NOT added to "feature_cols".
        # ==================================================================

        combined_df = pd.concat(
            [
                dev_df,
                mon_df,
            ],
            ignore_index=True,
        )

        shap_cols = (
            auto_detect_shap_columns(
                combined_df
            )
        )

        if shap_cols:

            shap_shift = detect_shap_shift(
                dev_df,
                mon_df,
                shap_cols,
            )

            logger.info(
                "Feature Binning detected SHAP shift: %s",
                shap_shift[
                    "top_shift_feature"
                ],
            )

        # ==================================================================
        # VERIFY / FILTER LEGITIMATE FEATURES
        # ==================================================================

        features = (
            get_candidate_features(
                dev_df,
                allowed_features=feature_cols,
            )
        )

        # Defensive logging.
        logger.info(
            "Feature Binning candidate features: %s",
            features,
        )

        # ==================================================================
        # SELECT STRONGEST FEATURES
        # ==================================================================

        selected = (
            select_top_features(
                dev_df,
                features,
                cfg.max_features,
            )
        )

        logger.info(
            "Feature Binning selected features: %s",
            selected,
        )

        # ==================================================================
        # BUILD BIN DEFINITIONS
        # ==================================================================

        bin_defs = []

        for feature in selected:

            if pd.api.types.is_numeric_dtype(
                dev_df[feature]
            ):

                bin_defs.extend(
                    build_numeric_bins(
                        dev_df,
                        feature,
                        max_bins=cfg.max_bins,
                        min_bin_pct=cfg.min_bin_pct,
                    )
                )

            else:

                bin_defs.extend(
                    build_categorical_bins(
                        dev_df,
                        feature,
                        max_bins=cfg.max_bins,
                        min_bin_pct=cfg.min_bin_pct,
                    )
                )

        # ==================================================================
        # POINT ESTIMATE EVALUATION
        # ==================================================================

        point_rows = []

        for bin_def in bin_defs:

            result = (
                point_estimate_metrics(
                    bin_def,
                    dev_df,
                    mon_df,
                    total_dev,
                    total_mon,
                    "target",
                    "score",
                    (
                        "ead"
                        if weight_col
                        else None
                    ),
                    min_bin_pct=(
                        cfg.min_bin_pct
                    ),
                    max_segment_pct=(
                        cfg.max_segment_pct
                    ),
                )
            )

            if result is not None:

                point_rows.append(
                    result
                )

        if not point_rows:

            exec_time = (
                time.perf_counter()
                - start_time
            )

            return TechniqueResult(
                technique_name=self.name,
                overall={
                    "dev_count": total_dev,
                    "mon_count": total_mon,
                    "features_evaluated": len(
                        features
                    ),
                    "candidate_bins": len(
                        bin_defs
                    ),
                    "bootstrapped": 0,
                },
                segments_df=pd.DataFrame(),
                execution_time=exec_time,
                schema=schema,
            )

        # ==================================================================
        # ROUGH EFFECT FOR BOOTSTRAP PRIORITIZATION
        # ==================================================================

        def rough_effect(
            row
        ):

            values = []

            for key in (
                "delta_gini",
                "delta_ks",
                "delta_auc",
                "delta_br",
                "delta_ece",
            ):

                value = row.get(
                    key,
                    0,
                )

                if value is None:
                    continue

                try:

                    if not np.isnan(
                        value
                    ):

                        values.append(
                            abs(
                                value
                            )
                        )

                except Exception:
                    continue

            if not values:
                return 0.0

            return float(
                np.nanmean(
                    values
                )
            )

        point_rows.sort(
            key=rough_effect,
            reverse=True,
        )

        to_bootstrap = (
            point_rows[
                : cfg.bootstrap_top_n
            ]
        )

        remainder = (
            point_rows[
                cfg.bootstrap_top_n:
            ]
        )

        # ==================================================================
        # BOOTSTRAP SIGNIFICANCE
        # ==================================================================

        bootstrapped = [
            add_significance(
                dict(row),
                dev_df,
                mon_df,
                n_iter=cfg.n_iter,
            )
            for row in to_bootstrap
        ]

        # ==================================================================
        # REMOVE INTERNAL MASKS
        # ==================================================================

        for row in (
            bootstrapped
            + remainder
        ):

            row.pop(
                "_dev_mask",
                None,
            )

            row.pop(
                "_mon_mask",
                None,
            )

        # Non-bootstrapped candidates do not have significance estimates.
        for row in remainder:

            for name in (
                "gini",
                "ks",
                "auc",
                "br",
                "ece",
            ):

                row[
                    f"{name}_ci_low"
                ] = np.nan

                row[
                    f"{name}_ci_high"
                ] = np.nan

                row[
                    f"{name}_p"
                ] = np.nan

        # ==================================================================
        # SCORE BOOTSTRAPPED CANDIDATES
        # ==================================================================

        scored_boot = (
            score_segments(
                bootstrapped,
                apply_bh=True,
                alpha=cfg.alpha,
            )
            if bootstrapped
            else pd.DataFrame()
        )

        # ==================================================================
        # SCORE REMAINDER
        # ==================================================================

        scored_rest = (
            score_segments(
                remainder,
                apply_bh=False,
            )
            if remainder
            else pd.DataFrame()
        )

        # ==================================================================
        # COMBINE SCORES
        # ==================================================================

        if len(scored_rest):

            scored = pd.concat(
                [
                    scored_boot,
                    scored_rest,
                ],
                ignore_index=True,
            )

        else:

            scored = scored_boot

        if scored.empty:

            exec_time = (
                time.perf_counter()
                - start_time
            )

            return TechniqueResult(
                technique_name=self.name,
                overall={
                    "dev_count": total_dev,
                    "mon_count": total_mon,
                    "features_evaluated": len(
                        features
                    ),
                    "candidate_bins": len(
                        bin_defs
                    ),
                    "bootstrapped": len(
                        to_bootstrap
                    ),
                },
                segments_df=pd.DataFrame(),
                execution_time=exec_time,
                schema=schema,
            )

        # ==================================================================
        # FINAL SORTING
        # ==================================================================

        scored = (
            scored.sort_values(
                [
                    "final_score",
                    "deterioration_score",
                ],
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        scored[
            "rank"
        ] = np.arange(
            1,
            len(scored) + 1,
        )

        # ==================================================================
        # ROOT CAUSE
        # ==================================================================
        #
        # For Feature Binning, the candidate itself is a feature-defined
        # segment.
        #
        # Example:
        #
        #     income <= 49487
        #
        # Therefore the feature defining the segment is the primary
        # candidate root-cause feature.
        #
        # This is a candidate driver, NOT proof of causality.
        #
        # IMPORTANT:
        # Since "feature" came exclusively from schema["feature_cols"],
        # customer_id cannot appear here.
        # ==================================================================

        scored[
            "Root_Cause_Feature"
        ] = scored[
            "feature"
        ]

        scored[
            "Root_Cause_PSI"
        ] = scored[
            "psi"
        ]

        # IMPORTANT:
        #
        # In Feature Binning:
        #
        #     delta_gini = Dev_Gini - Mon_Gini
        #
        # Therefore a positive delta_gini means deterioration.
        #
        # The previous implementation used:
        #
        #     max(0, -delta_gini)
        #
        # which reversed the direction.
        #
        # Correct:
        #
        #     max(0, delta_gini)
        # ------------------------------------------------------------------

        gini_deterioration = (
            scored[
                "delta_gini"
            ]
            .apply(
                lambda g:
                max(
                    0.0,
                    g,
                )
                if not pd.isna(g)
                else 0.0
            )
        )

        scored[
            "Root_Cause_Score"
        ] = (
            scored[
                "psi"
            ]
            * (
                1.0
                + gini_deterioration
            )
            * scored[
                "mon_pct"
            ]
        ).round(
            6
        )

        # ==================================================================
        # TOP N
        # ==================================================================

        scored = scored.head(
            cfg.top_n
        )

        # ==================================================================
        # STANDARDIZE SIGN CONVENTION FOR PUBLIC OUTPUT
        # ==================================================================
        # Internally this module uses delta = dev - mon (positive =
        # deterioration, see point_estimate_metrics docstring). Every other
        # technique's public output uses delta = mon - dev (negative =
        # deterioration). Flip here, at the output boundary only, so
        # Delta_AUC/Delta_Gini/Delta_KS mean the same thing regardless of
        # technique. All internal scoring above (deterioration_score,
        # Root_Cause_Score, rank, final_score) already consumed the
        # dev-mon values and is unaffected by this flip.

        for _col in ("delta_auc", "delta_gini", "delta_ks"):
            if _col in scored.columns:
                scored[_col] = -scored[_col]

        # ==================================================================
        # STANDARDIZED EXPORT-FRIENDLY COLUMNS
        # ==================================================================

        rename_columns = {
            "feature": (
                "Feature"
            ),
            "segment": (
                "Segment_Definition"
            ),
            "bin_type": (
                "Bin_Type"
            ),
            "dev_count": (
                "Dev_Count"
            ),
            "mon_count": (
                "Mon_Count"
            ),
            "dev_pct": (
                "Dev_Pct"
            ),
            "mon_pct": (
                "Mon_Pct"
            ),
            "psi": (
                "PSI"
            ),
            "psi_interpretation": (
                "PSI_Interpretation"
            ),
            "auc_dev": (
                "Dev_AUC"
            ),
            "auc_mon": (
                "Mon_AUC"
            ),
            "delta_auc": (
                "Delta_AUC"
            ),
            "gini_dev": (
                "Dev_Gini"
            ),
            "gini_mon": (
                "Mon_Gini"
            ),
            "delta_gini": (
                "Delta_Gini"
            ),
            "ks_dev": (
                "Dev_KS"
            ),
            "ks_mon": (
                "Mon_KS"
            ),
            "delta_ks": (
                "Delta_KS"
            ),
            "br_dev": (
                "Dev_BR"
            ),
            "br_mon": (
                "Mon_BR"
            ),
            "delta_br": (
                "Delta_BR"
            ),
            "cal_dev_intercept": (
                "Dev_Calibration_Intercept"
            ),
            "cal_mon_intercept": (
                "Mon_Calibration_Intercept"
            ),
            "delta_intercept": (
                "Delta_Calibration_Intercept"
            ),
            "cal_dev_slope": (
                "Dev_Calibration_Slope"
            ),
            "cal_mon_slope": (
                "Mon_Calibration_Slope"
            ),
            "delta_slope": (
                "Delta_Calibration_Slope"
            ),
            "ece_dev": (
                "Dev_ECE"
            ),
            "ece_mon": (
                "Mon_ECE"
            ),
            "delta_ece": (
                "Delta_ECE"
            ),
            "ead_dev": (
                "Dev_EAD"
            ),
            "ead_mon": (
                "Mon_EAD"
            ),
            "exposure_pct": (
                "Exposure_Pct"
            ),
            "default_count": (
                "Mon_Default_Count"
            ),
            "deterioration_score": (
                "Deterioration_Score"
            ),
            "confidence_score": (
                "Confidence_Score"
            ),
            "significant": (
                "Significant"
            ),
            "sis_raw": (
                "SIS_Raw"
            ),
            "sis_business_impact": (
                "SIS_Business_Impact"
            ),
            "dis_raw": (
                "DIS_Raw"
            ),
            "business_impact_score": (
                "Business_Impact_Score"
            ),
            "final_score": (
                "Overall_Score"
            ),
            "rank": (
                "Rank"
            ),
            "Root_Cause_Feature": (
                "Root_Cause_Feature"
            ),
            "Root_Cause_PSI": (
                "Root_Cause_PSI"
            ),
            "Root_Cause_Score": (
                "Root_Cause_Score"
            ),
        }

        scored = scored.rename(
            columns=rename_columns
        )

        # ==================================================================
        # EXECUTION SUMMARY
        # ==================================================================

        exec_time = (
            time.perf_counter()
            - start_time
        )

        overall = {
            "dev_count": total_dev,

            "mon_count": total_mon,

            "features_evaluated": len(
                features
            ),

            "candidate_bins": len(
                bin_defs
            ),

            "bootstrapped": len(
                to_bootstrap
            ),

            "candidates_returned": len(
                scored
            ),
        }

        # ==================================================================
        # RETURN
        # ==================================================================

        return TechniqueResult(
            technique_name=self.name,
            overall=overall,
            segments_df=scored,
            execution_time=exec_time,
            schema=schema,
        )


# ============================================================================
# FUNCTIONAL WRAPPER
# ============================================================================

def run_feature_binning_segmentation(
    dev_path: Union[
        str,
        pd.DataFrame,
    ],
    mon_path: Union[
        str,
        pd.DataFrame,
    ],
    cfg: Optional[
        FeatureBinningConfig
    ] = None,
) -> dict:
    """
    Functional wrapper for Feature Binning execution.

    Parameter optimization is handled by the common optimization framework.
    """

    from core.parameter_optimization import (
        optimize_parameters,
    )

    cfg = (
        cfg
        or FeatureBinningConfig()
    )

    dev_df = (
        pd.read_csv(dev_path)
        if isinstance(
            dev_path,
            str,
        )
        else dev_path.copy()
    )

    mon_df = (
        pd.read_csv(mon_path)
        if isinstance(
            mon_path,
            str,
        )
        else mon_path.copy()
    )

    def _run_single(
        d_df,
        m_df,
        config,
    ):

        technique = (
            FeatureBinningTechnique()
        )

        return technique.run(
            d_df,
            m_df,
            config=config,
        )

    res = optimize_parameters(
        _run_single,
        cfg,
        dev_df,
        mon_df,
    )

    return res.to_dict()