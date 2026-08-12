"""
core/candidate_evaluation.py
============================

Shared candidate-segment evaluation pipeline.

Computes:

    - Performance metrics:
        AUC, Gini, KS
    - Drift metrics:
        Segment population-share PSI,
        feature-level PSI,
        bad-rate shift
    - Statistical significance:
        two-proportion z-test,
        KS-test
    - Business metrics:
        exposure drift
    - Root Cause Analysis:
        feature-level PSI within the candidate segment

Important design principles
----------------------------

1. Customer/account/application IDs are NEVER considered root-cause features.
2. Target, model score, exposure/weight, time/vintage and SHAP columns are
   excluded from normal feature-level RCA.
3. Root cause is based on feature-level distributional drift within the
   affected segment. It identifies a CANDIDATE DRIVER, not causal proof.
4. All delta metrics use the convention:

       Delta = Monitoring - Development

   Therefore:
       Negative Delta -> deterioration for AUC/Gini/KS
       Positive Delta -> deterioration for bad rate
5. If feature_cols are supplied by schema detection, they are still sanitized
   defensively before RCA.
6. customer_id is explicitly blocked at multiple levels:
       - feature sanitization
       - feature-level RCA
       - final runtime assertion
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from metrics.performance_metrics import (
    calculate_auc_gini,
    calculate_ks,
    calculate_calibration_drift,
)

from metrics.drift_metrics import (
    calculate_psi,
    compute_psi_for_feature,
    build_decile_bins,
    detect_shap_shift,
)

from metrics.significance import (
    two_proportion_ztest_pvalue,
    ks_score_shift_pvalue,
)


# ---------------------------------------------------------------------------
# Column safety helpers
# ---------------------------------------------------------------------------

_ID_PATTERNS = re.compile(
    r"(^id$|_id$|^id_|customer_?id|account_?id|application_?id|"
    r"loan_?id|transaction_?id|uuid|identifier)",
    re.I,
)

_TIME_PATTERNS = re.compile(
    r"(month|period|date|vintage|snapshot|as_of|year|timestamp)",
    re.I,
)


def _looks_like_id_column(column: str) -> bool:
    """
    Return True when a column name looks like an identifier.

    Examples:
        customer_id
        account_id
        application_id
        loan_id
        transaction_id
        uuid
        identifier
    """
    return bool(
        _ID_PATTERNS.search(str(column))
    )


def _looks_like_time_column(column: str) -> bool:
    """
    Return True when a column name looks like a time/vintage column.
    """
    return bool(
        _TIME_PATTERNS.search(str(column))
    )


def _looks_like_shap_column(column: str) -> bool:
    """
    Return True when a column looks like a SHAP-value column.
    """
    return bool(
        re.search(
            r"^shap[_\-]",
            str(column),
            re.I,
        )
    )


# ---------------------------------------------------------------------------
# Feature sanitization
# ---------------------------------------------------------------------------

def _sanitize_feature_columns(
    dev_df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]],
    target_col: str,
    score_col: str,
    weight_col: Optional[str],
    shap_cols: Optional[Sequence[str]],
    excluded_feature_cols: Optional[Sequence[str]] = None,
) -> list[str]:
    """
    Safely determine the feature columns that are allowed to participate
    in root-cause analysis.

    This is deliberately defensive.

    Even if a caller accidentally passes customer_id or another metadata
    column in feature_cols, it will be removed here.

    Excluded categories:
        - target
        - model score
        - exposure / weight
        - explicitly excluded columns
        - ID-like columns
        - time/vintage columns
        - SHAP columns
    """

    # -----------------------------------------------------------------------
    # Determine candidate columns
    # -----------------------------------------------------------------------

    if feature_cols is None:
        candidates = list(
            dev_df.columns
        )
    else:
        candidates = list(
            feature_cols
        )

    # -----------------------------------------------------------------------
    # Explicit exclusions
    # -----------------------------------------------------------------------

    excluded = {
        target_col,
        score_col,
    }

    if weight_col:
        excluded.add(
            weight_col
        )

    if shap_cols:
        excluded.update(
            shap_cols
        )

    if excluded_feature_cols:
        excluded.update(
            excluded_feature_cols
        )

    # -----------------------------------------------------------------------
    # Build safe feature list
    # -----------------------------------------------------------------------

    safe_features: list[str] = []

    for col in candidates:

        # Column must exist in development data.
        if col not in dev_df.columns:
            continue

        # Explicit exclusions.
        if col in excluded:
            continue

        # Identifier protection.
        if _looks_like_id_column(col):
            continue

        # Time/vintage protection.
        if _looks_like_time_column(col):
            continue

        # SHAP protection.
        if _looks_like_shap_column(col):
            continue

        safe_features.append(
            col
        )

    return safe_features


# ---------------------------------------------------------------------------
# Feature-level drift / Root Cause Analysis
# ---------------------------------------------------------------------------

def _compute_feature_drift_within_segment(
    dev_sub: pd.DataFrame,
    mon_sub: pd.DataFrame,
    feature_cols: Sequence[str],
) -> dict:
    """
    Compute feature-level PSI for every allowed feature within the candidate
    segment.

    Development segment rows are compared against monitoring segment rows.

    For numeric features:
        - combined development + monitoring values are used to construct
          common decile bins
        - PSI is then calculated using those common bins

    For categorical features:
        - category-frequency PSI is calculated across the union of categories

    Returns
    -------
    dict
        {
            "top_drift_feature": str | None,
            "top_drift_psi": float,
            "feature_drift_details": [
                {"feature": str, "psi": float},
                ...
            ]
        }

    Important:
        This identifies a candidate root-cause DRIVER based on distributional
        drift. It does not establish causal relationships.

    Additional protection:
        Even if this function is called directly and receives an unsafe
        feature list, ID-like, time-like and SHAP columns are rejected again.
    """

    # -----------------------------------------------------------------------
    # Empty segment / feature checks
    # -----------------------------------------------------------------------

    if (
        not feature_cols
        or len(dev_sub) == 0
        or len(mon_sub) == 0
    ):
        return {
            "top_drift_feature": None,
            "top_drift_psi": 0.0,
            "feature_drift_details": [],
        }

    details: list[dict] = []

    # -----------------------------------------------------------------------
    # Evaluate every feature
    # -----------------------------------------------------------------------

    for col in feature_cols:

        # ---------------------------------------------------------------
        # Defensive ID protection
        # ---------------------------------------------------------------

        if _looks_like_id_column(col):
            continue

        # ---------------------------------------------------------------
        # Defensive time protection
        # ---------------------------------------------------------------

        if _looks_like_time_column(col):
            continue

        # ---------------------------------------------------------------
        # Defensive SHAP protection
        # ---------------------------------------------------------------

        if _looks_like_shap_column(col):
            continue

        # ---------------------------------------------------------------
        # Column availability
        # ---------------------------------------------------------------

        if (
            col not in dev_sub.columns
            or col not in mon_sub.columns
        ):
            continue

        try:

            dev_series = (
                dev_sub[col]
                .dropna()
            )

            mon_series = (
                mon_sub[col]
                .dropna()
            )

            # -----------------------------------------------------------
            # Avoid unstable estimates on extremely small samples.
            # -----------------------------------------------------------

            if (
                len(dev_series) < 5
                or len(mon_series) < 5
            ):
                continue

            # -----------------------------------------------------------
            # Numeric feature
            # -----------------------------------------------------------

            if pd.api.types.is_numeric_dtype(
                dev_series
            ):

                combined = pd.concat(
                    [
                        dev_series,
                        mon_series,
                    ],
                    ignore_index=True,
                )

                # If the feature has insufficient variation, there is no
                # meaningful distributional drift to calculate.
                if combined.nunique(
                    dropna=True
                ) < 2:

                    psi = 0.0

                else:

                    bins = build_decile_bins(
                        combined
                    )

                    psi = compute_psi_for_feature(
                        dev_series,
                        mon_series,
                        bins,
                    )

            # -----------------------------------------------------------
            # Categorical feature
            # -----------------------------------------------------------

            else:

                all_cats = (
                    set(
                        dev_series.unique()
                    )
                    |
                    set(
                        mon_series.unique()
                    )
                )

                dev_counts = (
                    dev_series.value_counts()
                )

                mon_counts = (
                    mon_series.value_counts()
                )

                # Small epsilon prevents log(0).
                epsilon = 1e-6

                dev_denominator = (
                    len(dev_series)
                    + epsilon
                )

                mon_denominator = (
                    len(mon_series)
                    + epsilon
                )

                psi = 0.0

                for category in all_cats:

                    dev_pct = (
                        dev_counts.get(
                            category,
                            0,
                        )
                        + epsilon
                    ) / dev_denominator

                    mon_pct = (
                        mon_counts.get(
                            category,
                            0,
                        )
                        + epsilon
                    ) / mon_denominator

                    psi += (
                        (mon_pct - dev_pct)
                        * np.log(
                            mon_pct / dev_pct
                        )
                    )

                psi = max(
                    0.0,
                    float(psi),
                )

            # -----------------------------------------------------------
            # Store result
            # -----------------------------------------------------------

            details.append(
                {
                    "feature": col,
                    "psi": float(psi),
                }
            )

        except Exception:
            # One problematic feature should never stop evaluation of the
            # entire candidate segment.
            continue

    # -----------------------------------------------------------------------
    # No valid features
    # -----------------------------------------------------------------------

    if not details:
        return {
            "top_drift_feature": None,
            "top_drift_psi": 0.0,
            "feature_drift_details": [],
        }

    # -----------------------------------------------------------------------
    # Sort by PSI
    # -----------------------------------------------------------------------

    details = sorted(
        details,
        key=lambda item: item["psi"],
        reverse=True,
    )

    top = details[0]

    return {
        "top_drift_feature": top["feature"],
        "top_drift_psi": float(
            top["psi"]
        ),
        "feature_drift_details": details,
    }


# ---------------------------------------------------------------------------
# Main candidate evaluation
# ---------------------------------------------------------------------------

def evaluate_segment(
    dev_df: pd.DataFrame,
    mon_df: pd.DataFrame,
    dev_mask: np.ndarray,
    mon_mask: np.ndarray,
    target_col: str,
    score_col: str,
    weight_col: Optional[str],
    total_dev_n: int,
    total_mon_n: int,
    total_dev_weight: float,
    total_mon_weight: float,
    shap_cols: Optional[list[str]] = None,
    feature_cols: Optional[list[str]] = None,
    excluded_feature_cols: Optional[list[str]] = None,
) -> Optional[dict]:
    """
    Compute all performance, drift, significance, business and root-cause
    metrics for one candidate segment.

    Parameters
    ----------
    dev_df:
        Development dataset.

    mon_df:
        Monitoring dataset.

    dev_mask:
        Boolean mask identifying the candidate segment in development data.

    mon_mask:
        Boolean mask identifying the candidate segment in monitoring data.

    target_col:
        Binary target/event column.

    score_col:
        Model prediction / PD / score column.

    weight_col:
        Exposure / EAD / weight column. Can be None.

    total_dev_n:
        Total number of rows in development dataset.

    total_mon_n:
        Total number of rows in monitoring dataset.

    total_dev_weight:
        Total development exposure.

    total_mon_weight:
        Total monitoring exposure.

    shap_cols:
        SHAP-value columns. These are explicitly excluded from ordinary
        feature-level root-cause analysis.

    feature_cols:
        Legitimate model/business feature columns to inspect for RCA.

    excluded_feature_cols:
        Additional columns that the caller explicitly wants excluded.

    Returns
    -------
    dict | None
        Flat dictionary containing all metrics.
        Returns None if either segment is empty.

    Notes
    -----
    Root-cause analysis here means:

        "Which legitimate feature has the highest distributional drift
         within this affected segment?"

    It does NOT mean causal proof.
    """

    # -----------------------------------------------------------------------
    # DEBUG 1: Confirm the actual Python file being executed
    # -----------------------------------------------------------------------
    #
    # This is intentionally included while diagnosing the customer_id issue.
    # It tells us exactly which candidate_evaluation.py is loaded.
    #
    # Remove/comment these DEBUG statements after the issue is resolved.
    # -----------------------------------------------------------------------

    print(
        "\n"
        "============================================================\n"
        "CANDIDATE EVALUATION DEBUG\n"
        "============================================================"
    )

    print(
        "candidate_evaluation.py loaded from:"
    )

    try:
        print(
            Path(__file__).resolve()
        )
    except Exception:
        print(
            __file__
        )

    # -----------------------------------------------------------------------
    # Create segment subsets
    # -----------------------------------------------------------------------

    dev_sub = dev_df.loc[
        dev_mask
    ]

    mon_sub = mon_df.loc[
        mon_mask
    ]

    n_dev = int(
        np.sum(dev_mask)
    )

    n_mon = int(
        np.sum(mon_mask)
    )

    if n_dev == 0 or n_mon == 0:
        return None

    if (
        total_dev_n <= 0
        or total_mon_n <= 0
    ):
        return None

    # -----------------------------------------------------------------------
    # Population share
    # -----------------------------------------------------------------------

    pct_dev = (
        n_dev / total_dev_n
    )

    pct_mon = (
        n_mon / total_mon_n
    )

    # -----------------------------------------------------------------------
    # Segment population-share PSI
    # -----------------------------------------------------------------------
    #
    # IMPORTANT:
    # This is a single-segment population-share PSI contribution.
    # It is not a full multi-bin distribution PSI.
    #
    # The output key remains "psi" for backward compatibility.
    # -----------------------------------------------------------------------

    psi = float(
        calculate_psi(
            pct_dev,
            pct_mon,
        )
    )

    # -----------------------------------------------------------------------
    # Bad-rate shift + two-proportion z-test
    # -----------------------------------------------------------------------

    x_dev = int(
        dev_sub[target_col].sum()
    )

    x_mon = int(
        mon_sub[target_col].sum()
    )

    br_dev = float(
        dev_sub[target_col].mean()
    )

    br_mon = float(
        mon_sub[target_col].mean()
    )

    # Convention:
    #
    #     delta_br = Monitoring - Development
    #
    # Positive = bad-rate increase / deterioration.

    delta_br = (
        br_mon - br_dev
    )

    br_pvalue = (
        two_proportion_ztest_pvalue(
            x_dev,
            n_dev,
            x_mon,
            n_mon,
        )
    )

    # -----------------------------------------------------------------------
    # AUC / Gini
    # -----------------------------------------------------------------------

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

    if (
        np.isnan(auc_dev)
        or np.isnan(auc_mon)
    ):
        delta_auc = np.nan
    else:
        # Negative = deterioration.
        delta_auc = (
            auc_mon - auc_dev
        )

    if (
        np.isnan(gini_dev)
        or np.isnan(gini_mon)
    ):
        delta_gini = np.nan
    else:
        # Negative = deterioration.
        delta_gini = (
            gini_mon - gini_dev
        )

    # -----------------------------------------------------------------------
    # Calibration drift
    # -----------------------------------------------------------------------

    dev_actual = float(
        dev_sub[target_col].sum()
    )

    dev_expected = float(
        dev_sub[score_col].sum()
    )

    mon_actual = float(
        mon_sub[target_col].sum()
    )

    mon_expected = float(
        mon_sub[score_col].sum()
    )

    calibration_drift = (
        calculate_calibration_drift(
            dev_actual,
            dev_expected,
            mon_actual,
            mon_expected,
        )
    )

    # -----------------------------------------------------------------------
    # KS
    # -----------------------------------------------------------------------

    ks_dev = calculate_ks(
        dev_sub[target_col],
        dev_sub[score_col],
    )

    ks_mon = calculate_ks(
        mon_sub[target_col],
        mon_sub[score_col],
    )

    if (
        np.isnan(ks_dev)
        or np.isnan(ks_mon)
    ):
        delta_ks = np.nan
    else:
        # Negative = deterioration.
        delta_ks = (
            ks_mon - ks_dev
        )

    # -----------------------------------------------------------------------
    # Model score-distribution shift
    # -----------------------------------------------------------------------

    score_shift_pvalue = (
        ks_score_shift_pvalue(
            dev_sub[score_col].values,
            mon_sub[score_col].values,
        )
    )

    # -----------------------------------------------------------------------
    # Exposure / weight metrics
    # -----------------------------------------------------------------------

    if (
        weight_col
        and weight_col in dev_sub.columns
    ):
        w_dev = float(
            dev_sub[weight_col].sum()
        )
    else:
        w_dev = 0.0

    if (
        weight_col
        and weight_col in mon_sub.columns
    ):
        w_mon = float(
            mon_sub[weight_col].sum()
        )
    else:
        w_mon = 0.0

    if total_dev_weight > 0:
        dev_weight_pct = (
            w_dev / total_dev_weight
        )
    else:
        dev_weight_pct = 0.0

    if total_mon_weight > 0:
        mon_weight_pct = (
            w_mon / total_mon_weight
        )
    else:
        mon_weight_pct = 0.0

    # Positive = monitoring exposure share increased.
    exposure_drift = (
        mon_weight_pct
        - dev_weight_pct
    )

    # -----------------------------------------------------------------------
    # Root Cause Analysis
    # -----------------------------------------------------------------------
    #
    # We do NOT reconstruct arbitrary numeric columns here.
    #
    # We first sanitize the feature list using:
    #
    #   - target exclusion
    #   - score exclusion
    #   - weight exclusion
    #   - ID exclusion
    #   - time exclusion
    #   - SHAP exclusion
    #
    # This prevents customer_id from being selected as a root cause.
    # -----------------------------------------------------------------------

    print(
        "\nInput feature_cols:"
    )

    print(
        feature_cols
    )

    if feature_cols is not None:
        print(
            "\ncustomer_id in input feature_cols?:",
            "customer_id" in feature_cols,
        )
    else:
        print(
            "\nInput feature_cols is None."
        )
        print(
            "Sanitizer will inspect dataframe columns and apply exclusions."
        )

    # -----------------------------------------------------------------------
    # Sanitize feature columns
    # -----------------------------------------------------------------------

    safe_feature_cols = (
        _sanitize_feature_columns(
            dev_df=dev_df,
            feature_cols=feature_cols,
            target_col=target_col,
            score_col=score_col,
            weight_col=weight_col,
            shap_cols=shap_cols,
            excluded_feature_cols=excluded_feature_cols,
        )
    )

    # -----------------------------------------------------------------------
    # DEBUG 2: Show sanitized feature list
    # -----------------------------------------------------------------------

    print(
        "\nSanitized RCA feature_cols:"
    )

    print(
        safe_feature_cols
    )

    print(
        "\ncustomer_id in sanitized feature_cols?:",
        "customer_id" in safe_feature_cols,
    )

    # -----------------------------------------------------------------------
    # HARD SAFETY CHECK
    # -----------------------------------------------------------------------
    #
    # customer_id must NEVER reach root-cause analysis.
    #
    # If this assertion fails, the program stops immediately instead of
    # silently generating an incorrect Root_Cause_Feature.
    # -----------------------------------------------------------------------

    assert (
        "customer_id"
        not in safe_feature_cols
    ), (
        "\n\n"
        "============================================================\n"
        "FATAL RCA SAFETY ERROR\n"
        "============================================================\n"
        "customer_id reached the sanitized RCA feature list.\n\n"
        f"Input feature_cols:\n{feature_cols}\n\n"
        f"Sanitized feature_cols:\n{safe_feature_cols}\n\n"
        "customer_id MUST NOT be used as a root-cause feature.\n"
        "Check schema_detection.py and the caller of evaluate_segment().\n"
        "============================================================\n"
    )

    # -----------------------------------------------------------------------
    # Additional safety check using the ID detector itself.
    # -----------------------------------------------------------------------

    unsafe_id_features = [
        col
        for col in safe_feature_cols
        if _looks_like_id_column(col)
    ]

    assert not unsafe_id_features, (
        "\n\n"
        "============================================================\n"
        "FATAL RCA SAFETY ERROR\n"
        "============================================================\n"
        "One or more ID-like columns reached RCA:\n"
        f"{unsafe_id_features}\n\n"
        f"All RCA features:\n{safe_feature_cols}\n"
        "============================================================\n"
    )

    # -----------------------------------------------------------------------
    # Compute feature-level drift
    # -----------------------------------------------------------------------

    rc = (
        _compute_feature_drift_within_segment(
            dev_sub=dev_sub,
            mon_sub=mon_sub,
            feature_cols=safe_feature_cols,
        )
    )

    top_drift_feature = (
        rc["top_drift_feature"]
    )

    top_drift_psi = float(
        rc["top_drift_psi"]
    )

    # -----------------------------------------------------------------------
    # FINAL RCA SAFETY CHECK
    # -----------------------------------------------------------------------
    #
    # This catches any unexpected behavior before the result is returned.
    # -----------------------------------------------------------------------

    assert (
        top_drift_feature != "customer_id"
    ), (
        "\n\n"
        "============================================================\n"
        "FATAL RCA OUTPUT ERROR\n"
        "============================================================\n"
        "Root Cause Feature was calculated as customer_id.\n\n"
        f"Top drift feature: {top_drift_feature}\n"
        f"Top drift PSI: {top_drift_psi}\n"
        f"Safe feature columns: {safe_feature_cols}\n"
        f"Feature drift details: {rc['feature_drift_details']}\n"
        "============================================================\n"
    )

    # -----------------------------------------------------------------------
    # DEBUG 3: Show final RCA result
    # -----------------------------------------------------------------------

    print(
        "\nFinal RCA result:"
    )

    print(
        "Root_Cause_Feature:",
        top_drift_feature,
    )

    print(
        "Root_Cause_PSI:",
        top_drift_psi,
    )

    print(
        "\nTop feature drift details:"
    )

    print(
        rc["feature_drift_details"][:10]
    )

    print(
        "============================================================\n"
    )

    # -----------------------------------------------------------------------
    # Root Cause Score
    # -----------------------------------------------------------------------
    #
    # This is a prioritization score, NOT a causal score.
    #
    # Formula:
    #
    #     Feature PSI
    #       ×
    #     (1 + Gini deterioration)
    #       ×
    #     Monitoring population share
    #
    # Gini deterioration is represented as a positive value when Gini falls.
    # -----------------------------------------------------------------------

    if np.isnan(delta_gini):
        gini_drop = 0.0
    else:
        gini_drop = max(
            0.0,
            -float(delta_gini),
        )

    root_cause_score = (
        top_drift_psi
        * (1.0 + gini_drop)
        * pct_mon
    )

    root_cause_score = round(
        float(root_cause_score),
        6,
    )

    # -----------------------------------------------------------------------
    # SHAP-based behavioural drift within this segment
    # -----------------------------------------------------------------------
    #
    # Distinct from the feature-PSI root cause above: this measures how
    # much the model's SHAP attributions shifted for rows inside this
    # specific segment (dev_sub vs mon_sub), not the whole population.
    # -----------------------------------------------------------------------

    shap_shift = detect_shap_shift(
        dev_sub, mon_sub, shap_cols or []
    )

    return {

        # -------------------------------------------------------------------
        # Segment support
        # -------------------------------------------------------------------

        "n_dev": n_dev,
        "n_mon": n_mon,
        "pct_dev": pct_dev,
        "pct_mon": pct_mon,

        # -------------------------------------------------------------------
        # Events
        # -------------------------------------------------------------------

        "x_dev": x_dev,
        "x_mon": x_mon,

        # -------------------------------------------------------------------
        # Population drift
        # -------------------------------------------------------------------

        "psi": psi,

        # -------------------------------------------------------------------
        # Bad rate
        # -------------------------------------------------------------------

        "br_dev": br_dev,
        "br_mon": br_mon,
        "delta_br": delta_br,
        "br_pvalue": br_pvalue,

        # -------------------------------------------------------------------
        # AUC
        # -------------------------------------------------------------------

        "auc_dev": auc_dev,
        "auc_mon": auc_mon,
        "delta_auc": delta_auc,

        # -------------------------------------------------------------------
        # Gini
        # -------------------------------------------------------------------

        "gini_dev": gini_dev,
        "gini_mon": gini_mon,
        "delta_gini": delta_gini,

        # -------------------------------------------------------------------
        # KS
        # -------------------------------------------------------------------

        "ks_dev": ks_dev,
        "ks_mon": ks_mon,
        "delta_ks": delta_ks,

        # -------------------------------------------------------------------
        # Score distribution shift
        # -------------------------------------------------------------------

        "score_shift_pvalue": score_shift_pvalue,

        # -------------------------------------------------------------------
        # Exposure
        # -------------------------------------------------------------------

        "weight_dev": w_dev,
        "weight_mon": w_mon,
        "dev_weight_pct": dev_weight_pct,
        "mon_weight_pct": mon_weight_pct,
        "exposure_drift": exposure_drift,

        # -------------------------------------------------------------------
        # Calibration
        # -------------------------------------------------------------------

        "calibration_drift": calibration_drift,

        # -------------------------------------------------------------------
        # Candidate root cause
        # -------------------------------------------------------------------

        "top_drift_feature": top_drift_feature,
        "top_drift_psi": top_drift_psi,
        "root_cause_score": root_cause_score,

        # -------------------------------------------------------------------
        # Full feature-level drift details
        # -------------------------------------------------------------------

        # Serialized to a JSON string here (not left as a list of dicts):
        # pandas .to_csv() would otherwise write the Python repr of the
        # list, and Streamlit's dataframe grid stringifies each dict via
        # JS as "[object Object]". A JSON string displays and exports
        # correctly in both paths.
        "feature_drift_details": json.dumps(
            rc["feature_drift_details"]
        ),

        # -------------------------------------------------------------------
        # SHAP-based behavioural drift, computed within this segment only
        # (dev_sub/mon_sub), distinct from the feature-PSI root cause above.
        # -------------------------------------------------------------------

        "top_shap_shift_feature": shap_shift["top_shift_feature"],
        "top_shap_shift_psi": shap_shift["top_shift_psi"],
    }