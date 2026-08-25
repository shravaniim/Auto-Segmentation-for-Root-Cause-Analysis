"""
techniques/autoslicer.py
========================

AutoSlicer Sub-group Discovery Technique.

Identifies multi-variable sub-populations experiencing performance
degradation or drift using adaptive beam search and hypothesis testing.

The technique generates:
    - Single-feature candidate segments
    - Multi-feature conjunctions
    - Performance/drift metrics
    - Business impact metrics
    - Candidate root-cause features through feature-level PSI

Important:
    Schema detection is centralized in utils.schema_detection.

    Only schema["feature_cols"] are used for segmentation and candidate
    root-cause analysis.

    IDs, targets, model scores, exposure, time columns and SHAP columns
    are excluded from normal feature analysis.
"""

from __future__ import annotations

import time
from typing import Optional, Union

import numpy as np
import pandas as pd

from models.config import SlicerConfig
from models.result import TechniqueResult
from techniques.base import BaseSegmentationTechnique
from utils.logging_config import get_logger
from utils.preprocessing import format_number

from utils.schema_detection import (
    detect_schema,
    auto_detect_shap_columns,
)

from metrics.drift_metrics import detect_shap_shift

from core.candidate_evaluation import evaluate_segment

from core.candidate_filtering import (
    derive_min_abs_count,
    derive_min_support,
    passes_support_filter,
    passes_event_count_filter,
)

from core.candidate_ranking import (
    compute_severity_scores,
    rank_records_significance_first,
    build_portfolio_view,
)

from core.candidate_deduplication import (
    deduplicate_by_jaccard,
)

from utils.exports import (
    export_results_csv,
    export_portfolio_view,
)


logger = get_logger(__name__)


# ============================================================================
# FEATURE BINNING
# ============================================================================

def build_feature_bins(
    series: pd.Series,
    n_bins: int = 4,
) -> list[tuple[float, float, str]]:
    """
    Discretise a numeric feature into quantile bins.

    Explicit -inf / +inf endpoints are used so that every valid observation
    belongs to exactly one generated interval.

    Parameters
    ----------
    series:
        Numeric pandas Series.

    n_bins:
        Requested number of quantile bins.

    Returns
    -------
    list of tuples:
        Each tuple contains:

            (lower_bound, upper_bound, human_readable_label)
    """

    clean = (
        series
        .dropna()
        .to_numpy(dtype=float)
    )

    if clean.size == 0:
        return []

    quantiles = np.quantile(
        clean,
        np.linspace(
            0.0,
            1.0,
            n_bins + 1,
        ),
    )

    unique_q = np.unique(
        quantiles
    )

    # If all observations have the same value, no meaningful bin can
    # be generated.
    if len(unique_q) < 2:
        return []

    edges = unique_q.tolist()

    edges[0] = -np.inf
    edges[-1] = np.inf

    bins = []

    for i in range(
        len(edges) - 1
    ):
        low = edges[i]
        high = edges[i + 1]

        if low == -np.inf:

            label = (
                f"{series.name} < {format_number(high)}"
            )

        elif high == np.inf:

            label = (
                f"{series.name} >= {format_number(low)}"
            )

        else:

            label = (
                f"{format_number(low)} <= "
                f"{series.name} < "
                f"{format_number(high)}"
            )

        bins.append(
            (
                low,
                high,
                label,
            )
        )

    return bins


# ============================================================================
# SINGLE FEATURE PREDICATES
# ============================================================================

def generate_single_feature_predicates(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_bins: int = 4,
) -> dict[str, list[dict]]:
    """
    Generate candidate single-variable predicates for numeric and
    categorical features.

    Numeric features:
        Quantile-based interval predicates.

    Categorical features:
        One predicate per category, provided the feature has between
        2 and 20 distinct categories.

    Parameters
    ----------
    df:
        Dataset used to generate candidate predicates.

    feature_cols:
        Authoritative list of legitimate segmentation features.

    n_bins:
        Number of bins for numeric variables.

    Returns
    -------
    dict
        Mapping:

            feature -> list of predicate dictionaries
    """

    predicates_by_feature: dict[
        str,
        list[dict],
    ] = {}

    for col in feature_cols:

        # Defensive check in case schema contains a column that isn't
        # available in the dataframe.
        if col not in df.columns:
            continue

        series = df[col]

        preds = []

        # ------------------------------------------------------------------
        # Numeric feature
        # ------------------------------------------------------------------

        if pd.api.types.is_numeric_dtype(
            series
        ):

            for (
                low,
                high,
                label,
            ) in build_feature_bins(
                series,
                n_bins,
            ):

                preds.append(
                    {
                        "feature": col,
                        "type": "numeric",
                        "lower": low,
                        "upper": high,
                        "label": label,
                    }
                )

        # ------------------------------------------------------------------
        # Categorical feature
        # ------------------------------------------------------------------

        else:

            categories = (
                series
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            # Avoid exploding the search space for high-cardinality
            # categorical variables.
            if (
                1 < len(categories) <= 20
            ):

                for cat in categories:

                    preds.append(
                        {
                            "feature": col,
                            "type": "categorical",
                            "value": cat,
                            "label": (
                                f"{col} = {cat}"
                            ),
                        }
                    )

        if preds:
            predicates_by_feature[
                col
            ] = preds

    return predicates_by_feature


# ============================================================================
# PREDICATE MASK EVALUATION
# ============================================================================

def eval_predicate_mask(
    df: pd.DataFrame,
    pred: dict,
) -> np.ndarray:
    """
    Evaluate a boolean mask for a single predicate.
    """

    col = pred["feature"]

    if col not in df.columns:
        return np.zeros(
            len(df),
            dtype=bool,
        )

    # ------------------------------------------------------------------
    # Numeric predicate
    # ------------------------------------------------------------------

    if pred["type"] == "numeric":

        s = df[col]

        mask = (
            s.notna()
            .to_numpy()
            .copy()
        )

        if pred["lower"] != -np.inf:

            mask &= (
                s >= pred["lower"]
            ).to_numpy()

        if pred["upper"] != np.inf:

            mask &= (
                s < pred["upper"]
            ).to_numpy()

        return mask

    # ------------------------------------------------------------------
    # Categorical predicate
    # ------------------------------------------------------------------

    return (
        df[col]
        .astype(str)
        == pred["value"]
    ).to_numpy()


# ============================================================================
# COMBINATION MASK
# ============================================================================

def eval_combo_mask(
    df: pd.DataFrame,
    combo: tuple[dict, ...],
) -> np.ndarray:
    """
    Evaluate a combined boolean mask for a conjunction of predicates.

    Example:

        age < 47
        AND
        income < 75000
        AND
        region = West
    """

    mask = np.ones(
        len(df),
        dtype=bool,
    )

    for pred in combo:

        mask &= eval_predicate_mask(
            df,
            pred,
        )

    return mask


# ============================================================================
# HEURISTIC SCORE
# ============================================================================

def heuristic_score(
    m: dict,
) -> float:
    """
    Fast depth-comparable screening metric used to drive beam-search
    expansion.

    The heuristic combines:

        PSI
        Monitoring exposure share
        Gini deterioration
        KS deterioration
        Bad-rate shift

    This is used only for beam-search prioritisation.
    Final ranking is handled separately by candidate_ranking.
    """

    # Negative Gini delta means deterioration.
    gini_drop = (
        max(
            0.0,
            -m["delta_gini"],
        )
        if not np.isnan(
            m["delta_gini"]
        )
        else 0.0
    )

    # Negative KS delta means deterioration.
    ks_drop = (
        max(
            0.0,
            -m["delta_ks"],
        )
        if not np.isnan(
            m["delta_ks"]
        )
        else 0.0
    )

    # Absolute bad-rate shift is used for screening.
    br_shift = (
        abs(m["delta_br"])
        if not np.isnan(
            m["delta_br"]
        )
        else 0.0
    )

    return (
        m["psi"] * 0.30
        + m["mon_weight_pct"] * 0.25
        + gini_drop * 0.20
        + ks_drop * 0.15
        + br_shift * 0.10
    )


# ============================================================================
# AUTOSLICER TECHNIQUE
# ============================================================================

class AutoSlicerTechnique(
    BaseSegmentationTechnique
):
    """
    AutoSlicer Sub-group Discovery Technique.

    Uses adaptive beam search to discover candidate sub-populations with
    model performance degradation, population drift and business impact.
    """

    @property
    def name(self) -> str:
        return "AutoSlicer"

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
            SlicerConfig
        ] = None,
    ) -> TechniqueResult:

        cfg = (
            config
            or SlicerConfig()
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
        # Use the authoritative feature_cols returned by schema_detection.
        #
        # This list excludes:
        #   - customer_id
        #   - account_id
        #   - target
        #   - model score
        #   - exposure
        #   - time/vintage
        #   - SHAP columns
        #
        # Do NOT reconstruct this as:
        #
        #   numeric_cols + categorical_cols
        #
        # because feature_cols is now the single source of truth.
        feature_cols = schema[
            "feature_cols"
        ]

        # ==================================================================
        # SHAP DETECTION
        # ==================================================================
        #
        # SHAP analysis remains separate from normal segmentation.
        #
        # SHAP columns are detected only for the existing SHAP-shift
        # diagnostic.
        #
        # They are NOT passed as segmentation features.
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
                "AutoSlicer detected SHAP shift: %s",
                shap_shift[
                    "top_shift_feature"
                ],
            )

        # ==================================================================
        # DATASET TOTALS
        # ==================================================================

        total_dev_n = len(
            dev_df
        )

        total_mon_n = len(
            mon_df
        )

        total_dev_weight = (
            float(
                dev_df[
                    weight_col
                ].sum()
            )
            if (
                weight_col
                and weight_col
                in dev_df.columns
            )
            else 0.0
        )

        total_mon_weight = (
            float(
                mon_df[
                    weight_col
                ].sum()
            )
            if (
                weight_col
                and weight_col
                in mon_df.columns
            )
            else 0.0
        )

        dev_event_rate = float(
            dev_df[
                target_col
            ].mean()
        )

        # ==================================================================
        # SUPPORT / EVENT THRESHOLDS
        # ==================================================================

        min_abs_count = (
            derive_min_abs_count(
                dev_event_rate,
                cfg.min_events_per_slice,
                cfg.min_abs_count,
            )
        )

        min_support = (
            derive_min_support(
                min_abs_count,
                total_dev_n,
                total_mon_n,
                cfg.min_support,
            )
        )

        # ==================================================================
        # GENERATE PREDICATES
        # ==================================================================

        preds_by_feature = (
            generate_single_feature_predicates(
                dev_df,
                feature_cols,
                n_bins=cfg.numeric_bins,
            )
        )

        avail_features = list(
            preds_by_feature.keys()
        )

        # ==================================================================
        # CANDIDATE STORAGE
        # ==================================================================

        candidates_evaluated: list[
            dict
        ] = []

        evaluated_combos: set[
            frozenset
        ] = set()

        current_beam: list[
            tuple[
                tuple[dict, ...],
                dict,
                float,
            ]
        ] = []

        # ==================================================================
        # DEPTH 1
        # ==================================================================

        for (
            feat,
            pred_list,
        ) in preds_by_feature.items():

            for pred in pred_list:

                combo = (
                    pred,
                )

                combo_key = frozenset(
                    (
                        p["feature"],
                        p["label"],
                    )
                    for p in combo
                )

                if (
                    combo_key
                    in evaluated_combos
                ):
                    continue

                evaluated_combos.add(
                    combo_key
                )

                # ----------------------------------------------------------
                # Segment masks
                # ----------------------------------------------------------

                d_mask = eval_combo_mask(
                    dev_df,
                    combo,
                )

                m_mask = eval_combo_mask(
                    mon_df,
                    combo,
                )

                n_dev = int(
                    d_mask.sum()
                )

                n_mon = int(
                    m_mask.sum()
                )

                pct_dev = (
                    n_dev
                    / total_dev_n
                )

                pct_mon = (
                    n_mon
                    / total_mon_n
                )

                # ----------------------------------------------------------
                # Support filter
                # ----------------------------------------------------------

                if not passes_support_filter(
                    n_dev,
                    n_mon,
                    pct_dev,
                    pct_mon,
                    min_abs_count,
                    min_support,
                    cfg.max_segment_pct,
                ):
                    continue

                # ----------------------------------------------------------
                # Event filter
                # ----------------------------------------------------------

                x_dev = int(
                    dev_df.loc[
                        d_mask,
                        target_col,
                    ].sum()
                )

                x_mon = int(
                    mon_df.loc[
                        m_mask,
                        target_col,
                    ].sum()
                )

                if not passes_event_count_filter(
                    x_dev,
                    x_mon,
                    cfg.min_events_per_slice,
                ):
                    continue

                # ----------------------------------------------------------
                # CENTRAL CANDIDATE EVALUATION
                # ----------------------------------------------------------
                #
                # IMPORTANT:
                # Named arguments are used deliberately.
                #
                # feature_cols contains only legitimate model/business
                # features.
                #
                # shap_cols are passed separately.
                # ----------------------------------------------------------

                eval_res = evaluate_segment(
                    dev_df=dev_df,
                    mon_df=mon_df,
                    dev_mask=d_mask,
                    mon_mask=m_mask,
                    target_col=target_col,
                    score_col=score_col,
                    weight_col=weight_col,
                    total_dev_n=total_dev_n,
                    total_mon_n=total_mon_n,
                    total_dev_weight=total_dev_weight,
                    total_mon_weight=total_mon_weight,
                    shap_cols=shap_cols,
                    feature_cols=feature_cols,
                )

                if eval_res is None:
                    continue

                # ----------------------------------------------------------
                # Heuristic screening score
                # ----------------------------------------------------------

                h_score = heuristic_score(
                    eval_res
                )

                rule_str = pred[
                    "label"
                ]

                cand_record = {
                    **eval_res,
                    "Segment_Definition": rule_str,
                    "heuristic_score": h_score,
                    "_mon_mask": m_mask,
                    "combo": combo,
                }

                candidates_evaluated.append(
                    cand_record
                )

                current_beam.append(
                    (
                        combo,
                        cand_record,
                        h_score,
                    )
                )

        # ==================================================================
        # INITIAL BEAM SORT
        # ==================================================================

        current_beam.sort(
            key=lambda x: x[2],
            reverse=True,
        )

        best_score_so_far = (
            current_beam[0][2]
            if current_beam
            else 0.0
        )

        # ==================================================================
        # MULTI-DEPTH BEAM SEARCH
        # ==================================================================

        for depth in range(
            2,
            cfg.max_combo_depth + 1,
        ):

            if not current_beam:
                break

            # --------------------------------------------------------------
            # Retain high-quality parents
            # --------------------------------------------------------------

            top_h_depth = (
                current_beam[0][2]
            )

            cutoff_h = (
                top_h_depth
                * cfg.beam_retain_ratio
            )

            retained_beam = [
                b
                for b in current_beam
                if b[2] >= cutoff_h
            ]

            if (
                len(retained_beam)
                < cfg.min_beam_width
            ):
                retained_beam = (
                    current_beam[
                        : cfg.min_beam_width
                    ]
                )

            if (
                len(retained_beam)
                > cfg.beam_width
            ):
                retained_beam = (
                    retained_beam[
                        : cfg.beam_width
                    ]
                )

            next_beam: list[
                tuple[
                    tuple[dict, ...],
                    dict,
                    float,
                ]
            ] = []

            # --------------------------------------------------------------
            # Expand retained combinations
            # --------------------------------------------------------------

            for (
                combo,
                parent_record,
                parent_h,
            ) in retained_beam:

                used_feats = {
                    p["feature"]
                    for p in combo
                }

                candidate_next_feats = [
                    f
                    for f in avail_features
                    if f not in used_feats
                ]

                for nf in candidate_next_feats:

                    for pred in preds_by_feature[
                        nf
                    ]:

                        new_combo = tuple(
                            sorted(
                                combo + (
                                    pred,
                                ),
                                key=lambda p:
                                p["feature"],
                            )
                        )

                        combo_key = frozenset(
                            (
                                p["feature"],
                                p["label"],
                            )
                            for p in new_combo
                        )

                        if (
                            combo_key
                            in evaluated_combos
                        ):
                            continue

                        evaluated_combos.add(
                            combo_key
                        )

                        # --------------------------------------------------
                        # Segment masks
                        # --------------------------------------------------

                        d_mask = eval_combo_mask(
                            dev_df,
                            new_combo,
                        )

                        m_mask = eval_combo_mask(
                            mon_df,
                            new_combo,
                        )

                        n_dev = int(
                            d_mask.sum()
                        )

                        n_mon = int(
                            m_mask.sum()
                        )

                        pct_dev = (
                            n_dev
                            / total_dev_n
                        )

                        pct_mon = (
                            n_mon
                            / total_mon_n
                        )

                        # --------------------------------------------------
                        # Support filter
                        # --------------------------------------------------

                        if not passes_support_filter(
                            n_dev,
                            n_mon,
                            pct_dev,
                            pct_mon,
                            min_abs_count,
                            min_support,
                            cfg.max_segment_pct,
                        ):
                            continue

                        # --------------------------------------------------
                        # Event filter
                        # --------------------------------------------------

                        x_dev = int(
                            dev_df.loc[
                                d_mask,
                                target_col,
                            ].sum()
                        )

                        x_mon = int(
                            mon_df.loc[
                                m_mask,
                                target_col,
                            ].sum()
                        )

                        if not passes_event_count_filter(
                            x_dev,
                            x_mon,
                            cfg.min_events_per_slice,
                        ):
                            continue

                        # --------------------------------------------------
                        # CENTRAL CANDIDATE EVALUATION
                        # --------------------------------------------------

                        eval_res = evaluate_segment(
                            dev_df=dev_df,
                            mon_df=mon_df,
                            dev_mask=d_mask,
                            mon_mask=m_mask,
                            target_col=target_col,
                            score_col=score_col,
                            weight_col=weight_col,
                            total_dev_n=total_dev_n,
                            total_mon_n=total_mon_n,
                            total_dev_weight=total_dev_weight,
                            total_mon_weight=total_mon_weight,
                            shap_cols=shap_cols,
                            feature_cols=feature_cols,
                        )

                        if eval_res is None:
                            continue

                        # --------------------------------------------------
                        # Heuristic score
                        # --------------------------------------------------

                        h_score = heuristic_score(
                            eval_res
                        )

                        rule_str = (
                            " AND ".join(
                                p["label"]
                                for p in new_combo
                            )
                        )

                        cand_record = {
                            **eval_res,
                            "Segment_Definition": rule_str,
                            "heuristic_score": h_score,
                            "_mon_mask": m_mask,
                            "combo": new_combo,
                        }

                        candidates_evaluated.append(
                            cand_record
                        )

                        next_beam.append(
                            (
                                new_combo,
                                cand_record,
                                h_score,
                            )
                        )

            # --------------------------------------------------------------
            # Stop if no further candidates
            # --------------------------------------------------------------

            if not next_beam:
                break

            # --------------------------------------------------------------
            # Rank newly generated beam
            # --------------------------------------------------------------

            next_beam.sort(
                key=lambda x: x[2],
                reverse=True,
            )

            best_score_this_depth = (
                next_beam[0][2]
            )

            # --------------------------------------------------------------
            # Early stopping
            # --------------------------------------------------------------

            relative_improvement = (
                (
                    best_score_this_depth
                    - best_score_so_far
                )
                / max(
                    abs(best_score_so_far),
                    1e-6,
                )
            )

            if (
                relative_improvement
                < cfg.depth_improvement_tolerance
            ):
                break

            best_score_so_far = max(
                best_score_so_far,
                best_score_this_depth,
            )

            current_beam = next_beam

        # ==================================================================
        # NO CANDIDATES
        # ==================================================================

        if not candidates_evaluated:

            exec_time = (
                time.perf_counter()
                - start_time
            )

            return TechniqueResult(
                technique_name=self.name,
                overall={
                    "total_dev_n": total_dev_n,
                    "total_mon_n": total_mon_n,
                    "candidates_evaluated": 0,
                    "candidates_returned": 0,
                },
                segments_df=pd.DataFrame(),
                execution_time=exec_time,
                schema=schema,
            )

        # ==================================================================
        # FINAL SEVERITY SCORING
        # ==================================================================

        scored_records = compute_severity_scores(
            candidates_evaluated,
            w_psi=cfg.w_psi,
            w_business_impact=cfg.w_business_impact,
            w_gini_drop=cfg.w_gini_drop,
            w_ks_drop=cfg.w_ks_drop,
            w_br_shift=cfg.w_br_shift,
            significance_alpha=cfg.significance_alpha,
        )

        # ==================================================================
        # SIGNIFICANCE-FIRST RANKING
        # ==================================================================

        ranked_records = (
            rank_records_significance_first(
                scored_records
            )
        )

        # ==================================================================
        # REMOVE HIGHLY OVERLAPPING SEGMENTS
        # ==================================================================

        dedup_records = (
            deduplicate_by_jaccard(
                ranked_records,
                mask_key="_mon_mask",
                overlap_threshold=(
                    cfg.overlap_jaccard_threshold
                ),
                pool_size=30,
            )
        )

        # ==================================================================
        # TOP RESULTS
        # ==================================================================

        top_records = dedup_records[
            : cfg.top_n
        ]

        for i, record in enumerate(
            top_records,
            1,
        ):

            record["Rank"] = i

            # Internal objects should not go into final DataFrame output.
            record.pop(
                "combo",
                None,
            )

            record.pop(
                "heuristic_score",
                None,
            )

        segments_df = pd.DataFrame(
            top_records
        )

        # ==================================================================
        # STANDARDIZE OUTPUT COLUMN NAMES
        # ==================================================================

        col_rename = {

            # Segment support
            "n_dev": "Dev_Count",
            "n_mon": "Mon_Count",
            "pct_dev": "Dev_Pct",
            "pct_mon": "Mon_Pct",

            # Population drift
            "psi": "PSI",

            # Bad rate
            "br_dev": "Dev_BR",
            "br_mon": "Mon_BR",
            "delta_br": "Delta_BR",

            # AUC
            "auc_dev": "Dev_AUC",
            "auc_mon": "Mon_AUC",
            "delta_auc": "Delta_AUC",

            # Gini
            "gini_dev": "Dev_Gini",
            "gini_mon": "Mon_Gini",
            "delta_gini": "Delta_Gini",

            # KS
            "ks_dev": "Dev_KS",
            "ks_mon": "Mon_KS",
            "delta_ks": "Delta_KS",

            # Exposure
            "weight_dev": "Dev_EAD",
            "weight_mon": "Mon_EAD",
            "dev_weight_pct": "Dev_Exposure_Pct",
            "mon_weight_pct": "Mon_Exposure_Pct",
            "exposure_drift": "Exposure_Drift",

            # Calibration
            "calibration_drift": (
                "Calibration_Drift"
            ),

            # Root cause
            "top_drift_feature": (
                "Root_Cause_Feature"
            ),
            "top_drift_psi": (
                "Root_Cause_PSI"
            ),
            "root_cause_score": (
                "Root_Cause_Score"
            ),

            # Backward compatibility
            "top_shap_shift_feature": (
                "Top_SHAP_Feature"
            ),
            "top_shap_shift_psi": (
                "Top_SHAP_PSI"
            ),
        }

        segments_df = (
            segments_df.rename(
                columns=col_rename
            )
        )

        # ==================================================================
        # PORTFOLIO VIEW
        # ==================================================================

        portfolio_df = (
            build_portfolio_view(
                segments_df,
                cfg.top_n,
            )
        )

        # ==================================================================
        # EXECUTION SUMMARY
        # ==================================================================

        exec_time = (
            time.perf_counter()
            - start_time
        )

        overall = {
            "total_dev_n": total_dev_n,
            "total_mon_n": total_mon_n,
            "candidates_evaluated": (
                len(
                    candidates_evaluated
                )
            ),
            "candidates_returned": (
                len(
                    segments_df
                )
            ),
        }

        # ==================================================================
        # FINAL RESULT
        # ==================================================================

        return TechniqueResult(
            technique_name=self.name,
            overall=overall,
            segments_df=segments_df,
            execution_time=exec_time,
            schema=schema,
            portfolio_view_df=portfolio_df,
        )


# ============================================================================
# FUNCTIONAL WRAPPER
# ============================================================================

def run_autoslicer_segmentation(
    dev_path: Union[
        str,
        pd.DataFrame,
    ],
    mon_path: Union[
        str,
        pd.DataFrame,
    ],
    cfg: Optional[
        SlicerConfig
    ] = None,
) -> dict:
    """
    Functional wrapper for AutoSlicer segmentation execution.

    Parameter optimization is handled externally by
    core.parameter_optimization.
    """

    from core.parameter_optimization import (
        optimize_parameters,
    )

    cfg = (
        cfg
        or SlicerConfig()
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
            AutoSlicerTechnique()
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