"""
techniques/gradient_boosting.py
===============================

Gradient Boosting (GBDT) Segmentation Technique.

Fits an ensemble of shallow trees to discriminate Development vs. Monitoring,
extracting leaf path rules across trees and ranking them by severity and
statistical significance.

The technique uses:

    1. Centralized schema detection
    2. Gradient Boosting classifier for Development vs Monitoring separation
    3. Tree leaf path extraction
    4. Centralized candidate evaluation
    5. Centralized candidate filtering
    6. Centralized severity scoring
    7. Jaccard-based candidate deduplication
    8. Feature-level PSI for Root Cause analysis

Root Cause analysis is performed by candidate_evaluation.py using the
schema-derived feature columns. SHAP is not required.
"""

from __future__ import annotations

import time
from typing import Optional, Union

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

from models.config import GBConfig
from models.result import TechniqueResult

from techniques.base import BaseSegmentationTechnique

from utils.logging_config import get_logger
from utils.schema_detection import (
    detect_schema,
    auto_detect_shap_columns,
)

from utils.preprocessing import (
    build_feature_matrix,
    extract_tree_rules,
    canonicalize_path_conditions,
    simplify_rule_list,
)

from metrics.drift_metrics import (
    detect_shap_shift,
)

from core.candidate_evaluation import (
    evaluate_segment,
)

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


logger = get_logger(__name__)


# ============================================================================
# GRADIENT BOOSTING TECHNIQUE
# ============================================================================

class GradientBoostingSegmentationTechnique(
    BaseSegmentationTechnique
):
    """
    Gradient Boosting Segmentation Technique.

    Uses GradientBoostingClassifier to discriminate Development observations
    from Monitoring observations.

    Regions identified by tree leaves are evaluated for:

        - Population PSI
        - Bad-rate shift
        - AUC
        - Gini
        - KS
        - Calibration drift
        - Score-distribution significance
        - Exposure drift
        - Feature-level Root Cause PSI

    Root Cause analysis is centralized in candidate_evaluation.py.
    """

    @property
    def name(self) -> str:
        return "Gradient Boosting"

    # ----------------------------------------------------------------------
    # RUN
    # ----------------------------------------------------------------------

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
            GBConfig
        ] = None,
    ) -> TechniqueResult:

        cfg = (
            config
            or GBConfig()
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

        # ------------------------------------------------------------------
        # IMPORTANT
        #
        # Use the centralized schema feature list for Root Cause analysis.
        #
        # Do not allow candidate_evaluation.py to infer feature columns
        # independently because that could accidentally include IDs or
        # metadata columns.
        # ------------------------------------------------------------------

        feature_cols = (
            schema["numeric_cols"]
            + schema["categorical_cols"]
        )

        logger.info(
            "GBDT schema — target=%r, score=%r, weight=%r",
            target_col,
            score_col,
            weight_col,
        )

        logger.info(
            "GBDT Root Cause features: %s",
            feature_cols,
        )

        # ==================================================================
        # BASIC DATA COUNTS
        # ==================================================================

        total_dev_n = len(
            dev_df
        )

        total_mon_n = len(
            mon_df
        )

        # ==================================================================
        # WEIGHT / EXPOSURE TOTALS
        # ==================================================================

        if (
            weight_col
            and weight_col
            in dev_df.columns
        ):

            total_dev_weight = float(
                dev_df[
                    weight_col
                ].sum()
            )

        else:

            total_dev_weight = 0.0

        if (
            weight_col
            and weight_col
            in mon_df.columns
        ):

            total_mon_weight = float(
                mon_df[
                    weight_col
                ].sum()
            )

        else:

            total_mon_weight = 0.0

        # ==================================================================
        # DEVELOPMENT EVENT RATE
        # ==================================================================

        dev_event_rate = float(
            dev_df[
                target_col
            ].mean()
        )

        # ==================================================================
        # MINIMUM SUPPORT / EVENT COUNTS
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

        logger.info(
            "GBDT filtering — min_abs_count=%s, min_support=%s",
            min_abs_count,
            min_support,
        )

        # ==================================================================
        # CREATE DEVELOPMENT / MONITORING LABEL
        # ==================================================================
        #
        # IMPORTANT:
        #
        # is_monitoring is a temporary modelling column.
        #
        # It is NOT included in feature_cols, so it cannot contaminate
        # Root Cause PSI analysis.
        # ==================================================================

        dev_model_df = dev_df.copy()
        mon_model_df = mon_df.copy()

        dev_model_df[
            "is_monitoring"
        ] = 0

        mon_model_df[
            "is_monitoring"
        ] = 1

        combined_df = pd.concat(
            [
                dev_model_df,
                mon_model_df,
            ],
            ignore_index=True,
        )

        # ==================================================================
        # SHAP DETECTION
        # ==================================================================
        #
        # SHAP columns are used only for diagnostics/logging.
        #
        # Root Cause analysis itself is now performed using feature-level
        # PSI in candidate_evaluation.py.
        # ==================================================================

        shap_cols = (
            auto_detect_shap_columns(
                combined_df
            )
        )

        if shap_cols:

            try:

                shap_shift = (
                    detect_shap_shift(
                        dev_df,
                        mon_df,
                        shap_cols,
                    )
                )

                logger.info(
                    "GBDT detected SHAP shift: %s",
                    shap_shift.get(
                        "top_shift_feature"
                    ),
                )

            except Exception as exc:

                logger.warning(
                    "GBDT SHAP shift detection failed: %s",
                    exc,
                )

        # ==================================================================
        # BUILD FEATURE MATRIX
        # ==================================================================
        #
        # Only schema-detected numeric/categorical features are supplied.
        #
        # Therefore:
        #
        #     target
        #     score
        #     weight
        #     IDs
        #     dates
        #     SHAP columns
        #
        # are not automatically used as GBDT segmentation predictors.
        # ==================================================================

        X_df, onehot_map = (
            build_feature_matrix(
                combined_df,
                schema["numeric_cols"],
                schema["categorical_cols"],
            )
        )

        y = (
            combined_df[
                "is_monitoring"
            ].values
        )

        # ==================================================================
        # CHECK FEATURE MATRIX
        # ==================================================================

        if X_df.empty or X_df.shape[1] == 0:

            logger.warning(
                "GBDT could not build a usable feature matrix."
            )

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
        # FIT GRADIENT BOOSTING MODEL
        # ==================================================================

        logger.info(
            "Fitting GBDT — n_estimators=%s, max_depth=%s, "
            "learning_rate=%s, min_samples_leaf=%s",
            cfg.n_estimators,
            cfg.max_depth,
            cfg.learning_rate,
            cfg.min_samples_leaf,
        )

        gbdt = GradientBoostingClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            min_samples_leaf=cfg.min_samples_leaf,
            random_state=cfg.random_state,
        )

        try:

            gbdt.fit(
                X_df,
                y,
            )

        except Exception as exc:

            logger.exception(
                "GBDT model fitting failed."
            )

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
                    "error": str(exc),
                },
                segments_df=pd.DataFrame(),
                execution_time=exec_time,
                schema=schema,
            )

        # ==================================================================
        # DRIFT PROBABILITY
        # ==================================================================
        #
        # Probability of an observation being Monitoring.
        #
        # This is useful as a diagnostic but is not directly used as the
        # final severity score because candidate_evaluation.py provides the
        # centralized evaluation metrics.
        # ==================================================================

        try:

            drift_probs = (
                gbdt.predict_proba(
                    X_df
                )[:, 1]
            )

        except Exception as exc:

            logger.warning(
                "Unable to calculate GBDT drift probabilities: %s",
                exc,
            )

            drift_probs = np.zeros(
                len(combined_df)
            )

        combined_df[
            "drift_prob"
        ] = drift_probs

        # ==================================================================
        # TREE SCANNING
        # ==================================================================

        candidates_evaluated: list[
            dict
        ] = []

        seen_rules: set[
            str
        ] = set()

        if (
            cfg.max_trees_scanned
            and cfg.max_trees_scanned > 0
        ):

            trees_to_scan = (
                gbdt.estimators_[
                    : cfg.max_trees_scanned
                ]
            )

        else:

            trees_to_scan = (
                gbdt.estimators_
            )

        logger.info(
            "GBDT scanning %s trees.",
            len(trees_to_scan),
        )

        # ==================================================================
        # LOOP THROUGH TREES
        # ==================================================================

        for tree_idx, tree_estimator in enumerate(
            trees_to_scan
        ):

            # GradientBoostingClassifier stores binary classifiers as
            # arrays of shape (1, 1) for each boosting stage.
            if isinstance(
                tree_estimator,
                np.ndarray,
            ):

                if (
                    tree_estimator.size
                    == 0
                ):
                    continue

                tree_obj = (
                    tree_estimator[
                        0
                    ]
                )

            else:

                tree_obj = (
                    tree_estimator
                )

            # --------------------------------------------------------------
            # Extract raw tree rules
            # --------------------------------------------------------------

            try:

                leaf_rules_raw = (
                    extract_tree_rules(
                        tree_obj,
                        list(
                            X_df.columns
                        ),
                    )
                )

            except Exception as exc:

                logger.warning(
                    "Could not extract rules from GBDT tree %s: %s",
                    tree_idx,
                    exc,
                )

                continue

            # --------------------------------------------------------------
            # Leaf assignments for the complete dataset
            # --------------------------------------------------------------

            try:

                leaf_assignments = (
                    tree_obj.apply(
                        X_df.values
                    )
                )

            except Exception as exc:

                logger.warning(
                    "Could not apply GBDT tree %s: %s",
                    tree_idx,
                    exc,
                )

                continue

            # --------------------------------------------------------------
            # Split leaf assignments back into Development / Monitoring
            # --------------------------------------------------------------

            dev_indicator = (
                combined_df[
                    "is_monitoring"
                ].values
                == 0
            )

            mon_indicator = (
                combined_df[
                    "is_monitoring"
                ].values
                == 1
            )

            dev_leaves = (
                leaf_assignments[
                    dev_indicator
                ]
            )

            mon_leaves = (
                leaf_assignments[
                    mon_indicator
                ]
            )

            # --------------------------------------------------------------
            # Evaluate each leaf
            # --------------------------------------------------------------

            for (
                leaf_id,
                raw_rules,
            ) in leaf_rules_raw.items():

                # ==========================================================
                # CANONICALIZE RULE
                # ==========================================================

                try:

                    canonical = (
                        canonicalize_path_conditions(
                            raw_rules,
                            onehot_map,
                        )
                    )

                except Exception as exc:

                    logger.warning(
                        "Could not canonicalize GBDT tree %s leaf %s: %s",
                        tree_idx,
                        leaf_id,
                        exc,
                    )

                    continue

                if canonical is None:
                    continue

                # ==========================================================
                # HUMAN-READABLE RULE
                # ==========================================================

                try:

                    rule_str = (
                        simplify_rule_list(
                            canonical,
                            root_label=(
                                "All Customers (Root)"
                            ),
                        )
                    )

                except Exception:

                    continue

                if not rule_str:
                    continue

                # ----------------------------------------------------------
                # Avoid evaluating the same rule repeatedly across trees.
                # ----------------------------------------------------------

                if (
                    rule_str
                    in seen_rules
                ):
                    continue

                seen_rules.add(
                    rule_str
                )

                # ==========================================================
                # SEGMENT MASKS
                # ==========================================================

                d_mask = (
                    dev_leaves
                    == leaf_id
                )

                m_mask = (
                    mon_leaves
                    == leaf_id
                )

                n_dev = int(
                    d_mask.sum()
                )

                n_mon = int(
                    m_mask.sum()
                )

                # ----------------------------------------------------------
                # Empty leaf protection
                # ----------------------------------------------------------

                if (
                    n_dev == 0
                    or n_mon == 0
                ):
                    continue

                pct_dev = (
                    n_dev
                    / total_dev_n
                    if total_dev_n > 0
                    else 0.0
                )

                pct_mon = (
                    n_mon
                    / total_mon_n
                    if total_mon_n > 0
                    else 0.0
                )

                # ==========================================================
                # SUPPORT FILTER
                # ==========================================================

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

                # ==========================================================
                # EVENT COUNT FILTER
                # ==========================================================

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

                # ==========================================================
                # CENTRALIZED CANDIDATE EVALUATION
                # ==========================================================
                #
                # IMPORTANT:
                #
                # feature_cols is explicitly supplied.
                #
                # This is the key integration with the new
                # candidate_evaluation.py.
                #
                # Root Cause PSI will therefore be calculated only for:
                #
                #     schema["numeric_cols"]
                #     +
                #     schema["categorical_cols"]
                #
                # rather than attempting automatic inference.
                # ==========================================================

                eval_res = evaluate_segment(
                    dev_df,
                    mon_df,
                    d_mask,
                    m_mask,
                    target_col,
                    score_col,
                    weight_col,
                    total_dev_n,
                    total_mon_n,
                    total_dev_weight,
                    total_mon_weight,
                    shap_cols=shap_cols,
                    feature_cols=feature_cols,
                )

                if eval_res is None:
                    continue

                # ==========================================================
                # CANDIDATE RECORD
                # ==========================================================

                cand_record = {
                    **eval_res,

                    "Tree_Idx": (
                        tree_idx
                    ),

                    "Leaf_ID": (
                        leaf_id
                    ),

                    "Segment_Definition": (
                        rule_str
                    ),

                    # Keep the monitoring mask for Jaccard deduplication.
                    "_mon_mask": (
                        m_mask
                    ),
                }

                candidates_evaluated.append(
                    cand_record
                )

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
        # CENTRALIZED SEVERITY SCORING
        # ==================================================================

        scored_records = (
            compute_severity_scores(
                candidates_evaluated,

                significance_alpha=(
                    cfg.significance_alpha
                ),
            )
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
        # JACCARD DEDUPLICATION
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
        # TOP N
        # ==================================================================

        top_records = (
            dedup_records[
                : cfg.top_n
            ]
        )

        for i, record in enumerate(
            top_records,
            1,
        ):

            record[
                "Rank"
            ] = i

        # ==================================================================
        # DATAFRAME
        # ==================================================================

        segments_df = pd.DataFrame(
            top_records
        )

        # ==================================================================
        # STANDARDIZED OUTPUT COLUMN NAMES
        # ==================================================================

        col_rename = {

            # --------------------------------------------------------------
            # Population
            # --------------------------------------------------------------

            "n_dev":
                "Dev_Count",

            "n_mon":
                "Mon_Count",

            "pct_dev":
                "Dev_Pct",

            "pct_mon":
                "Mon_Pct",

            # --------------------------------------------------------------
            # Population PSI
            # --------------------------------------------------------------

            "psi":
                "PSI",

            # --------------------------------------------------------------
            # Bad Rate
            # --------------------------------------------------------------

            "br_dev":
                "Dev_BR",

            "br_mon":
                "Mon_BR",

            "delta_br":
                "Delta_BR",

            "br_pvalue":
                "BR_PValue",

            # --------------------------------------------------------------
            # AUC
            # --------------------------------------------------------------

            "auc_dev":
                "Dev_AUC",

            "auc_mon":
                "Mon_AUC",

            "delta_auc":
                "Delta_AUC",

            # --------------------------------------------------------------
            # Gini
            # --------------------------------------------------------------

            "gini_dev":
                "Dev_Gini",

            "gini_mon":
                "Mon_Gini",

            "delta_gini":
                "Delta_Gini",

            # --------------------------------------------------------------
            # KS
            # --------------------------------------------------------------

            "ks_dev":
                "Dev_KS",

            "ks_mon":
                "Mon_KS",

            "delta_ks":
                "Delta_KS",

            "score_shift_pvalue":
                "Score_Shift_PValue",

            # --------------------------------------------------------------
            # Exposure
            # --------------------------------------------------------------

            "weight_dev":
                "Dev_EAD",

            "weight_mon":
                "Mon_EAD",

            "dev_weight_pct":
                "Dev_Exposure_Pct",

            "mon_weight_pct":
                "Mon_Exposure_Pct",

            "exposure_drift":
                "Exposure_Drift",

            # --------------------------------------------------------------
            # Calibration
            # --------------------------------------------------------------

            "calibration_drift":
                "Calibration_Drift",

            # --------------------------------------------------------------
            # Root Cause
            # --------------------------------------------------------------

            "top_drift_feature":
                "Root_Cause_Feature",

            "top_drift_psi":
                "Root_Cause_PSI",

            "root_cause_score":
                "Root_Cause_Score",

            # --------------------------------------------------------------
            # Backward-compatible SHAP columns
            # --------------------------------------------------------------

            "top_shap_shift_feature":
                "Top_SHAP_Feature",

            "top_shap_shift_psi":
                "Top_SHAP_PSI",
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

            "total_dev_n":
                total_dev_n,

            "total_mon_n":
                total_mon_n,

            "candidates_evaluated":
                len(
                    candidates_evaluated
                ),

            "candidates_returned":
                len(
                    segments_df
                ),
        }

        # ==================================================================
        # RETURN TECHNIQUE RESULT
        # ==================================================================

        return TechniqueResult(
            technique_name=self.name,

            overall=overall,

            segments_df=segments_df,

            execution_time=exec_time,

            schema=schema,

            portfolio_view_df=(
                portfolio_df
            ),
        )


# ============================================================================
# FUNCTIONAL WRAPPER
# ============================================================================

def run_gradient_boosting_segmentation(
    dev_path: Union[
        str,
        pd.DataFrame,
    ],
    mon_path: Union[
        str,
        pd.DataFrame,
    ],
    cfg: Optional[
        GBConfig
    ] = None,
) -> dict:
    """
    Functional wrapper for Gradient Boosting segmentation execution.

    Uses the common parameter optimization framework.
    """

    from core.parameter_optimization import (
        optimize_parameters,
    )

    cfg = (
        cfg
        or GBConfig()
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
            GradientBoostingSegmentationTechnique()
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