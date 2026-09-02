"""
techniques/drift_tree.py
========================

Drift Localization Tree (DLT) Segmentation Technique.

Uses a supervised decision tree trained to discriminate Development vs.
Monitoring samples to discover regions of high population drift and
performance degradation.

The tree is trained on legitimate business/model features only.

Root-cause analysis is handled centrally by candidate_evaluation.py using
feature-level PSI within each discovered segment.

Important:
    - customer_id / account_id / application_id are excluded by schema
      detection.
    - target / score / exposure / time columns are excluded.
    - SHAP columns are excluded from normal feature analysis.
    - "is_monitoring" and "leaf" are internal DLT columns and are not used
      for root-cause analysis.
"""

from __future__ import annotations

import time
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from models.config import DLTConfig
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


logger = get_logger(__name__)


class DriftLocalizationTreeTechnique(
    BaseSegmentationTechnique
):
    """
    Drift Localization Tree (DLT) technique.

    A supervised decision tree is trained to distinguish Development
    observations from Monitoring observations.

    Tree leaves are interpreted as candidate drift segments.
    """

    @property
    def name(self) -> str:
        return "Drift Localization Tree"

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
            DLTConfig
        ] = None,
    ) -> TechniqueResult:

        cfg = (
            config
            or DLTConfig()
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
        # feature_cols is the authoritative list of legitimate business /
        # model features.
        #
        # It excludes:
        #   - customer_id
        #   - account_id
        #   - application_id
        #   - target
        #   - model score
        #   - exposure
        #   - time/vintage columns
        #   - SHAP columns
        #
        # This list is used for candidate root-cause analysis.
        feature_cols = schema[
            "feature_cols"
        ]

        # Numeric/categorical lists are still required separately by
        # build_feature_matrix().
        numeric_cols = schema[
            "numeric_cols"
        ]

        categorical_cols = schema[
            "categorical_cols"
        ]

        # ==================================================================
        # SHAP DETECTION
        # ==================================================================
        #
        # SHAP analysis is kept separate from the normal segmentation
        # feature list.
        #
        # SHAP columns may be used by detect_shap_shift(), but they are NOT
        # supplied as normal RCA features.
        # ==================================================================

        combined_for_shap = pd.concat(
            [
                dev_df,
                mon_df,
            ],
            ignore_index=True,
        )

        shap_cols = (
            auto_detect_shap_columns(
                combined_for_shap
            )
        )

        if shap_cols:

            shap_shift = detect_shap_shift(
                dev_df,
                mon_df,
                shap_cols,
            )

            logger.info(
                "DLT detected SHAP shift: %s",
                shap_shift[
                    "top_shift_feature"
                ],
            )

        # ==================================================================
        # CREATE MONITORING LABEL
        # ==================================================================
        #
        # This column is internal to DLT.
        #
        # 0 = Development
        # 1 = Monitoring
        #
        # It is intentionally added AFTER schema detection so that it
        # cannot accidentally become a feature.
        # ==================================================================

        dev_df["is_monitoring"] = 0
        mon_df["is_monitoring"] = 1

        combined_df = pd.concat(
            [
                dev_df,
                mon_df,
            ],
            ignore_index=True,
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
        # BUILD TREE FEATURE MATRIX
        # ==================================================================
        #
        # build_feature_matrix() needs numeric and categorical feature
        # lists separately.
        #
        # These lists come from schema detection and therefore do not
        # contain customer_id or other excluded columns.
        # ==================================================================

        X_df, onehot_map = build_feature_matrix(
            combined_df,
            numeric_cols,
            categorical_cols,
        )

        y = combined_df[
            "is_monitoring"
        ].values

        # ==================================================================
        # TRAIN DECISION TREE
        # ==================================================================

        clf = DecisionTreeClassifier(
            max_depth=cfg.max_depth,
            min_samples_leaf=cfg.min_samples_leaf,
            random_state=cfg.random_state,
        )

        clf.fit(
            X_df,
            y,
        )

        # ==================================================================
        # ASSIGN LEAF IDS
        # ==================================================================

        combined_df["leaf"] = (
            clf.apply(X_df)
        )

        dev_df["leaf"] = (
            combined_df.loc[
                combined_df[
                    "is_monitoring"
                ]
                == 0,
                "leaf",
            ].values
        )

        mon_df["leaf"] = (
            combined_df.loc[
                combined_df[
                    "is_monitoring"
                ]
                == 1,
                "leaf",
            ].values
        )

        # ==================================================================
        # EXTRACT TREE RULES
        # ==================================================================

        leaf_rules_raw = (
            extract_tree_rules(
                clf,
                list(
                    X_df.columns
                ),
            )
        )

        # ==================================================================
        # EVALUATE TREE LEAVES
        # ==================================================================

        candidates_evaluated: list[
            dict
        ] = []

        for (
            leaf_id,
            raw_rules,
        ) in leaf_rules_raw.items():

            # --------------------------------------------------------------
            # Convert encoded tree conditions back to original features
            # --------------------------------------------------------------

            canonical = (
                canonicalize_path_conditions(
                    raw_rules,
                    onehot_map,
                )
            )

            if canonical is None:
                continue

            rule_str = (
                simplify_rule_list(
                    canonical,
                    root_label=(
                        "All Customers (Root)"
                    ),
                )
            )

            # --------------------------------------------------------------
            # Segment masks
            # --------------------------------------------------------------

            d_mask = (
                dev_df["leaf"]
                == leaf_id
            ).values

            m_mask = (
                mon_df["leaf"]
                == leaf_id
            ).values

            n_dev = int(
                d_mask.sum()
            )

            n_mon = int(
                m_mask.sum()
            )

            if total_dev_n <= 0 or total_mon_n <= 0:
                continue

            pct_dev = (
                n_dev
                / total_dev_n
            )

            pct_mon = (
                n_mon
                / total_mon_n
            )

            # --------------------------------------------------------------
            # Support filter
            # --------------------------------------------------------------

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

            # --------------------------------------------------------------
            # Event filter
            # --------------------------------------------------------------

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

            # --------------------------------------------------------------
            # CENTRAL CANDIDATE EVALUATION
            # --------------------------------------------------------------
            #
            # IMPORTANT:
            #
            # feature_cols is passed explicitly to the centralized
            # evaluator.
            #
            # This prevents customer_id and other metadata columns from
            # being considered root causes.
            #
            # SHAP columns are passed separately.
            # --------------------------------------------------------------

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

            # --------------------------------------------------------------
            # Candidate record
            # --------------------------------------------------------------

            cand_record = {
                **eval_res,
                "Leaf_ID": leaf_id,
                "Segment_Definition": rule_str,
                "_mon_mask": m_mask,
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
        # SEVERITY SCORING
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
        # TOP RESULTS
        # ==================================================================

        top_records = (
            ranked_records[
                : cfg.top_n
            ]
        )

        for i, record in enumerate(
            top_records,
            1,
        ):

            record["Rank"] = i

            # Internal mask is required for deduplication / processing but
            # should not be exposed in the final output.
            record.pop(
                "_mon_mask",
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
            "dev_weight_pct": (
                "Dev_Exposure_Pct"
            ),
            "mon_weight_pct": (
                "Mon_Exposure_Pct"
            ),
            "exposure_drift": (
                "Exposure_Drift"
            ),

            # Calibration
            "calibration_drift": (
                "Calibration_Drift"
            ),

            # Root Cause
            "top_drift_feature": (
                "Root_Cause_Feature"
            ),
            "top_drift_psi": (
                "Root_Cause_PSI"
            ),
            "root_cause_score": (
                "Root_Cause_Score"
            ),

            # Backward-compatible SHAP fields
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
        # RETURN
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

def run_drift_localization(
    dev_path: Union[
        str,
        pd.DataFrame,
    ],
    mon_path: Union[
        str,
        pd.DataFrame,
    ],
    cfg: Optional[
        DLTConfig
    ] = None,
) -> dict:
    """
    Functional wrapper for DLT execution.

    Uses the common parameter-optimization framework.
    """

    from core.parameter_optimization import (
        optimize_parameters,
    )

    cfg = (
        cfg
        or DLTConfig()
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
            DriftLocalizationTreeTechnique()
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