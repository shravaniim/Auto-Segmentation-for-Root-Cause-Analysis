"""
techniques/gradient_boosting.py
===============================
Gradient Boosting (GBDT) Segmentation Technique.

Fits an ensemble of shallow trees to discriminate Development vs. Monitoring,
extracting leaf path rules across trees and ranking them by severity & significance.
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
from utils.schema_detection import detect_schema, auto_detect_shap_columns
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
from core.candidate_deduplication import deduplicate_by_jaccard

logger = get_logger(__name__)


class GradientBoostingSegmentationTechnique(BaseSegmentationTechnique):
    """Gradient Boosting Segmentation Technique implementation."""

    @property
    def name(self) -> str:
        return "Gradient Boosting"

    def run(
        self,
        dev_data: Union[str, pd.DataFrame],
        mon_data: Union[str, pd.DataFrame],
        config: Optional[GBConfig] = None,
    ) -> TechniqueResult:
        cfg = config or GBConfig()
        start_time = time.perf_counter()

        dev_df = pd.read_csv(dev_data) if isinstance(dev_data, str) else dev_data.copy()
        mon_df = pd.read_csv(mon_data) if isinstance(mon_data, str) else mon_data.copy()

        schema = detect_schema(dev_df, cfg.schema)
        target_col = schema["target_col"]
        score_col = schema["score_col"]
        weight_col = schema["weight_col"]

        dev_df["is_monitoring"] = 0
        mon_df["is_monitoring"] = 1
        combined_df = pd.concat([dev_df, mon_df], ignore_index=True)

        shap_cols = auto_detect_shap_columns(combined_df)
        if shap_cols:
            shap_shift = detect_shap_shift(dev_df, mon_df, shap_cols)
            logger.info("GBDT detected SHAP shift: %s", shap_shift["top_shift_feature"])

        total_dev_n = len(dev_df)
        total_mon_n = len(mon_df)
        total_dev_weight = (
            float(dev_df[weight_col].sum())
            if weight_col and weight_col in dev_df.columns
            else 0.0
        )
        total_mon_weight = (
            float(mon_df[weight_col].sum())
            if weight_col and weight_col in mon_df.columns
            else 0.0
        )
        dev_event_rate = float(dev_df[target_col].mean())

        min_abs_count = derive_min_abs_count(
            dev_event_rate, cfg.min_events_per_slice, cfg.min_abs_count
        )
        min_support = derive_min_support(
            min_abs_count, total_dev_n, total_mon_n, cfg.min_support
        )

        X_df, onehot_map = build_feature_matrix(
            combined_df, schema["numeric_cols"], schema["categorical_cols"]
        )
        y = combined_df["is_monitoring"].values

        gbdt = GradientBoostingClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            min_samples_leaf=cfg.min_samples_leaf,
            random_state=cfg.random_state,
        )
        gbdt.fit(X_df, y)

        # Obtain drift predictions to prune low-drift trees
        drift_probs = gbdt.predict_proba(X_df)[:, 1]
        combined_df["drift_prob"] = drift_probs
        dev_df["drift_prob"] = combined_df.loc[combined_df["is_monitoring"] == 0, "drift_prob"].values
        mon_df["drift_prob"] = combined_df.loc[combined_df["is_monitoring"] == 1, "drift_prob"].values

        candidates_evaluated: list[dict] = []
        seen_rules: set[str] = set()

        trees_to_scan = (
            gbdt.estimators_[: cfg.max_trees_scanned]
            if cfg.max_trees_scanned
            else gbdt.estimators_
        )

        for tree_idx, tree_estimator in enumerate(trees_to_scan):
            tree_obj = tree_estimator[0] if isinstance(tree_estimator, np.ndarray) else tree_estimator
            leaf_rules_raw = extract_tree_rules(tree_obj, list(X_df.columns))

            # Apply tree masks
            leaf_assignments = tree_obj.apply(X_df.values)
            dev_leaves = leaf_assignments[combined_df["is_monitoring"] == 0]
            mon_leaves = leaf_assignments[combined_df["is_monitoring"] == 1]

            for leaf_id, raw_rules in leaf_rules_raw.items():
                canonical = canonicalize_path_conditions(raw_rules, onehot_map)
                if canonical is None:
                    continue
                rule_str = simplify_rule_list(canonical, root_label="All Customers (Root)")
                if rule_str in seen_rules:
                    continue
                seen_rules.add(rule_str)

                d_mask = (dev_leaves == leaf_id)
                m_mask = (mon_leaves == leaf_id)
                n_dev, n_mon = int(d_mask.sum()), int(m_mask.sum())
                pct_dev, pct_mon = n_dev / total_dev_n, n_mon / total_mon_n

                if not passes_support_filter(
                    n_dev, n_mon, pct_dev, pct_mon, min_abs_count, min_support, cfg.max_segment_pct
                ):
                    continue

                x_dev = int(dev_df.loc[d_mask, target_col].sum())
                x_mon = int(mon_df.loc[m_mask, target_col].sum())
                if not passes_event_count_filter(x_dev, x_mon, cfg.min_events_per_slice):
                    continue

                eval_res = evaluate_segment(
                    dev_df, mon_df, d_mask, m_mask, target_col, score_col, weight_col,
                    total_dev_n, total_mon_n, total_dev_weight, total_mon_weight, shap_cols
                )
                if eval_res is None:
                    continue

                cand_record = {
                    **eval_res,
                    "Tree_Idx": tree_idx,
                    "Leaf_ID": leaf_id,
                    "Segment_Definition": rule_str,
                    "_mon_mask": m_mask,
                }
                candidates_evaluated.append(cand_record)

        if not candidates_evaluated:
            exec_time = time.perf_counter() - start_time
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

        scored_records = compute_severity_scores(
            candidates_evaluated,
            w_psi=cfg.w_psi,
            w_business_impact=cfg.w_business_impact,
            w_gini_drop=cfg.w_gini_drop,
            w_ks_drop=cfg.w_ks_drop,
            w_br_shift=cfg.w_br_shift,
            significance_alpha=cfg.significance_alpha,
        )

        ranked_records = rank_records_significance_first(scored_records)

        dedup_records = deduplicate_by_jaccard(
            ranked_records,
            mask_key="_mon_mask",
            overlap_threshold=cfg.overlap_jaccard_threshold,
            pool_size=30,
        )

        top_records = dedup_records[: cfg.top_n]
        for i, r in enumerate(top_records, 1):
            r["Rank"] = i

        segments_df = pd.DataFrame(top_records)

        col_rename = {
            "n_dev": "Dev_Count",
            "n_mon": "Mon_Count",
            "pct_dev": "Dev_Pct",
            "pct_mon": "Mon_Pct",
            "psi": "PSI",
            "br_dev": "Dev_BR",
            "br_mon": "Mon_BR",
            "delta_br": "Delta_BR",
            "auc_dev": "Dev_AUC",
            "auc_mon": "Mon_AUC",
            "delta_auc": "Delta_AUC",
            "gini_dev": "Dev_Gini",
            "gini_mon": "Mon_Gini",
            "delta_gini": "Delta_Gini",
            "ks_dev": "Dev_KS",
            "ks_mon": "Mon_KS",
            "delta_ks": "Delta_KS",
            "weight_dev": "Dev_EAD",
            "weight_mon": "Mon_EAD",
            # New metric columns
            "dev_weight_pct": "Dev_Exposure_Pct",
            "mon_weight_pct": "Mon_Exposure_Pct",
            "exposure_drift": "Exposure_Drift",
            "calibration_drift": "Calibration_Drift",
            "top_drift_feature": "Root_Cause_Feature",
            "top_drift_psi": "Root_Cause_PSI",
            "root_cause_score": "Root_Cause_Score",
            "top_shap_shift_feature": "Top_SHAP_Feature",
            "top_shap_shift_psi": "Top_SHAP_PSI",
        }
        segments_df = segments_df.rename(columns=col_rename)

        portfolio_df = build_portfolio_view(segments_df, cfg.top_n)

        exec_time = time.perf_counter() - start_time
        overall = {
            "total_dev_n": total_dev_n,
            "total_mon_n": total_mon_n,
            "candidates_evaluated": len(candidates_evaluated),
            "candidates_returned": len(segments_df),
        }

        return TechniqueResult(
            technique_name=self.name,
            overall=overall,
            segments_df=segments_df,
            execution_time=exec_time,
            schema=schema,
            portfolio_view_df=portfolio_df,
        )


def run_gradient_boosting_segmentation(
    dev_path: Union[str, pd.DataFrame],
    mon_path: Union[str, pd.DataFrame],
    cfg: Optional[GBConfig] = None,
) -> dict:
    """Functional wrapper for Gradient Boosting segmentation execution."""
    from core.parameter_optimization import optimize_parameters
    
    cfg = cfg or GBConfig()
    dev_df = pd.read_csv(dev_path) if isinstance(dev_path, str) else dev_path.copy()
    mon_df = pd.read_csv(mon_path) if isinstance(mon_path, str) else mon_path.copy()
    
    def _run_single(d_df, m_df, config):
        technique = GradientBoostingSegmentationTechnique()
        return technique.run(d_df, m_df, config=config)
        
    res = optimize_parameters(_run_single, cfg, dev_df, mon_df)
    return res.to_dict()
