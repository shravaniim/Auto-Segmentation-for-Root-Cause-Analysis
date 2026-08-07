"""
techniques/autoslicer.py
========================
AutoSlicer Sub-group Discovery Technique.

Identifies multi-variable sub-populations experiencing performance degradation or drift
using adaptive beam search and hypothesis testing.
"""

from __future__ import annotations

import itertools
import time
from typing import Optional, Union

import numpy as np
import pandas as pd

from models.config import SlicerConfig
from models.result import TechniqueResult
from techniques.base import BaseSegmentationTechnique
from utils.logging_config import get_logger
from utils.schema_detection import detect_schema, auto_detect_shap_columns
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
from utils.exports import export_results_csv, export_portfolio_view

logger = get_logger(__name__)


def build_feature_bins(
    series: pd.Series, n_bins: int = 4
) -> list[tuple[float, float, str]]:
    """Discretise numeric feature into quantile bins with explicit -inf / +inf endpoints."""
    clean = series.dropna().to_numpy(dtype=float)
    if clean.size == 0:
        return []
    quantiles = np.quantile(clean, np.linspace(0.0, 1.0, n_bins + 1))
    unique_q = np.unique(quantiles)
    if len(unique_q) < 2:
        return []
    edges = unique_q.tolist()
    edges[0] = -np.inf
    edges[-1] = np.inf

    bins = []
    for i in range(len(edges) - 1):
        low, high = edges[i], edges[i + 1]
        if low == -np.inf:
            lbl = f"{series.name} < {high:.4g}"
        elif high == np.inf:
            lbl = f"{series.name} >= {low:.4g}"
        else:
            lbl = f"{low:.4g} <= {series.name} < {high:.4g}"
        bins.append((low, high, lbl))
    return bins


def generate_single_feature_predicates(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_bins: int = 4,
) -> dict[str, list[dict]]:
    """Generate candidate single-variable predicates for numeric and categorical features."""
    predicates_by_feature: dict[str, list[dict]] = {}
    for col in feature_cols:
        series = df[col]
        preds = []
        if pd.api.types.is_numeric_dtype(series):
            for low, high, label in build_feature_bins(series, n_bins):
                preds.append(
                    {
                        "feature": col,
                        "type": "numeric",
                        "lower": low,
                        "upper": high,
                        "label": label,
                    }
                )
        else:
            categories = series.dropna().astype(str).unique().tolist()
            if 1 < len(categories) <= 20:
                for cat in categories:
                    preds.append(
                        {
                            "feature": col,
                            "type": "categorical",
                            "value": cat,
                            "label": f"{col} = {cat}",
                        }
                    )
        if preds:
            predicates_by_feature[col] = preds
    return predicates_by_feature


def eval_predicate_mask(df: pd.DataFrame, pred: dict) -> np.ndarray:
    """Evaluate boolean mask for a single predicate."""
    col = pred["feature"]
    if pred["type"] == "numeric":
        s = df[col]
        mask = s.notna().to_numpy().copy()
        if pred["lower"] != -np.inf:
            mask &= (s >= pred["lower"]).to_numpy()
        if pred["upper"] != np.inf:
            mask &= (s < pred["upper"]).to_numpy()
        return mask
    else:
        return (df[col].astype(str) == pred["value"]).to_numpy()


def eval_combo_mask(df: pd.DataFrame, combo: tuple[dict, ...]) -> np.ndarray:
    """Evaluate combined boolean mask for a conjunction of predicates."""
    mask = np.ones(len(df), dtype=bool)
    for pred in combo:
        mask &= eval_predicate_mask(df, pred)
    return mask


def heuristic_score(m: dict) -> float:
    """Fast depth-comparable screening metric used to drive beam search expansion."""
    gini_drop = max(0.0, -m["delta_gini"]) if not np.isnan(m["delta_gini"]) else 0.0
    ks_drop = max(0.0, -m["delta_ks"]) if not np.isnan(m["delta_ks"]) else 0.0
    br_shift = abs(m["delta_br"]) if not np.isnan(m["delta_br"]) else 0.0
    return (
        m["psi"] * 0.30
        + m["mon_weight_pct"] * 0.25
        + gini_drop * 0.20
        + ks_drop * 0.15
        + br_shift * 0.10
    )


class AutoSlicerTechnique(BaseSegmentationTechnique):
    """AutoSlicer Sub-group Discovery Technique class implementation."""

    @property
    def name(self) -> str:
        return "AutoSlicer"

    def run(
        self,
        dev_data: Union[str, pd.DataFrame],
        mon_data: Union[str, pd.DataFrame],
        config: Optional[SlicerConfig] = None,
    ) -> TechniqueResult:
        cfg = config or SlicerConfig()
        start_time = time.perf_counter()

        dev_df = pd.read_csv(dev_data) if isinstance(dev_data, str) else dev_data.copy()
        mon_df = pd.read_csv(mon_data) if isinstance(mon_data, str) else mon_data.copy()

        schema = detect_schema(dev_df, cfg.schema)
        target_col = schema["target_col"]
        score_col = schema["score_col"]
        weight_col = schema["weight_col"]
        feature_cols = schema["numeric_cols"] + schema["categorical_cols"]

        combined_df = pd.concat([dev_df, mon_df], ignore_index=True)
        shap_cols = auto_detect_shap_columns(combined_df)
        if shap_cols:
            shap_shift = detect_shap_shift(dev_df, mon_df, shap_cols)
            logger.info("AutoSlicer detected SHAP shift: %s", shap_shift["top_shift_feature"])

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

        preds_by_feature = generate_single_feature_predicates(
            dev_df, feature_cols, n_bins=cfg.numeric_bins
        )
        avail_features = list(preds_by_feature.keys())

        candidates_evaluated: list[dict] = []
        evaluated_combos: set[frozenset] = set()

        current_beam: list[tuple[tuple[dict, ...], dict, float]] = []

        # --- Depth 1 ---
        for feat, pred_list in preds_by_feature.items():
            for pred in pred_list:
                combo = (pred,)
                combo_key = frozenset((p["feature"], p["label"]) for p in combo)
                if combo_key in evaluated_combos:
                    continue
                evaluated_combos.add(combo_key)

                d_mask = eval_combo_mask(dev_df, combo)
                m_mask = eval_combo_mask(mon_df, combo)
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

                h_score = heuristic_score(eval_res)
                rule_str = pred["label"]
                cand_record = {
                    **eval_res,
                    "Segment_Definition": rule_str,
                    "heuristic_score": h_score,
                    "_mon_mask": m_mask,
                    "combo": combo,
                }
                candidates_evaluated.append(cand_record)
                current_beam.append((combo, cand_record, h_score))

        current_beam.sort(key=lambda x: x[2], reverse=True)
        best_score_so_far = current_beam[0][2] if current_beam else 0.0

        # --- Multi-depth expansion ---
        for depth in range(2, cfg.max_combo_depth + 1):
            if not current_beam:
                break

            top_h_depth = current_beam[0][2]
            cutoff_h = top_h_depth * cfg.beam_retain_ratio
            retained_beam = [b for b in current_beam if b[2] >= cutoff_h]
            if len(retained_beam) < cfg.min_beam_width:
                retained_beam = current_beam[: cfg.min_beam_width]
            if len(retained_beam) > cfg.beam_width:
                retained_beam = retained_beam[: cfg.beam_width]

            next_beam: list[tuple[tuple[dict, ...], dict, float]] = []

            for combo, parent_record, parent_h in retained_beam:
                used_feats = {p["feature"] for p in combo}
                candidate_next_feats = [f for f in avail_features if f not in used_feats]

                for nf in candidate_next_feats:
                    for pred in preds_by_feature[nf]:
                        new_combo = tuple(sorted(combo + (pred,), key=lambda p: p["feature"]))
                        combo_key = frozenset((p["feature"], p["label"]) for p in new_combo)
                        if combo_key in evaluated_combos:
                            continue
                        evaluated_combos.add(combo_key)

                        d_mask = eval_combo_mask(dev_df, new_combo)
                        m_mask = eval_combo_mask(mon_df, new_combo)
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

                        h_score = heuristic_score(eval_res)
                        rule_str = " AND ".join(p["label"] for p in new_combo)
                        cand_record = {
                            **eval_res,
                            "Segment_Definition": rule_str,
                            "heuristic_score": h_score,
                            "_mon_mask": m_mask,
                            "combo": new_combo,
                        }
                        candidates_evaluated.append(cand_record)
                        next_beam.append((new_combo, cand_record, h_score))

            if not next_beam:
                break

            next_beam.sort(key=lambda x: x[2], reverse=True)
            best_score_this_depth = next_beam[0][2]

            if (best_score_this_depth - best_score_so_far) / max(abs(best_score_so_far), 1e-6) < cfg.depth_improvement_tolerance:
                break

            best_score_so_far = max(best_score_so_far, best_score_this_depth)
            current_beam = next_beam

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
            r.pop("combo", None)
            r.pop("heuristic_score", None)

        segments_df = pd.DataFrame(top_records)

        # Standardize column naming for export
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
            # backward compat
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


def run_autoslicer_segmentation(
    dev_path: Union[str, pd.DataFrame],
    mon_path: Union[str, pd.DataFrame],
    cfg: Optional[SlicerConfig] = None,
) -> dict:
    """Functional wrapper for AutoSlicer segmentation execution."""
    from core.parameter_optimization import optimize_parameters
    
    cfg = cfg or SlicerConfig()
    dev_df = pd.read_csv(dev_path) if isinstance(dev_path, str) else dev_path.copy()
    mon_df = pd.read_csv(mon_path) if isinstance(mon_path, str) else mon_path.copy()
    
    def _run_single(d_df, m_df, config):
        technique = AutoSlicerTechnique()
        return technique.run(d_df, m_df, config=config)
        
    res = optimize_parameters(_run_single, cfg, dev_df, mon_df)
    return res.to_dict()

