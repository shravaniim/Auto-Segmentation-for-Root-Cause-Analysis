"""
techniques/feature_binning_adapter.py
======================================
PROTOTYPE -- not wired into the app or run_feature_binning_segmentation.

Demonstrates the "adapter" architecture the manager proposed: Feature
Binning's own candidate-generation logic (build_numeric_bins /
build_categorical_bins) is kept exactly as-is, but every candidate is then
routed through the SAME shared evaluation/filtering/dedup/scoring pipeline
used by AutoSlicer, Drift Tree, Gradient Boosting, and K-Means
(core.candidate_evaluation.evaluate_segment, core.candidate_filtering,
core.candidate_deduplication, core.candidate_ranking) instead of Feature
Binning's own parallel implementation of those steps.

Run standalone to compare against the existing techniques/feature_binning.py
output:

    python -m techniques.feature_binning_adapter
"""

from __future__ import annotations

import time
from typing import Union

import pandas as pd

from models.config import FeatureBinningConfig
from models.result import TechniqueResult
from utils.schema_detection import detect_schema, auto_detect_shap_columns

from core.candidate_evaluation import evaluate_segment
from core.candidate_filtering import (
    derive_min_abs_count,
    derive_min_support,
    passes_support_filter,
)
from core.candidate_deduplication import deduplicate_by_jaccard
from core.candidate_ranking import compute_severity_scores, rank_records_significance_first

from techniques.feature_binning import (
    get_candidate_features,
    select_top_features,
    build_numeric_bins,
    build_categorical_bins,
    bin_mask,
)


def run_feature_binning_adapter(
    dev_path: Union[str, pd.DataFrame],
    mon_path: Union[str, pd.DataFrame],
    cfg: FeatureBinningConfig | None = None,
) -> dict:
    cfg = cfg or FeatureBinningConfig()
    start_time = time.perf_counter()

    dev_df = pd.read_csv(dev_path) if isinstance(dev_path, str) else dev_path.copy()
    mon_df = pd.read_csv(mon_path) if isinstance(mon_path, str) else mon_path.copy()

    # --- Schema (identical to every other technique) ---
    schema = detect_schema(dev_df, cfg.schema)
    target_col = schema["target_col"]
    score_col = schema["score_col"]
    weight_col = schema["weight_col"]
    feature_cols = schema["feature_cols"]

    shap_cols = auto_detect_shap_columns(pd.concat([dev_df, mon_df], ignore_index=True))

    total_dev_n = len(dev_df)
    total_mon_n = len(mon_df)
    total_dev_weight = float(dev_df[weight_col].sum()) if weight_col and weight_col in dev_df.columns else 0.0
    total_mon_weight = float(mon_df[weight_col].sum()) if weight_col and weight_col in mon_df.columns else 0.0
    dev_event_rate = float(dev_df[target_col].mean())

    min_abs_count = derive_min_abs_count(dev_event_rate, 30, None)
    min_support = derive_min_support(min_abs_count, total_dev_n, total_mon_n, None)

    # --- Candidate generation: UNCHANGED Feature Binning discovery logic ---
    features = get_candidate_features(dev_df, allowed_features=feature_cols)
    selected = select_top_features(dev_df, features, cfg.max_features)

    bin_defs = []
    for feature in selected:
        if pd.api.types.is_numeric_dtype(dev_df[feature]):
            bin_defs.extend(build_numeric_bins(dev_df, feature, max_bins=cfg.max_bins, min_bin_pct=cfg.min_bin_pct))
        else:
            bin_defs.extend(build_categorical_bins(dev_df, feature, max_bins=cfg.max_bins, min_bin_pct=cfg.min_bin_pct))

    # --- Everything below is the SAME shared pipeline the other 4 techniques use ---
    candidates_evaluated = []
    for bin_def in bin_defs:
        d_mask = bin_mask(dev_df, bin_def)
        m_mask = bin_mask(mon_df, bin_def)
        n_dev, n_mon = int(d_mask.sum()), int(m_mask.sum())
        if n_dev == 0 or n_mon == 0:
            continue
        pct_dev, pct_mon = n_dev / total_dev_n, n_mon / total_mon_n

        if not passes_support_filter(
            n_dev, n_mon, pct_dev, pct_mon, min_abs_count, min_support, cfg.max_segment_pct
        ):
            continue

        eval_res = evaluate_segment(
            dev_df=dev_df, mon_df=mon_df, dev_mask=d_mask, mon_mask=m_mask,
            target_col=target_col, score_col=score_col, weight_col=weight_col,
            total_dev_n=total_dev_n, total_mon_n=total_mon_n,
            total_dev_weight=total_dev_weight, total_mon_weight=total_mon_weight,
            shap_cols=shap_cols, feature_cols=feature_cols,
        )
        if eval_res is None:
            continue

        candidates_evaluated.append({
            **eval_res,
            "Segment_Definition": bin_def["label"],
            "_mon_mask": m_mask,
        })

    exec_time = time.perf_counter() - start_time
    if not candidates_evaluated:
        return TechniqueResult(
            technique_name="Feature Binning (adapter)",
            overall={"total_dev_n": total_dev_n, "total_mon_n": total_mon_n,
                     "candidates_evaluated": 0, "candidates_returned": 0},
            segments_df=pd.DataFrame(), execution_time=exec_time, schema=schema,
        ).to_dict()

    deduped = deduplicate_by_jaccard(
        candidates_evaluated, mask_key="_mon_mask",
        overlap_threshold=cfg.overlap_jaccard_threshold, pool_size=len(candidates_evaluated),
    )
    scored = compute_severity_scores(deduped)
    ranked = rank_records_significance_first(scored)
    top_records = ranked[: cfg.top_n]

    segments_df = pd.DataFrame(top_records)
    result = TechniqueResult(
        technique_name="Feature Binning (adapter)",
        overall={
            "total_dev_n": total_dev_n, "total_mon_n": total_mon_n,
            "candidates_evaluated": len(candidates_evaluated),
            "candidates_returned": len(segments_df),
        },
        segments_df=segments_df, execution_time=exec_time, schema=schema,
    )
    return result.to_dict()


if __name__ == "__main__":
    from compare_segmentation_techniques import DEV_FILE, MON_FILE, build_feature_schema

    dev_df = pd.read_csv(DEV_FILE)
    mon_df = pd.read_csv(MON_FILE)
    cfg = FeatureBinningConfig(schema=build_feature_schema(dev_df))

    res = run_feature_binning_adapter(dev_df, mon_df, cfg=cfg)
    print(f"Candidates evaluated: {res['overall']['candidates_evaluated']}")
    print(f"Top segments returned: {res['overall']['candidates_returned']}")
    print(f"Execution time: {res['execution_time']:.2f}s")
    cols = [c for c in ["Segment_Definition", "PSI", "Delta_Gini", "Root_Cause_Score", "Severity_Score"]
            if c in res["segments"].columns]
    if not res["segments"].empty:
        print(res["segments"][cols].head(10).to_string(index=False))
