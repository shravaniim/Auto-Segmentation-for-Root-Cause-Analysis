"""
techniques/kmeans.py
=====================
K-Means Clustering Segmentation Technique.

Discovers sub-populations by clustering feature space on Development data,
assigning Monitoring data to nearest centroids, and distilling human-readable
tree rules for each cluster.
"""

from __future__ import annotations

import time
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score,
    davies_bouldin_score, adjusted_rand_score
)

from models.config import KMeansConfig
from models.result import TechniqueResult
from techniques.base import BaseSegmentationTechnique
from utils.logging_config import get_logger
from utils.schema_detection import detect_schema, auto_detect_shap_columns
from utils.preprocessing import format_condition
from metrics.drift_metrics import detect_shap_shift
from core.candidate_evaluation import evaluate_segment
from core.candidate_ranking import (
    compute_severity_scores,
    rank_records_significance_first,
    build_portfolio_view,
)

logger = get_logger(__name__)

STABILITY_SEED = 777
LARGE_DATA_ROWS = 200_000


def fit_encoder(dev_df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> dict:
    """Fit scaling/encoding on DEV ONLY, reused unchanged for monitoring."""
    scaler = StandardScaler()
    num_fit = scaler.fit(dev_df[numeric_cols].fillna(dev_df[numeric_cols].median())) if numeric_cols else None
    cat_values = {c: sorted(dev_df[c].astype(str).fillna("__MISSING__").unique()) for c in categorical_cols}
    return {
        "scaler": num_fit,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "cat_values": cat_values,
        "numeric_medians": dev_df[numeric_cols].median() if numeric_cols else None,
    }


def transform(df: pd.DataFrame, enc: dict) -> np.ndarray:
    parts = []
    if enc["numeric_cols"]:
        X_num = df[enc["numeric_cols"]].fillna(enc["numeric_medians"])
        parts.append(enc["scaler"].transform(X_num))
    for c in enc["categorical_cols"]:
        vals = df[c].astype(str).fillna("__MISSING__")
        onehot = pd.get_dummies(vals).reindex(columns=enc["cat_values"][c], fill_value=0)
        parts.append(onehot.to_numpy(dtype=float))
    return np.hstack(parts) if parts else np.zeros((len(df), 0))


def _kmeans_cls(n_rows: int):
    return MiniBatchKMeans if n_rows > LARGE_DATA_ROWS else KMeans


def select_optimal_k(
    X: np.ndarray,
    k_range: range,
    min_cluster_pct: float,
    seed: int,
    X_mon: Optional[np.ndarray] = None,
    drift_aware: bool = True,
    drift_weight: float = 0.4,
) -> tuple[int, pd.DataFrame]:
    """Select optimal k using internal validity metrics, stability, and drift sensitivity."""
    n = len(X)
    Estimator = _kmeans_cls(n)
    rows = []
    for k in k_range:
        if k >= n:
            continue

        km_a = Estimator(n_clusters=k, random_state=seed, n_init=10 if Estimator is KMeans else 3)
        labels_a = km_a.fit_predict(X)

        sizes = np.bincount(labels_a) / n
        if sizes.min() < min_cluster_pct:
            rows.append({
                "k": k, "silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan,
                "stability_ari": np.nan, "max_cluster_drift": np.nan, "min_cluster_pct": sizes.min(),
                "composite": -np.inf,
                "rejected_reason": f"smallest cluster {sizes.min()*100:.1f}% < floor {min_cluster_pct*100:.0f}%",
            })
            continue

        km_b = Estimator(n_clusters=k, random_state=STABILITY_SEED, n_init=10 if Estimator is KMeans else 3)
        labels_b = km_b.fit_predict(X)
        stability = adjusted_rand_score(labels_a, labels_b)

        sil = silhouette_score(X, labels_a, sample_size=min(n, 10000), random_state=seed)
        ch = calinski_harabasz_score(X, labels_a)
        db = davies_bouldin_score(X, labels_a)

        max_drift = np.nan
        if drift_aware and X_mon is not None:
            mon_labels = km_a.predict(X_mon)
            dev_pcts = np.bincount(labels_a, minlength=k) / len(labels_a)
            mon_pcts = np.bincount(mon_labels, minlength=k) / len(mon_labels)
            drifts = np.abs(mon_pcts - dev_pcts) / np.maximum(dev_pcts, 1e-6)
            max_drift = float(np.max(drifts))

        rows.append({
            "k": k, "silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db,
            "stability_ari": stability, "max_cluster_drift": max_drift, "min_cluster_pct": sizes.min(),
            "rejected_reason": None,
        })

    diag = pd.DataFrame(rows)
    valid = diag[diag["rejected_reason"].isna()].copy()
    if valid.empty:
        raise ValueError("No candidate k satisfied the minimum cluster-size floor.")

    def norm(series):
        return (series - series.min()) / (series.max() - series.min()) if series.max() > series.min() else series * 0 + 1.0

    valid["sil_n"] = norm(valid["silhouette"])
    valid["ch_n"] = norm(valid["calinski_harabasz"])
    valid["db_n"] = 1 - norm(valid["davies_bouldin"])
    valid["stability_n"] = norm(valid["stability_ari"])
    internal_validity = valid[["sil_n", "ch_n", "db_n"]].mean(axis=1)

    if drift_aware and X_mon is not None:
        valid["drift_n"] = norm(valid["max_cluster_drift"])
        valid["composite"] = (
            (1 - drift_weight) * (0.5 * internal_validity + 0.5 * valid["stability_n"])
            + drift_weight * valid["drift_n"]
        )
    else:
        valid["composite"] = 0.5 * internal_validity + 0.5 * valid["stability_n"]

    diag = diag.merge(valid[["k", "composite"]], on="k", how="left")
    best_k = int(valid.loc[valid["composite"].idxmax(), "k"])
    return best_k, diag.sort_values("k").reset_index(drop=True)


def distill_cluster_rules(raw_df, cluster_labels, numeric_cols, categorical_cols, max_depth=4):
    """Distills human-readable decision rules for each cluster using a decision tree."""
    X = pd.DataFrame(index=raw_df.index)
    for c in numeric_cols:
        X[c] = raw_df[c].fillna(raw_df[c].median())

    onehot_cols = {}
    for c in categorical_cols:
        vals = raw_df[c].astype(str).fillna("__MISSING__")
        for v in sorted(vals.unique()):
            col = f"{c}__{v}"
            X[col] = (vals == v).astype(int)
            onehot_cols[col] = (c, v)

    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=max(20, int(0.01 * len(X))),
        random_state=42,
    )
    tree.fit(X, cluster_labels)
    leaf_id = tree.apply(X)
    t = tree.tree_

    def path_conditions(leaf):
        def walk(node, path):
            if node == leaf:
                return path
            if t.children_left[node] == t.children_right[node]:
                return None
            f = X.columns[t.feature[node]]
            thr = t.threshold[node]
            left_path = walk(t.children_left[node], path + [(f, "<=", thr)])
            if left_path is not None:
                return left_path
            return walk(t.children_right[node], path + [(f, ">", thr)])
        return walk(0, [])

    results = {}
    for cluster_id in np.unique(cluster_labels):
        mask = cluster_labels == cluster_id
        leaves_for_cluster = pd.Series(leaf_id[mask]).value_counts()
        best_leaf = leaves_for_cluster.index[0]
        leaf_mask = leaf_id == best_leaf
        precision = (cluster_labels[leaf_mask] == cluster_id).mean() if leaf_mask.sum() else 0.0
        recall = leaves_for_cluster.iloc[0] / mask.sum()

        conds = path_conditions(best_leaf) or []
        rendered = []
        for feature_name, direction, threshold in conds:
            rendered.append(format_condition(feature_name, threshold, direction, onehot_cols))

        results[cluster_id] = {
            "rule_text": " AND ".join(rendered) if rendered else f"Cluster_{cluster_id} (centroid)",
            "precision": float(precision),
            "recall": float(recall),
        }
    return results


class KMeansSegmentationTechnique(BaseSegmentationTechnique):
    """K-Means Segmentation Technique implementation."""

    @property
    def name(self) -> str:
        return "K-Means Clustering"

    def run(
        self,
        dev_data: Union[str, pd.DataFrame],
        mon_data: Union[str, pd.DataFrame],
        config: Optional[KMeansConfig] = None,
    ) -> TechniqueResult:
        cfg = config or KMeansConfig()
        start_time = time.perf_counter()

        dev_df = pd.read_csv(dev_data) if isinstance(dev_data, str) else dev_data.copy()
        mon_df = pd.read_csv(mon_data) if isinstance(mon_data, str) else mon_data.copy()

        schema = detect_schema(dev_df, cfg.schema)
        target_col = schema["target_col"]
        score_col = schema["score_col"]
        weight_col = schema["weight_col"]
        num_cols = schema["numeric_cols"]
        cat_cols = schema["categorical_cols"]

        combined_df = pd.concat([dev_df, mon_df], ignore_index=True)
        shap_cols = auto_detect_shap_columns(combined_df)
        if shap_cols:
            shap_shift = detect_shap_shift(dev_df, mon_df, shap_cols)
            logger.info("K-Means detected SHAP shift: %s", shap_shift["top_shift_feature"])

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

        enc = fit_encoder(dev_df, num_cols, cat_cols)
        X_dev = transform(dev_df, enc)
        X_mon = transform(mon_df, enc)

        best_k, k_diagnostics = select_optimal_k(
            X_dev, cfg.k_range, cfg.min_cluster_pct, cfg.seed,
            X_mon=X_mon, drift_aware=cfg.drift_aware, drift_weight=cfg.drift_weight
        )

        Estimator = _kmeans_cls(len(X_dev))
        km = Estimator(n_clusters=best_k, random_state=cfg.seed, n_init=10 if Estimator is KMeans else 3)
        dev_clusters = km.fit_predict(X_dev)
        mon_clusters = km.predict(X_mon)

        rules = distill_cluster_rules(
            dev_df, dev_clusters, num_cols, cat_cols, max_depth=cfg.max_tree_depth
        )

        candidates_evaluated: list[dict] = []
        for cluster_id in range(best_k):
            d_mask = (dev_clusters == cluster_id)
            m_mask = (mon_clusters == cluster_id)

            eval_res = evaluate_segment(
                dev_df, mon_df, d_mask, m_mask, target_col, score_col, weight_col,
                total_dev_n, total_mon_n, total_dev_weight, total_mon_weight, shap_cols
            )
            if eval_res is None:
                continue

            rule_info = rules.get(cluster_id, {"rule_text": f"Cluster_{cluster_id}"})
            cand_record = {
                **eval_res,
                "Cluster_ID": cluster_id,
                "Segment_Definition": f"Cluster_{cluster_id}: {rule_info['rule_text']}",
                "Rule_Precision": rule_info.get("precision", 1.0),
                "Rule_Recall": rule_info.get("recall", 1.0),
                "_mon_mask": m_mask,
            }
            candidates_evaluated.append(cand_record)

        scored_records = compute_severity_scores(
            candidates_evaluated,
            significance_alpha=0.05,
        )

        ranked_records = rank_records_significance_first(scored_records)

        for i, r in enumerate(ranked_records, 1):
            r["Rank"] = i
            r.pop("_mon_mask", None)

        segments_df = pd.DataFrame(ranked_records)

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

        portfolio_df = build_portfolio_view(segments_df, len(segments_df))

        exec_time = time.perf_counter() - start_time
        overall = {
            "total_dev_n": total_dev_n,
            "total_mon_n": total_mon_n,
            "optimal_k": best_k,
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
            extra={"k_diagnostics": k_diagnostics},
        )


def run_kmeans_segmentation(
    dev_path: Union[str, pd.DataFrame],
    mon_path: Union[str, pd.DataFrame],
    cfg: Optional[KMeansConfig] = None,
) -> dict:
    """Functional wrapper for K-Means segmentation execution."""
    from core.parameter_optimization import optimize_parameters
    
    cfg = cfg or KMeansConfig()
    dev_df = pd.read_csv(dev_path) if isinstance(dev_path, str) else dev_path.copy()
    mon_df = pd.read_csv(mon_path) if isinstance(mon_path, str) else mon_path.copy()
    
    def _run_single(d_df, m_df, config):
        technique = KMeansSegmentationTechnique()
        return technique.run(d_df, m_df, config=config)
        
    res = optimize_parameters(_run_single, cfg, dev_df, mon_df)
    return res.to_dict()
