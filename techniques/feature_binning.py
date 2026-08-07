"""
techniques/feature_binning.py
=============================
Multi-Feature Binning Segmentation Technique.

Discovers univariate and pairwise interaction bins where model performance deteriorates,
applying bootstrap resampling and FDR correction.
"""

from __future__ import annotations

import itertools
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
from utils.schema_detection import detect_schema, auto_detect_shap_columns
from metrics.performance_metrics import calculate_auc_gini, calculate_ks, safe_auc
from metrics.drift_metrics import calculate_psi, interpret_psi, detect_shap_shift
from metrics.business_metrics import calculate_sis, calculate_dis
from metrics.significance import adjust_p_values_bh, min_max_normalize

logger = get_logger(__name__)

MISSING_LABEL = "__MISSING__"
RANDOM_STATE = 42
IGNORE_FEATURES = {"target", "score", "ead", "is_monitoring", "leaf_id", "customer_id"}


def safe_ece(df, score_col="score", target_col="target", n_bins=10):
    if len(df) == 0:
        return np.nan
    scores = np.clip(df[score_col].astype(float), 1e-6, 1 - 1e-6)
    try:
        bins = min(n_bins, len(scores))
        df2 = df.copy()
        df2["score_bin"] = pd.qcut(scores, q=bins, duplicates="drop")
        grouped = df2.groupby("score_bin", observed=True).agg(
            p_pred=(score_col, "mean"), p_obs=(target_col, "mean"), n=(target_col, "size")
        )
        if grouped.empty:
            return np.nan
        weights = grouped["n"] / grouped["n"].sum()
        return float((weights * np.abs(grouped["p_pred"] - grouped["p_obs"])).sum())
    except Exception:
        return np.nan


def fast_ece(scores, targets, edges):
    if len(scores) == 0:
        return np.nan
    bin_idx = np.clip(np.digitize(scores, edges) - 1, 0, len(edges) - 2)
    total = 0.0
    n = len(scores)
    for b in np.unique(bin_idx):
        mask = bin_idx == b
        w = mask.sum() / n
        total += w * abs(scores[mask].mean() - targets[mask].mean())
    return float(total)


def _make_unregularized_logreg():
    return LogisticRegression(C=1e9, solver="lbfgs", max_iter=200)


def compute_calibration_stats(df, score_col="score", target_col="target", n_bins=10):
    result = {"intercept": np.nan, "slope": np.nan, "ece": np.nan}
    if len(df) == 0:
        return result
    try:
        X = np.clip(df[score_col].astype(float), 1e-6, 1 - 1e-6).values.reshape(-1, 1)
        y = df[target_col].astype(int).values
        if len(np.unique(y)) < 2:
            result["ece"] = safe_ece(df, score_col, target_col, n_bins)
            return result
        lr = _make_unregularized_logreg()
        lr.fit(X, y)
        result["intercept"] = float(lr.intercept_[0])
        result["slope"] = float(lr.coef_[0][0])
    except Exception:
        pass
    result["ece"] = safe_ece(df, score_col, target_col, n_bins)
    return result


def get_candidate_features(df, excluded_cols=None):
    excluded_cols = set(excluded_cols or [])
    return [
        col for col in df.columns
        if col not in IGNORE_FEATURES and col not in excluded_cols and not col.startswith("shap_")
    ]


def feature_strength(df, feature):
    if pd.api.types.is_numeric_dtype(df[feature]):
        vals = df[feature]
        mask = vals.notna()
        return abs(safe_auc(df.loc[mask, "target"], vals[mask]) - 0.5)
    values = df[feature].astype(str).fillna(MISSING_LABEL)
    agg = df.groupby(values)["target"].mean()
    return abs(safe_auc(df["target"], values.map(agg)) - 0.5)


def select_top_features(df, features, max_features=30):
    scores = []
    for feature in features:
        try:
            scores.append((feature, feature_strength(df, feature)))
        except Exception:
            scores.append((feature, 0.0))
    return [f for f, _ in sorted(scores, key=lambda x: x[1], reverse=True)[:max_features]]


def build_numeric_bins(df, feature, max_bins=8, min_bin_pct=0.01):
    values = df[feature]
    non_missing = values.notna()
    bins = []

    n_missing = int((~non_missing).sum())
    if n_missing > 0 and n_missing / len(df) >= min_bin_pct:
        bins.append({"feature": feature, "type": "missing", "label": f"{feature} is Missing"})

    X = values[non_missing].values.reshape(-1, 1)
    y = df.loc[non_missing, "target"].values
    if len(X) == 0:
        return bins
    min_samples_leaf = max(int(len(df) * min_bin_pct), 20)
    if len(X) < 2 * min_samples_leaf:
        return bins

    clf = DecisionTreeClassifier(
        max_leaf_nodes=max(2, max_bins), min_samples_leaf=min_samples_leaf, random_state=RANDOM_STATE
    )
    clf.fit(X, y)

    intervals = []
    tree = clf.tree_

    def recurse(node, lower, upper):
        if tree.children_left[node] == tree.children_right[node]:
            intervals.append((lower, upper))
            return
        threshold = tree.threshold[node]
        recurse(tree.children_left[node], lower, min(upper, threshold))
        recurse(tree.children_right[node], max(lower, threshold), upper)

    recurse(0, -np.inf, np.inf)
    for lower, upper in intervals:
        if lower == -np.inf and upper == np.inf:
            label = f"{feature} (all non-missing values)"
        elif lower == -np.inf:
            label = f"{feature} <= {round(upper, 3)}"
        elif upper == np.inf:
            label = f"{feature} > {round(lower, 3)}"
        else:
            label = f"{round(lower, 3)} < {feature} <= {round(upper, 3)}"
        bins.append({"feature": feature, "type": "numeric", "lower": lower, "upper": upper, "label": label})
    return bins


def build_categorical_bins(df, feature, max_bins=8, min_bin_pct=0.01):
    values = df[feature].astype(str).fillna(MISSING_LABEL)
    counts = df.assign(_v=values).groupby("_v")["target"].agg(["size", "mean"]).rename(
        columns={"size": "count", "mean": "bad_rate"}
    )
    counts = counts.sort_values("bad_rate")
    total = len(df)
    counts["pct"] = counts["count"] / total
    if len(counts) <= max_bins:
        return [
            {
                "feature": feature,
                "type": "missing" if cat == MISSING_LABEL else "categorical",
                "categories": [cat],
                "label": (f"{feature} is Missing" if cat == MISSING_LABEL else f"{feature} = {cat}"),
            }
            for cat in counts.index if counts.loc[cat, "pct"] >= min_bin_pct
        ]
    counts["group"] = pd.qcut(counts["bad_rate"].rank(method="first"), q=max_bins, labels=False, duplicates="drop")
    bins = []
    for group_id, group_rows in counts.groupby("group"):
        cats = list(group_rows.index)
        if cats == [MISSING_LABEL]:
            label = f"{feature} is Missing"
        else:
            label = f"{feature} IN ({', '.join(map(str, cats))})"
        bins.append({"feature": feature, "type": "categorical", "categories": cats, "label": label})
    return bins


def _single_condition_mask(df, cond):
    feat = cond["feature"]
    if cond["type"] == "missing":
        return (
            df[feat].isna().values if pd.api.types.is_numeric_dtype(df[feat])
            else (df[feat].astype(str).isna() | (df[feat].astype(str) == MISSING_LABEL)).values
        )
    if cond["type"] == "numeric":
        mask = df[feat].notna().to_numpy().copy()
        if cond["lower"] != -np.inf:
            mask &= (df[feat] > cond["lower"]).values
        if cond["upper"] != np.inf:
            mask &= (df[feat] <= cond["upper"]).values
        return mask
    cats = cond["categories"]
    values = df[feat].astype(str).fillna(MISSING_LABEL)
    return values.isin(cats).values


def bin_mask(df, bin_def):
    if bin_def["type"] == "interaction":
        mask = np.ones(len(df), dtype=bool)
        for cond in bin_def["conditions"]:
            mask &= _single_condition_mask(df, cond)
        return mask
    return _single_condition_mask(df, bin_def)


def point_estimate_metrics(bin_def, dev_df, mon_df, total_dev, total_mon, target_col, score_col, weight_col, min_bin_pct=0.01):
    dev_mask = bin_mask(dev_df, bin_def)
    mon_mask = bin_mask(mon_df, bin_def)
    n_dev, n_mon = int(dev_mask.sum()), int(mon_mask.sum())
    if n_dev < max(30, int(total_dev * min_bin_pct)) or n_mon < max(30, int(total_mon * min_bin_pct)):
        return None
    dev_sub, mon_sub = dev_df.loc[dev_mask], mon_df.loc[mon_mask]

    auc_dev, gini_dev = calculate_auc_gini(dev_sub[target_col], dev_sub[score_col])
    auc_mon, gini_mon = calculate_auc_gini(mon_sub[target_col], mon_sub[score_col])
    ks_dev, ks_mon = calculate_ks(dev_sub[target_col], dev_sub[score_col]), calculate_ks(mon_sub[target_col], mon_sub[score_col])
    br_dev, br_mon = float(dev_sub[target_col].mean()), float(mon_sub[target_col].mean())
    cal_dev, cal_mon = compute_calibration_stats(dev_sub, score_col=score_col, target_col=target_col), compute_calibration_stats(mon_sub, score_col=score_col, target_col=target_col)

    ead_dev = float(dev_sub[weight_col].sum()) if weight_col and weight_col in dev_sub.columns else np.nan
    ead_mon = float(mon_sub[weight_col].sum()) if weight_col and weight_col in mon_sub.columns else np.nan
    total_ead_mon = float(mon_df[weight_col].sum()) if weight_col and weight_col in mon_df.columns else np.nan
    exposure_pct = ead_mon / max(total_ead_mon, 1.0) if not np.isnan(ead_mon) else 0.0

    return {
        "feature": bin_def["feature"], "segment": bin_def["label"], "bin_type": bin_def["type"],
        "dev_count": n_dev, "mon_count": n_mon,
        "dev_pct": n_dev / total_dev if total_dev else 0.0, "mon_pct": n_mon / total_mon if total_mon else 0.0,
        "psi": calculate_psi(n_dev / total_dev if total_dev else 0.0, n_mon / total_mon if total_mon else 0.0),
        "psi_interpretation": interpret_psi(calculate_psi(n_dev / total_dev if total_dev else 0.0, n_mon / total_mon if total_mon else 0.0)),
        "auc_dev": auc_dev, "auc_mon": auc_mon,
        "delta_auc": (auc_dev - auc_mon) if not (np.isnan(auc_dev) or np.isnan(auc_mon)) else np.nan,
        "gini_dev": gini_dev, "gini_mon": gini_mon,
        "delta_gini": (gini_dev - gini_mon) if not (np.isnan(gini_dev) or np.isnan(gini_mon)) else np.nan,
        "ks_dev": ks_dev, "ks_mon": ks_mon,
        "delta_ks": (ks_dev - ks_mon) if not (np.isnan(ks_dev) or np.isnan(ks_mon)) else np.nan,
        "br_dev": br_dev, "br_mon": br_mon, "delta_br": br_mon - br_dev,
        "cal_dev_intercept": cal_dev["intercept"], "cal_mon_intercept": cal_mon["intercept"],
        "delta_intercept": (cal_mon["intercept"] - cal_dev["intercept"]) if not (np.isnan(cal_dev["intercept"]) or np.isnan(cal_mon["intercept"])) else np.nan,
        "cal_dev_slope": cal_dev["slope"], "cal_mon_slope": cal_mon["slope"],
        "delta_slope": (cal_mon["slope"] - cal_dev["slope"]) if not (np.isnan(cal_dev["slope"]) or np.isnan(cal_mon["slope"])) else np.nan,
        "ece_dev": cal_dev["ece"], "ece_mon": cal_mon["ece"],
        "delta_ece": (cal_mon["ece"] - cal_dev["ece"]) if not (np.isnan(cal_dev["ece"]) or np.isnan(cal_mon["ece"])) else np.nan,
        "ead_dev": ead_dev, "ead_mon": ead_mon, "exposure_pct": exposure_pct,
        "default_count": int(mon_sub["target"].sum()),
        "_dev_mask": dev_mask, "_mon_mask": mon_mask,
    }


def bootstrap_signed_difference(dev_sub, mon_sub, func, n_iter=50):
    if len(dev_sub) < 30 or len(mon_sub) < 30:
        return np.nan, np.nan, np.nan, 1.0
    rng = np.random.RandomState(RANDOM_STATE)
    diffs = []
    for _ in range(n_iter):
        dev_sample = dev_sub.sample(n=len(dev_sub), replace=True, random_state=rng.randint(1_000_000))
        mon_sample = mon_sub.sample(n=len(mon_sub), replace=True, random_state=rng.randint(1_000_000))
        val = func(dev_sample, mon_sample)
        if not np.isnan(val):
            diffs.append(val)
    if not diffs:
        return np.nan, np.nan, np.nan, 1.0
    diffs = np.asarray(diffs, dtype=float)
    lower, upper = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    point = float(np.mean(diffs))
    p_le = np.mean(diffs <= 0.0)
    p_ge = np.mean(diffs >= 0.0)
    p_two_sided = float(min(1.0, 2.0 * min(p_le, p_ge)))
    return lower, upper, point, p_two_sided


def add_significance(row, dev_df, mon_df, n_iter=50):
    dev_sub, mon_sub = dev_df.loc[row["_dev_mask"]], mon_df.loc[row["_mon_mask"]]

    def gini_diff(dv, mv): return (2 * safe_auc(mv["target"], mv["score"]) - 1) - (2 * safe_auc(dv["target"], dv["score"]) - 1)
    def ks_diff(dv, mv): return calculate_ks(mv["target"], mv["score"]) - calculate_ks(dv["target"], dv["score"])
    def auc_diff(dv, mv): return safe_auc(mv["target"], mv["score"]) - safe_auc(dv["target"], dv["score"])
    def br_diff(dv, mv): return float(mv["target"].mean()) - float(dv["target"].mean())

    all_scores = pd.concat([dev_sub["score"], mon_sub["score"]])
    edges = np.quantile(np.clip(all_scores, 1e-6, 1 - 1e-6), np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf

    def ece_diff(dv, mv):
        return fast_ece(mv["score"].values, mv["target"].values, edges) - fast_ece(dv["score"].values, dv["target"].values, edges)

    for name, fn in [("gini", gini_diff), ("ks", ks_diff), ("auc", auc_diff), ("br", br_diff), ("ece", ece_diff)]:
        lo, hi, point, p = bootstrap_signed_difference(dev_sub, mon_sub, fn, n_iter=n_iter)
        row[f"{name}_ci_low"], row[f"{name}_ci_high"], row[f"{name}_p"] = lo, hi, p
    return row


def score_segments(segments, apply_bh=True, alpha=0.10):
    df = pd.DataFrame(segments)
    df["abs_delta_br"] = np.abs(df["delta_br"])
    df["abs_delta_ece"] = np.abs(df["delta_ece"])
    df["gini_score"] = min_max_normalize(df["delta_gini"])
    df["ks_score"] = min_max_normalize(df["delta_ks"])
    df["auc_score"] = min_max_normalize(df["delta_auc"])
    df["br_score"] = min_max_normalize(df["abs_delta_br"])
    df["ece_score"] = min_max_normalize(df["abs_delta_ece"])
    df["deterioration_score"] = df[["gini_score", "ks_score", "auc_score", "br_score", "ece_score"]].mean(axis=1)

    p_cols = ["gini_p", "ks_p", "auc_p", "br_p", "ece_p"]
    if apply_bh:
        for c in p_cols:
            if c in df.columns:
                df[f"{c}_adj"] = adjust_p_values_bh(df[c].values)
        p_adj_cols = [f"{c}_adj" for c in p_cols if f"{c}_adj" in df.columns]
        df["min_p_adj"] = df[p_adj_cols].min(axis=1, skipna=True)
        df["confidence_score"] = df[p_adj_cols].apply(
            lambda row: np.nanmean(1.0 - np.clip(row.fillna(1.0), 0.0, 1.0)), axis=1
        )
        df["significant"] = df["min_p_adj"] <= alpha
    else:
        df["confidence_score"] = 0.5
        df["significant"] = False

    df["psi_interpretation"] = df["psi"].apply(interpret_psi) if "psi" in df.columns else "Stable"
    df["sis_raw"] = df.apply(lambda row: calculate_sis(row["psi"], row["delta_gini"], row["delta_ks"], row["delta_br"], row["mon_pct"], row.get("exposure_pct", 0.0))["raw"] if not np.isnan(row["psi"]) else np.nan, axis=1)
    df["sis_business_impact"] = df.apply(lambda row: calculate_sis(row["psi"], row["delta_gini"], row["delta_ks"], row["delta_br"], row["mon_pct"], row.get("exposure_pct", 0.0))["business_impact"] if not np.isnan(row["psi"]) else np.nan, axis=1)
    df["dis_raw"] = df.apply(lambda row: calculate_dis(row["psi"], row["delta_gini"], row["delta_ks"], row["delta_br"]) if not np.isnan(row["psi"]) else np.nan, axis=1)
    df["business_impact_score"] = df["exposure_pct"] if "exposure_pct" in df.columns else 0.0
    df["final_score"] = df["deterioration_score"] * df["confidence_score"] * (0.5 + 0.5 * df["business_impact_score"])
    df["rank"] = df["final_score"].rank(ascending=False, method="min").astype(int)
    return df.sort_values(["final_score", "deterioration_score"], ascending=False).reset_index(drop=True)


class FeatureBinningTechnique(BaseSegmentationTechnique):
    """Feature Binning Technique class implementation."""

    @property
    def name(self) -> str:
        return "Feature Binning"

    def run(
        self,
        dev_data: Union[str, pd.DataFrame],
        mon_data: Union[str, pd.DataFrame],
        config: Optional[FeatureBinningConfig] = None,
    ) -> TechniqueResult:
        cfg = config or FeatureBinningConfig()
        start_time = time.perf_counter()

        dev_df = pd.read_csv(dev_data) if isinstance(dev_data, str) else dev_data.copy()
        mon_df = pd.read_csv(mon_data) if isinstance(mon_data, str) else mon_data.copy()

        schema = detect_schema(dev_df, cfg.schema)
        target_col = schema["target_col"]
        score_col = schema["score_col"]
        weight_col = schema["weight_col"]
        total_dev, total_mon = len(dev_df), len(mon_df)

        dev_df = dev_df.rename(columns={target_col: "target", score_col: "score"})
        mon_df = mon_df.rename(columns={target_col: "target", score_col: "score"})
        if weight_col:
            dev_df = dev_df.rename(columns={weight_col: "ead"})
            mon_df = mon_df.rename(columns={weight_col: "ead"})

        combined_df = pd.concat([dev_df, mon_df], ignore_index=True)
        shap_cols = auto_detect_shap_columns(combined_df)
        if shap_cols:
            shap_shift = detect_shap_shift(dev_df, mon_df, shap_cols)
            logger.info("Feature Binning detected SHAP shift: %s", shap_shift["top_shift_feature"])

        features = get_candidate_features(dev_df, excluded_cols=schema["excluded_cols"])
        selected = select_top_features(dev_df, features, cfg.max_features)
        bin_defs = []
        for feature in selected:
            if pd.api.types.is_numeric_dtype(dev_df[feature]):
                bin_defs.extend(build_numeric_bins(dev_df, feature, max_bins=cfg.max_bins, min_bin_pct=cfg.min_bin_pct))
            else:
                bin_defs.extend(build_categorical_bins(dev_df, feature, max_bins=cfg.max_bins, min_bin_pct=cfg.min_bin_pct))

        point_rows = []
        for bin_def in bin_defs:
            m = point_estimate_metrics(
                bin_def, dev_df, mon_df, total_dev, total_mon, "target", "score",
                "ead" if weight_col else None, min_bin_pct=cfg.min_bin_pct
            )
            if m is not None:
                point_rows.append(m)

        stage1_time = time.perf_counter() - start_time

        if not point_rows:
            return TechniqueResult(
                technique_name=self.name,
                overall={"dev_count": total_dev, "mon_count": total_mon, "candidate_bins": len(bin_defs)},
                segments_df=pd.DataFrame(),
                execution_time=time.perf_counter() - start_time,
                schema=schema,
            )

        def rough_effect(r):
            vals = [abs(r.get(k, 0) or 0) for k in ("delta_gini", "delta_ks", "delta_auc", "delta_br", "delta_ece")]
            return np.nanmean([v for v in vals if not np.isnan(v)]) if vals else 0.0

        point_rows.sort(key=rough_effect, reverse=True)
        to_bootstrap = point_rows[: cfg.bootstrap_top_n]
        remainder = point_rows[cfg.bootstrap_top_n:]

        bootstrapped = [add_significance(dict(r), dev_df, mon_df, n_iter=cfg.n_iter) for r in to_bootstrap]
        for r in bootstrapped + remainder:
            r.pop("_dev_mask", None)
            r.pop("_mon_mask", None)
        for r in remainder:
            for name in ("gini", "ks", "auc", "br", "ece"):
                r[f"{name}_ci_low"] = r[f"{name}_ci_high"] = r[f"{name}_p"] = np.nan

        scored_boot = score_segments(bootstrapped, apply_bh=True, alpha=cfg.alpha) if bootstrapped else pd.DataFrame()
        scored_rest = score_segments(remainder, apply_bh=False) if remainder else pd.DataFrame()
        scored = pd.concat([scored_boot, scored_rest], ignore_index=True) if len(scored_rest) else scored_boot
        scored = scored.sort_values(["final_score", "deterioration_score"], ascending=False).reset_index(drop=True)
        scored["rank"] = np.arange(1, len(scored) + 1)
        scored["Root_Cause_Feature"] = scored["feature"]
        scored["Root_Cause_PSI"] = scored["psi"]
        scored["Root_Cause_Score"] = (scored["psi"] * (1.0 + scored["delta_gini"].apply(lambda g: max(0.0, -g) if not pd.isna(g) else 0.0)) * scored["mon_pct"]).round(6)
        scored = scored.head(cfg.top_n)

        exec_time = time.perf_counter() - start_time
        overall = {
            "dev_count": total_dev,
            "mon_count": total_mon,
            "features_evaluated": len(get_candidate_features(dev_df, excluded_cols=schema["excluded_cols"])),
            "candidate_bins": len(bin_defs),
            "bootstrapped": len(to_bootstrap),
        }

        return TechniqueResult(
            technique_name=self.name,
            overall=overall,
            segments_df=scored,
            execution_time=exec_time,
            schema=schema,
        )


def run_feature_binning_segmentation(
    dev_path: Union[str, pd.DataFrame],
    mon_path: Union[str, pd.DataFrame],
    cfg: Optional[FeatureBinningConfig] = None,
) -> dict:
    """Functional wrapper for Feature Binning execution."""
    from core.parameter_optimization import optimize_parameters
    
    cfg = cfg or FeatureBinningConfig()
    dev_df = pd.read_csv(dev_path) if isinstance(dev_path, str) else dev_path.copy()
    mon_df = pd.read_csv(mon_path) if isinstance(mon_path, str) else mon_path.copy()
    
    def _run_single(d_df, m_df, config):
        technique = FeatureBinningTechnique()
        return technique.run(d_df, m_df, config=config)
        
    res = optimize_parameters(_run_single, cfg, dev_df, mon_df)
    return res.to_dict()
