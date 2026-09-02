"""
core/candidate_ranking.py
=========================
Severity scoring, significance-first ranking, and portfolio-impact view
construction.

Centralises the ranking pipeline that was duplicated across autoslicer,
drift_tree, and gradient_boosting technique files.  Numerical behaviour is
**identical** to the original implementations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from metrics.significance import (
    adjust_p_values_bh,
    min_max_normalize,
)
from metrics.business_metrics import (
    calculate_confidence,
    calculate_sis,
    calculate_dis,
    calculate_dis_symmetric,
    calculate_root_cause_score,
)
from metrics.drift_metrics import interpret_psi


# ---------------------------------------------------------------------------
# Severity scoring
# ---------------------------------------------------------------------------

def compute_severity_scores(
    evaluated: list[dict],
    significance_alpha: float = 0.05,
) -> list[dict]:
    """Compute BH-adjusted p-values, severity scores, significance flags,
    SIS/DIS, and drift explanations for every evaluated candidate.

    Returns a list of fully-scored record dicts ready for DataFrame conversion
    and downstream ranking.

    This function is the shared scoring pipeline used by autoslicer,
    drift_tree, and gradient_boosting.  Feature-binning uses its own
    bootstrap-based scoring and is NOT routed through here.
    """
    if not evaluated:
        return []

    # --- BH correction across the full evaluated pool ---
    raw_p_br = np.array([e["br_pvalue"] for e in evaluated], dtype=float)
    raw_p_ks = np.array([e["score_shift_pvalue"] for e in evaluated], dtype=float)
    adj_p_br = adjust_p_values_bh(raw_p_br)
    adj_p_ks = adjust_p_values_bh(raw_p_ks)

    # --- Business_Impact_Score: unrelated to SIS/DIS/Root Cause Score below,
    # kept as its own diagnostic (population share x exposure share x drift). ---
    bus_impacts = [
        (e["pct_mon"] * e["mon_weight_pct"])
        * (
            1.0
            + e["psi"]
            + (
                max(0.0, -e["delta_gini"])
                if not np.isnan(e["delta_gini"])
                else 0.0
            )
        )
        for e in evaluated
    ]

    # --- Baseline-aligned SIS / DIS / Root Cause Score ---
    # (segment_discovery/segment_root_cause_analysis.py is the reference
    # implementation these formulas are ported from.)
    auc_drops = [
        max(0.0, -e["delta_auc"]) if not np.isnan(e["delta_auc"]) else 0.0
        for e in evaluated
    ]
    confidences = [calculate_confidence(e["n_mon"]) for e in evaluated]
    # No weight column -> mon_weight_pct is always 0.0; treat exposure as
    # neutral (1.0) rather than zeroing out every score.
    exposure_factors = [
        e["mon_weight_pct"] if e["mon_weight_pct"] > 0 else 1.0 for e in evaluated
    ]
    population_drifts = [e["pct_mon"] - e["pct_dev"] for e in evaluated]
    psi_vals = [e["mean_feature_psi"] for e in evaluated]
    shap_vals = [e["mean_shap_shift"] for e in evaluated]

    sis_vals = [
        calculate_sis(auc_drops[i], exposure_factors[i], e["pct_mon"], confidences[i])
        for i, e in enumerate(evaluated)
    ]
    dis_vals = [
        calculate_dis(population_drifts[i], auc_drops[i], exposure_factors[i], psi_vals[i])
        for i in range(len(evaluated))
    ]
    dis_symmetric_vals = [
        calculate_dis_symmetric(population_drifts[i], auc_drops[i], exposure_factors[i], psi_vals[i])
        for i in range(len(evaluated))
    ]

    norm_sis = min_max_normalize(sis_vals)
    norm_dis = min_max_normalize(dis_vals)
    norm_psi = min_max_normalize(psi_vals)
    norm_shap = min_max_normalize(shap_vals)

    records: list[dict] = []
    for i, e in enumerate(evaluated):
        root_cause_score = calculate_root_cause_score(
            norm_sis[i], norm_dis[i], norm_psi[i], norm_shap[i]
        )
        # Same scale (0-20) the rest of the app already uses for
        # within-technique ranking; Root_Cause_Score (0-1) is what's
        # comparable across techniques (see cross_technique_analysis.py).
        severity = root_cause_score * 20.0

        p_br_adj = adj_p_br[i] if not np.isnan(adj_p_br[i]) else 1.0
        p_ks_adj = adj_p_ks[i] if not np.isnan(adj_p_ks[i]) else 1.0
        is_significant = (
            p_br_adj <= significance_alpha
        ) or (p_ks_adj <= significance_alpha)

        # --- Drift explanation ---
        reasons = _build_drift_reasons(e, p_br_adj, significance_alpha)
        explanation = " | ".join(reasons) if reasons else "Segment performance stable"
        portfolio_impact = (
            e["pct_mon"] * abs(e["delta_gini"])
            if not np.isnan(e["delta_gini"])
            else 0.0
        )

        record = {
            **e,  # carry forward all raw metrics
            "BadRate_pvalue_adj": p_br_adj,
            "ScoreShift_pvalue_adj": p_ks_adj,
            "Statistically_Significant": bool(is_significant),
            "Business_Impact_Score": round(bus_impacts[i], 4),
            "SIS_Raw": round(sis_vals[i], 6),
            "DIS_Raw": round(dis_vals[i], 6),
            "DIS_Symmetric": round(dis_symmetric_vals[i], 6),
            "Confidence_Score": round(confidences[i], 4),
            "PSI_Interpretation": f"Population Share: {interpret_psi(e['psi'])}",
            "root_cause_score": round(root_cause_score, 6),
            "Severity_Score": round(severity, 4),
            "Portfolio_Impact_Score": round(portfolio_impact, 6),
            "Drift_Explanation": explanation,
        }
        records.append(record)

    return records


def _build_drift_reasons(
    e: dict, p_br_adj: float, significance_alpha: float
) -> list[str]:
    """Build human-readable drift-explanation reason strings."""
    reasons: list[str] = []
    if e["psi"] > 0.10:
        reasons.append(
            f"Population shift (Dev {e['pct_dev']:.1%} -> Mon {e['pct_mon']:.1%}, "
            f"PSI {e['psi']:.4f})"
        )
    if not np.isnan(e["delta_gini"]) and e["delta_gini"] < -0.05:
        reasons.append(
            f"Gini drop {abs(e['delta_gini']):.4f} "
            f"(Dev {e['gini_dev']:.4f} -> Mon {e['gini_mon']:.4f})"
        )
    if not np.isnan(e["delta_ks"]) and e["delta_ks"] < -0.05:
        reasons.append(
            f"KS drop {abs(e['delta_ks']):.4f} "
            f"(Dev {e['ks_dev']:.4f} -> Mon {e['ks_mon']:.4f})"
        )
    if (
        not np.isnan(e["delta_br"])
        and abs(e["delta_br"]) > 0.03
        and p_br_adj <= significance_alpha
    ):
        reasons.append(
            f"Significant bad-rate shift "
            f"(Dev {e['br_dev']:.2%} -> Mon {e['br_mon']:.2%}, p_adj={p_br_adj:.4f})"
        )
    if (
        e["psi"] < 0.10
        and not np.isnan(e["delta_gini"])
        and abs(e["delta_gini"]) >= 0.05
    ):
        reasons.append(
            "Population share stable (low PSI) but the score no longer separates "
            "risk inside this segment -- concept drift, not exposure drift"
        )
    return reasons


# ---------------------------------------------------------------------------
# Significance-first ranking
# ---------------------------------------------------------------------------

def rank_records_significance_first(records: list[dict]) -> list[dict]:
    """Sort records by (Statistically_Significant DESC, Severity_Score DESC).

    This is the primary ranking used by autoslicer, drift_tree, and
    gradient_boosting, ensuring significant segments always rank above
    non-significant ones."""
    return sorted(
        records,
        key=lambda r: (r["Statistically_Significant"], r["Severity_Score"]),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Portfolio-impact view
# ---------------------------------------------------------------------------

def build_portfolio_view(
    pool_df: pd.DataFrame, top_n: int
) -> pd.DataFrame:
    """Build the business-ranked portfolio-impact view from a deduplicated
    pool of significant candidates."""
    if pool_df.empty:
        return pd.DataFrame()

    sig_pool = pool_df[pool_df["Statistically_Significant"]]
    if sig_pool.empty:
        return pd.DataFrame()

    portfolio_df = (
        sig_pool.sort_values("Portfolio_Impact_Score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    if "Rank" in portfolio_df.columns:
        portfolio_df["Rank"] = portfolio_df.index + 1
    else:
        portfolio_df.insert(0, "Rank", portfolio_df.index + 1)

    columns_to_keep = [
        "Rank",
        "Segment_Definition",
        "Mon_Pct",
        "Delta_Gini",
        "Delta_KS",
        "Delta_BR",
        "Portfolio_Impact_Score",
        "SIS_Raw",
        "Statistically_Significant",
    ]
    columns_to_keep = [c for c in columns_to_keep if c in portfolio_df.columns]

    portfolio_df = portfolio_df[columns_to_keep].rename(
        columns={
            "Segment_Definition": "Segment",
            "Mon_Pct": "Population_Pct",
            "Portfolio_Impact_Score": "Severity_PopulationXGini",
        }
    )
    return portfolio_df
