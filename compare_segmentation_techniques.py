"""
Auto Segmentation Techniques Comparison & Benchmarking Suite
Author: Antigravity AI
Description: Benchmarks and compares 5 Auto Segmentation techniques with a director-ready
             interpretable summary table featuring ranked overall scores, segment counts,
             reproducibility flags, and human-readable explainability ratings.
             
             Includes full segment-level metric pipeline:
             - Population%, Exposure%, AUC, Gini, Bad Rate, SHAP Metrics
             - Drift Detection: Population Drift, Exposure Drift, PSI, SHAP Shift
             - Performance Degradation: AUC Drop, Gini Drop, Calibration Drift
             - Impact Scores: SIS, DIS, Root Cause Score
             - SHAP Analysis for worst segments
             - Executive Summary & Insights
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path

# Import 5 Segmentation Modules
from drift_localization_tree import run_drift_localization, DLTConfig
from kmeans_segmentation import run_kmeans_segmentation, KMeansConfig
from autoslicer_segmentation import run_autoslicer_segmentation, SlicerConfig
from feature_binning_segmentation import run_feature_binning_segmentation, FeatureBinningConfig
from gradient_boosting_segmentation import run_gradient_boosting_segmentation, GBConfig

from core.segment_insights import (
    compute_root_cause_scores,
    generate_executive_summary,
)
from utils.logging_config import get_logger

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DEV_FILE = str(DATA_DIR / "development_data_NEW.csv")
MON_FILE = str(DATA_DIR / "monitoring_2026_01.csv")

# --------------------------------------------------------------------------
# Static properties per technique (not data-driven)
# --------------------------------------------------------------------------
TECHNIQUE_PROPERTIES = {
    "Drift Localization Tree": {"Deterministic": "Yes",  "Explainability_Label": "High"},
    "K-Means Clustering":      {"Deterministic": "Yes*", "Explainability_Label": "Medium"},
    "AutoSlicer":              {"Deterministic": "Yes",  "Explainability_Label": "Medium"},
    "Feature Binning":         {"Deterministic": "Yes",  "Explainability_Label": "High"},
    "Gradient Boosting":       {"Deterministic": "Yes",  "Explainability_Label": "Medium"},
}


def overall_score_100(row, col_weights):
    total = 0.0
    for col, weight in col_weights.items():
        val = row.get(col, 0.0)
        if isinstance(val, str) or np.isnan(float(val) if isinstance(val, (int, float)) else 0):
            val = 0.0
        total += float(val) * weight
    return round(total * 100.0, 1)


def min_max_normalize(series):
    mn, mx = series.min(), series.max()
    if mx - mn < 1e-9:
        return series * 0.0
    return (series - mn) / (mx - mn)


def _resolve_segment_definition(row):
    for key in ['Segment_Definition', 'segment_definition', 'rule_text', 'segment', 'Segment']:
        if key in row and not pd.isna(row[key]):
            return str(row[key])
    return 'No segment description available'


def _first_nonnull_value(row, keys, default=0.0):
    for key in keys:
        if key in row and not pd.isna(row[key]):
            return row[key]
    return default


def _max_metric(df, keys, default=0.0):
    values = []
    for key in keys:
        if key in df.columns:
            values.extend(df[key].dropna().astype(float).tolist())
    return float(np.nanmax(values)) if values else default


def _min_metric(df, keys, default=np.nan):
    values = []
    for key in keys:
        if key in df.columns:
            values.extend(df[key].dropna().astype(float).tolist())
    return float(np.nanmin(values)) if values else default


def _max_abs_metric(df, keys, default=0.0):
    values = []
    for key in keys:
        if key in df.columns:
            values.extend(df[key].dropna().astype(float).tolist())
    if not values:
        return default
    return float(max(values, key=abs))






def standardize_columns(df, tech_name):
    """Standardizes column names so the concatenated dataframe has uniform columns."""
    if df.empty:
        return df

    col_map = {
        # Segment definition (Feature Binning uses 'segment')
        'Segment_Definition': 'Segment_Definition',
        'segment_definition': 'Segment_Definition',
        'segment': 'Segment_Definition',
        'rule_text': 'Segment_Definition',
        # PSI
        'PSI': 'PSI', 'psi': 'PSI',
        # AUC
        'Dev_AUC': 'Dev_AUC', 'auc_dev': 'Dev_AUC',
        'Mon_AUC': 'Mon_AUC', 'auc_mon': 'Mon_AUC',
        'Delta_AUC': 'Delta_AUC', 'delta_auc': 'Delta_AUC',
        # Gini
        'Dev_Gini': 'Dev_Gini', 'gini_dev': 'Dev_Gini',
        'Mon_Gini': 'Mon_Gini', 'gini_mon': 'Mon_Gini',
        'Delta_Gini': 'Delta_Gini', 'delta_gini': 'Delta_Gini',
        # KS
        'Dev_KS': 'Dev_KS', 'ks_dev': 'Dev_KS',
        'Mon_KS': 'Mon_KS', 'ks_mon': 'Mon_KS',
        'Delta_KS': 'Delta_KS', 'delta_ks': 'Delta_KS',
        # Bad Rate
        'Dev_BR': 'Dev_BR', 'br_dev': 'Dev_BR',
        'Mon_BR': 'Mon_BR', 'br_mon': 'Mon_BR',
        'Delta_BR': 'Delta_BR', 'delta_br': 'Delta_BR',
        # Population %
        'Dev_Pct': 'Dev_Pct', 'dev_pct': 'Dev_Pct',
        'Mon_Pct': 'Mon_Pct', 'mon_pct': 'Mon_Pct',
        # Counts
        'Dev_Count': 'Dev_Count', 'dev_count': 'Dev_Count', 'n_dev': 'Dev_Count',
        'Mon_Count': 'Mon_Count', 'mon_count': 'Mon_Count', 'n_mon': 'Mon_Count',
        # Exposure (EAD)
        'Dev_EAD': 'Dev_EAD', 'weight_dev': 'Dev_EAD', 'ead_dev': 'Dev_EAD',
        'Mon_EAD': 'Mon_EAD', 'weight_mon': 'Mon_EAD', 'ead_mon': 'Mon_EAD',
        # Exposure %
        'Dev_Exposure_Pct': 'Dev_Exposure_Pct', 'dev_weight_pct': 'Dev_Exposure_Pct',
        'Mon_Exposure_Pct': 'Mon_Exposure_Pct', 'mon_weight_pct': 'Mon_Exposure_Pct',
        'exposure_pct': 'Mon_Exposure_Pct',  # Feature Binning uses this
        # Drift metrics
        'Exposure_Drift': 'Exposure_Drift', 'exposure_drift': 'Exposure_Drift',
        'Calibration_Drift': 'Calibration_Drift', 'calibration_drift': 'Calibration_Drift',
        # Feature Binning calibration columns
        'delta_intercept': 'Calibration_Drift',  # A/E intercept shift = calibration drift proxy
        # Root Cause (feature-level PSI)
        'Root_Cause_Feature': 'Root_Cause_Feature', 'top_drift_feature': 'Root_Cause_Feature',
        'Root_Cause_PSI': 'Root_Cause_PSI', 'top_drift_psi': 'Root_Cause_PSI',
        'Root_Cause_Score': 'Root_Cause_Score', 'root_cause_score': 'Root_Cause_Score',
        # backward compat SHAP
        'top_shap_shift_feature': 'Top_SHAP_Feature',
        'top_shap_shift_psi': 'Top_SHAP_PSI',
        # Scores
        'Business_Impact_Score': 'Business_Impact_Score',
        'business_impact_score': 'Business_Impact_Score',
        'SIS_Raw': 'SIS_Raw', 'sis_raw': 'SIS_Raw',
        'SIS_Business_Impact': 'SIS_Business_Impact', 'sis_business_impact': 'SIS_Business_Impact',
        'DIS_Raw': 'DIS_Raw', 'dis_raw': 'DIS_Raw',
        'Severity_Score': 'Severity_Score',
        # Feature Binning final_score = severity proxy
        'final_score': 'Severity_Score',
        'deterioration_score': 'Deterioration_Score',
        'Drift_Explanation': 'Drift_Explanation',
        'Rank': 'Rank', 'rank': 'Rank',
    }

    # Apply rename only for existing columns
    rename = {k: v for k, v in col_map.items() if k in df.columns and k != v}
    df_renamed = df.rename(columns=rename)

    if 'Technique' not in df_renamed.columns:
        df_renamed['Technique'] = tech_name

    return df_renamed


def benchmark_all_techniques():
    print("=" * 115)
    print("SAS RISK MANAGEMENT - AUTO SEGMENTATION TECHNIQUES BENCHMARKING & COMPARISON")
    print("=" * 115)
    print("Running 5 Segmentation Algorithms on Development & Monitoring Datasets...\n")

    # Load base data (only dev_NEW and mon_2026_01)
    dev_df = pd.read_csv(DEV_FILE)
    mon_df = pd.read_csv(MON_FILE)
    print(f"  Dev data: {len(dev_df)} rows | Mon data: {len(mon_df)} rows")
    print()

    print("--> [1/5] Running Drift Localization Tree (DLT)...")
    res_dlt = run_drift_localization(dev_df, mon_df, cfg=DLTConfig(max_depth=3, min_samples_leaf=0.05))

    print("--> [2/5] Running K-Means Clustering...")
    res_kmeans = run_kmeans_segmentation(dev_df, mon_df, cfg=KMeansConfig())

    print("--> [3/5] Running AutoSlicer (Sub-group Discovery)...")
    slicer_cfg = SlicerConfig(
        max_combo_depth=2,
        beam_width=10,
        param_grid={
            "max_combo_depth": [2, 3],
            "beam_width": [10, 15]
        }
    )
    res_slicer = run_autoslicer_segmentation(dev_df, mon_df, cfg=slicer_cfg)

    print("--> [4/5] Running Multi-Feature Binning...")
    res_binning = run_feature_binning_segmentation(dev_df, mon_df, cfg=FeatureBinningConfig())

    print("--> [5/5] Running Gradient Boosting (GBDT)...")
    res_gbdt = run_gradient_boosting_segmentation(dev_df, mon_df, cfg=GBConfig(n_estimators=5, max_depth=3))

    techniques_runs = [
        ("Drift Localization Tree", res_dlt),
        ("K-Means Clustering",     res_kmeans),
        ("AutoSlicer",             res_slicer),
        ("Feature Binning",        res_binning),
        ("Gradient Boosting",      res_gbdt),
    ]

    all_segments_list = []
    raw_metrics = []

    for tech_name, res in techniques_runs:
        df_seg = res.get('segments', pd.DataFrame())
        exec_time = res.get('execution_time', 0.0)

        if not df_seg.empty:
            df_std = standardize_columns(df_seg, tech_name)
            all_segments_list.append(df_std)

            top_segment = df_seg.iloc[0]
            max_psi = _max_metric(df_seg, ['PSI', 'psi'])

            min_dg = _min_metric(df_seg, ['Delta_Gini', 'delta_gini'])
            max_gini_drop = abs(min_dg) if (not np.isnan(min_dg) and min_dg < 0) else 0.0

            min_dk = _min_metric(df_seg, ['Delta_KS', 'delta_ks'])
            max_ks_drop = abs(min_dk) if (not np.isnan(min_dk) and min_dk < 0) else 0.0

            max_br_shift = _max_abs_metric(df_seg, ['Delta_BR', 'delta_br'])

            bus_impact_raw = _first_nonnull_value(
                top_segment,
                ['Business_Impact_Score', 'SIS_Business_Impact', 'business_impact', 'business_impact_score']
            )
            top_ead = _first_nonnull_value(top_segment, ['Mon_EAD', 'Weight_Mon', 'ead_mon', 'weight_mon'])

            calibration_drift = _first_nonnull_value(top_segment, ['Calibration_Drift', 'calibration_drift', 'delta_intercept'])
            exposure_drift = _first_nonnull_value(top_segment, ['Exposure_Drift', 'exposure_drift'])

            # Root Cause from feature-level PSI within segment
            root_cause_feature = _first_nonnull_value(
                top_segment,
                ['Root_Cause_Feature', 'top_drift_feature', 'Top_SHAP_Feature', 'top_shap_shift_feature'],
                default='N/A'
            )
            root_cause_score = _first_nonnull_value(
                top_segment,
                ['Root_Cause_Score', 'root_cause_score'],
                default=0.0
            )

            # Population % and Exposure % (top segment)
            pop_pct = _first_nonnull_value(top_segment, ['Mon_Pct', 'pct_mon', 'mon_pct'])
            exp_pct = _first_nonnull_value(top_segment, ['Mon_Exposure_Pct', 'mon_weight_pct', 'exposure_pct'])

            top_segment_definition = _resolve_segment_definition(top_segment)
            num_segments = len(df_seg)
        else:
            top_segment_definition = 'No stable drift segments found'
            max_psi = 0.0
            max_gini_drop = 0.0
            max_ks_drop = 0.0
            max_br_shift = 0.0
            exposure_drift = 0.0
            calibration_drift = 0.0
            root_cause_feature = 'N/A'
            root_cause_score = 0.0
            bus_impact_raw = 0.0
            top_ead = 0.0
            pop_pct = 0.0
            exp_pct = 0.0
            num_segments = 0

        raw_metrics.append({
            'Technique':              tech_name,
            'Top_Segment_Definition': top_segment_definition,
            'Segments_Generated':     num_segments,
            'Top_Segment_Pop_Pct':    round(float(pop_pct) * 100, 2) if not pd.isna(pop_pct) else 0.0,
            'Top_Segment_Exp_Pct':    round(float(exp_pct) * 100, 2) if not pd.isna(exp_pct) else 0.0,
            'Max_PSI':                round(max_psi, 4),
            'Max_Gini_Drop':          round(max_gini_drop, 4),
            'Max_KS_Drop':            round(max_ks_drop, 4),
            'Max_BR_Shift_Pct':       f"{round(max_br_shift * 100, 2)}%",
            '_max_br_shift_raw':      abs(max_br_shift),
            'Exposure_Drift':         round(exposure_drift, 4) if not pd.isna(exposure_drift) else 0.0,
            'Calibration_Drift':      round(calibration_drift, 4) if not pd.isna(calibration_drift) else 0.0,
            'Root_Cause_Feature':     str(root_cause_feature) if root_cause_feature and not pd.isna(root_cause_feature) else 'N/A',
            'Root_Cause_Score':       round(root_cause_score, 4),
            'Business_Impact_Score':  round(bus_impact_raw, 4) if not pd.isna(bus_impact_raw) else 0.0,
            'Top_Segment_EAD':        f"${top_ead:,.0f}" if isinstance(top_ead, (int, float)) and not np.isnan(top_ead) else "N/A",
            'Execution_Time_Sec':     round(exec_time, 4),
            'Deterministic':          TECHNIQUE_PROPERTIES[tech_name]['Deterministic'],
            'Explainability':         TECHNIQUE_PROPERTIES[tech_name]['Explainability_Label'],
        })

    summary_df = pd.DataFrame(raw_metrics)

    summary_df['_norm_psi']        = min_max_normalize(summary_df['Max_PSI'])
    summary_df['_norm_gini']       = min_max_normalize(summary_df['Max_Gini_Drop'])
    summary_df['_norm_ks']         = min_max_normalize(summary_df['Max_KS_Drop'])
    summary_df['_norm_br']         = min_max_normalize(summary_df['_max_br_shift_raw'])
    summary_df['_norm_bus']        = min_max_normalize(summary_df['Business_Impact_Score'].astype(float))

    weights = {
        '_norm_psi':  0.30,   # Population shift sensitivity
        '_norm_gini': 0.25,   # Model discrimination degradation
        '_norm_ks':   0.20,   # KS rank-ordering degradation
        '_norm_br':   0.15,   # Bad-rate drift
        '_norm_bus':  0.10,   # Financial exposure weighting
    }

    summary_df['Overall_Score_100'] = summary_df.apply(
        lambda row: overall_score_100(row, weights), axis=1
    )
    summary_df['Severity_Rank'] = summary_df['Overall_Score_100'].rank(ascending=False, method='min').astype(int)
    summary_df.drop(columns=[c for c in summary_df.columns if c.startswith('_')], inplace=True)

    # Combine all segments with standardized columns
    if all_segments_list:
        combined_segments_df = pd.concat(all_segments_list, ignore_index=True)
        # Compute Root_Cause_Score for every segment
        combined_segments_df = compute_root_cause_scores(combined_segments_df)
        # Ensure Technique is first column
        cols = combined_segments_df.columns.tolist()
        if 'Technique' in cols:
            cols.insert(0, cols.pop(cols.index('Technique')))
            combined_segments_df = combined_segments_df[cols]
    else:
        combined_segments_df = pd.DataFrame()

    # Executive summary (no SHAP analysis — using dev_NEW/mon_2026_01 only)
    exec_summary_df = generate_executive_summary(summary_df, combined_segments_df, None)

    def safe_to_csv(df_to_save, filepath):
        try:
            df_to_save.to_csv(filepath, index=False)
            print(f"  Saved: {filepath}")
        except PermissionError:
            # File is open in Excel/another program — save to alternative filename
            alt_path = str(filepath).replace('.csv', '_latest.csv')
            df_to_save.to_csv(alt_path, index=False)
            print(f"  [WARNING] {filepath} was locked by another process. Saved to {alt_path} instead.")

    # Save all outputs
    safe_to_csv(summary_df, OUTPUT_DIR / 'segmentation_comparison_summary.csv')
    safe_to_csv(combined_segments_df, OUTPUT_DIR / 'all_techniques_segments_output.csv')

    if not exec_summary_df.empty:
        safe_to_csv(exec_summary_df, OUTPUT_DIR / 'executive_summary.csv')

    ranked = summary_df.sort_values('Severity_Rank')
    top_segments_df = ranked[[
        'Severity_Rank', 'Technique', 'Top_Segment_Definition', 'Segments_Generated',
        'Top_Segment_Pop_Pct', 'Top_Segment_Exp_Pct',
        'Max_PSI', 'Max_Gini_Drop', 'Max_KS_Drop', 'Max_BR_Shift_Pct',
        'Exposure_Drift', 'Calibration_Drift', 'Root_Cause_Feature', 'Root_Cause_Score',
        'Business_Impact_Score', 'Top_Segment_EAD', 'Execution_Time_Sec',
        'Deterministic', 'Explainability', 'Overall_Score_100'
    ]].copy()
    safe_to_csv(top_segments_df, OUTPUT_DIR / 'top_technique_segments_summary.csv')

    # ── Print results ──────────────────────────────────────────────────────
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 1200)

    print("\n" + "=" * 115)
    print("DIRECTOR-READY: HEAD-TO-HEAD TECHNIQUE COMPARISON TABLE")
    print("=" * 115)

    display_cols = [
        'Severity_Rank', 'Technique', 'Overall_Score_100',
        'Segments_Generated', 'Max_PSI', 'Max_Gini_Drop', 'Max_KS_Drop',
        'Max_BR_Shift_Pct', 'Exposure_Drift', 'Calibration_Drift',
        'Root_Cause_Feature', 'Root_Cause_Score', 'Business_Impact_Score',
        'Deterministic', 'Explainability', 'Execution_Time_Sec'
    ]
    print(summary_df.sort_values('Severity_Rank')[display_cols].to_string(index=False))

    print("\n" + "=" * 115)
    print("TOP DRIFT SEGMENT IDENTIFIED BY EACH TECHNIQUE")
    print("=" * 115)
    for _, row in ranked.iterrows():
        print(f"\n  [Rank #{row['Severity_Rank']}] {row['Technique']}"
              f"  |  Overall Score: {row['Overall_Score_100']}/100"
              f"  |  Deterministic: {row['Deterministic']}"
              f"  |  Explainability: {row['Explainability']}")
        print(f"    Segment        : {row['Top_Segment_Definition']}")
        print(f"    Population %   : {row['Top_Segment_Pop_Pct']}%  |  Exposure %: {row['Top_Segment_Exp_Pct']}%")
        print(f"    PSI            : {row['Max_PSI']}  |  Gini Drop: {row['Max_Gini_Drop']}"
              f"  |  KS Drop: {row['Max_KS_Drop']}  |  BR Shift: {row['Max_BR_Shift_Pct']}")
        print(f"    Exposure Drift : {row['Exposure_Drift']}  |  Calibration Drift: {row['Calibration_Drift']}")
        print(f"    Root Cause     : Feature={row['Root_Cause_Feature']}  |  Score={row['Root_Cause_Score']}")
        print(f"    EAD            : {row['Top_Segment_EAD']}  |  Segments: {row['Segments_Generated']}  |  Runtime: {row['Execution_Time_Sec']}s")

    # Print executive summary
    if not exec_summary_df.empty:
        print("\n" + "=" * 115)
        print("EXECUTIVE SUMMARY")
        print("=" * 115)
        for section in exec_summary_df['Section'].unique():
            print(f"\n  --- {section} ---")
            sect_rows = exec_summary_df[exec_summary_df['Section'] == section]
            for _, r in sect_rows.iterrows():
                print(f"    {r['Key']:<45}  {r['Value']}")

    winner = ranked.iloc[0]
    print("\n" + "=" * 115)
    print("RECOMMENDATION")
    print("=" * 115)
    print(f"  Best Technique  : {winner['Technique']}  (Overall Score: {winner['Overall_Score_100']}/100, Rank #1)")
    print(f"  Key Strengths   : Max PSI={winner['Max_PSI']}, Max Gini Drop={winner['Max_Gini_Drop']}, "
          f"KS Drop={winner['Max_KS_Drop']}, Deterministic={winner['Deterministic']}")
    print(f"  Notes: Scores normalized across all 5 techniques. 100 = best drift detection. 0 = worst.")
    print("=" * 115)

    return summary_df, combined_segments_df


if __name__ == '__main__':
    benchmark_all_techniques()
