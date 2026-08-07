"""
core/segment_insights.py
========================
Generates SHAP-based root cause analysis, executive summary, and insight narratives
for the worst-performing segments across all segmentation techniques.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from metrics.drift_metrics import detect_shap_shift, interpret_psi
from metrics.business_metrics import calculate_root_cause_score
from utils.logging_config import get_logger

logger = get_logger(__name__)


def compute_root_cause_scores(segments_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Root_Cause_Score for each segment row.
    Uses existing Root_Cause_Score if present, or calculates from Root_Cause_PSI / top_drift_psi.
    """
    df = segments_df.copy()

    if 'Root_Cause_Score' in df.columns and df['Root_Cause_Score'].abs().sum() > 0:
        return df

    # Resolve drift PSI column
    psi_col = next(
        (c for c in ['Root_Cause_PSI', 'top_drift_psi', 'Top_SHAP_PSI', 'top_shap_shift_psi'] if c in df.columns), None
    )
    # Resolve severity / business impact column
    impact_col = next(
        (c for c in ['Severity_Score', 'Business_Impact_Score', 'SIS_Business_Impact', 'business_impact_score']
         if c in df.columns), None
    )

    if psi_col and impact_col:
        df['Root_Cause_Score'] = df.apply(
            lambda row: round(
                (float(row[psi_col]) if not pd.isna(row[psi_col]) else 0.0) *
                (float(row[impact_col]) if not pd.isna(row[impact_col]) else 0.0), 6
            ),
            axis=1,
        )
    elif 'Root_Cause_Score' not in df.columns:
        df['Root_Cause_Score'] = 0.0

    return df


def perform_shap_analysis_worst_segments(
    segments_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    mon_df: pd.DataFrame,
    shap_cols: list[str],
    n_worst: int = 5,
) -> pd.DataFrame:
    """
    For the N worst segments (lowest Business_Impact_Score or highest Severity_Score),
    performs deep SHAP shift analysis and returns a DataFrame with per-feature SHAP details.
    """
    if not shap_cols or segments_df.empty:
        return pd.DataFrame()

    # Rank by severity — pick worst performers
    rank_col = next(
        (c for c in ['Severity_Score', 'Business_Impact_Score', 'SIS_Raw'] if c in segments_df.columns),
        None
    )
    if rank_col is None:
        worst_segs = segments_df.head(n_worst)
    else:
        ascending = rank_col == 'Business_Impact_Score'  # lower is worse for this
        worst_segs = segments_df.nlargest(n_worst, rank_col)

    rows = []
    for idx, seg_row in worst_segs.iterrows():
        seg_def = next(
            (seg_row[c] for c in ['Segment_Definition', 'segment_definition', 'rule_text'] if c in seg_row and not pd.isna(seg_row[c])),
            f"Segment {idx}"
        )
        technique = seg_row.get('Technique', 'Unknown')

        shap_result = detect_shap_shift(dev_df, mon_df, shap_cols)
        for detail in shap_result.get('details', []):
            rows.append({
                'Segment_Definition': seg_def,
                'Technique': technique,
                'SHAP_Feature': detail['feature'],
                'Dev_Mean_SHAP': round(detail['dev_mean'], 6) if not np.isnan(detail['dev_mean']) else None,
                'Mon_Mean_SHAP': round(detail['mon_mean'], 6) if not np.isnan(detail['mon_mean']) else None,
                'SHAP_Delta': round(detail['delta'], 6) if not np.isnan(detail['delta']) else None,
                'SHAP_PSI': round(detail['psi'], 6),
                'SHAP_Interpretation': detail['interpretation'],
                'Is_Top_Driver': detail['feature'] == shap_result.get('top_shift_feature'),
            })

    return pd.DataFrame(rows)


def generate_executive_summary(
    summary_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    shap_analysis_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Generates a structured executive summary with:
    - Overall portfolio health
    - Top 5 worst segments ranked by Severity_Score (not Business_Impact_Score)
    - Root cause features identified via feature-level PSI
    - Key recommendations
    Returns a DataFrame for CSV export.
    """
    if summary_df.empty:
        return pd.DataFrame()

    ranked = summary_df.sort_values('Severity_Rank')
    best_tech = ranked.iloc[0]['Technique']
    best_score = ranked.iloc[0]['Overall_Score_100']

    # Rank worst segments: Severity_Score first, then by absolute Gini Drop
    seg_rank_col = next(
        (c for c in ['Severity_Score', 'Business_Impact_Score', 'Root_Cause_Score'] if c in segments_df.columns),
        None
    )
    worst_segs = []
    if seg_rank_col and not segments_df.empty:
        # Prefer Severity_Score; compute absolute Gini Drop for tie-breaking
        _df = segments_df.copy()
        if 'Delta_Gini' in _df.columns:
            _df['_abs_gini_drop'] = _df['Delta_Gini'].apply(
                lambda x: abs(x) if (not pd.isna(x) and x < 0) else 0.0
            )
        else:
            _df['_abs_gini_drop'] = 0.0
        sort_cols = [col for col in [seg_rank_col, '_abs_gini_drop'] if col in _df.columns]
        top_n = _df.nlargest(5, sort_cols[0]) if sort_cols else _df.head(5)

        for _, r in top_n.iterrows():
            seg_def = next(
                (r[c] for c in ['Segment_Definition', 'segment_definition', 'segment'] if c in r and not pd.isna(r[c])),
                'Unknown'
            )
            root_feat = next(
                (str(r[c]) for c in ['Root_Cause_Feature', 'top_drift_feature', 'Top_SHAP_Feature']
                 if c in r and not pd.isna(r[c]) and str(r[c]) not in ('None', 'nan', 'N/A')),
                'N/A'
            )
            worst_segs.append({
                'Segment': seg_def,
                'Technique': r.get('Technique', 'Unknown'),
                'Severity_Score': r.get(seg_rank_col, 0.0),
                'Gini_Drop': r.get('Delta_Gini', r.get('delta_gini', np.nan)),
                'Exposure_Drift': r.get('Exposure_Drift', r.get('exposure_drift', np.nan)),
                'Calibration_Drift': r.get('Calibration_Drift', r.get('calibration_drift', np.nan)),
                'Root_Cause_Feature': root_feat,
                'Root_Cause_Score': r.get('Root_Cause_Score', r.get('root_cause_score', 0.0)),
                'Root_Cause_PSI': r.get('Root_Cause_PSI', r.get('top_drift_psi', np.nan)),
            })

    # Aggregate root cause features across all segments
    rc_feature_col = next(
        (c for c in ['Root_Cause_Feature', 'top_drift_feature'] if c in segments_df.columns), None
    )
    top_rc_features = []
    if rc_feature_col and not segments_df.empty:
        valid_rc = segments_df[rc_feature_col].dropna()
        valid_rc = valid_rc[~valid_rc.isin(['None', 'N/A', 'nan'])]
        if not valid_rc.empty:
            top_rc_features = valid_rc.value_counts().head(3).index.tolist()

    # Overall health assessment
    max_gini_drop = summary_df['Max_Gini_Drop'].max() if 'Max_Gini_Drop' in summary_df.columns else 0
    max_psi = summary_df['Max_PSI'].max() if 'Max_PSI' in summary_df.columns else 0
    max_cal = summary_df['Calibration_Drift'].abs().max() if 'Calibration_Drift' in summary_df.columns else 0

    if max_gini_drop > 0.20 or max_psi > 0.25 or max_cal > 0.20:
        health = "CRITICAL - Significant model degradation and/or population drift detected."
    elif max_gini_drop > 0.10 or max_psi > 0.10 or max_cal > 0.10:
        health = "WARNING - Moderate drift detected. Monitoring escalation recommended."
    else:
        health = "STABLE - Drift signals within acceptable thresholds."

    summary_rows = []
    summary_rows.append({'Section': 'PORTFOLIO HEALTH', 'Key': 'Overall Status', 'Value': health})
    summary_rows.append({'Section': 'PORTFOLIO HEALTH', 'Key': 'Best Technique (Drift Detection)', 'Value': f"{best_tech} (Score: {best_score}/100)"})
    summary_rows.append({'Section': 'PORTFOLIO HEALTH', 'Key': 'Max Gini Drop Detected', 'Value': f"{max_gini_drop:.4f}"})
    summary_rows.append({'Section': 'PORTFOLIO HEALTH', 'Key': 'Max PSI Detected', 'Value': f"{max_psi:.4f}"})
    summary_rows.append({'Section': 'PORTFOLIO HEALTH', 'Key': 'Max Calibration Drift', 'Value': f"{max_cal:.4f}"})
    summary_rows.append({'Section': 'PORTFOLIO HEALTH', 'Key': 'Total Segments Analyzed', 'Value': str(len(segments_df))})

    for i, seg in enumerate(worst_segs, 1):
        summary_rows.append({
            'Section': 'TOP WORST SEGMENTS',
            'Key': f"Rank {i}: {seg['Technique']}",
            'Value': seg['Segment'],
        })
        summary_rows.append({
            'Section': 'TOP WORST SEGMENTS',
            'Key': f"  -> Gini Drop",
            'Value': f"{seg['Gini_Drop']:.4f}" if not pd.isna(seg.get('Gini_Drop', np.nan)) else 'N/A',
        })
        summary_rows.append({
            'Section': 'TOP WORST SEGMENTS',
            'Key': f"  -> Exposure Drift",
            'Value': f"{seg['Exposure_Drift']:.4f}" if not pd.isna(seg.get('Exposure_Drift', np.nan)) else 'N/A',
        })
        summary_rows.append({
            'Section': 'TOP WORST SEGMENTS',
            'Key': f"  -> Calibration Drift",
            'Value': f"{seg['Calibration_Drift']:.4f}" if not pd.isna(seg.get('Calibration_Drift', np.nan)) else 'N/A',
        })
        summary_rows.append({
            'Section': 'TOP WORST SEGMENTS',
            'Key': f"  -> Root Cause Feature",
            'Value': str(seg['Root_Cause_Feature']),
        })
        summary_rows.append({
            'Section': 'TOP WORST SEGMENTS',
            'Key': f"  -> Root Cause Score",
            'Value': f"{seg['Root_Cause_Score']:.4f}" if not pd.isna(seg.get('Root_Cause_Score', np.nan)) else 'N/A',
        })

    if top_rc_features:
        summary_rows.append({
            'Section': 'ROOT CAUSE ANALYSIS',
            'Key': 'Top Drifting Features (across all segments)',
            'Value': ', '.join(top_rc_features),
        })
    elif shap_analysis_df is not None and not shap_analysis_df.empty:
        top_drivers = shap_analysis_df[shap_analysis_df.get('Is_Top_Driver', False) == True]
        feature_freq = top_drivers['SHAP_Feature'].value_counts()
        top_shap = feature_freq.head(3).index.tolist()
        summary_rows.append({
            'Section': 'ROOT CAUSE ANALYSIS',
            'Key': 'Top Contributing Features (SHAP)',
            'Value': ', '.join(top_shap) if top_shap else 'N/A',
        })
    else:
        summary_rows.append({
            'Section': 'ROOT CAUSE ANALYSIS',
            'Key': 'Top Drifting Features (Feature PSI)',
            'Value': 'No significant feature drift identified in segment subsets',
        })

    # Recommendations
    top_seg_def = worst_segs[0]['Segment'] if worst_segs else 'N/A'
    top_rc_feat = worst_segs[0]['Root_Cause_Feature'] if worst_segs else 'N/A'
    summary_rows.append({'Section': 'RECOMMENDATIONS', 'Key': '1. Immediate Action',
                          'Value': f"Investigate '{top_seg_def}' - highest severity score"})
    summary_rows.append({'Section': 'RECOMMENDATIONS', 'Key': '2. Root Cause Investigation',
                          'Value': f"Focus on feature '{top_rc_feat}' - highest drift within worst segment"})
    summary_rows.append({'Section': 'RECOMMENDATIONS', 'Key': '3. Recalibration',
                          'Value': 'Consider model recalibration for segments with calibration drift > 0.05'})
    summary_rows.append({'Section': 'RECOMMENDATIONS', 'Key': '4. Monitoring Thresholds',
                          'Value': 'Set up automated alerts for PSI > 0.10 and Gini Drop > 0.10'})
    summary_rows.append({'Section': 'RECOMMENDATIONS', 'Key': '5. Best Technique',
                          'Value': f"Use {best_tech} for ongoing monitoring - highest sensitivity"})

    return pd.DataFrame(summary_rows)

