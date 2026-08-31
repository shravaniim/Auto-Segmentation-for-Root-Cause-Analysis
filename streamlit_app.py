import streamlit as st
import pandas as pd
from pathlib import Path

from models.config import (
    SlicerConfig,
    DLTConfig,
    GBConfig,
    KMeansConfig,
    FeatureBinningConfig,
    SchemaConfig,
    TrendAnalysisConfig,
)
from core.multi_period_analysis import (
    compute_trend_metrics,
    build_segment_time_series,
    forecast_segment_scores,
    reindex_segment_time_series,
)
from core.cross_technique_analysis import build_cross_technique_top10

from autoslicer_segmentation import run_autoslicer_segmentation
from kmeans_segmentation import run_kmeans_segmentation
from drift_localization_tree import run_drift_localization
from feature_binning_segmentation import run_feature_binning_segmentation
from gradient_boosting_segmentation import run_gradient_boosting_segmentation

from compare_segmentation_techniques import (
    benchmark_all_techniques,
    build_feature_schema,
    standardize_columns,
    DEV_FILE,
    MON_FILE,
    REQUESTED_FEATURES,
    DATA_DIR,
)

from llm.insight_generator import generate_insight, answer_data_question
from utils.charts import (
    segment_metric_heatmap,
    segment_bubble_chart,
    sis_waterfall_chart,
    segment_metric_time_series_chart,
    segment_all_metrics_chart,
    early_warning_urgency_chart,
)


# ============================================================
# Technique Parameter Registry
# ============================================================
#
# Approach 1 is fixed to development_data_5000_shap.csv /
# monitoring_data_5000_shap.csv. Every technique's search space is
# restricted (via build_feature_schema) to age/income/region/occupation.
# target, score, ead, shap_*, customer_id are never used as
# segmentation features -- only as metric/exposure inputs.
#
# "Decision Tree" is the user-facing label for the Gradient Boosting
# technique (a tree-ensemble implementation already in techniques/).

PARAM_SPECS = {
    "AutoSlicer": {
        "runner": run_autoslicer_segmentation,
        "config_cls": SlicerConfig,
        "rule_length_param": "max_combo_depth",
        "rule_length_default": 3,
        "rule_length_range": (1, 6),
        "percentile_param": "numeric_bins",
        "percentile_label": "Numeric Percentile Bins",
        "percentile_kind": "slider",
        "percentile_default": 4,
        "percentile_range": (2, 10),
        "auto_grid": {"max_combo_depth": [2, 3, 4], "beam_width": [10, 20, 30], "numeric_bins": [3, 4, 6]},
    },
    "Feature Binning": {
        "runner": run_feature_binning_segmentation,
        "config_cls": FeatureBinningConfig,
        "rule_length_param": None,
        "percentile_param": "max_bins",
        "percentile_label": "Bin Granularity (Max Bins)",
        "percentile_kind": "slider",
        "percentile_default": 8,
        "percentile_range": (3, 15),
        "auto_grid": {"max_bins": [5, 8, 10, 12], "min_bin_pct": [0.01, 0.02, 0.03]},
    },
    "Decision Tree": {
        "runner": run_gradient_boosting_segmentation,
        "config_cls": GBConfig,
        "rule_length_param": "max_depth",
        "rule_length_default": 3,
        "rule_length_range": (1, 6),
        "percentile_param": "drift_quantiles",
        "percentile_label": "Drift Scan Percentiles",
        "percentile_kind": "multiselect",
        "percentile_options": [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99],
        "percentile_default": [0.70, 0.85, 0.95],
        "auto_grid": {
            "max_depth": [2, 3, 4],
            "n_estimators": [50, 100, 150],
            "drift_quantiles": [(0.70, 0.85, 0.95), (0.60, 0.80, 0.95), (0.75, 0.90, 0.99)],
        },
    },
    "Clustering": {
        "runner": run_kmeans_segmentation,
        "config_cls": KMeansConfig,
        "rule_length_param": "max_tree_depth",
        "rule_length_default": 4,
        "rule_length_range": (1, 6),
        "percentile_param": None,
        "auto_grid": {"drift_weight": [0.2, 0.4, 0.6], "max_tree_depth": [3, 4, 5]},
    },
    "Drift Localization Tree": {
        "runner": run_drift_localization,
        "config_cls": DLTConfig,
        "rule_length_param": "max_depth",
        "rule_length_default": 3,
        "rule_length_range": (1, 6),
        "percentile_param": None,
        "auto_grid": {"max_depth": [2, 3, 4]},
    },
}


def build_config(tech, min_segment_size, ui_values, auto_optimize, dev_df, schema_cfg):
    """Translate user inputs into the technique's real Config dataclass."""
    spec = PARAM_SPECS[tech]
    kwargs = {"schema": schema_cfg, "min_abs_count": int(min_segment_size)}

    if tech == "Clustering":
        kwargs["min_cluster_pct"] = round(max(0.01, min_segment_size / len(dev_df)), 4)

    if tech == "Feature Binning":
        # FeatureBinningConfig has no min_abs_count field; its equivalent
        # minimum-segment-size gate is min_bin_pct (fraction of population).
        kwargs.pop("min_abs_count")
        kwargs["min_bin_pct"] = round(max(0.001, min_segment_size / len(dev_df)), 4)

    if auto_optimize:
        if spec["auto_grid"]:
            kwargs["param_grid"] = spec["auto_grid"]
    else:
        if spec["rule_length_param"]:
            kwargs[spec["rule_length_param"]] = ui_values["rule_length"]

        if spec["percentile_param"] == "drift_quantiles":
            values = ui_values.get("percentiles") or spec["percentile_default"]
            kwargs["drift_quantiles"] = tuple(sorted(values))
        elif spec["percentile_param"]:
            kwargs[spec["percentile_param"]] = ui_values["percentile_value"]

    return spec["config_cls"](**kwargs)


def render_technique_results(tech, result):
    st.subheader(f"📊 Execution Summary — {tech}")

    overall = result.get("overall", {})
    for key, value in overall.items():
        st.write(f"**{key}** : {value}")

    if "execution_time" in result:
        st.write("**Execution Time** :", round(result["execution_time"], 2), "seconds")

    if result.get("selected_params"):
        st.write(
            f"**Auto-selected parameters** (best of {result.get('params_evaluated', '?')} "
            f"combinations, optimization score {result.get('optimization_score', 0):.4f}):"
        )
        st.json(result["selected_params"])

    segments_df = result.get("segments", pd.DataFrame())

    st.subheader(f"📋 Top Segments — {tech}")
    st.dataframe(segments_df, use_container_width=True)

    if not segments_df.empty:
        st.download_button(
            label="📥 Download Results CSV",
            data=segments_df.to_csv(index=False),
            file_name=f"{tech.replace(' ', '_')}_results.csv",
            mime="text/csv",
            key=f"download_{tech}",
        )

        chart_df = standardize_columns(segments_df.copy(), tech)

        st.subheader(f"🌡️ Segment x Metric Heatmap — {tech}")
        fig = segment_metric_heatmap(chart_df)
        if fig is not None:
            st.pyplot(fig)

        st.subheader(f"🫧 Bubble Chart — {tech}")
        fig = segment_bubble_chart(chart_df)
        if fig is not None:
            st.pyplot(fig)

        st.subheader(f"💧 SIS Waterfall — {tech} (Top Segment)")
        fig = sis_waterfall_chart(chart_df.iloc[0])
        if fig is not None:
            st.pyplot(fig)

        try:
            top_segment = segments_df.iloc[0]

            segment_info = {
                "Segment_Definition": str(
                    top_segment.get("Segment_Definition", top_segment.get("segment", "Unknown Segment"))
                ),
                "PSI": float(top_segment.get("PSI", top_segment.get("psi", 0))),
                "Delta_Gini": float(top_segment.get("Delta_Gini", top_segment.get("delta_gini", 0))),
                "Delta_BR": float(top_segment.get("Delta_BR", 0)),
                "Root_Cause_Feature": str(top_segment.get("Root_Cause_Feature", "Not Available")),
                "Severity_Score": float(
                    top_segment.get("Severity_Score", top_segment.get("final_score", 0))
                ),
                "Business_Impact_Score": float(top_segment.get("Business_Impact_Score", 0)),
            }

            with st.spinner("Generating AI Executive Summary..."):
                llm_summary = generate_insight(segment_info)

            st.subheader(f"🤖 AI Generated Executive Summary — {tech}")
            st.markdown(llm_summary)

        except Exception as e:
            st.warning(f"Could not generate LLM summary: {e}")


# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title="Auto Segmentation Framework",
    layout="wide"
)

st.title("Auto-Segmentation Framework")


# ============================================================
# Tabs
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "Segmentation Analysis",
        "Technique Comparison",
        "Cross-Technique Insights",
    ]
)

# ============================================================
# TAB 1
# ============================================================

with tab1:

    st.header("Segmentation Analysis")

    dev_df = pd.read_csv(DEV_FILE)
    mon_df = pd.read_csv(MON_FILE)
    schema_cfg = build_feature_schema(dev_df)

    st.caption(
        f"Development: `{Path(DEV_FILE).name}` ({len(dev_df):,} rows) · "
        f"Monitoring: `{Path(MON_FILE).name}` ({len(mon_df):,} rows). "
        f"Segmentation features: **{', '.join(REQUESTED_FEATURES)}**. "
        "`target`, `score`, `ead`, `shap_*` and `customer_id` are excluded from "
        "segment rule discovery for every technique below — they are only used "
        "to compute performance, drift and exposure metrics."
    )

    selected_techniques = st.multiselect(
        "1. Segmentation Technique(s)",
        list(PARAM_SPECS.keys()),
        default=["AutoSlicer"],
    )

    min_segment_size = st.number_input(
        "2. Minimum Segment Size (rows)",
        min_value=10,
        max_value=int(len(dev_df)),
        value=150,
        step=10,
        help="Candidate segments with fewer development-period rows than this "
             "are discarded, regardless of technique.",
    )

    auto_optimize = st.checkbox(
        "Auto-generate optimal values for Max Rule Length / Percentiles",
        value=False,
        help="Runs a grid search (core.parameter_optimization) over each "
             "technique's search-space parameters and keeps the combination "
             "with the highest aggregate Business Impact Score, instead of "
             "using the manual values set below.",
    )

    ui_values = {}

    for technique in selected_techniques:
        spec = PARAM_SPECS[technique]
        with st.expander(f"{technique} Parameters", expanded=True):
            values = {}

            if auto_optimize:
                st.info("Max Rule Length / Percentiles will be auto-selected via grid search.")
            else:
                if spec["rule_length_param"]:
                    lo, hi = spec["rule_length_range"]
                    values["rule_length"] = st.slider(
                        "Max Rule Length",
                        min_value=lo,
                        max_value=hi,
                        value=spec["rule_length_default"],
                        key=f"rule_{technique}",
                    )
                else:
                    st.caption("Max Rule Length — not applicable (single-feature segments only).")

                if spec["percentile_param"] == "drift_quantiles":
                    values["percentiles"] = st.multiselect(
                        spec["percentile_label"],
                        spec["percentile_options"],
                        default=spec["percentile_default"],
                        key=f"perc_{technique}",
                    )
                elif spec["percentile_param"]:
                    lo, hi = spec["percentile_range"]
                    values["percentile_value"] = st.slider(
                        spec["percentile_label"],
                        min_value=lo,
                        max_value=hi,
                        value=spec["percentile_default"],
                        key=f"perc_{technique}",
                    )
                else:
                    st.caption("Percentiles — not applicable for this technique.")

            ui_values[technique] = values

    if st.button("Run Segment Analysis"):

        if not selected_techniques:
            st.error("Select at least one segmentation technique.")
            st.stop()

        for technique in selected_techniques:
            spec = PARAM_SPECS[technique]

            cfg = build_config(
                technique,
                min_segment_size,
                ui_values.get(technique, {}),
                auto_optimize,
                dev_df,
                schema_cfg,
            )

            with st.spinner(f"Running {technique}..."):
                result = spec["runner"](dev_df, mon_df, cfg=cfg)

            st.success(f"{technique} Completed Successfully")

            render_technique_results(technique, result)

            st.divider()


# ============================================================
# TAB 2
# ============================================================

def _build_schema_for_dev_file(dev_df: pd.DataFrame) -> SchemaConfig:
    """Fully automatic SchemaConfig for whatever dev file the user picked in
    trend mode -- numeric/categorical/target/score/id/time all inferred
    from the data by utils.schema_detection.detect_schema, so a dataset
    that later gets new columns (e.g. v2_1M added marital_status/channel/
    product_type/sanctioned_amount on top of the original v2 schema) is
    picked up automatically, with no code change needed here.

    Two manual overrides on top of the generic detection, both found by
    testing against the real v2_1M data:

    - weight_col is pinned to "ead" when present. detect_schema's
      weight/exposure pattern matches any column *containing* "amount" --
      v2_1M's new sanctioned_amount column matches it and, because it sits
      earlier in the column order than ead, wins the auto-detection,
      silently using the wrong exposure basis for every EAD-weighted
      metric (Business_Impact_Score, Mon_Exposure_Pct, etc.).
    - lgd_actual (loss given default) is excluded even though its name
      matches no target/score pattern -- it's a model-output field in the
      same leakage category as target/score, just not named obviously
      enough for the generic detector to catch on its own.
    """
    from utils.schema_detection import detect_schema

    cols = set(dev_df.columns)
    seed_cfg = SchemaConfig(weight_col="ead") if "ead" in cols else SchemaConfig()
    detected = detect_schema(dev_df, seed_cfg)

    exclude_cols = list(detected["excluded_cols"])
    numeric_cols = list(detected["numeric_cols"])
    categorical_cols = list(detected["categorical_cols"])
    if "lgd_actual" in cols and "lgd_actual" not in exclude_cols:
        exclude_cols.append("lgd_actual")
        numeric_cols = [c for c in numeric_cols if c != "lgd_actual"]
        categorical_cols = [c for c in categorical_cols if c != "lgd_actual"]

    return SchemaConfig(
        target_col=detected["target_col"],
        score_col=detected["score_col"],
        weight_col=detected["weight_col"],
        id_cols=detected["id_cols"],
        exclude_cols=exclude_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )


def _build_qa_context() -> str:
    """Compact text summary of whatever run results are currently in
    session_state -- classic and/or trend, whichever the user last ran --
    for the free-form Q&A assistant. Kept small (top rows only) to stay
    within a reasonable prompt size."""
    parts = []

    summary_df = st.session_state.get("summary_df", pd.DataFrame())
    if not summary_df.empty:
        cols = [c for c in ["Technique", "Overall_Score_100", "Max_PSI", "Max_Gini_Drop",
                             "Root_Cause_Feature", "Root_Cause_Score"] if c in summary_df.columns]
        parts.append("=== Technique Comparison (classic run) ===\n" + summary_df[cols].to_string(index=False))

    cross_top10_df = st.session_state.get("cross_top10_df", pd.DataFrame())
    if not cross_top10_df.empty:
        cols = [c for c in ["Overall_Rank", "Technique", "Segment_Definition",
                             "Normalized_Root_Cause_Score"] if c in cross_top10_df.columns]
        parts.append("=== Cross-Technique Top 10 (classic run) ===\n" + cross_top10_df[cols].to_string(index=False))

    trend_df = st.session_state.get("trend_df", pd.DataFrame())
    if not trend_df.empty:
        cols = [c for c in ["Technique", "Segment_Definition", "Periods_Appeared", "Total_Periods",
                             "Frequency", "Consistency_Score", "Trend_Impact_Score",
                             "Severity_Score_Trend"] if c in trend_df.columns]
        top = trend_df.nlargest(15, "Severity_Score_Trend")[cols]
        parts.append("=== Trend Analysis Summary (top 15 by Severity_Score_Trend) ===\n" + top.to_string(index=False))

    trend_cross_top10_df = st.session_state.get("trend_cross_top10_df", pd.DataFrame())
    if not trend_cross_top10_df.empty:
        cols = [c for c in ["Overall_Rank", "Technique", "Segment_Definition",
                             "Normalized_Root_Cause_Score"] if c in trend_cross_top10_df.columns]
        parts.append("=== Cross-Technique Trend Top 10 ===\n" + trend_cross_top10_df[cols].to_string(index=False))

    time_series_df = st.session_state.get("trend_time_series_df", pd.DataFrame())
    if not time_series_df.empty:
        forecast_df = forecast_segment_scores(time_series_df, TrendAnalysisConfig())
        if not forecast_df.empty:
            parts.append("=== Early Warning Forecast ===\n" + forecast_df.to_string(index=False))

    return "\n\n".join(parts) if parts else ""


with tab2:

    st.header("🏆 Compare All Segmentation Techniques")

    trend_enabled = st.radio(
        "Run Trend Analysis (multiple monitoring months)?",
        options=["No", "Yes"],
        horizontal=True,
        index=0,
        key="trend_analysis_toggle",
        help="No (default): today's single dev-vs-monitoring comparison, unchanged. "
             "Yes: supply 1 development file + several monitoring files (one per month) "
             "to additionally see which segments recur as worst performers over time.",
    ) == "Yes"

    if trend_enabled:

        csv_files = sorted(p.name for p in DATA_DIR.glob("*.csv"))
        dev_like = [f for f in csv_files if "dev" in f.lower()]
        mon_like = [f for f in csv_files if "mon" in f.lower()]
        v2_mon_defaults = sorted(f for f in mon_like if f.lower().startswith("v2_monitoring"))

        trend_dev_file = st.selectbox(
            "Development dataset",
            options=dev_like or csv_files,
            index=(dev_like or csv_files).index("v2_development_data.csv")
            if "v2_development_data.csv" in (dev_like or csv_files) else 0,
        )
        trend_mon_files = st.multiselect(
            "Monitoring datasets, in chronological order (one per month)",
            options=mon_like or csv_files,
            default=v2_mon_defaults,
        )

        run_trend_clicked = st.button("Run Trend Analysis", disabled=len(trend_mon_files) < 2)
        if len(trend_mon_files) == 1:
            st.caption("Pick at least 2 monitoring months to compute recurrence trends.")

        if run_trend_clicked:
            dev_df = pd.read_csv(DATA_DIR / trend_dev_file)
            trend_schema = _build_schema_for_dev_file(dev_df)

            period_results = {}
            progress = st.progress(0.0, text="Starting trend analysis...")
            for i, mon_file in enumerate(trend_mon_files):
                progress.progress(
                    i / len(trend_mon_files),
                    text=f"Running all 5 techniques for {mon_file} ({i + 1}/{len(trend_mon_files)})...",
                )
                mon_df = pd.read_csv(DATA_DIR / mon_file)
                _, combined_segments_df, _, _ = benchmark_all_techniques(
                    dev_df=dev_df, mon_df=mon_df, save_outputs=False, schema_cfg=trend_schema,
                )
                period_results[mon_file] = combined_segments_df
            progress.progress(1.0, text="Computing trend metrics...")

            trend_df = compute_trend_metrics(period_results, TrendAnalysisConfig())
            progress.empty()

            # Everything below is *display*, which must not live inside this
            # `if run_trend_clicked:` block: st.button() only returns True on
            # the exact rerun where it was clicked, so on the very next
            # rerun (e.g. the user touching the segment picker below) it
            # reverts to False and this whole block would be skipped --
            # making the summary table disappear. Storing results in
            # session_state and rendering them separately (same pattern
            # Tab 3 already uses for cross_top10_df) keeps them visible
            # across unrelated widget interactions.
            st.session_state["trend_df"] = trend_df
            st.session_state["trend_months_count"] = len(trend_mon_files)
            st.session_state["trend_periods"] = list(period_results.keys())

            if trend_df.empty:
                st.session_state["trend_cross_top10_df"] = pd.DataFrame()
                st.session_state["trend_time_series_df"] = pd.DataFrame()
            else:
                # Cross-technique step (taskflow step 4, trend version): pool
                # each technique's worst trend segments and normalize
                # Root_Cause_Score_Trend across all of them, same as the
                # classic cross-technique top-10 but on trend-boosted scores.
                # build_cross_technique_top10 looks for columns literally
                # named Severity_Score/Root_Cause_Score, so the trend-boosted
                # values are substituted in on a copy before calling it.
                trend_for_cross = trend_df.copy()
                trend_for_cross["Severity_Score"] = trend_for_cross["Severity_Score_Trend"]
                trend_for_cross["Root_Cause_Score"] = trend_for_cross["Root_Cause_Score_Trend"]
                st.session_state["trend_cross_top10_df"] = build_cross_technique_top10(trend_for_cross)

                # Add-on requested by the manager: the raw month-by-month
                # metrics (not just the trend-boosted summary), for whichever
                # segment the user wants to inspect.
                st.session_state["trend_time_series_df"] = build_segment_time_series(
                    period_results, TrendAnalysisConfig()
                )

        # ====================================================
        # Display (reads from session_state, not run_trend_clicked, so it
        # survives reruns triggered by widgets below it -- e.g. the segment
        # picker further down).
        # ====================================================

        trend_df = st.session_state.get("trend_df", pd.DataFrame())

        if trend_df.empty:
            if run_trend_clicked:
                st.warning("No segments to report -- check that the selected files produced valid candidates.")
        else:
            st.success(
                f"Trend Analysis Completed Successfully "
                f"({st.session_state.get('trend_months_count', '?')} months, {trend_df['Technique'].nunique()} techniques)"
            )

            st.subheader("📈 Trend Analysis Summary")
            st.caption(
                "One row per segment matched across the selected months (by rule text) within its "
                "own technique. Frequency = share of months it ranked in that technique's worst-10. "
                "Recency_Factor = how recently it last appeared (1.0 = most recent month). "
                "Consistency_Score = longest unbroken streak of months it appeared in, divided by "
                "total months -- a segment that appears then disappears then reappears scores lower "
                "here than one that stayed continuously present, even at the same overall frequency. "
                "Trend_Impact_Score blends the three; SIS_Trend/Root_Cause_Score_Trend/"
                "Severity_Score_Trend are the existing formulas with SIS boosted by that trend score."
            )

            display_cols = [
                "Technique", "Segment_Definition", "Periods_Appeared", "Total_Periods",
                "Appeared_In", "Frequency", "Recency_Factor", "Consistency_Score",
                "Trend_Impact_Score", "SIS_Raw", "SIS_Trend", "Root_Cause_Score_Trend",
                "Severity_Score_Trend",
            ]
            st.dataframe(trend_df[display_cols], use_container_width=True)
            st.download_button(
                label="📥 Download Trend Analysis CSV",
                data=trend_df.to_csv(index=False),
                file_name="trend_analysis_summary.csv",
                mime="text/csv",
                key="download_trend_analysis",
            )

            trend_top10 = trend_df.nlargest(10, "Severity_Score_Trend")

            st.subheader("🌡️ Segment x Metric Heatmap (Top 10 by Severity_Score_Trend)")
            fig = segment_metric_heatmap(trend_top10)
            if fig is not None:
                st.pyplot(fig)

            st.subheader("🫧 Bubble Chart — Population vs Gini Drop (size = exposure)")
            fig = segment_bubble_chart(trend_top10)
            if fig is not None:
                st.pyplot(fig)

            st.subheader("💧 SIS Waterfall — #1 Most Persistent Root Cause")
            fig = sis_waterfall_chart(trend_top10.iloc[0])
            if fig is not None:
                st.pyplot(fig)

            st.subheader("🏅 Most Persistent Root Causes (Top 5 by Severity_Score_Trend)")
            top5 = trend_df.nlargest(5, "Severity_Score_Trend")
            for _, r in top5.iterrows():
                st.write(
                    f"**{r['Technique']}** — {r['Segment_Definition']}  \n"
                    f"Appeared in {r['Periods_Appeared']}/{r['Total_Periods']} months "
                    f"({r['Appeared_In']}) · Trend Impact Score {r['Trend_Impact_Score']:.2f} · "
                    f"Severity_Score_Trend {r['Severity_Score_Trend']:.2f}"
                )

        # ====================================================
        # Metric Time Series (add-on, requested by manager) -- also reads
        # from session_state for the same reason as above.
        # ====================================================

        time_series_df = st.session_state.get("trend_time_series_df", pd.DataFrame())
        forecast_df = forecast_segment_scores(time_series_df, TrendAnalysisConfig()) if not time_series_df.empty else pd.DataFrame()

        if not time_series_df.empty:
            st.subheader("📉 Metric Time Series (per segment)")
            st.caption(
                "Add-on to the summary table above (unchanged) -- pick one segment to see its "
                "actual month-by-month numbers (population %, AUC, KS, bad rate, SIS, DIS, "
                "Root_Cause_Score) across *every* month analyzed, not just the single "
                "trend-boosted score. Months where the segment didn't rank in that technique's "
                "worst-10 show as a blank row / a gap in the line -- not skipped or interpolated -- "
                "so the recurrence pattern (continuous vs. on-and-off) is visible at a glance."
            )

            segment_options = (
                time_series_df[["Technique", "Segment_Definition"]]
                .drop_duplicates()
                .apply(lambda r: f"{r['Technique']} — {r['Segment_Definition']}", axis=1)
                .tolist()
            )
            selected = st.selectbox(
                "Select a segment to view its time series",
                options=["-- Select a segment --"] + sorted(segment_options),
                index=0,
                key="time_series_segment_picker",
            )

            if selected != "-- Select a segment --":
                sel_tech, sel_seg = selected.split(" — ", 1)
                seg_ts = time_series_df[
                    (time_series_df["Technique"] == sel_tech)
                    & (time_series_df["Segment_Definition"] == sel_seg)
                ].sort_values("Period_Index")

                all_periods = st.session_state.get("trend_periods", [])
                seg_ts_full = reindex_segment_time_series(seg_ts, all_periods) if all_periods else seg_ts

                display_ts = seg_ts_full.drop(columns=["Period_Index"], errors="ignore").copy()
                if "Root_Cause_Score" in display_ts.columns:
                    display_ts.insert(
                        1, "Appeared_This_Month",
                        display_ts["Root_Cause_Score"].notna().map({True: "✓", False: "—"}),
                    )
                st.dataframe(display_ts, use_container_width=True)

                forecast_point = None
                if not forecast_df.empty:
                    match = forecast_df[
                        (forecast_df["Technique"] == sel_tech)
                        & (forecast_df["Segment_Definition"] == sel_seg)
                    ]
                    if not match.empty:
                        forecast_point = float(match.iloc[0]["Predicted_Next_Root_Cause_Score"])

                st.markdown("**All key metrics together, full month range:**")
                fig = segment_all_metrics_chart(seg_ts_full, sel_seg)
                if fig is not None:
                    st.pyplot(fig)
                else:
                    st.info("Not enough numeric data across the selected months to plot this segment.")

                col1, col2 = st.columns(2)
                with col1:
                    fig = segment_metric_time_series_chart(seg_ts_full, "Root_Cause_Score", sel_seg, forecast_point=forecast_point)
                    if fig is not None:
                        st.pyplot(fig)
                with col2:
                    fig = segment_metric_time_series_chart(seg_ts_full, "Mon_AUC", sel_seg)
                    if fig is not None:
                        st.pyplot(fig)

            st.download_button(
                label="📥 Download Full Time-Series CSV (all segments)",
                data=time_series_df.to_csv(index=False),
                file_name="trend_metric_time_series.csv",
                mime="text/csv",
                key="download_trend_time_series",
            )

        # ====================================================
        # Early Warning (new feature): plain linear extrapolation of each
        # segment's Root_Cause_Score, flagging ones on track to cross the
        # alert threshold soon. Pure math on already-computed
        # trend_time_series_df -- no new pipeline runs.
        # ====================================================

        if not forecast_df.empty:
            st.subheader("🔮 Early Warning: Segments Projected to Worsen")
            _cfg = TrendAnalysisConfig()
            st.caption(
                f"Simple straight-line projection of each segment's Root_Cause_Score across the "
                f"months it appeared -- not a statistical model, just \"is this getting worse, and "
                f"how fast.\" Flagged when the trend is worsening and projected to cross "
                f"{_cfg.forecast_alert_threshold} within {_cfg.forecast_horizon_months} months. "
                f"2-point forecasts are lower confidence than 3+ points -- treat as \"worth watching,\" "
                f"not a precise prediction. Threshold/horizon are tunable defaults, not a validated cutoff."
            )
            n_flagged = int(forecast_df["Early_Warning"].sum())
            if n_flagged:
                st.warning(f"⚠️ {n_flagged} segment(s) projected to breach the threshold within the forecast horizon.")

                st.markdown("**Segments to Watch (most urgent first):**")
                watch_list = forecast_df[forecast_df["Early_Warning"]].sort_values("Months_To_Breach").head(5)
                for r in watch_list.itertuples():
                    st.write(
                        f"**{r.Technique}** — {r.Segment_Definition}  \n"
                        f"Currently at {r.Current_Root_Cause_Score:.2f}, projected to reach "
                        f"{r.Predicted_Next_Root_Cause_Score:.2f} next month · last seen in "
                        f"*{r.Last_Appeared_Period}* · expected to cross the alert threshold in "
                        f"~{r.Months_To_Breach} month(s) · {r.Confidence.lower()}"
                    )

                fig = early_warning_urgency_chart(forecast_df)
                if fig is not None:
                    st.pyplot(fig)
            else:
                st.info("No segments currently projected to breach the threshold within the forecast horizon.")

            show_only_flagged = st.checkbox(
                "Show only flagged (Early_Warning) segments in the table below",
                value=n_flagged > 0,
                key="early_warning_only_flagged",
            )
            table_df = forecast_df[forecast_df["Early_Warning"]] if show_only_flagged and n_flagged else forecast_df
            st.dataframe(table_df, use_container_width=True)

    elif st.button("Run Benchmark"):

        with st.spinner(
            "Running all segmentation techniques..."
        ):

            summary_df, combined_segments_df, cross_top10_df, cross_exec_summary_df = (
                benchmark_all_techniques()
            )

        st.session_state["cross_top10_df"] = cross_top10_df
        st.session_state["cross_exec_summary_df"] = cross_exec_summary_df
        st.session_state["summary_df"] = summary_df

        st.success(
            "Benchmark Completed Successfully"
        )

        # ====================================================
        # Sort By Overall Score
        # ====================================================

        ranking_df = (
            summary_df
            .sort_values(
                by="Overall_Score_100",
                ascending=False
            )
            .reset_index(drop=True)
        )

        ranking_df["Rank"] = (
            ranking_df.index + 1
        )

        ranking_df = ranking_df[
            ["Rank"] +
            [c for c in ranking_df.columns if c != "Rank"]
        ]

        # ====================================================
        # Ranking Table
        # ====================================================

        st.subheader(
            "📊 Technique Ranking"
        )

        st.dataframe(
            ranking_df,
            use_container_width=True
        )

        # ====================================================
        # Top 10 Per Technique (5 x 10 = 50 rows)
        # ====================================================

        st.subheader("📋 Top 10 Segments per Technique")
        st.caption(
            "Full metric set, same columns as Cross-Technique Insights (Tab 3). "
            "A technique shows fewer than 10 rows only if it did not discover "
            "that many candidates passing the significance/support gates -- "
            "counts are not padded to 10."
        )

        if not combined_segments_df.empty:
            _rank_col = (
                "Severity_Score" if "Severity_Score" in combined_segments_df.columns
                else "Business_Impact_Score"
            )
            groups = []
            for _, g in combined_segments_df.groupby("Technique"):
                top = g.nlargest(10, _rank_col).reset_index(drop=True)
                top.insert(1, "Rank_Within_Technique", top.index + 1)
                groups.append(top)
            top10_per_technique = pd.concat(groups, ignore_index=True)

            st.dataframe(top10_per_technique, use_container_width=True)
            st.download_button(
                label="📥 Download Top 10 x 5 CSV",
                data=top10_per_technique.to_csv(index=False),
                file_name="top10_per_technique.csv",
                mime="text/csv",
                key="download_top10_per_technique",
            )

        # ====================================================
        # Overall Score Chart
        # ====================================================

        if "Overall_Score_100" in ranking_df.columns:

            st.subheader(
                "🏅 Overall Score Comparison"
            )

            st.bar_chart(
                ranking_df.set_index(
                    "Technique"
                )["Overall_Score_100"]
            )

        # ====================================================
        # PSI Chart
        # ====================================================

        if "Max_PSI" in ranking_df.columns:

            st.subheader(
                "📈 PSI Comparison"
            )

            st.bar_chart(
                ranking_df.set_index(
                    "Technique"
                )["Max_PSI"]
            )

        # ====================================================
        # Runtime Chart
        # ====================================================

        if "Execution_Time_Sec" in ranking_df.columns:

            st.subheader(
                "⏱ Runtime Comparison"
            )

            st.bar_chart(
                ranking_df.set_index(
                    "Technique"
                )["Execution_Time_Sec"]
            )

        # ====================================================
        # Winner Based On Overall Score
        # ====================================================

        winner = ranking_df.iloc[0]

        st.subheader(
            "🥇 Most Sensitive Technique (Drift Detection)"
        )

        st.write(
            f"**Technique:** {winner['Technique']}"
        )

        st.write(
            f"**Overall Score:** {winner['Overall_Score_100']}"
        )

        st.write(
            f"**Rank:** {winner['Rank']}"
        )

        st.write(
            f"**Root Cause Feature:** {winner['Root_Cause_Feature']}"
        )

        # ====================================================
        # AI Recommendation
        # ====================================================

        try:

            benchmark_info = {

                "Technique":
                    winner["Technique"],

                "Overall_Score":
                    winner["Overall_Score_100"],

                "Max_PSI":
                    winner["Max_PSI"],

                "Max_Gini_Drop":
                    winner["Max_Gini_Drop"],

                "Max_KS_Drop":
                    winner["Max_KS_Drop"],

                "Root_Cause_Feature":
                    winner["Root_Cause_Feature"]
            }

            with st.spinner(
                "Generating AI Recommendation..."
            ):

                benchmark_summary = (
                    generate_insight(
                        benchmark_info
                    )
                )

            st.subheader(
                "🤖 AI Recommendation"
            )

            st.markdown(
                benchmark_summary
            )

        except Exception as e:

            st.warning(
                f"Could not generate benchmark summary: {e}"
            )

    # ====================================================
    # Ask About These Results (new feature): free-form Q&A over whatever
    # results are currently in session_state, classic and/or trend.
    # Placed at the top level of Tab 2 (outside both branches above) so it
    # shows regardless of which mode was last run.
    # ====================================================

    if _build_qa_context():
        st.divider()
        st.subheader("💬 Ask About These Results")
        st.caption(
            "Ask a free-form question about whatever results are currently on screen. Each "
            "question is answered fresh from the actual numbers above -- not a running "
            "conversation the model remembers turn to turn."
        )

        if "qa_history" not in st.session_state:
            st.session_state["qa_history"] = []

        for q, a in st.session_state["qa_history"]:
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                st.write(a)

        question = st.chat_input("Ask a question about the results above...")
        if question:
            try:
                with st.spinner("Thinking..."):
                    answer = answer_data_question(question, _build_qa_context())
            except Exception as e:
                answer = f"Could not generate an answer: {e}"
            st.session_state["qa_history"].append((question, answer))
            st.rerun()


# ============================================================
# TAB 3 — Cross-Technique Insights (task-flow step 4)
# ============================================================

with tab3:

    st.header("🔎 Cross-Technique Insights")
    st.caption(
        "Top-10 root-cause segments across all techniques, ranked by "
        "normalized Root_Cause_Score — a blend of drift, performance "
        "decay and population impact, not performance decay alone "
        "(task-flow step 4). Run the benchmark in 'Technique Comparison' first."
    )

    cross_top10_df = st.session_state.get("cross_top10_df", pd.DataFrame())
    cross_exec_summary_df = st.session_state.get("cross_exec_summary_df", pd.DataFrame())
    trend_cross_top10_df = st.session_state.get("trend_cross_top10_df", pd.DataFrame())

    if cross_top10_df.empty and trend_cross_top10_df.empty:
        st.info("No data yet — run the benchmark or trend analysis in the 'Technique Comparison' tab first.")
    elif cross_top10_df.empty:
        pass  # only a trend run has been done -- the trend section below covers it
    else:
        st.subheader("📋 Top 10 Root-Cause Segments (All Techniques)")
        st.dataframe(cross_top10_df, use_container_width=True)

        if not cross_exec_summary_df.empty:
            st.subheader("📝 Cross-Technique Executive Summary")
            for section in cross_exec_summary_df["Section"].unique():
                st.markdown(f"**{section}**")
                sect_rows = cross_exec_summary_df[cross_exec_summary_df["Section"] == section]
                for _, r in sect_rows.iterrows():
                    st.write(f"{r['Key']}: {r['Value']}")

        st.subheader("🌡️ Segment x Metric Heatmap")
        fig = segment_metric_heatmap(cross_top10_df)
        if fig is not None:
            st.pyplot(fig)

        st.subheader("🫧 Bubble Chart — Population vs Gini Drop (size = exposure)")
        fig = segment_bubble_chart(cross_top10_df)
        if fig is not None:
            st.pyplot(fig)

        st.subheader("💧 SIS Waterfall — #1 Worst Segment")
        fig = sis_waterfall_chart(cross_top10_df.iloc[0])
        if fig is not None:
            st.pyplot(fig)

    # ====================================================
    # Cross-Technique Trend Insights (taskflow step 4, trend version) --
    # separate from the classic section above, shown when a trend-analysis
    # run has populated it, regardless of whether a classic run has too.
    # ====================================================

    if not trend_cross_top10_df.empty:
        st.divider()
        st.header("🔎📈 Cross-Technique Trend Insights")
        st.caption(
            "Top-10 root-cause segments across all techniques and all selected months, "
            "ranked by normalized Root_Cause_Score_Trend (Root_Cause_Score with SIS "
            "boosted by each segment's Trend_Impact_Score). Run trend analysis in "
            "'Technique Comparison' first."
        )

        trend_display_cols = [
            c for c in [
                "Overall_Rank", "Technique", "Discovered_By", "Segment_Definition",
                "Periods_Appeared", "Total_Periods", "Appeared_In", "Frequency",
                "Recency_Factor", "Consistency_Score", "Trend_Impact_Score",
                "Normalized_Root_Cause_Score",
            ] if c in trend_cross_top10_df.columns
        ]
        st.subheader("📋 Top 10 Root-Cause Segments (All Techniques, Trend-Adjusted)")
        st.dataframe(trend_cross_top10_df[trend_display_cols], use_container_width=True)

        st.subheader("🌡️ Segment x Metric Heatmap (Trend)")
        fig = segment_metric_heatmap(trend_cross_top10_df)
        if fig is not None:
            st.pyplot(fig)

        st.subheader("🫧 Bubble Chart — Population vs Gini Drop (Trend)")
        fig = segment_bubble_chart(trend_cross_top10_df)
        if fig is not None:
            st.pyplot(fig)

        st.subheader("💧 SIS Waterfall — #1 Most Persistent Root Cause")
        fig = sis_waterfall_chart(trend_cross_top10_df.iloc[0])
        if fig is not None:
            st.pyplot(fig)
