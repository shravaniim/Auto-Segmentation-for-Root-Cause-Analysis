import streamlit as st
import pandas as pd

from models.config import (
    SlicerConfig,
    DLTConfig,
    GBConfig,
    KMeansConfig,
    FeatureBinningConfig,
)

from autoslicer_segmentation import run_autoslicer_segmentation
from kmeans_segmentation import run_kmeans_segmentation
from drift_localization_tree import run_drift_localization
from feature_binning_segmentation import run_feature_binning_segmentation
from gradient_boosting_segmentation import run_gradient_boosting_segmentation

from compare_segmentation_techniques import (
    benchmark_all_techniques,
    build_feature_schema,
    DEV_FILE,
    MON_FILE,
    REQUESTED_FEATURES,
)

from llm.insight_generator import generate_insight
from utils.charts import segment_metric_heatmap, segment_bubble_chart, sis_waterfall_chart


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
        f"Development: `development_data_5000_shap.csv` ({len(dev_df):,} rows) · "
        f"Monitoring: `monitoring_data_5000_shap.csv` ({len(mon_df):,} rows). "
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

with tab2:

    st.header("🏆 Compare All Segmentation Techniques")

    if st.button("Run Benchmark"):

        with st.spinner(
            "Running all segmentation techniques..."
        ):

            summary_df, combined_segments_df, cross_top10_df, cross_exec_summary_df = (
                benchmark_all_techniques()
            )

        st.session_state["cross_top10_df"] = cross_top10_df
        st.session_state["cross_exec_summary_df"] = cross_exec_summary_df

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

    if cross_top10_df.empty:
        st.info("No data yet — run the benchmark in the 'Technique Comparison' tab first.")
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
