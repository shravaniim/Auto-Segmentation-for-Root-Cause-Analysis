import streamlit as st
import tempfile
import pandas as pd

from autoslicer_segmentation import run_autoslicer_segmentation
from kmeans_segmentation import run_kmeans_segmentation
from drift_localization_tree import run_drift_localization
from feature_binning_segmentation import run_feature_binning_segmentation
from gradient_boosting_segmentation import run_gradient_boosting_segmentation

from compare_segmentation_techniques import benchmark_all_techniques

from llm.insight_generator import generate_insight


# ============================================================
# Technique Mapping
# ============================================================

TECHNIQUE_MAP = {
    "AutoSlicer": run_autoslicer_segmentation,
    "KMeans": run_kmeans_segmentation,
    "Drift Tree": run_drift_localization,
    "Feature Binning": run_feature_binning_segmentation,
    "Gradient Boosting": run_gradient_boosting_segmentation
}


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

tab1, tab2 = st.tabs(
    [
        "Segmentation Analysis",
        "Technique Comparison"
    ]
)

# ============================================================
# TAB 1
# ============================================================

with tab1:

    st.header("Segmentation Analysis")

    dev_file = st.file_uploader(
        "Upload Development Dataset",
        type=["csv"],
        key="dev_file"
    )

    mon_file = st.file_uploader(
        "Upload Monitoring Dataset",
        type=["csv"],
        key="mon_file"
    )

    technique = st.selectbox(
        "Segmentation Technique",
        [
            "AutoSlicer",
            "KMeans",
            "Drift Tree",
            "Feature Binning",
            "Gradient Boosting"
        ]
    )

    if st.button("Run Analysis"):

        if dev_file is None or mon_file is None:
            st.error("Please upload both datasets.")
            st.stop()

        with st.spinner(f"Running {technique}..."):

            dev_temp = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".csv"
            )

            mon_temp = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".csv"
            )

            dev_temp.write(dev_file.getvalue())
            mon_temp.write(mon_file.getvalue())

            dev_temp.close()
            mon_temp.close()

            selected_runner = TECHNIQUE_MAP[technique]

            result = selected_runner(
                dev_temp.name,
                mon_temp.name
            )

        st.success(f"{technique} Completed Successfully")

        # ====================================================
        # Execution Summary
        # ====================================================

        st.subheader("📊 Execution Summary")

        overall = result.get("overall", {})

        for key, value in overall.items():
            st.write(f"**{key}** : {value}")

        if "execution_time" in result:
            st.write(
                "**Execution Time** :",
                round(result["execution_time"], 2),
                "seconds"
            )

        # ====================================================
        # Segment Table
        # ====================================================

        segments_df = result.get(
            "segments",
            pd.DataFrame()
        )

        st.subheader("📋 Top Segments")

        st.dataframe(
            segments_df,
            use_container_width=True
        )

        # ====================================================
        # Download
        # ====================================================

        if not segments_df.empty:

            st.download_button(
                label="📥 Download Results CSV",
                data=segments_df.to_csv(index=False),
                file_name=f"{technique}_results.csv",
                mime="text/csv"
            )

        # ====================================================
        # LLM Summary
        # ====================================================

        if not segments_df.empty:

            try:

                top_segment = segments_df.iloc[0]

                segment_info = {

                    "Segment_Definition":
                        str(
                            top_segment.get(
                                "Segment_Definition",
                                top_segment.get(
                                    "segment",
                                    "Unknown Segment"
                                )
                            )
                        ),

                    "PSI":
                        float(
                            top_segment.get(
                                "PSI",
                                top_segment.get(
                                    "psi",
                                    0
                                )
                            )
                        ),

                    "Delta_Gini":
                        float(
                            top_segment.get(
                                "Delta_Gini",
                                top_segment.get(
                                    "delta_gini",
                                    0
                                )
                            )
                        ),

                    "Delta_BR":
                        float(
                            top_segment.get(
                                "Delta_BR",
                                0
                            )
                        ),

                    "Root_Cause_Feature":
                        str(
                            top_segment.get(
                                "Root_Cause_Feature",
                                "Not Available"
                            )
                        ),

                    "Severity_Score":
                        float(
                            top_segment.get(
                                "Severity_Score",
                                top_segment.get(
                                    "final_score",
                                    0
                                )
                            )
                        ),

                    "Business_Impact_Score":
                        float(
                            top_segment.get(
                                "Business_Impact_Score",
                                0
                            )
                        )
                }

                with st.spinner(
                    "Generating AI Executive Summary..."
                ):

                    llm_summary = generate_insight(
                        segment_info
                    )

                st.subheader(
                    "🤖 AI Generated Executive Summary"
                )

                st.markdown(
                    llm_summary
                )

            except Exception as e:

                st.warning(
                    f"Could not generate LLM summary: {e}"
                )


# ============================================================
# TAB 2
# ============================================================

# ============================================================
# TAB 2
# ============================================================

with tab2:

    st.header("🏆 Compare All Segmentation Techniques")

    if st.button("Run Benchmark"):

        with st.spinner(
            "Running all segmentation techniques..."
        ):

            summary_df, combined_segments_df = (
                benchmark_all_techniques()
            )

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
            "🥇 Recommended Technique"
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