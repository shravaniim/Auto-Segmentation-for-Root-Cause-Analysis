# Auto-Segmentation for Root Cause Analysis

An engine that automatically finds **which customer segments are responsible for a risk model's performance decline** — replacing the manual, one-segment-at-a-time investigation analysts do today with a run that discovers, scores, and ranks candidate segments across five different techniques.

> See [`docs/architecture_high_level.svg`](docs/architecture_high_level.svg) and [`docs/architecture_low_level.svg`](docs/architecture_low_level.svg) for diagrams of the flow, or open [`docs/architecture_preview.html`](docs/architecture_preview.html) to view both together with a glossary.

## The problem

When a PD, credit-scoring, or collections model's monitoring metrics (AUC, Gini, KS, PSI, bad rate) decline, analysts manually guess candidate segments ("Age < 30", "Self-employed", "West region"), recompute metrics for each one, and compare against development-period benchmarks. It's slow, expertise-dependent, and can't realistically cover more than a handful of the possible segment combinations — so real drivers get missed.

## What this does instead

1. **Discovers candidate segments automatically** — no analyst has to guess them — using five different techniques so no single method's blind spot hides the real driver.
2. **Scores every candidate the same way**: population drift (PSI), performance decay (ΔAUC/Gini/KS), bad-rate shift, calibration drift, exposure concentration, and a per-segment SHAP-based behavioral check.
3. **Ranks candidates** by business impact, both within each technique and across all five combined.
4. **Explains the result** in plain language via an executive summary, with heatmap, bubble, and waterfall visualizations.

## The five techniques

| Technique | How it finds segments |
|---|---|
| **AutoSlicer** | Beam search over combinations of feature rules (up to N-way) |
| **Feature Binning** | Bins each feature independently, ranks by deterioration |
| **Decision Tree** (Gradient Boosting) | Leaves of a gradient-boosted tree ensemble |
| **Clustering** (K-Means) | Drift-aware clusters, converted to explainable rules |
| **Drift Localization Tree** | A tree trained to separate development rows from monitoring rows directly |

Every technique is fed the same leakage-safe feature set (`age`, `income`, `region`, `occupation`) — `target`, `score`, `ead`, `shap_*`, and `customer_id` are excluded from segment discovery and only used to *compute* metrics, never to define a segment.

## Key metrics, defined once

| Term | Meaning |
|---|---|
| **PSI** | Population Stability Index — how much a segment's population share shifted between development and monitoring |
| **Bad Rate** | Share of defaults/events in a segment, dev vs. monitoring |
| **AUC / Gini / KS** | Model discrimination, dev vs. monitoring — the drop is what matters |
| **Calibration Drift** | Change in the Actual/Expected ratio (signed: monitoring − development) |
| **SIS** (Segment Impact Score) | Weighted composite of drift + performance decay + business impact for one segment |
| **DIS** (Drift Impact Score) | Additive composite of population and performance drift |
| **Root Cause Score** | `feature-level PSI × (1 + Gini decay) × population share` — the *only* metric with the same formula and scale across all five techniques, so it's what cross-technique ranking uses |
| **Severity Score** | Each technique's own internal ranking score — comparable **within** that technique, not across techniques (see the low-level diagram) |

All deltas follow one convention: `monitoring − development`. A metric like "Gini Drop" is reported as `max(0, -Δ)` — positive only when things got worse; an improvement never shows up as a drop.

## Project layout

```
streamlit_app.py               UI: run techniques, compare them, cross-technique insights
compare_segmentation_techniques.py   Benchmarks all 5 techniques, builds the executive summary
techniques/                    One file per technique (autoslicer, feature_binning, kmeans, drift_tree, gradient_boosting)
core/
  candidate_evaluation.py        Shared per-segment metric computation (4 of 5 techniques)
  candidate_filtering.py         Min-size / max-35%-population gates
  candidate_deduplication.py     Jaccard-overlap dedup (shared by all 5 techniques)
  candidate_ranking.py           Severity scoring + significance-first ranking
  cross_technique_analysis.py    Pools each technique's worst-10, ranks the overall top-10
  multi_period_analysis.py       Tracks segments that recur as "worst" across scoring dates
  segment_insights.py            Executive summary generation
metrics/                        PSI, AUC/Gini/KS, calibration, SIS/DIS, significance testing
models/config.py                All technique parameters, in one place
llm/                            Azure OpenAI wiring for the AI-generated narrative summary
utils/charts.py                 Heatmap / bubble / waterfall chart builders
data/                           development_data_5000_shap.csv, monitoring_data_5000_shap.csv
outputs/                        Generated CSVs (per-technique results, cross-technique top-10, executive summaries)
tests/                          Regression tests for the metric/ranking correctness fixes
```

## Running it

```bash
pip install -r requirements.txt

# Interactive UI: pick technique(s), set parameters, view results and charts
streamlit run streamlit_app.py

# Or run everything headless and write CSVs to outputs/
python compare_segmentation_techniques.py
```

Azure OpenAI credentials for the AI narrative summary go in a `.env` file (gitignored):
```
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_API_VERSION=...
```

## Current status

**Working:** all 5 techniques, candidate filtering/dedup, full metric suite, SIS/DIS/Root Cause Score, per-segment SHAP, cross-technique top-10 ranking, executive summary, Streamlit UI with heatmap/bubble/waterfall charts.


