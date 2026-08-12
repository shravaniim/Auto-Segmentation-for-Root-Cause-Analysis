# Metric Reference & Baseline Validation

Every formula below was checked against the manager's baseline
(`segment_discovery/segment_root_cause_analysis.py`) line by line. Where a
metric exists in both, the formula is now byte-identical (verified earlier
by running both on the same segment: SIS matched to 6 decimal places).
Where it doesn't, it's marked **[Ours only]** and is an addition, not a
substitute for anything the baseline computes.

One convention holds everywhere unless noted: **delta = monitoring − development**.
"Drop" fields are `max(0, -delta)` for performance metrics (positive only
when things got worse) — this is deliberate, not the same thing as `abs(delta)`.

---

## 1. Population & Exposure

| Metric | Formula | Range | Source |
|---|---|---|---|
| `Dev_Pct` / `Mon_Pct` | `n / total_n` for that dataset | [0, 1] | shared |
| `Population_Drift` | `Mon_Pct − Dev_Pct` | [−1, 1] | shared, matches baseline |
| `Dev_Exposure_Pct` / `Mon_Exposure_Pct` | `Σ(weight) / total_weight` | [0, 1] | shared |
| `Exposure_Drift` | `Mon_Exposure_Pct − Dev_Exposure_Pct` | [−1, 1] | shared, matches baseline |
| `PSI` (segment-level) | `calculate_psi(Dev_Pct, Mon_Pct)`, decile-log formula, floored at 0 | typically [0, ~1], unbounded above | shared. **Not the same thing as baseline's `total_psi`** — see §3 |

## 2. Performance (dev vs. monitoring)

| Metric | Formula | Range | Source |
|---|---|---|---|
| `AUC` | `roc_auc_score(target, score)` | [0, 1] | shared |
| `Gini` | `2·AUC − 1` | [−1, 1] | shared |
| `KS` | max separation of cumulative bad/good rate curves | [0, 1] | **[Ours only]** — baseline does not compute KS |
| `Bad_Rate` | `mean(target)` | [0, 1] | shared |
| `Delta_AUC` / `Delta_Gini` / `Delta_KS` / `Delta_BR` | `monitoring − development` | [−1, 1] each | shared convention |
| `Calibration_Drift` | `(mon_actual/mon_expected) − (dev_actual/dev_expected)`, signed A/E-ratio shift | unbounded, typically small | **[Ours only]** — baseline does not compute calibration |

## 3. Drift diagnostics feeding the composite scores

| Metric | Formula | Range | Source |
|---|---|---|---|
| `Root_Cause_Feature` / `Root_Cause_PSI` | the single feature (of age/income/region/occupation) with the **highest** value-distribution PSI inside this segment | PSI part: [0, ~1] | shared — narrative field, not used in scoring |
| `mean_feature_psi` (baseline calls this `total_psi`) | **mean** value-distribution PSI across those same features | [0, ~1] | ported from baseline, adapted to our 4-feature schema (baseline uses age/income/score) |
| `Top_SHAP_Feature` / `Top_SHAP_PSI` | feature with the highest PSI of its SHAP-value distribution, computed on this segment's own rows | PSI part: [0, ~1] | shared — narrative field, not used in scoring |
| `mean_shap_shift` (baseline: `total_shap_shift`) | mean of `\|mon_mean_shap − dev_mean_shap\|` across features | ≥ 0, unbounded, typically small | ported from baseline exactly |
| `feature_drift_details` | JSON list of `{feature, psi}` for every allowed feature in this segment | — | shared |

**Important distinction:** `PSI` (§1, population-share) answers "did this segment get bigger/smaller as a share of the portfolio?" `mean_feature_psi`/`Root_Cause_PSI` (§3) answer "did the *people inside* this segment change?" They are different questions computed by the same underlying PSI math, and both baseline and this project keep them separate.

## 4. Composite Impact Scores — the ones the manager asked to match

| Metric | Formula | Range | Baseline match |
|---|---|---|---|
| `Confidence` | `min(1, sqrt(Mon_Count / 100))` | [0, 1] | **exact match** |
| `SIS` (Segment Impact Score) | `auc_drop × exposure_factor × Mon_Pct × Confidence`, where `auc_drop = max(0, -Delta_AUC)` | [0, 1], compounds to small numbers in practice (~0–0.003 observed) | **exact match** — verified to 6 decimals on an identical segment |
| `DIS` (Drift Impact Score) | `max(0, Population_Drift × auc_drop × exposure_factor × (1 + mean_feature_psi))` | [0, ∞) in theory, small in practice (~0–0.001 observed) | **exact match** in structure; tiny numeric gap (~1.6%) from `mean_feature_psi` using our 4-feature set vs. baseline's age/income/score |
| `Root_Cause_Score` | `0.35·norm(SIS) + 0.30·norm(DIS) + 0.20·norm(mean_feature_psi) + 0.15·norm(mean_shap_shift)`, each `norm(...)` min-max scaled to [0,1] **within that technique's own candidate pool** | **[0, 1], strictly** (weights sum to 1.0 over four [0,1] terms) | **exact formula and weights match**. Absolute numbers won't match baseline 1:1 for the same segment — see note below |
| `Severity_Score` | `Root_Cause_Score × 20` | [0, 20], strictly | not a baseline concept — our addition for a within-technique 0–20 display scale |
| `exposure_factor` | `Mon_Exposure_Pct` if a weight column exists and is > 0, else `1.0` | [0, 1] | matches baseline's `use_ead_in_scoring` fallback behavior |

**Why `Root_Cause_Score` won't be numerically identical to baseline for the same segment, even with an identical formula:** it's normalized against *whatever candidates were evaluated in that run*. Baseline brute-forces every combination up to 4-way (≈1,400 candidates); our techniques use pruned search (beam search, bins, tree leaves, clusters — ≈10–50 candidates). A segment normalized against 1,400 competitors scores differently than the same segment normalized against 20. This was checked directly: for `region = West`, baseline scored it 0.11 (rank 30/1432); ours scores it 0.6–0.9 (rank 1–2 of ~15). The *inputs* (SIS, DIS) matched exactly — only the normalization pool differs, which is an expected consequence of using efficient search instead of exhaustive combinatorics.

## 5. Statistical significance (not present in baseline)

| Metric | Formula | Range | Source |
|---|---|---|---|
| `br_pvalue` | two-proportion z-test, dev bad rate vs. mon bad rate | [0, 1] | **[Ours only]** |
| `score_shift_pvalue` | two-sample KS test, dev score distribution vs. mon | [0, 1] | **[Ours only]** |
| `BadRate_pvalue_adj` / `ScoreShift_pvalue_adj` | Benjamini-Hochberg FDR correction across all candidates in one run | [0, 1] | **[Ours only]** |
| `Statistically_Significant` | `BadRate_p_adj ≤ 0.05 OR ScoreShift_p_adj ≤ 0.05` | boolean | **[Ours only]** |

## 6. Other business-facing scores (not present in baseline)

| Metric | Formula | Range | Notes |
|---|---|---|---|
| `Business_Impact_Score` | `Mon_Pct × Mon_Exposure_Pct × (1 + PSI + gini_penalty)` | ≥ 0, typically [0, 1] | population-share PSI here, not `mean_feature_psi` |
| `Portfolio_Impact_Score` | `Mon_Pct × \|Delta_Gini\|` | ≥ 0, typically [0, 1] | used by the portfolio view, not the ranking |
| `Normalized_Root_Cause_Score` | min-max of `Root_Cause_Score` **across the pooled cross-technique candidates** | [0, 1], strictly | the only score safe to compare *across* techniques — `Severity_Score` is technique-local |

---

## Quick sanity ranges (what "normal" looks like on this dataset)

| Metric | Typical observed range |
|---|---|
| PSI (population-share) | 0.00 – 0.24 |
| Delta_Gini / Delta_KS | −0.33 – +0.17 |
| Delta_BR | −0.64 – +0.07 (a few segments swing hard — verified as a real, concentrated data effect, not a bug) |
| Confidence | usually 1.0 (segments here mostly exceed 100 monitoring rows) |
| SIS_Raw | 0.0000 – 0.003 |
| DIS_Raw | 0.0000 – 0.001 (frequently exactly 0 — happens whenever the segment shrank or AUC improved) |
| Root_Cause_Score | 0.0006 – 0.93 |
| Severity_Score | 0.01 – 18.5 |
