# Drift Segment Discovery

This folder contains the datasets and an analysis script for automated segment discovery using drift localization and k-means.

## Files

- `development_data_5000_shap.csv` — development dataset
- `monitoring_data_5000_shap.csv` — monitoring dataset
- `drift_segment_analysis.py` — analysis script
- `requirements.txt` — optional package dependencies

## Run analysis

From this folder execute:

```bash
python drift_segment_analysis.py
```

The script will:

- summarize numeric and categorical distributions for each dataset
- compare drift in means and category ratios
- identify top drift segments by categorical and numeric splits
- run K-means segment discovery on combined data

## Optional dependencies

Install optional packages if you want to extend the analysis in the future:

```bash
pip install -r requirements.txt
```
