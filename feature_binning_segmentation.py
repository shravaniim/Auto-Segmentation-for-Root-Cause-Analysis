"""
feature_binning_segmentation.py
===============================
Backwards-compatibility wrapper for Feature Binning Segmentation technique.
Delegates to techniques.feature_binning.
"""

from __future__ import annotations

from models.config import FeatureBinningConfig
from techniques.feature_binning import (
    FeatureBinningTechnique,
    run_feature_binning_segmentation,
)

if __name__ == "__main__":
    DEV_FILE = "data/development_data_NEW.csv"
    MON_FILE = "data/monitoring_2026_01.csv"
    
    res = run_feature_binning_segmentation(DEV_FILE, MON_FILE)
    print("=" * 80)
    print("FEATURE BINNING SEGMENTATION COMPLETED")
    print("=" * 80)
    print(f"Candidate Bins Generated: {res['overall']['candidate_bins']}")
    print(f"Execution Time: {res['execution_time']:.2f}s")
    if not res["segments"].empty:
        print("\nTop Segment:")
        print(res["segments"].iloc[0][["segment", "psi", "delta_gini", "final_score"]])
