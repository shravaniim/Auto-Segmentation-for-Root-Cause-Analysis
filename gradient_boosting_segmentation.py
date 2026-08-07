"""
gradient_boosting_segmentation.py
=================================
Backwards-compatibility wrapper for Gradient Boosting Segmentation technique.
Delegates to techniques.gradient_boosting.
"""

from __future__ import annotations

from models.config import GBConfig
from techniques.gradient_boosting import (
    GradientBoostingSegmentationTechnique,
    run_gradient_boosting_segmentation,
)

if __name__ == "__main__":
    DEV_FILE = "data/development_data_NEW.csv"
    MON_FILE = "data/monitoring_2026_01.csv"
    
    res = run_gradient_boosting_segmentation(DEV_FILE, MON_FILE)
    print("=" * 80)
    print("GRADIENT BOOSTING SEGMENTATION COMPLETED")
    print("=" * 80)
    print(f"Candidates Evaluated: {res['overall']['candidates_evaluated']}")
    print(f"Top Segments Returned: {res['overall']['candidates_returned']}")
    print(f"Execution Time: {res['execution_time']:.2f}s")
    if not res["segments"].empty:
        print("\nTop Segment:")
        print(res["segments"].iloc[0][["Segment_Definition", "PSI", "Delta_Gini", "Severity_Score"]])
