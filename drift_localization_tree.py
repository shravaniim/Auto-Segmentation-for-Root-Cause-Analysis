"""
drift_localization_tree.py
==========================
Backwards-compatibility wrapper for Drift Localization Tree technique.
Delegates to techniques.drift_tree.
"""

from __future__ import annotations

from models.config import DLTConfig
from techniques.drift_tree import (
    DriftLocalizationTreeTechnique,
    run_drift_localization,
)

if __name__ == "__main__":
    DEV_FILE = "data/development_data_NEW.csv"
    MON_FILE = "data/monitoring_2026_01.csv"
    
    res = run_drift_localization(DEV_FILE, MON_FILE)
    print("=" * 80)
    print("DRIFT LOCALIZATION TREE COMPLETED")
    print("=" * 80)
    print(f"Candidates Evaluated: {res['overall']['candidates_evaluated']}")
    print(f"Top Segments Returned: {res['overall']['candidates_returned']}")
    print(f"Execution Time: {res['execution_time']:.2f}s")
    if not res["segments"].empty:
        print("\nTop Segment:")
        print(res["segments"].iloc[0][["Segment_Definition", "PSI", "Delta_Gini", "Severity_Score"]])
