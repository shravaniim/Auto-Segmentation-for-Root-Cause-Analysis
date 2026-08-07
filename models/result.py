"""
models/result.py
================
Standardized technique result data model returned by all segmentation techniques.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import pandas as pd


@dataclass
class TechniqueResult:
    """Standardized output structure for any segmentation technique run."""
    technique_name: str
    overall: dict[str, Any]
    segments_df: pd.DataFrame
    execution_time: float
    schema: dict[str, Any]
    portfolio_view_df: Optional[pd.DataFrame] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to standard dictionary format expected by downstream scripts."""
        res = {
            "overall": self.overall,
            "segments": self.segments_df,
            "execution_time": self.execution_time,
            "schema": self.schema,
        }
        if self.portfolio_view_df is not None:
            res["portfolio_view"] = self.portfolio_view_df
        for k, v in self.extra.items():
            res[k] = v
        return res
