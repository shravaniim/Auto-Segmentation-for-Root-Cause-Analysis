"""
models/segment.py
=================
Data classes representing individual candidate segments and evaluation results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Segment:
    """Represents a candidate segment definition (rules / conditions)."""
    segment_definition: str
    feature_names: list[str] = field(default_factory=list)
    conditions: list[dict[str, Any]] = field(default_factory=list)
    bin_type: str = "custom"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmentResult:
    """Represents a fully evaluated and scored segment with statistical & business metrics."""
    segment: Segment
    dev_count: int
    mon_count: int
    dev_pct: float
    mon_pct: float
    psi: float
    psi_interpretation: str
    
    # Model performance metrics
    auc_dev: Optional[float] = None
    auc_mon: Optional[float] = None
    delta_auc: Optional[float] = None
    gini_dev: Optional[float] = None
    gini_mon: Optional[float] = None
    delta_gini: Optional[float] = None
    ks_dev: Optional[float] = None
    ks_mon: Optional[float] = None
    delta_ks: Optional[float] = None
    
    # Bad rate metrics
    br_dev: Optional[float] = None
    br_mon: Optional[float] = None
    delta_br: Optional[float] = None
    br_pvalue_adj: Optional[float] = None
    
    # Score shift metrics
    score_shift_pvalue_adj: Optional[float] = None
    statistically_significant: bool = False
    
    # Business impact metrics
    business_impact_score: float = 0.0
    sis_raw: float = 0.0
    dis_raw: float = 0.0
    confidence_score: float = 0.5
    severity_score: float = 0.0
    rank: int = 0
    drift_explanation: str = ""
    
    # Raw record dictionary for legacy compatibility
    raw_record: dict[str, Any] = field(default_factory=dict)
