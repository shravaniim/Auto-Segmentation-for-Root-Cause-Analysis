"""
core/candidate_filtering.py
============================
Candidate-level filtering gates: minimum support, minimum event counts,
maximum segment size, and auto-derived thresholds.

Centralises the filtering logic that was previously copy-pasted identically
in autoslicer, drift_tree, and gradient_boosting pipeline functions.
"""

from __future__ import annotations

import numpy as np


def derive_min_abs_count(
    dev_event_rate: float,
    min_events_per_slice: int,
    explicit_min_abs_count: int | None,
) -> int:
    """Derive the minimum absolute row-count floor from the event rate,
    unless an explicit value was provided."""
    if explicit_min_abs_count is not None:
        return explicit_min_abs_count
    return max(50, int(np.ceil(min_events_per_slice / max(dev_event_rate, 1e-4))))


def derive_min_support(
    min_abs_count: int,
    total_dev_n: int,
    total_mon_n: int,
    explicit_min_support: float | None,
) -> float:
    """Derive the minimum population-fraction support from the absolute
    count, unless an explicit value was provided."""
    if explicit_min_support is not None:
        return explicit_min_support
    return min(0.30, max(0.02, min_abs_count / min(total_dev_n, total_mon_n)))


def passes_support_filter(
    n_dev: int,
    n_mon: int,
    pct_dev: float,
    pct_mon: float,
    min_abs_count: int,
    min_support: float,
    max_segment_pct: float,
) -> bool:
    """Return True if the segment passes all population-based gates."""
    if n_dev < min_abs_count or n_mon < min_abs_count:
        return False
    if pct_dev < min_support or pct_mon < min_support:
        return False
    if pct_dev > max_segment_pct or pct_mon > max_segment_pct:
        return False
    return True


def passes_event_count_filter(
    x_dev: int, x_mon: int, min_events_per_slice: int
) -> bool:
    """Return True if both dev and mon sides have enough positive events."""
    return x_dev >= min_events_per_slice and x_mon >= min_events_per_slice
