"""
core/candidate_deduplication.py
===============================
Jaccard-overlap-based deduplication of candidate segments.

Centralises the dedup logic that was copy-pasted across autoslicer and
gradient_boosting technique files.
"""

from __future__ import annotations

import numpy as np


def deduplicate_by_jaccard(
    records: list[dict],
    mask_key: str = "_mon_mask",
    overlap_threshold: float = 0.70,
    pool_size: int = 30,
) -> list[dict]:
    """Select up to *pool_size* diverse candidates from *records* (assumed
    pre-sorted by desired priority) by dropping near-duplicates whose
    monitoring-side boolean masks have Jaccard similarity ≥
    *overlap_threshold*.

    The ``mask_key`` field is removed from each selected record before
    returning.
    """
    selected: list[dict] = []
    for cand in records:
        cand_mask = cand[mask_key]
        is_redundant = False
        for kept in selected:
            kept_mask = kept[mask_key]
            inter = np.logical_and(cand_mask, kept_mask).sum()
            union = np.logical_or(cand_mask, kept_mask).sum()
            jaccard = inter / union if union > 0 else 0.0
            if jaccard >= overlap_threshold:
                is_redundant = True
                break
        if not is_redundant:
            selected.append(cand)
        if len(selected) >= pool_size:
            break

    # Clean up internal mask field
    for r in selected:
        r.pop(mask_key, None)

    return selected
