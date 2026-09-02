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
    containment_threshold: float = 0.90,
) -> list[dict]:
    """Select up to *pool_size* diverse candidates from *records* (assumed
    pre-sorted by desired priority) by dropping near-duplicates whose
    monitoring-side boolean masks have Jaccard similarity ≥
    *overlap_threshold*.

    Also drops a candidate whenever it and an already-kept, higher-priority
    candidate are near-total subsets of one another in *either* direction
    (containment ≥ *containment_threshold*), even when Jaccard is well under
    *overlap_threshold* -- Jaccard is symmetric, so one segment nested
    inside a much larger one (e.g. "age<54 AND income<70k" vs. "age<54"
    alone) can score a low Jaccard purely because the union is dominated by
    the larger segment. Checking both directions catches this regardless of
    which of the pair happens to rank higher and get kept first: a broader
    segment that's mostly just a diluted restatement of an already-flagged
    sharper sub-segment is dropped too, not only the reverse. Defaults
    higher (0.90) than overlap_threshold (0.70) so this only catches
    near-total nesting, not genuinely-different-but-overlapping segments.

    The ``mask_key`` field is removed from each selected record before
    returning.
    """
    selected: list[dict] = []
    for cand in records:
        cand_mask = cand[mask_key]
        cand_size = cand_mask.sum()
        is_redundant = False
        for kept in selected:
            kept_mask = kept[mask_key]
            kept_size = kept_mask.sum()
            inter = np.logical_and(cand_mask, kept_mask).sum()
            union = np.logical_or(cand_mask, kept_mask).sum()
            jaccard = inter / union if union > 0 else 0.0
            containment_cand_in_kept = inter / cand_size if cand_size > 0 else 0.0
            containment_kept_in_cand = inter / kept_size if kept_size > 0 else 0.0
            if (
                jaccard >= overlap_threshold
                or containment_cand_in_kept >= containment_threshold
                or containment_kept_in_cand >= containment_threshold
            ):
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
