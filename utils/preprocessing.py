"""
utils/preprocessing.py
======================
Shared feature-encoding and rule-extraction utilities used by tree-based
segmentation techniques (DLT, Gradient Boosting).

Functions are extracted from the duplicate copies in ``drift_localization_tree.py``
and ``gradient_boosting_segmentation.py`` — the logic is identical in both,
now maintained in one place.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Feature matrix construction (one-hot encoding for trees)
# ---------------------------------------------------------------------------

def build_feature_matrix(
    combined_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, dict]:
    """Build a numeric feature matrix suitable for tree-based classifiers.

    Returns ``(X_df, onehot_map)`` where *onehot_map* maps each generated
    one-hot column name back to ``(original_col, category_value)``."""
    X = pd.DataFrame(index=combined_df.index)
    for c in numeric_cols:
        X[c] = combined_df[c].fillna(combined_df[c].median())
    onehot_map: dict[str, tuple[str, str]] = {}
    for c in categorical_cols:
        vals = combined_df[c].astype(str).fillna("__MISSING__")
        for v in sorted(vals.unique()):
            col = f"{c}__{v}"
            X[col] = (vals == v).astype(int)
            onehot_map[col] = (c, v)
    return X, onehot_map


# ---------------------------------------------------------------------------
# Tree-rule formatting and canonicalization
# ---------------------------------------------------------------------------

def format_number(value: float) -> str:
    """Format a numeric segment-rule threshold for display: no scientific
    notation, thousands separators for readability (e.g. 60810.0 -> "60,810",
    44.5 -> "44.5"), regardless of magnitude."""
    if not np.isfinite(value):
        return str(value)
    rounded = round(float(value), 2)
    if rounded == int(rounded):
        return f"{int(rounded):,}"
    return f"{rounded:,.2f}".rstrip("0").rstrip(".")


def format_condition(
    feature_name: str,
    threshold: float,
    direction: str,
    onehot_map: dict,
) -> str:
    """Format a single tree split condition into a human-readable string."""
    if feature_name in onehot_map:
        col, val = onehot_map[feature_name]
        return f"{col} != {val}" if direction == "<=" else f"{col} = {val}"
    val_fmt = format_number(threshold)
    return (
        f"{feature_name} < {val_fmt}"
        if direction == "<="
        else f"{feature_name} >= {val_fmt}"
    )


def canonicalize_path_conditions(
    rule_list: list[tuple[str, float, str]],
    onehot_map: dict,
) -> Optional[list[str]]:
    """Merge redundant same-feature bounds in a tree-path condition list.

    A single tree path CAN legitimately revisit the same numeric feature at
    different depths.  This function collapses all visits to the tightest
    bound, keeping categorical (one-hot) conditions as-is.

    Returns ``None`` if the path contains contradictory bounds."""
    numeric_bounds: dict[str, dict] = {}
    # original categorical column -> {"positive": value or None, "negatives": {values}}
    categorical_by_col: dict[str, dict] = {}

    for feature_name, threshold, direction in rule_list:
        if feature_name in onehot_map:
            col, val = onehot_map[feature_name]
            entry = categorical_by_col.setdefault(col, {"positive": None, "negatives": set()})
            if direction == "<=":
                entry["negatives"].add(val)
            else:
                entry["positive"] = val
            continue
        bounds = numeric_bounds.setdefault(feature_name, {})
        if direction == "<=":
            bounds["upper"] = (
                threshold
                if "upper" not in bounds
                else min(bounds["upper"], threshold)
            )
        else:
            bounds["lower"] = (
                threshold
                if "lower" not in bounds
                else max(bounds["lower"], threshold)
            )

    # A positive "= value" assignment for a categorical column already
    # implies every "!= other_value" for that same column -- keep only the
    # positive assignment when one is present, instead of showing both
    # (e.g. "region != South AND region = North" -> just "region = North").
    categorical_parts: list[str] = []
    for col, entry in categorical_by_col.items():
        if entry["positive"] is not None:
            categorical_parts.append(f"{col} = {entry['positive']}")
        else:
            for val in sorted(entry["negatives"]):
                categorical_parts.append(f"{col} != {val}")

    numeric_parts: list[str] = []
    for feature_name, bounds in numeric_bounds.items():
        lower, upper = bounds.get("lower"), bounds.get("upper")
        if lower is not None and upper is not None and lower >= upper:
            return None  # contradictory
        if lower is not None:
            numeric_parts.append(f"{feature_name} >= {format_number(lower)}")
        if upper is not None:
            numeric_parts.append(f"{feature_name} < {format_number(upper)}")

    return categorical_parts + numeric_parts


def simplify_rule_list(rules: list[str], root_label: str = "All Customers (Root)") -> str:
    """Join condition strings into a single AND-clause, placing equality
    conditions first for readability."""
    equals_rules = [r for r in rules if " = " in r and "!=" not in r]
    range_rules = [r for r in rules if r not in equals_rules]
    combined = equals_rules + range_rules
    return " AND ".join(combined) if combined else root_label


# ---------------------------------------------------------------------------
# Tree-rule extraction (recursive walk)
# ---------------------------------------------------------------------------

def extract_tree_rules(tree_estimator, feature_names: list[str]) -> dict:
    """Extract all root-to-leaf paths from a fitted sklearn tree.

    Returns ``{leaf_node_id: [(feature, threshold, direction), ...]}``."""
    tree = tree_estimator.tree_
    leaf_rules: dict[int, list] = {}

    def recurse(node_id: int, current_rules: list) -> None:
        if tree.children_left[node_id] == tree.children_right[node_id]:
            leaf_rules[node_id] = list(current_rules)
            return
        feat_idx = tree.feature[node_id]
        feature_name = feature_names[feat_idx]
        threshold = tree.threshold[node_id]
        recurse(
            tree.children_left[node_id],
            current_rules + [(feature_name, threshold, "<=")],
        )
        recurse(
            tree.children_right[node_id],
            current_rules + [(feature_name, threshold, ">")],
        )

    recurse(0, [])
    return leaf_rules
