"""
utils/exports.py
================
CSV export helpers and output-formatting utilities shared across techniques.

All outputs are written to the ``outputs/`` directory by default, following
the new project directory convention:
  data/     — raw input datasets
  outputs/  — all generated CSV result files
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

from utils.logging_config import get_logger

logger = get_logger(__name__)

# Resolve outputs/ relative to this file's package root (two levels up from utils/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = _PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


def resolve_output_path(filename: str) -> Path:
    """Return the canonical path inside outputs/ for a given filename.

    If *filename* is already an absolute path, it is returned as-is so
    callers can still override the destination.
    """
    p = Path(filename)
    if p.is_absolute():
        return p
    return OUTPUTS_DIR / p


def export_results_csv(
    df: pd.DataFrame,
    filename: Union[str, Path],
    label: str = "results",
) -> Path:
    """Write *df* to outputs/<filename> as CSV, logging the action.

    Returns the resolved Path where the file was written.
    """
    if df.empty:
        logger.warning("Skipping export of empty %s DataFrame.", label)
        return resolve_output_path(str(filename))
    out_path = resolve_output_path(str(filename))
    df.to_csv(out_path, index=False)
    logger.info("Exported %s (%d rows) → %s", label, len(df), out_path)
    return out_path


def export_portfolio_view(
    df: pd.DataFrame,
    base_filename: Union[str, Path],
) -> Path:
    """Write the portfolio-impact view CSV alongside the main results CSV."""
    if df.empty:
        return resolve_output_path(str(base_filename))
    stem = Path(str(base_filename)).stem
    portfolio_filename = f"{stem}_portfolio_view.csv"
    return export_results_csv(df, portfolio_filename, label="portfolio view")
