"""
utils/logging_config.py
=======================
Centralized structured logging configuration, replacing print() statements
across the codebase.  Provides a single ``get_logger`` factory that all modules
import so log format, level, and output destination are consistent project-wide.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


_CONFIGURED = False

LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> None:
    """One-time global logging setup.  Safe to call multiple times; only the
    first call has effect."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(console)

    # Optional file handler
    if log_file:
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    """Returns a named logger.  Automatically calls ``configure_logging`` on
    first use so modules don't need to worry about setup order."""
    configure_logging()
    return logging.getLogger(name)
