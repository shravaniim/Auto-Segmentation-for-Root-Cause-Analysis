"""
core/parameter_optimization.py
==============================
Provides hyperparameter grid search capabilities for segmentation techniques.
Iterates over a predefined parameter grid in the configuration, runs the segmentation
technique, and selects the parameter combination that yields the highest cumulative
Business Impact Score (or other primary metric) across the top N generated segments.
"""

from __future__ import annotations

import itertools
from copy import deepcopy
from typing import Callable, Any

import pandas as pd

from utils.logging_config import get_logger

logger = get_logger(__name__)


def generate_param_grid(param_grid: dict) -> list[dict]:
    """Expands a dict of lists into a list of dicts representing all combinations."""
    if not param_grid:
        return [{}]
    
    # Ensure all values in the dict are iterables (lists/tuples)
    clean_grid = {k: (v if isinstance(v, (list, tuple)) else [v]) for k, v in param_grid.items()}
    keys, values = zip(*clean_grid.items())
    
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def apply_params_to_config(config: Any, params: dict) -> Any:
    """Returns a deep copy of config with updated parameters."""
    new_config = deepcopy(config)
    for k, v in params.items():
        if hasattr(new_config, k):
            setattr(new_config, k, v)
        else:
            logger.warning("Config %s has no attribute %r (skipping)", type(new_config).__name__, k)
    return new_config


def optimize_parameters(
    technique_func: Callable, 
    config: Any, 
    dev_df: pd.DataFrame, 
    mon_df: pd.DataFrame, 
    *args, 
    **kwargs
) -> Any:
    """
    Executes the technique_func across all parameter combinations in config.param_grid.
    Returns the result object from the best performing configuration based
    on the sum of Business_Impact_Score.
    
    technique_func must accept (dev_df, mon_df, config, *args, **kwargs).
    """
    param_grid = getattr(config, 'param_grid', None)
    
    if not param_grid:
        # No grid specified, run normally.
        return technique_func(dev_df, mon_df, config, *args, **kwargs)

    grid = generate_param_grid(param_grid)
    if len(grid) <= 1:
        return technique_func(dev_df, mon_df, config, *args, **kwargs)
        
    logger.info("Starting parameter optimization over %d combinations.", len(grid))
    
    best_score = -float('inf')
    best_result = pd.DataFrame()
    best_params = None

    for i, params in enumerate(grid, 1):
        logger.info("Testing combination %d/%d: %s", i, len(grid), params)
        test_config = apply_params_to_config(config, params)
        
        try:
            result = technique_func(dev_df, mon_df, test_config, *args, **kwargs)
            
            segments_df = getattr(result, "segments_df", result)
            
            if not isinstance(segments_df, pd.DataFrame) or segments_df.empty:
                score = 0.0
            elif 'Business_Impact_Score' in segments_df.columns:
                score = float(segments_df['Business_Impact_Score'].sum())
            else:
                score = 0.0
                
            logger.info("Combination %d yielded aggregate score: %.4f", i, score)
            
            if score > best_score:
                best_score = score
                best_result = result
                best_params = params
                
        except Exception as e:
            logger.error("Error during optimization combination %s: %s", params, e)

    logger.info("Optimization complete. Best score: %.4f with params: %s", best_score, best_params)
    
    if best_result is None:
        logger.warning("All combinations failed or yielded no segments. Falling back to default.")
        return technique_func(dev_df, mon_df, config, *args, **kwargs)
        
    return best_result
