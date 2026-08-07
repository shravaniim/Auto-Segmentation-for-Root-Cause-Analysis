"""
techniques/base.py
==================
Abstract base class defining the contract for all auto-segmentation techniques.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Union
import pandas as pd

from models.result import TechniqueResult


class BaseSegmentationTechnique(ABC):
    """Abstract base class that all segmentation techniques implement.
    
    Provides standard interface: fit / fit_predict / run.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the technique."""
        pass

    @abstractmethod
    def run(
        self,
        dev_data: Union[str, pd.DataFrame],
        mon_data: Union[str, pd.DataFrame],
        config: Optional[Any] = None,
    ) -> TechniqueResult:
        """Execute the segmentation technique on development and monitoring datasets.
        
        Args:
            dev_data: Path to CSV or loaded DataFrame for development dataset.
            mon_data: Path to CSV or loaded DataFrame for monitoring dataset.
            config: Optional technique-specific configuration object.
            
        Returns:
            TechniqueResult container with summary metrics and candidate segments.
        """
        pass
