"""Core policy components for adaptive continual recommendation.

This package contains only the method-level policies. Model training,
data loading, and serving integrations are intentionally left to the caller.
"""

from .clock import AdaptiveStageController, MeanCUSUMSignal
from .utility import PairedUtilitySignal

__all__ = [
    "AdaptiveStageController",
    "MeanCUSUMSignal",
    "PairedUtilitySignal",
]
