"""
Pydantic Models for IRB Segmentation

This module provides type-safe, validated data models for the segmentation toolkit:
- Configuration and parameters
- Results and outputs
- Segment information and statistics
- Adjustment logs and history

All models use Pydantic for validation and serialization.
"""

from .params import IRBSegmentationParams
from .results import SegmentationResult, ValidationResult
from .segment_info import SegmentStatistics, SegmentValidation
from .adjustments import MergeLog, SplitLog, ForcedSplitLog, MonotonicityViolation, AdjustmentHistory

__all__ = [
    # Parameters
    'IRBSegmentationParams',

    # Results
    'SegmentationResult',
    'ValidationResult',

    # Segment Information
    'SegmentStatistics',
    'SegmentValidation',

    # Adjustments
    'MergeLog',
    'SplitLog',
    'ForcedSplitLog',
    'MonotonicityViolation',
    'AdjustmentHistory',
]

__version__ = '2.0.0'  # Major version bump for Pydantic refactor
