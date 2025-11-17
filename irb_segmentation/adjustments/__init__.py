"""
Segment Adjustment Functions

Modular, standalone functions for adjusting segments to meet IRB requirements.

This module replaces the static methods in the old SegmentAdjuster class with
granular, testable functions organized by purpose.

Main Functions:
    - merge_small_segments: Merge segments below density/defaults thresholds
    - split_large_segments: Split segments above density threshold
    - apply_forced_splits: Apply business-mandated splits
    - enforce_monotonicity: Check monotonicity constraints

Usage:
    >>> from irb_segmentation.adjustments import merge_small_segments
    >>> adjusted_segments, logs = merge_small_segments(X, segments, y)
"""

# Import main entry point functions
from .merging import merge_small_segments
from .splitting import split_large_segments
from .forced_splits import apply_forced_splits
from .monotonicity import enforce_monotonicity

# Import granular helper functions (for advanced usage and testing)
from .merging import (
    identify_segments_to_merge,
    find_most_similar_segment,
    merge_single_segment
)

from .splitting import (
    identify_segments_to_split,
    find_best_split,
    split_single_segment
)

from .forced_splits import (
    check_numeric_split_applicable,
    apply_numeric_forced_split,
    check_categorical_split_applicable,
    apply_categorical_forced_split
)

from .monotonicity import (
    calculate_segment_feature_stats,
    check_monotonicity_between_segments,
    check_feature_monotonicity
)

# Import utility functions
from .utils import (
    relabel_segments,
    calculate_default_rate,
    calculate_segment_density,
    get_segment_statistics,
    gini_impurity
)

# Backward compatibility wrapper for old SegmentAdjuster class
class SegmentAdjuster:
    """
    DEPRECATED: Backward compatibility wrapper for old static method interface.

    Use the module-level functions instead:
    - SegmentAdjuster.merge_small_segments() -> merge_small_segments()
    - SegmentAdjuster.split_large_segments() -> split_large_segments()
    - SegmentAdjuster.apply_forced_splits() -> apply_forced_splits()
    - SegmentAdjuster.enforce_monotonicity() -> enforce_monotonicity()
    """

    @staticmethod
    def merge_small_segments(*args, **kwargs):
        """DEPRECATED: Use merge_small_segments() instead."""
        return merge_small_segments(*args, **kwargs)

    @staticmethod
    def split_large_segments(*args, **kwargs):
        """DEPRECATED: Use split_large_segments() instead."""
        return split_large_segments(*args, **kwargs)

    @staticmethod
    def apply_forced_splits(*args, **kwargs):
        """DEPRECATED: Use apply_forced_splits() instead."""
        return apply_forced_splits(*args, **kwargs)

    @staticmethod
    def enforce_monotonicity(*args, **kwargs):
        """DEPRECATED: Use enforce_monotonicity() instead."""
        return enforce_monotonicity(*args, **kwargs)

    @staticmethod
    def _find_best_split(*args, **kwargs):
        """DEPRECATED: Use find_best_split() instead."""
        return find_best_split(*args, **kwargs)

    @staticmethod
    def _gini_impurity(*args, **kwargs):
        """DEPRECATED: Use gini_impurity() instead."""
        return gini_impurity(*args, **kwargs)

    @staticmethod
    def _relabel_segments(*args, **kwargs):
        """DEPRECATED: Use relabel_segments() instead."""
        return relabel_segments(*args, **kwargs)


__all__ = [
    # Main entry points (most commonly used)
    'merge_small_segments',
    'split_large_segments',
    'apply_forced_splits',
    'enforce_monotonicity',

    # Backward compatibility
    'SegmentAdjuster',

    # Merging helpers
    'identify_segments_to_merge',
    'find_most_similar_segment',
    'merge_single_segment',

    # Splitting helpers
    'identify_segments_to_split',
    'find_best_split',
    'split_single_segment',

    # Forced split helpers
    'check_numeric_split_applicable',
    'apply_numeric_forced_split',
    'check_categorical_split_applicable',
    'apply_categorical_forced_split',

    # Monotonicity helpers
    'calculate_segment_feature_stats',
    'check_monotonicity_between_segments',
    'check_feature_monotonicity',

    # Utilities
    'relabel_segments',
    'calculate_default_rate',
    'calculate_segment_density',
    'get_segment_statistics',
    'gini_impurity',
]

__version__ = '2.0.0'  # Major version for modular refactor
