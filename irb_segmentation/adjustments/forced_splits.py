"""
Forced Split Functions

Functions for applying business-mandated splits on specific features.
Supports both numeric thresholds and categorical membership splits.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from ..logger import get_logger
from .utils import relabel_segments

logger = get_logger(__name__)


def check_numeric_split_applicable(
    X: np.ndarray,
    segments: np.ndarray,
    segment_label: int,
    feature_idx: int,
    threshold: float
) -> bool:
    """
    Check if a numeric split is applicable to a segment.

    A split is applicable if the segment spans the threshold
    (i.e., has values both below and above the threshold).

    Args:
        X: Feature matrix
        segments: Array of segment labels
        segment_label: Segment to check
        feature_idx: Feature index
        threshold: Split threshold

    Returns:
        True if split is applicable, False otherwise

    Example:
        >>> X = np.array([[1], [2], [3], [4]])
        >>> segments = np.array([0, 0, 0, 0])
        >>> check_numeric_split_applicable(X, segments, 0, 0, 2.5)
        True  # Segment has values on both sides of 2.5
    """
    seg_mask = segments == segment_label
    seg_values = X[seg_mask, feature_idx]

    # Check if segment spans the threshold
    return seg_values.min() < threshold < seg_values.max()


def apply_numeric_forced_split(
    X: np.ndarray,
    segments: np.ndarray,
    segment_label: int,
    feature_idx: int,
    threshold: float,
    new_segment_id: int,
    feature_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, dict]:
    """
    Apply a numeric threshold split to a segment.

    Observations >= threshold are assigned to the new segment.

    Args:
        X: Feature matrix
        segments: Array of segment labels
        segment_label: Segment to split
        feature_idx: Feature index to split on
        threshold: Split threshold
        new_segment_id: ID for new segment
        feature_names: Optional feature names for logging

    Returns:
        Tuple of (updated_segments, split_log_entry)

    Example:
        >>> X = np.array([[1], [2], [3], [4]])
        >>> segments = np.array([0, 0, 0, 0])
        >>> new_segs, log = apply_numeric_forced_split(
        ...     X, segments, 0, 0, 2.5, 1
        ... )
        >>> np.array_equal(new_segs, np.array([0, 0, 1, 1]))
        True
    """
    updated_segments = segments.copy()
    seg_mask = segments == segment_label

    # Create split mask (values >= threshold go to new segment)
    split_mask = seg_mask & (X[:, feature_idx] >= threshold)

    # Apply split
    updated_segments[split_mask] = new_segment_id

    # Create log entry
    feature_name = (
        feature_names[feature_idx]
        if feature_names and feature_idx < len(feature_names)
        else f'feature_{feature_idx}'
    )

    log_entry = {
        'segment': int(segment_label),
        'new_segment': int(new_segment_id),
        'feature_idx': int(feature_idx),
        'feature_name': feature_name,
        'split_type': 'numeric',
        'threshold': float(threshold),
        'reason': 'forced_split'
    }

    return updated_segments, log_entry


def check_categorical_split_applicable(
    categorical_values: np.ndarray,
    segments: np.ndarray,
    segment_label: int,
    target_categories: List[str]
) -> bool:
    """
    Check if a categorical split is applicable to a segment.

    A split is applicable if the segment contains both target
    categories and non-target categories.

    Args:
        categorical_values: Array of categorical values
        segments: Array of segment labels
        segment_label: Segment to check
        target_categories: Categories to split on

    Returns:
        True if split is applicable, False otherwise

    Example:
        >>> cat_vals = np.array(['A', 'A', 'B', 'B'])
        >>> segments = np.array([0, 0, 0, 0])
        >>> check_categorical_split_applicable(cat_vals, segments, 0, ['A'])
        True  # Segment has both A and non-A values
    """
    seg_mask = segments == segment_label
    seg_cat_values = categorical_values[seg_mask]

    # Check if segment has both IN and NOT IN target categories
    has_target = np.any(np.isin(seg_cat_values, target_categories))
    has_non_target = np.any(~np.isin(seg_cat_values, target_categories))

    return has_target and has_non_target


def apply_categorical_forced_split(
    categorical_values: np.ndarray,
    segments: np.ndarray,
    segment_label: int,
    feature_idx: int,
    target_categories: List[str],
    new_segment_id: int,
    feature_name: str
) -> Tuple[np.ndarray, dict]:
    """
    Apply a categorical membership split to a segment.

    Observations in target categories are assigned to the new segment.

    Args:
        categorical_values: Array of categorical values
        segments: Array of segment labels
        segment_label: Segment to split
        feature_idx: Feature index (for logging)
        target_categories: Categories to move to new segment
        new_segment_id: ID for new segment
        feature_name: Feature name for logging

    Returns:
        Tuple of (updated_segments, split_log_entry)

    Example:
        >>> cat_vals = np.array(['A', 'A', 'B', 'B'])
        >>> segments = np.array([0, 0, 0, 0])
        >>> new_segs, log = apply_categorical_forced_split(
        ...     cat_vals, segments, 0, 0, ['A'], 1, 'category'
        ... )
        >>> np.array_equal(new_segs, np.array([1, 1, 0, 0]))
        True
    """
    updated_segments = segments.copy()
    seg_mask = segments == segment_label

    # Create split mask (observations IN target categories go to new segment)
    in_category_mask = seg_mask & np.isin(categorical_values, target_categories)

    # Apply split
    updated_segments[in_category_mask] = new_segment_id

    # Create log entry
    log_entry = {
        'segment': int(segment_label),
        'new_segment': int(new_segment_id),
        'feature_idx': int(feature_idx),
        'feature_name': feature_name,
        'split_type': 'categorical',
        'categories': target_categories,
        'reason': 'forced_categorical_split'
    }

    return updated_segments, log_entry


def apply_forced_splits(
    X: np.ndarray,
    segments: np.ndarray,
    forced_splits: Dict[int, Union[float, List[str]]],
    feature_names: Optional[List[str]] = None,
    X_categorical: Optional[Dict[str, np.ndarray]] = None
) -> Tuple[np.ndarray, List[dict]]:
    """
    Apply mandatory split points for specific features.

    Supports both numeric thresholds and categorical membership splits.
    This is the main entry point for forced splits.

    Args:
        X: Feature matrix (numeric features only)
        segments: Current segment labels
        forced_splits: Dict mapping feature index to:
            - float: numeric threshold for continuous features
            - list: categorical values for membership splits
        feature_names: Optional feature names for logging
        X_categorical: Optional dict mapping feature names to categorical arrays

    Returns:
        Tuple of (adjusted_segments, split_log)
            - adjusted_segments: Updated segment labels with contiguous numbering
            - split_log: List of dictionaries tracking each forced split

    Example:
        >>> X = np.array([[1, 10], [2, 20], [3, 30], [4, 40]])
        >>> segments = np.array([0, 0, 0, 0])
        >>> forced_splits = {0: 2.5}  # Split feature 0 at 2.5
        >>> new_segments, logs = apply_forced_splits(X, segments, forced_splits)
        >>> len(np.unique(new_segments))
        2
    """
    adjusted_segments = segments.copy()
    split_log = []
    next_segment_id = int(np.max(adjusted_segments)) + 1

    for feature_idx, split_value in forced_splits.items():
        # Get current unique segments (may change during iteration)
        unique_segments = np.unique(adjusted_segments)

        # Determine if numeric or categorical split
        is_categorical = isinstance(split_value, list)

        for seg in unique_segments:
            if is_categorical:
                # Categorical split
                feature_name = (
                    feature_names[feature_idx]
                    if feature_names and feature_idx < len(feature_names)
                    else f'feature_{feature_idx}'
                )

                # Get categorical values if provided
                if X_categorical and feature_name in X_categorical:
                    cat_values = X_categorical[feature_name]

                    # Check if split is applicable
                    if not check_categorical_split_applicable(
                        cat_values, adjusted_segments, seg, split_value
                    ):
                        continue

                    # Apply split
                    adjusted_segments, log_entry = apply_categorical_forced_split(
                        cat_values,
                        adjusted_segments,
                        seg,
                        feature_idx,
                        split_value,
                        next_segment_id,
                        feature_name
                    )

                    split_log.append(log_entry)
                    next_segment_id += 1

            else:
                # Numeric split
                threshold = float(split_value)

                # Check if split is applicable
                if not check_numeric_split_applicable(
                    X, adjusted_segments, seg, feature_idx, threshold
                ):
                    continue

                # Apply split
                adjusted_segments, log_entry = apply_numeric_forced_split(
                    X,
                    adjusted_segments,
                    seg,
                    feature_idx,
                    threshold,
                    next_segment_id,
                    feature_names
                )

                split_log.append(log_entry)
                next_segment_id += 1

    # Relabel to contiguous integers
    adjusted_segments = relabel_segments(adjusted_segments)

    return adjusted_segments, split_log
