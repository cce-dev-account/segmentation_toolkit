"""
Segment Splitting Functions

Functions for splitting large segments to meet IRB maximum density requirements.
"""

import numpy as np
from typing import List, Tuple, Optional
from ..logger import get_logger
from .utils import gini_impurity, relabel_segments

logger = get_logger(__name__)


def identify_segments_to_split(
    segments: np.ndarray,
    max_density: float
) -> List[int]:
    """
    Identify segments that exceed maximum density threshold.

    Args:
        segments: Array of segment labels
        max_density: Maximum segment density threshold

    Returns:
        List of segment labels that need splitting

    Example:
        >>> segments = np.repeat([0, 1], [80, 20])
        >>> identify_segments_to_split(segments, max_density=0.50)
        [0]  # Segment 0 has density 0.8, exceeds max
    """
    n_total = len(segments)
    unique_segments = np.unique(segments)
    segments_to_split = []

    for seg in unique_segments:
        mask = segments == seg
        density = np.sum(mask) / n_total

        if density > max_density:
            segments_to_split.append(int(seg))

    return segments_to_split


def find_best_split(
    X: np.ndarray,
    y: np.ndarray,
    min_samples_per_split: int = 50
) -> Optional[Tuple[int, float]]:
    """
    Find the best feature and threshold to split a segment.

    Uses Gini impurity as the splitting criterion. Evaluates splits
    at 25th, 50th, and 75th percentiles for each feature.

    Args:
        X: Feature matrix for the segment
        y: Binary outcomes for the segment
        min_samples_per_split: Minimum observations required in each split

    Returns:
        Tuple of (feature_index, threshold) or None if no good split found

    Example:
        >>> X = np.array([[1, 5], [2, 6], [3, 4], [4, 3]])
        >>> y = np.array([0, 0, 1, 1])
        >>> feature_idx, threshold = find_best_split(X, y, min_samples_per_split=2)
        >>> feature_idx in [0, 1]  # Returns valid feature
        True
    """
    if len(y) < min_samples_per_split * 2:
        # Too small to split
        return None

    best_gini = float('inf')
    best_split = None
    n_features = X.shape[1]

    for feature_idx in range(n_features):
        # Get feature values and candidate thresholds
        feature_values = X[:, feature_idx]
        thresholds = np.percentile(feature_values, [25, 50, 75])

        for threshold in thresholds:
            # Create split masks
            left_mask = feature_values <= threshold
            right_mask = ~left_mask

            n_left = np.sum(left_mask)
            n_right = np.sum(right_mask)

            # Check minimum samples constraint
            if n_left < min_samples_per_split or n_right < min_samples_per_split:
                continue

            # Calculate weighted Gini impurity
            left_gini = gini_impurity(y[left_mask])
            right_gini = gini_impurity(y[right_mask])
            weighted_gini = (n_left * left_gini + n_right * right_gini) / len(y)

            # Update best split
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_split = (feature_idx, float(threshold))

    return best_split


def split_single_segment(
    X: np.ndarray,
    segments: np.ndarray,
    y: np.ndarray,
    segment_to_split: int,
    feature_idx: int,
    threshold: float,
    new_segment_id: int,
    feature_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, dict]:
    """
    Split a segment at a specified feature threshold.

    Args:
        X: Feature matrix
        segments: Array of segment labels
        y: Array of binary outcomes
        segment_to_split: Segment to split
        feature_idx: Feature index to split on
        threshold: Split threshold
        new_segment_id: ID to assign to new segment
        feature_names: Optional feature names for logging

    Returns:
        Tuple of (updated_segments, split_log_entry)

    Example:
        >>> X = np.array([[1], [2], [3], [4]])
        >>> segments = np.array([0, 0, 0, 0])
        >>> y = np.array([0, 0, 1, 1])
        >>> new_segs, log = split_single_segment(
        ...     X, segments, y, segment_to_split=0,
        ...     feature_idx=0, threshold=2.5, new_segment_id=1
        ... )
        >>> np.array_equal(new_segs, np.array([1, 1, 0, 0]))
        True
    """
    n_total = len(segments)
    updated_segments = segments.copy()

    # Get segment mask and original density
    seg_mask = segments == segment_to_split
    original_density = np.sum(seg_mask) / n_total

    # Create split mask (values <= threshold go to new segment)
    split_mask = seg_mask & (X[:, feature_idx] <= threshold)

    # Apply split
    updated_segments[split_mask] = new_segment_id

    # Calculate new densities
    new_density_left = np.sum(split_mask) / n_total
    new_density_right = np.sum(seg_mask & ~split_mask) / n_total

    # Create log entry
    feature_name = (
        feature_names[feature_idx]
        if feature_names and feature_idx < len(feature_names)
        else f'feature_{feature_idx}'
    )

    log_entry = {
        'split_segment': int(segment_to_split),
        'new_segment': int(new_segment_id),
        'feature_idx': int(feature_idx),
        'feature_name': feature_name,
        'threshold': float(threshold),
        'original_density': float(original_density),
        'new_densities': [float(new_density_left), float(new_density_right)]
    }

    return updated_segments, log_entry


def split_large_segments(
    X: np.ndarray,
    segments: np.ndarray,
    y: np.ndarray,
    max_density: float = 0.50,
    feature_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[dict]]:
    """
    Split segments that exceed maximum density using secondary features.

    This is the main entry point for segment splitting. Identifies large
    segments and splits them using the best available feature/threshold.

    Args:
        X: Feature matrix
        segments: Current segment labels
        y: Binary outcomes
        max_density: Maximum segment density threshold
        feature_names: Optional feature names for logging

    Returns:
        Tuple of (adjusted_segments, split_log)
            - adjusted_segments: Updated segment labels with contiguous numbering
            - split_log: List of dictionaries tracking each split operation

    Example:
        >>> X = np.random.randn(100, 5)
        >>> segments = np.zeros(100, dtype=int)  # All in one segment
        >>> y = np.random.binomial(1, 0.1, 100)
        >>> new_segments, logs = split_large_segments(X, segments, y, max_density=0.50)
        >>> len(np.unique(new_segments)) > 1  # Segment was split
        True
    """
    adjusted_segments = segments.copy()
    split_log = []
    next_segment_id = int(np.max(adjusted_segments)) + 1

    # Identify segments needing split
    segments_to_split = identify_segments_to_split(adjusted_segments, max_density)

    for seg in segments_to_split:
        # Get segment data
        seg_mask = adjusted_segments == seg
        seg_X = X[seg_mask]
        seg_y = y[seg_mask]

        # Find best split point
        split_result = find_best_split(seg_X, seg_y)

        if split_result is None:
            logger.warning(
                f"Could not find valid split for segment {seg} "
                f"(size: {np.sum(seg_mask)})"
            )
            continue

        feature_idx, threshold = split_result

        # Apply split
        adjusted_segments, log_entry = split_single_segment(
            X, adjusted_segments, y,
            segment_to_split=seg,
            feature_idx=feature_idx,
            threshold=threshold,
            new_segment_id=next_segment_id,
            feature_names=feature_names
        )

        split_log.append(log_entry)
        next_segment_id += 1

    # Relabel to contiguous integers
    adjusted_segments = relabel_segments(adjusted_segments)

    return adjusted_segments, split_log
