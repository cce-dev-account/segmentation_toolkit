"""
Segment Merging Functions

Functions for merging small segments to meet IRB density and default requirements.
"""

import numpy as np
from typing import List, Tuple
from ..logger import get_logger
from .utils import calculate_default_rate, relabel_segments

logger = get_logger(__name__)


def identify_segments_to_merge(
    segments: np.ndarray,
    y: np.ndarray,
    min_density: float,
    min_defaults: int
) -> List[int]:
    """
    Identify segments that don't meet minimum requirements.

    Args:
        segments: Array of segment labels
        y: Array of binary outcomes
        min_density: Minimum segment density threshold
        min_defaults: Minimum defaults per segment threshold

    Returns:
        List of segment labels that need merging

    Example:
        >>> segments = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
        >>> y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        >>> identify_segments_to_merge(segments, y, min_density=0.15, min_defaults=5)
        [0]  # Segment 0 has density 0.2 but 0 defaults
    """
    n_total = len(segments)
    unique_segments = np.unique(segments)
    segments_to_merge = []

    for seg in unique_segments:
        mask = segments == seg
        density = np.sum(mask) / n_total
        n_defaults = np.sum(y[mask])

        if density < min_density or n_defaults < min_defaults:
            segments_to_merge.append(int(seg))

    return segments_to_merge


def find_most_similar_segment(
    segments: np.ndarray,
    y: np.ndarray,
    target_segment: int,
    exclude_segments: List[int] = None
) -> Tuple[int, float]:
    """
    Find the segment most similar to the target by default rate.

    Args:
        segments: Array of segment labels
        y: Array of binary outcomes
        target_segment: Segment to find neighbor for
        exclude_segments: Optional list of segments to exclude from search

    Returns:
        Tuple of (most_similar_segment_id, default_rate_diff)

    Raises:
        ValueError: If no valid segments available for merging

    Example:
        >>> segments = np.array([0, 0, 1, 1, 2, 2])
        >>> y = np.array([0, 1, 0, 0, 1, 1])
        >>> find_most_similar_segment(segments, y, target_segment=0)
        (1, 0.25)  # Segment 1 has DR=0.0, closest to seg 0's DR=0.5
    """
    target_rate = calculate_default_rate(segments, y, target_segment)
    exclude_segments = exclude_segments or []

    # Get all other segments
    unique_segments = np.unique(segments)
    candidate_segments = [
        s for s in unique_segments
        if s != target_segment and s not in exclude_segments
    ]

    if not candidate_segments:
        raise ValueError(
            f"Cannot merge segment {target_segment}: no other segments available"
        )

    # Calculate similarity (smaller diff = more similar)
    similarities = []
    for seg in candidate_segments:
        seg_rate = calculate_default_rate(segments, y, seg)
        diff = abs(seg_rate - target_rate)
        similarities.append((seg, diff))

    # Return most similar (minimum difference)
    most_similar_seg, min_diff = min(similarities, key=lambda x: x[1])
    return int(most_similar_seg), float(min_diff)


def merge_single_segment(
    segments: np.ndarray,
    y: np.ndarray,
    source_segment: int,
    target_segment: int
) -> Tuple[np.ndarray, dict]:
    """
    Merge source segment into target segment.

    Args:
        segments: Array of segment labels
        y: Array of binary outcomes
        source_segment: Segment to merge away
        target_segment: Segment to merge into

    Returns:
        Tuple of (updated_segments, merge_log_entry)

    Example:
        >>> segments = np.array([0, 0, 1, 1])
        >>> y = np.array([0, 1, 0, 0])
        >>> new_segs, log = merge_single_segment(segments, y, source_segment=1, target_segment=0)
        >>> np.array_equal(new_segs, np.array([0, 0, 0, 0]))
        True
    """
    n_total = len(segments)
    updated_segments = segments.copy()

    # Get source segment info before merge
    source_mask = segments == source_segment
    source_density = np.sum(source_mask) / n_total
    source_defaults = int(np.sum(y[source_mask]))
    source_rate = calculate_default_rate(segments, y, source_segment)
    target_rate = calculate_default_rate(segments, y, target_segment)

    # Perform merge
    updated_segments[source_mask] = target_segment

    # Create log entry
    log_entry = {
        'merged_segment': int(source_segment),
        'into_segment': int(target_segment),
        'reason': 'density' if source_density < 0.10 else 'defaults',
        'original_density': float(source_density),
        'original_defaults': source_defaults,
        'default_rate_diff': float(abs(source_rate - target_rate))
    }

    return updated_segments, log_entry


def merge_small_segments(
    X: np.ndarray,
    segments: np.ndarray,
    y: np.ndarray,
    min_density: float = 0.10,
    min_defaults: int = 20
) -> Tuple[np.ndarray, List[dict]]:
    """
    Merge segments that don't meet minimum density or default requirements.

    Small segments are merged with their most similar neighbor based on
    default rate proximity. This is the main entry point for segment merging.

    Args:
        X: Feature matrix (not used but kept for API compatibility)
        segments: Current segment labels
        y: Binary outcomes
        min_density: Minimum segment density threshold
        min_defaults: Minimum defaults per segment threshold

    Returns:
        Tuple of (adjusted_segments, merge_log)
            - adjusted_segments: Updated segment labels with contiguous numbering
            - merge_log: List of dictionaries tracking each merge operation

    Example:
        >>> X = np.random.randn(100, 5)
        >>> segments = np.repeat([0, 1, 2], [10, 80, 10])
        >>> y = np.random.binomial(1, 0.1, 100)
        >>> new_segments, logs = merge_small_segments(X, segments, y)
        >>> len(np.unique(new_segments)) < 3  # Some segments merged
        True
    """
    adjusted_segments = segments.copy()
    merge_log = []

    # Identify segments needing merge
    segments_to_merge = identify_segments_to_merge(
        adjusted_segments, y, min_density, min_defaults
    )

    # Merge each small segment
    for seg in segments_to_merge:
        # Check if still exists (may have been merged already)
        if seg not in np.unique(adjusted_segments):
            continue

        try:
            # Find most similar neighbor
            target_seg, rate_diff = find_most_similar_segment(
                adjusted_segments, y, seg
            )

            # Perform merge
            adjusted_segments, log_entry = merge_single_segment(
                adjusted_segments, y, seg, target_seg
            )

            merge_log.append(log_entry)

        except ValueError as e:
            logger.warning(f"Cannot merge segment {seg}: {e}")
            continue

    # Relabel to contiguous integers
    adjusted_segments = relabel_segments(adjusted_segments)

    return adjusted_segments, merge_log
