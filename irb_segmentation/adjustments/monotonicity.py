"""
Monotonicity Constraint Functions

Functions for checking and enforcing monotonicity constraints on default rates
with respect to feature values.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from ..logger import get_logger

logger = get_logger(__name__)


def calculate_segment_feature_stats(
    X: np.ndarray,
    y: np.ndarray,
    segments: np.ndarray,
    feature_idx: int
) -> List[Tuple[int, float, float]]:
    """
    Calculate feature median and default rate for each segment.

    Args:
        X: Feature matrix
        y: Binary outcomes
        segments: Array of segment labels
        feature_idx: Feature index to analyze

    Returns:
        List of tuples (segment_id, feature_median, default_rate)
        sorted by feature median

    Example:
        >>> X = np.array([[1], [2], [3], [4]])
        >>> y = np.array([0, 0, 1, 1])
        >>> segments = np.array([0, 0, 1, 1])
        >>> stats = calculate_segment_feature_stats(X, y, segments, 0)
        >>> stats[0][0]  # First segment ID
        0
        >>> stats[0][1]  # Feature median for first segment
        1.5
    """
    unique_segments = np.unique(segments)
    segment_stats = []

    for seg in unique_segments:
        mask = segments == seg
        default_rate = np.mean(y[mask])
        feature_median = np.median(X[mask, feature_idx])
        segment_stats.append((int(seg), float(feature_median), float(default_rate)))

    # Sort by feature median
    segment_stats.sort(key=lambda x: x[1])

    return segment_stats


def check_monotonicity_between_segments(
    seg1_id: int,
    seg1_median: float,
    seg1_rate: float,
    seg2_id: int,
    seg2_median: float,
    seg2_rate: float,
    direction: int
) -> bool:
    """
    Check if monotonicity holds between two adjacent segments.

    Args:
        seg1_id: First segment ID
        seg1_median: Feature median for first segment
        seg1_rate: Default rate for first segment
        seg2_id: Second segment ID
        seg2_median: Feature median for second segment
        seg2_rate: Default rate for second segment
        direction: Monotonicity direction (1: increasing, -1: decreasing)

    Returns:
        True if monotonicity is satisfied, False if violated

    Example:
        >>> # Increasing monotonicity: higher feature -> higher rate
        >>> check_monotonicity_between_segments(
        ...     0, 1.0, 0.1,  # seg 0: feature=1.0, rate=0.1
        ...     1, 2.0, 0.2,  # seg 1: feature=2.0, rate=0.2
        ...     direction=1   # increasing
        ... )
        True  # rate2 >= rate1, monotonicity satisfied
    """
    if direction == 1:
        # Increasing: higher feature value should have higher (or equal) default rate
        return seg2_rate >= seg1_rate
    elif direction == -1:
        # Decreasing: higher feature value should have lower (or equal) default rate
        return seg2_rate <= seg1_rate
    else:
        # No constraint
        return True


def create_violation_log_entry(
    feature_idx: int,
    feature_name: str,
    direction: int,
    seg1_id: int,
    seg1_median: float,
    seg1_rate: float,
    seg2_id: int,
    seg2_median: float,
    seg2_rate: float,
    violated: bool
) -> dict:
    """
    Create a log entry for monotonicity check.

    Args:
        feature_idx: Feature index
        feature_name: Feature name
        direction: Monotonicity direction (1 or -1)
        seg1_id: First segment ID
        seg1_median: Feature median for first segment
        seg1_rate: Default rate for first segment
        seg2_id: Second segment ID
        seg2_median: Feature median for second segment
        seg2_rate: Default rate for second segment
        violated: Whether monotonicity was violated

    Returns:
        Dictionary with violation details

    Example:
        >>> log = create_violation_log_entry(
        ...     0, 'age', 1, 0, 30.0, 0.1, 1, 40.0, 0.08, True
        ... )
        >>> log['violated']
        True
        >>> log['direction']
        'increasing'
    """
    return {
        'feature_idx': int(feature_idx),
        'feature_name': feature_name,
        'direction': 'increasing' if direction == 1 else 'decreasing',
        'segment1': int(seg1_id),
        'segment2': int(seg2_id),
        'feature_median1': float(seg1_median),
        'feature_median2': float(seg2_median),
        'default_rate1': float(seg1_rate),
        'default_rate2': float(seg2_rate),
        'violated': bool(violated)
    }


def check_feature_monotonicity(
    X: np.ndarray,
    y: np.ndarray,
    segments: np.ndarray,
    feature_idx: int,
    direction: int,
    feature_name: str
) -> List[dict]:
    """
    Check monotonicity constraint for a single feature.

    Args:
        X: Feature matrix
        y: Binary outcomes
        segments: Array of segment labels
        feature_idx: Feature index to check
        direction: Monotonicity direction (1: increasing, -1: decreasing)
        feature_name: Feature name for logging

    Returns:
        List of violation log entries (empty if no violations)

    Example:
        >>> X = np.array([[1], [2], [3], [4]])
        >>> y = np.array([0.0, 0.0, 1.0, 1.0])
        >>> segments = np.array([0, 0, 1, 1])
        >>> violations = check_feature_monotonicity(
        ...     X, y, segments, 0, direction=1, feature_name='feature_0'
        ... )
        >>> len(violations)  # No violations: feature increases, rate increases
        0
    """
    violation_log = []

    # Get segment statistics sorted by feature median
    segment_stats = calculate_segment_feature_stats(X, y, segments, feature_idx)

    # Check monotonicity between adjacent segments
    for i in range(len(segment_stats) - 1):
        seg1_id, seg1_median, seg1_rate = segment_stats[i]
        seg2_id, seg2_median, seg2_rate = segment_stats[i + 1]

        # Check if monotonicity is satisfied
        is_satisfied = check_monotonicity_between_segments(
            seg1_id, seg1_median, seg1_rate,
            seg2_id, seg2_median, seg2_rate,
            direction
        )

        violated = not is_satisfied

        # Log violation (or successful check if direction matches)
        if violated:
            log_entry = create_violation_log_entry(
                feature_idx, feature_name, direction,
                seg1_id, seg1_median, seg1_rate,
                seg2_id, seg2_median, seg2_rate,
                violated=True
            )
            violation_log.append(log_entry)

    return violation_log


def enforce_monotonicity(
    X: np.ndarray,
    segments: np.ndarray,
    y: np.ndarray,
    monotone_constraints: Dict[int, int],
    feature_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[dict]]:
    """
    Enforce monotonicity constraints on default rates with respect to features.

    This is the main entry point for monotonicity checking. Currently logs
    violations but does not automatically fix them, as fixes may conflict
    with other IRB constraints.

    Args:
        X: Feature matrix
        segments: Current segment labels
        y: Binary outcomes
        monotone_constraints: Dict mapping feature idx to direction
            (1: increasing, -1: decreasing)
        feature_names: Optional feature names for logging

    Returns:
        Tuple of (adjusted_segments, violation_log)
            - adjusted_segments: Unchanged segment labels
            - violation_log: List of dictionaries tracking violations

    Example:
        >>> X = np.array([[1], [2], [3], [4]])
        >>> y = np.array([0.2, 0.1, 0.05, 0.01])
        >>> segments = np.array([0, 0, 1, 1])
        >>> monotone_constraints = {0: -1}  # Feature 0 should decrease with rate
        >>> adjusted_segs, violations = enforce_monotonicity(
        ...     X, segments, y, monotone_constraints
        ... )
        >>> len(violations)  # Violations found
        0  # Or number of violations
    """
    adjusted_segments = segments.copy()
    violation_log = []

    for feature_idx, direction in monotone_constraints.items():
        # Get feature name
        feature_name = (
            feature_names[feature_idx]
            if feature_names and feature_idx < len(feature_names)
            else f'feature_{feature_idx}'
        )

        # Check monotonicity for this feature
        feature_violations = check_feature_monotonicity(
            X, y, adjusted_segments, feature_idx, direction, feature_name
        )

        violation_log.extend(feature_violations)

    # Log results
    if violation_log:
        logger.warning(
            f"Found {len(violation_log)} monotonicity violations across "
            f"{len(monotone_constraints)} constrained features"
        )
    else:
        logger.info(
            f"All {len(monotone_constraints)} monotonicity constraints satisfied"
        )

    # Note: This function logs violations but doesn't automatically fix them
    # Fixing would require merging segments, which may conflict with other constraints
    return adjusted_segments, violation_log
