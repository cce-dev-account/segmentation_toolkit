"""
Utility Functions for Segment Adjustments

Pure helper functions used across adjustment modules.
"""

import numpy as np
from typing import Dict


def relabel_segments(segments: np.ndarray) -> np.ndarray:
    """
    Relabel segments to be contiguous integers starting from 0.

    Args:
        segments: Array of segment labels (may have gaps)

    Returns:
        Array with relabeled segments (0, 1, 2, ...)

    Example:
        >>> segments = np.array([10, 10, 25, 25, 30])
        >>> relabel_segments(segments)
        array([0, 0, 1, 1, 2])
    """
    unique_segments = np.unique(segments)
    relabeled = segments.copy()

    for new_label, old_label in enumerate(unique_segments):
        relabeled[segments == old_label] = new_label

    return relabeled


def calculate_default_rate(
    segments: np.ndarray,
    y: np.ndarray,
    segment_label: int
) -> float:
    """
    Calculate default rate for a specific segment.

    Args:
        segments: Array of segment labels
        y: Array of binary outcomes (0/1)
        segment_label: Segment to calculate rate for

    Returns:
        Default rate (proportion of defaults) in [0, 1]

    Example:
        >>> segments = np.array([0, 0, 1, 1])
        >>> y = np.array([0, 1, 0, 0])
        >>> calculate_default_rate(segments, y, 0)
        0.5
    """
    mask = segments == segment_label
    n_obs = np.sum(mask)

    if n_obs == 0:
        return 0.0

    return float(np.mean(y[mask]))


def calculate_segment_density(
    segments: np.ndarray,
    segment_label: int
) -> float:
    """
    Calculate density (proportion of total observations) for a segment.

    Args:
        segments: Array of segment labels
        segment_label: Segment to calculate density for

    Returns:
        Segment density in (0, 1]

    Example:
        >>> segments = np.array([0, 0, 1, 1, 1])
        >>> calculate_segment_density(segments, 1)
        0.6
    """
    mask = segments == segment_label
    return float(np.sum(mask) / len(segments))


def get_segment_statistics(
    segments: np.ndarray,
    y: np.ndarray,
    segment_label: int
) -> Dict[str, float]:
    """
    Calculate comprehensive statistics for a segment.

    Args:
        segments: Array of segment labels
        y: Array of binary outcomes (0/1)
        segment_label: Segment to analyze

    Returns:
        Dictionary with:
            - n_observations: Number of observations
            - n_defaults: Number of default events
            - default_rate: Proportion of defaults
            - density: Proportion of total population

    Example:
        >>> segments = np.array([0, 0, 1, 1])
        >>> y = np.array([0, 1, 0, 0])
        >>> stats = get_segment_statistics(segments, y, 0)
        >>> stats['default_rate']
        0.5
    """
    mask = segments == segment_label
    n_obs = int(np.sum(mask))
    n_defaults = int(np.sum(y[mask]))

    return {
        'n_observations': n_obs,
        'n_defaults': n_defaults,
        'default_rate': float(np.mean(y[mask])) if n_obs > 0 else 0.0,
        'density': float(n_obs / len(segments))
    }


def gini_impurity(y: np.ndarray) -> float:
    """
    Calculate Gini impurity for binary outcomes.

    Gini impurity measures the probability of incorrectly classifying
    a randomly chosen element if it were randomly labeled according
    to the distribution in the subset.

    Args:
        y: Array of binary outcomes (0/1)

    Returns:
        Gini impurity in [0, 0.5]

    Example:
        >>> y = np.array([0, 0, 1, 1])
        >>> gini_impurity(y)
        0.5
    """
    if len(y) == 0:
        return 0.0

    p = np.mean(y)
    return float(2 * p * (1 - p))
