"""
Post-Processing Adjustments for IRB Segmentation

This module provides functions to adjust segmentation results to meet IRB requirements.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
from .logger import get_logger

# Module-level logger
logger = get_logger(__name__)


class SegmentAdjuster:
    """
    Post-processing adjustments to enforce IRB constraints on segments.
    """

    @staticmethod
    def merge_small_segments(
        X: np.ndarray,
        segments: np.ndarray,
        y: np.ndarray,
        min_density: float = 0.10,
        min_defaults: int = 20
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Merge segments that don't meet minimum density or default requirements.

        Small segments are merged with their most similar neighbor based on
        default rate proximity.

        Args:
            X: Feature matrix
            segments: Current segment labels
            y: Binary outcomes
            min_density: Minimum segment density
            min_defaults: Minimum defaults per segment

        Returns:
            Tuple of (adjusted_segments, merge_log)
        """
        n_total = len(segments)
        adjusted_segments = segments.copy()
        merge_log = []

        # Identify segments that need merging
        unique_segments = np.unique(adjusted_segments)
        segments_to_merge = []

        for seg in unique_segments:
            mask = adjusted_segments == seg
            density = np.sum(mask) / n_total
            n_defaults = np.sum(y[mask])

            if density < min_density or n_defaults < min_defaults:
                segments_to_merge.append(seg)

        # Calculate default rates for all segments
        def get_default_rate(seg_label):
            mask = adjusted_segments == seg_label
            return np.mean(y[mask]) if np.sum(mask) > 0 else 0

        # Merge small segments one by one
        for seg in segments_to_merge:
            if seg not in np.unique(adjusted_segments):
                continue  # Already merged

            seg_mask = adjusted_segments == seg
            seg_default_rate = get_default_rate(seg)

            # Find most similar neighbor by default rate
            other_segments = [s for s in np.unique(adjusted_segments) if s != seg]
            if not other_segments:
                logger.warning(f"Cannot merge segment {seg}: no other segments available")
                continue

            similarities = [
                (other_seg, abs(get_default_rate(other_seg) - seg_default_rate))
                for other_seg in other_segments
            ]
            most_similar = min(similarities, key=lambda x: x[1])[0]

            # Perform merge
            adjusted_segments[seg_mask] = most_similar

            merge_log.append({
                'merged_segment': int(seg),
                'into_segment': int(most_similar),
                'reason': 'density' if np.sum(seg_mask) / n_total < min_density else 'defaults',
                'original_density': float(np.sum(seg_mask) / n_total),
                'original_defaults': int(np.sum(y[seg_mask])),
                'default_rate_diff': float(similarities[0][1])
            })

        # Relabel segments to be contiguous
        adjusted_segments = SegmentAdjuster._relabel_segments(adjusted_segments)

        return adjusted_segments, merge_log

    @staticmethod
    def split_large_segments(
        X: np.ndarray,
        segments: np.ndarray,
        y: np.ndarray,
        max_density: float = 0.50,
        feature_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Split segments that exceed maximum density using secondary features.

        Args:
            X: Feature matrix
            segments: Current segment labels
            y: Binary outcomes
            max_density: Maximum segment density
            feature_names: Optional feature names for logging

        Returns:
            Tuple of (adjusted_segments, split_log)
        """
        n_total = len(segments)
        adjusted_segments = segments.copy()
        split_log = []
        next_segment_id = int(np.max(adjusted_segments)) + 1

        unique_segments = np.unique(adjusted_segments)

        for seg in unique_segments:
            mask = adjusted_segments == seg
            density = np.sum(mask) / n_total

            if density > max_density:
                # Find best feature and threshold to split on
                seg_X = X[mask]
                seg_y = y[mask]

                best_split = SegmentAdjuster._find_best_split(seg_X, seg_y)

                if best_split is not None:
                    feature_idx, threshold = best_split

                    # Apply split
                    split_mask = mask & (X[:, feature_idx] <= threshold)
                    adjusted_segments[split_mask] = next_segment_id

                    split_log.append({
                        'split_segment': int(seg),
                        'new_segment': next_segment_id,
                        'feature_idx': int(feature_idx),
                        'feature_name': feature_names[feature_idx] if feature_names else f'feature_{feature_idx}',
                        'threshold': float(threshold),
                        'original_density': float(density),
                        'new_densities': [
                            float(np.sum(split_mask) / n_total),
                            float(np.sum(mask & ~split_mask) / n_total)
                        ]
                    })

                    next_segment_id += 1

        # Relabel segments to be contiguous
        adjusted_segments = SegmentAdjuster._relabel_segments(adjusted_segments)

        return adjusted_segments, split_log

    @staticmethod
    def apply_forced_splits(
        X: np.ndarray,
        segments: np.ndarray,
        forced_splits: Dict[int, any],
        feature_names: Optional[List[str]] = None,
        X_categorical: Optional[Dict[str, np.ndarray]] = None
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Apply mandatory split points for specific features.

        Supports both numeric thresholds and categorical membership splits.

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
        """
        adjusted_segments = segments.copy()
        split_log = []
        next_segment_id = int(np.max(adjusted_segments)) + 1

        for feature_idx, split_value in forced_splits.items():
            # For each existing segment, check if it spans the forced split
            unique_segments = np.unique(adjusted_segments)

            # Determine if numeric or categorical split
            is_categorical = isinstance(split_value, list)

            for seg in unique_segments:
                mask = adjusted_segments == seg

                if is_categorical:
                    # Categorical split: membership condition
                    feature_name = feature_names[feature_idx] if feature_names else f'feature_{feature_idx}'

                    # Get categorical values (if provided in X_categorical)
                    if X_categorical and feature_name in X_categorical:
                        cat_values = X_categorical[feature_name]

                        # Split: observations IN specified categories vs NOT IN
                        in_category_mask = mask & np.isin(cat_values, split_value)

                        # Only split if segment contains both IN and NOT IN
                        if np.any(in_category_mask) and np.any(mask & ~np.isin(cat_values, split_value)):
                            adjusted_segments[in_category_mask] = next_segment_id

                            split_log.append({
                                'segment': int(seg),
                                'new_segment': next_segment_id,
                                'feature_idx': int(feature_idx),
                                'feature_name': feature_name,
                                'split_type': 'categorical',
                                'categories': split_value,
                                'reason': 'forced_categorical_split'
                            })

                            next_segment_id += 1
                else:
                    # Numeric split: threshold condition
                    threshold = float(split_value)
                    seg_values = X[mask, feature_idx]

                    # Check if segment spans the threshold
                    if seg_values.min() < threshold < seg_values.max():
                        # Split segment at threshold
                        split_mask = mask & (X[:, feature_idx] >= threshold)
                        adjusted_segments[split_mask] = next_segment_id

                        split_log.append({
                            'segment': int(seg),
                            'new_segment': next_segment_id,
                            'feature_idx': int(feature_idx),
                            'feature_name': feature_names[feature_idx] if feature_names else f'feature_{feature_idx}',
                            'split_type': 'numeric',
                            'threshold': float(threshold),
                            'reason': 'forced_split'
                        })

                        next_segment_id += 1

        # Relabel segments to be contiguous
        adjusted_segments = SegmentAdjuster._relabel_segments(adjusted_segments)

        return adjusted_segments, split_log

    @staticmethod
    def enforce_monotonicity(
        X: np.ndarray,
        segments: np.ndarray,
        y: np.ndarray,
        monotone_constraints: Dict[int, int],
        feature_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Enforce monotonicity constraints on default rates with respect to features.

        Args:
            X: Feature matrix
            segments: Current segment labels
            y: Binary outcomes
            monotone_constraints: Dict mapping feature idx to direction (1: increasing, -1: decreasing)
            feature_names: Optional feature names for logging

        Returns:
            Tuple of (adjusted_segments, violation_log)
        """
        adjusted_segments = segments.copy()
        violation_log = []

        for feature_idx, direction in monotone_constraints.items():
            # Get segment default rates and feature medians
            unique_segments = np.unique(adjusted_segments)
            segment_stats = []

            for seg in unique_segments:
                mask = adjusted_segments == seg
                default_rate = np.mean(y[mask])
                feature_median = np.median(X[mask, feature_idx])
                segment_stats.append((seg, feature_median, default_rate))

            # Sort by feature median
            segment_stats.sort(key=lambda x: x[1])

            # Check monotonicity
            for i in range(len(segment_stats) - 1):
                seg1, med1, rate1 = segment_stats[i]
                seg2, med2, rate2 = segment_stats[i + 1]

                # Check if monotonicity is violated
                if direction == 1 and rate2 < rate1:
                    violation_log.append({
                        'feature_idx': int(feature_idx),
                        'feature_name': feature_names[feature_idx] if feature_names else f'feature_{feature_idx}',
                        'direction': 'increasing',
                        'segment1': int(seg1),
                        'segment2': int(seg2),
                        'feature_median1': float(med1),
                        'feature_median2': float(med2),
                        'default_rate1': float(rate1),
                        'default_rate2': float(rate2),
                        'violated': True
                    })
                elif direction == -1 and rate2 > rate1:
                    violation_log.append({
                        'feature_idx': int(feature_idx),
                        'feature_name': feature_names[feature_idx] if feature_names else f'feature_{feature_idx}',
                        'direction': 'decreasing',
                        'segment1': int(seg1),
                        'segment2': int(seg2),
                        'feature_median1': float(med1),
                        'feature_median2': float(med2),
                        'default_rate1': float(rate1),
                        'default_rate2': float(rate2),
                        'violated': True
                    })

        # Note: This function logs violations but doesn't automatically fix them
        # Fixing would require merging segments, which may conflict with other constraints
        return adjusted_segments, violation_log

    @staticmethod
    def _find_best_split(
        X: np.ndarray,
        y: np.ndarray
    ) -> Optional[Tuple[int, float]]:
        """
        Find the best feature and threshold to split a segment.

        Uses Gini impurity as the splitting criterion.

        Args:
            X: Feature matrix for the segment
            y: Binary outcomes for the segment

        Returns:
            Tuple of (feature_index, threshold) or None if no good split found
        """
        if len(y) < 100:  # Don't split very small segments
            return None

        best_gini = float('inf')
        best_split = None

        n_features = X.shape[1]

        for feature_idx in range(n_features):
            # Get sorted unique values
            feature_values = X[:, feature_idx]
            thresholds = np.percentile(feature_values, [25, 50, 75])

            for threshold in thresholds:
                # Split
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < 50 or np.sum(right_mask) < 50:
                    continue  # Avoid tiny splits

                # Calculate Gini impurity
                left_gini = SegmentAdjuster._gini_impurity(y[left_mask])
                right_gini = SegmentAdjuster._gini_impurity(y[right_mask])

                # Weighted average
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                weighted_gini = (n_left * left_gini + n_right * right_gini) / len(y)

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_split = (feature_idx, threshold)

        return best_split

    @staticmethod
    def _gini_impurity(y: np.ndarray) -> float:
        """Calculate Gini impurity for a set of binary outcomes."""
        if len(y) == 0:
            return 0.0
        p = np.mean(y)
        return 2 * p * (1 - p)

    @staticmethod
    def _relabel_segments(segments: np.ndarray) -> np.ndarray:
        """Relabel segments to be contiguous integers starting from 0."""
        unique_segments = np.unique(segments)
        relabeled = segments.copy()

        for new_label, old_label in enumerate(unique_segments):
            relabeled[segments == old_label] = new_label

        return relabeled
