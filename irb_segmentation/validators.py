"""
Segment Validation Framework

This module provides comprehensive validation functions for IRB PD model segments.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings


class SegmentValidator:
    """
    Comprehensive validator for IRB PD model segments.

    Performs statistical and regulatory validation of segmentation results.
    """

    @staticmethod
    def validate_statistical_significance(
        segments: np.ndarray,
        y: np.ndarray,
        significance_level: float = 0.01,
        method: str = 'chi_squared'
    ) -> Dict[str, any]:
        """
        Test statistical significance of segments using pairwise comparisons.

        Args:
            segments: Array of segment labels
            y: Array of binary outcomes (0/1)
            significance_level: Significance level for tests (after Bonferroni correction)
            method: Statistical test method ('chi_squared' or 'fisher')

        Returns:
            Dictionary with validation results including:
                - passed: Boolean indicating if all segments are significantly different
                - p_values: Matrix of pairwise p-values
                - failed_pairs: List of segment pairs that are not significantly different
        """
        unique_segments = np.unique(segments)
        n_segments = len(unique_segments)
        n_comparisons = n_segments * (n_segments - 1) // 2

        # Bonferroni correction
        adjusted_alpha = significance_level / n_comparisons if n_comparisons > 0 else significance_level

        # Initialize results
        p_values = np.ones((n_segments, n_segments))
        failed_pairs = []

        # Perform pairwise comparisons
        for i, seg1 in enumerate(unique_segments):
            for j, seg2 in enumerate(unique_segments):
                if i >= j:
                    continue

                mask1 = segments == seg1
                mask2 = segments == seg2

                y1 = y[mask1]
                y2 = y[mask2]

                # Create contingency table
                defaults1, non_defaults1 = np.sum(y1), len(y1) - np.sum(y1)
                defaults2, non_defaults2 = np.sum(y2), len(y2) - np.sum(y2)

                contingency_table = np.array([
                    [defaults1, non_defaults1],
                    [defaults2, non_defaults2]
                ])

                # Perform test
                if method == 'chi_squared':
                    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                    p_values[i, j] = p_values[j, i] = p_value
                elif method == 'fisher':
                    odds_ratio, p_value = stats.fisher_exact(contingency_table)
                    p_values[i, j] = p_values[j, i] = p_value
                else:
                    raise ValueError(f"Unknown method: {method}")

                # Check if significantly different
                if p_value >= adjusted_alpha:
                    failed_pairs.append((int(seg1), int(seg2), p_value))

        return {
            'passed': len(failed_pairs) == 0,
            'p_values': p_values,
            'failed_pairs': failed_pairs,
            'adjusted_alpha': adjusted_alpha,
            'n_comparisons': n_comparisons,
            'method': method
        }

    @staticmethod
    def validate_minimum_defaults(
        segments: np.ndarray,
        y: np.ndarray,
        min_defaults: int = 20
    ) -> Dict[str, any]:
        """
        Check that each segment has sufficient default events.

        Args:
            segments: Array of segment labels
            y: Array of binary outcomes (0/1)
            min_defaults: Minimum number of defaults required per segment

        Returns:
            Dictionary with validation results including:
                - passed: Boolean indicating if all segments meet minimum
                - defaults_per_segment: Dict mapping segment to default count
                - failed_segments: List of segments below minimum
        """
        unique_segments = np.unique(segments)
        defaults_per_segment = {}
        failed_segments = []

        for seg in unique_segments:
            mask = segments == seg
            n_defaults = np.sum(y[mask])
            defaults_per_segment[int(seg)] = int(n_defaults)

            if n_defaults < min_defaults:
                failed_segments.append({
                    'segment': int(seg),
                    'defaults': int(n_defaults),
                    'required': min_defaults,
                    'shortfall': min_defaults - int(n_defaults)
                })

        return {
            'passed': len(failed_segments) == 0,
            'defaults_per_segment': defaults_per_segment,
            'failed_segments': failed_segments,
            'min_defaults': min_defaults
        }

    @staticmethod
    def validate_density(
        segments: np.ndarray,
        min_density: float = 0.10,
        max_density: float = 0.50
    ) -> Dict[str, any]:
        """
        Check segment size distribution against density constraints.

        Args:
            segments: Array of segment labels
            min_density: Minimum proportion of observations per segment
            max_density: Maximum proportion of observations per segment

        Returns:
            Dictionary with validation results including:
                - passed: Boolean indicating if all segments meet density requirements
                - density_per_segment: Dict mapping segment to density
                - failed_segments: List of segments outside density bounds
        """
        unique_segments = np.unique(segments)
        n_total = len(segments)
        density_per_segment = {}
        failed_segments = []

        for seg in unique_segments:
            count = np.sum(segments == seg)
            density = count / n_total
            density_per_segment[int(seg)] = density

            if density < min_density or density > max_density:
                failed_segments.append({
                    'segment': int(seg),
                    'density': density,
                    'count': int(count),
                    'min_density': min_density,
                    'max_density': max_density,
                    'violation': 'too_small' if density < min_density else 'too_large'
                })

        return {
            'passed': len(failed_segments) == 0,
            'density_per_segment': density_per_segment,
            'failed_segments': failed_segments,
            'min_density': min_density,
            'max_density': max_density
        }

    @staticmethod
    def calculate_psi(
        reference_segments: np.ndarray,
        current_segments: np.ndarray,
        threshold: float = 0.1
    ) -> Dict[str, any]:
        """
        Calculate Population Stability Index for temporal validation.

        PSI measures the shift in population distribution between two periods.
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.25: Small change
        PSI >= 0.25: Major change (requires investigation)

        Args:
            reference_segments: Segment labels from reference period
            current_segments: Segment labels from current period
            threshold: PSI threshold for stability (default 0.1)

        Returns:
            Dictionary with PSI calculation results
        """
        # Get unique segments from both periods
        all_segments = np.unique(np.concatenate([reference_segments, current_segments]))

        psi_total = 0.0
        psi_per_segment = {}

        for seg in all_segments:
            # Calculate proportions
            ref_count = np.sum(reference_segments == seg)
            cur_count = np.sum(current_segments == seg)

            ref_prop = ref_count / len(reference_segments)
            cur_prop = cur_count / len(current_segments)

            # Avoid log(0) by using small epsilon
            epsilon = 1e-10
            ref_prop = max(ref_prop, epsilon)
            cur_prop = max(cur_prop, epsilon)

            # PSI formula: (current% - reference%) * ln(current% / reference%)
            psi_segment = (cur_prop - ref_prop) * np.log(cur_prop / ref_prop)
            psi_per_segment[int(seg)] = psi_segment
            psi_total += psi_segment

        # Determine stability
        if psi_total < threshold:
            stability = 'stable'
        elif psi_total < 0.25:
            stability = 'minor_shift'
        else:
            stability = 'major_shift'

        return {
            'passed': psi_total < threshold,
            'psi': psi_total,
            'psi_per_segment': psi_per_segment,
            'threshold': threshold,
            'stability': stability
        }

    @staticmethod
    def validate_default_rate_differences(
        segments: np.ndarray,
        y: np.ndarray,
        min_diff: float = 0.001
    ) -> Dict[str, any]:
        """
        Validate that segments have meaningfully different default rates.

        Args:
            segments: Array of segment labels
            y: Array of binary outcomes (0/1)
            min_diff: Minimum PD difference between adjacent segments

        Returns:
            Dictionary with validation results
        """
        unique_segments = np.sort(np.unique(segments))
        default_rates = {}
        failed_pairs = []

        # Calculate default rates for each segment
        for seg in unique_segments:
            mask = segments == seg
            default_rate = np.mean(y[mask])
            default_rates[int(seg)] = default_rate

        # Check differences between all pairs
        for i in range(len(unique_segments) - 1):
            for j in range(i + 1, len(unique_segments)):
                seg1, seg2 = unique_segments[i], unique_segments[j]
                rate1, rate2 = default_rates[int(seg1)], default_rates[int(seg2)]
                diff = abs(rate2 - rate1)

                if diff < min_diff:
                    failed_pairs.append({
                        'segment1': int(seg1),
                        'segment2': int(seg2),
                        'rate1': rate1,
                        'rate2': rate2,
                        'difference': diff,
                        'min_required': min_diff
                    })

        return {
            'passed': len(failed_pairs) == 0,
            'default_rates': default_rates,
            'failed_pairs': failed_pairs,
            'min_diff': min_diff
        }

    @staticmethod
    def validate_binomial_confidence(
        segments: np.ndarray,
        y: np.ndarray,
        confidence_level: float = 0.99
    ) -> Dict[str, any]:
        """
        Calculate binomial confidence intervals for segment default rates.

        Args:
            segments: Array of segment labels
            y: Array of binary outcomes (0/1)
            confidence_level: Confidence level for intervals

        Returns:
            Dictionary with confidence intervals for each segment
        """
        from scipy.stats import binom

        unique_segments = np.unique(segments)
        results = {}

        for seg in unique_segments:
            mask = segments == seg
            n = np.sum(mask)
            k = np.sum(y[mask])

            # Calculate point estimate
            default_rate = k / n if n > 0 else 0

            # Calculate Clopper-Pearson (exact) confidence interval
            alpha = 1 - confidence_level
            if k == 0:
                lower = 0
            else:
                lower = stats.beta.ppf(alpha / 2, k, n - k + 1)

            if k == n:
                upper = 1
            else:
                upper = stats.beta.ppf(1 - alpha / 2, k + 1, n - k)

            results[int(seg)] = {
                'default_rate': default_rate,
                'n_observations': int(n),
                'n_defaults': int(k),
                'confidence_level': confidence_level,
                'lower_bound': lower,
                'upper_bound': upper,
                'width': upper - lower
            }

        return {
            'confidence_intervals': results,
            'confidence_level': confidence_level
        }

    @classmethod
    def run_all_validations(
        cls,
        segments: np.ndarray,
        y: np.ndarray,
        params: 'IRBSegmentationParams',
        reference_segments: Optional[np.ndarray] = None,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, any]:
        """
        Run all configured validation tests.

        Args:
            segments: Array of segment labels
            y: Array of binary outcomes
            params: IRBSegmentationParams object with validation configuration
            reference_segments: Optional reference segments for PSI calculation
            y_pred_proba: Optional predicted probabilities for performance metrics

        Returns:
            Dictionary with all validation results
        """
        results = {
            'all_passed': True,
            'validations': {}
        }

        # Run each configured validation test
        for test in params.validation_tests:
            if test == 'chi_squared':
                result = cls.validate_statistical_significance(
                    segments, y, params.significance_level, method='chi_squared'
                )
                results['validations']['chi_squared'] = result
                results['all_passed'] &= result['passed']

            elif test == 'psi' and reference_segments is not None:
                result = cls.calculate_psi(reference_segments, segments, threshold=0.1)
                results['validations']['psi'] = result
                results['all_passed'] &= result['passed']

            elif test == 'binomial':
                result = cls.validate_binomial_confidence(
                    segments, y, confidence_level=1 - params.significance_level
                )
                results['validations']['binomial'] = result

            elif test in ('gini', 'ks') and y_pred_proba is not None:
                # Import performance metrics module (lazy import to avoid circular dependency)
                try:
                    from irb_segmentation.validators import performance_metrics
                except ImportError:
                    # Fallback for relative import
                    from . import performance_metrics

                # Calculate all performance metrics
                perf_metrics = performance_metrics.calculate_all_metrics(
                    y, y_pred_proba,
                    gini_threshold=0.30,  # Basel II/III threshold
                    ks_threshold=0.20     # Basel II/III threshold
                )

                results['validations']['performance_metrics'] = perf_metrics.to_dict()
                results['all_passed'] &= perf_metrics.passed_thresholds

        # Always run core validations
        min_defaults = cls.validate_minimum_defaults(
            segments, y, params.min_defaults_per_leaf
        )
        results['validations']['min_defaults'] = min_defaults
        results['all_passed'] &= min_defaults['passed']

        density = cls.validate_density(
            segments, params.min_segment_density, params.max_segment_density
        )
        results['validations']['density'] = density
        results['all_passed'] &= density['passed']

        default_rate_diff = cls.validate_default_rate_differences(
            segments, y, params.min_default_rate_diff
        )
        results['validations']['default_rate_diff'] = default_rate_diff
        results['all_passed'] &= default_rate_diff['passed']

        return results
