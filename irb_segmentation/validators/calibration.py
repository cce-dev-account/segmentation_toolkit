"""
Calibration Tests for IRB Models

Validates that predicted probabilities match observed default rates.
Required for Basel II/III compliance per Article 185(e) CRR.

References:
- Article 185(e) CRR: Evidence that model assigns risk weights consistent with observed default rates
- EBA Guidelines on PD estimation (EBA/GL/2017/16)
- Hosmer, D.W. and Lemeshow, S. (1980) "A goodness-of-fit test for the multiple logistic regression model"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy import stats
from ..logger import get_logger

logger = get_logger(__name__)


@dataclass
class CalibrationResults:
    """
    Container for calibration test results.

    Attributes:
        hosmer_lemeshow_chi2: Chi-squared statistic from HL test
        hosmer_lemeshow_pvalue: P-value from HL test
        hosmer_lemeshow_passed: Whether HL test passed (p > 0.05)
        traffic_light_results: DataFrame with traffic light test per segment
        n_green: Number of segments in green status
        n_amber: Number of segments in amber status
        n_red: Number of segments in red status
        central_tendency_diff: Difference between predicted and actual mean PD
        overall_passed: Whether all calibration tests passed
    """
    hosmer_lemeshow_chi2: float
    hosmer_lemeshow_pvalue: float
    hosmer_lemeshow_passed: bool
    traffic_light_results: pd.DataFrame
    n_green: int
    n_amber: int
    n_red: int
    central_tendency_diff: float
    overall_passed: bool

    def __str__(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 70,
            "CALIBRATION TEST RESULTS",
            "=" * 70,
            "",
            "Hosmer-Lemeshow Test:",
            f"  Chi-squared: {self.hosmer_lemeshow_chi2:.4f}",
            f"  P-value: {self.hosmer_lemeshow_pvalue:.4f}",
            f"  Status: {'✓ PASSED' if self.hosmer_lemeshow_passed else '✗ FAILED'} "
            f"(p {'>' if self.hosmer_lemeshow_passed else '<'} 0.05)",
            "",
            "Traffic Light Test (Segment-level):",
            f"  🟢 GREEN: {self.n_green} segments (within 95% CI)",
            f"  🟡 AMBER: {self.n_amber} segments (within 99% CI)",
            f"  🔴 RED: {self.n_red} segments (outside 99% CI)",
        ]

        if self.n_red > 0:
            lines.append(f"  ⚠️ WARNING: {self.n_red} segment(s) failed calibration")

        lines.extend([
            "",
            "Central Tendency:",
            f"  Mean PD difference: {self.central_tendency_diff:.4f} "
            f"({'over' if self.central_tendency_diff > 0 else 'under'}-predicting)",
            "",
            f"Overall Status: {'✓ PASSED' if self.overall_passed else '✗ FAILED'}",
            "=" * 70
        ])

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'hosmer_lemeshow': {
                'chi2': float(self.hosmer_lemeshow_chi2),
                'pvalue': float(self.hosmer_lemeshow_pvalue),
                'passed': bool(self.hosmer_lemeshow_passed)
            },
            'traffic_light': {
                'n_green': int(self.n_green),
                'n_amber': int(self.n_amber),
                'n_red': int(self.n_red),
                'results': self.traffic_light_results.to_dict('records')
            },
            'central_tendency': {
                'mean_pd_diff': float(self.central_tendency_diff)
            },
            'overall_passed': bool(self.overall_passed)
        }


def hosmer_lemeshow_test(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, float]:
    """
    Hosmer-Lemeshow goodness-of-fit test for calibration.

    Tests whether predicted probabilities match observed frequencies
    across risk groups. A well-calibrated model should show no significant
    difference between predicted and observed rates.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred_proba: Predicted probabilities for positive class
        n_bins: Number of bins to create (default: 10)

    Returns:
        Tuple of (chi_squared_stat, p_value)

    Interpretation:
        p > 0.05: Model is well-calibrated (cannot reject H0)
        p < 0.05: Poor calibration (reject H0)

    Example:
        >>> y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 1])
        >>> y_pred = np.array([0.1, 0.2, 0.7, 0.8, 0.3, 0.6, 0.9, 0.1, 0.8, 0.7])
        >>> chi2, pval = hosmer_lemeshow_test(y_true, y_pred)
        >>> print(f"HL test: chi2={chi2:.2f}, p={pval:.3f}")

    Raises:
        ValueError: If inputs are invalid or insufficient data for bins

    Regulatory Context:
        The Hosmer-Lemeshow test is a standard calibration test for binary
        models, widely accepted by supervisory authorities for IRB validation.
    """
    # Validate inputs
    if len(y_true) != len(y_pred_proba):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} samples, "
            f"y_pred_proba has {len(y_pred_proba)} samples"
        )

    if len(y_true) < n_bins * 2:
        raise ValueError(
            f"Insufficient data for {n_bins} bins. "
            f"Need at least {n_bins * 2} samples, got {len(y_true)}"
        )

    # Create bins based on predicted probabilities
    bin_edges = np.percentile(y_pred_proba, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1e-8  # Include max value

    # Ensure unique bin edges
    bin_edges = np.unique(bin_edges)
    actual_n_bins = len(bin_edges) - 1

    bins = np.digitize(y_pred_proba, bin_edges) - 1

    observed = []
    expected = []
    totals = []

    for i in range(actual_n_bins):
        mask = bins == i
        n_total = mask.sum()

        if n_total == 0:
            continue

        n_observed = y_true[mask].sum()
        n_expected = y_pred_proba[mask].sum()

        observed.append(n_observed)
        expected.append(n_expected)
        totals.append(n_total)

    # Convert to arrays
    observed = np.array(observed)
    expected = np.array(expected)
    totals = np.array(totals)

    # Chi-squared test statistic
    # HL statistic: sum((O - E)^2 / (E * (1 - E/N)))
    # where O = observed, E = expected, N = total in bin

    # Avoid division by zero
    variance = expected * (1 - expected / totals)
    variance = np.maximum(variance, 1e-8)

    chi_squared = np.sum((observed - expected) ** 2 / variance)

    # Degrees of freedom = number of groups - 2
    df = len(observed) - 2
    df = max(df, 1)  # Ensure at least 1 df

    # Calculate p-value
    p_value = 1 - stats.chi2.cdf(chi_squared, df)

    logger.debug(
        f"Hosmer-Lemeshow test: chi2={chi_squared:.4f}, p={p_value:.4f}, "
        f"df={df}, bins={len(observed)}"
    )

    return float(chi_squared), float(p_value)


def traffic_light_test(
    segment_id: np.ndarray,
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    green_threshold: float = 1.96,  # 95% confidence
    amber_threshold: float = 2.58   # 99% confidence
) -> pd.DataFrame:
    """
    Binomial test with traffic light bands for each segment.

    Performs segment-level calibration testing using binomial confidence
    intervals. Each segment is classified as GREEN (acceptable), AMBER
    (investigate), or RED (failed) based on z-score.

    Args:
        segment_id: Array of segment assignments
        y_true: True binary labels (0 or 1)
        y_pred_proba: Predicted probabilities for positive class
        green_threshold: Z-score threshold for green (default: 1.96 for 95% CI)
        amber_threshold: Z-score threshold for amber (default: 2.58 for 99% CI)

    Returns:
        DataFrame with columns: [segment_id, n_observations, n_defaults,
                                 predicted_pd, actual_pd, z_score, status]

    Status interpretation:
        GREEN: Within 95% confidence interval (model OK)
        AMBER: Outside 95% but within 99% CI (investigate)
        RED: Outside 99% CI (model failed)

    Example:
        >>> segments = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        >>> y_true = np.array([0, 0, 1, 0, 1, 1, 1, 1, 1])
        >>> y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        >>> results = traffic_light_test(segments, y_true, y_pred)
        >>> print(results[['segment_id', 'status']])

    Raises:
        ValueError: If inputs have mismatched lengths

    Regulatory Context:
        Traffic light tests are commonly used in IRB back-testing to quickly
        identify segments with calibration issues. RED segments require
        immediate investigation and potential model recalibration.
    """
    # Validate inputs
    if not (len(segment_id) == len(y_true) == len(y_pred_proba)):
        raise ValueError(
            f"Length mismatch: segment_id={len(segment_id)}, "
            f"y_true={len(y_true)}, y_pred_proba={len(y_pred_proba)}"
        )

    results = []

    for seg_id in np.unique(segment_id):
        mask = segment_id == seg_id

        n_obs = mask.sum()
        n_defaults = y_true[mask].sum()
        predicted_pd = y_pred_proba[mask].mean()
        actual_pd = n_defaults / n_obs if n_obs > 0 else 0.0

        # Binomial test: z-score calculation
        # z = (observed - expected) / sqrt(variance)
        expected_defaults = predicted_pd * n_obs
        variance = n_obs * predicted_pd * (1 - predicted_pd)
        std_dev = np.sqrt(variance)

        if std_dev > 0:
            z_score = abs(n_defaults - expected_defaults) / std_dev
        else:
            z_score = 0.0

        # Assign traffic light color
        if z_score <= green_threshold:
            status = 'GREEN'
        elif z_score <= amber_threshold:
            status = 'AMBER'
        else:
            status = 'RED'

        results.append({
            'segment_id': int(seg_id),
            'n_observations': int(n_obs),
            'n_defaults': int(n_defaults),
            'predicted_pd': float(predicted_pd),
            'actual_pd': float(actual_pd),
            'z_score': float(z_score),
            'status': status
        })

    df = pd.DataFrame(results)

    logger.info(
        f"Traffic light test: {(df['status'] == 'GREEN').sum()} green, "
        f"{(df['status'] == 'AMBER').sum()} amber, "
        f"{(df['status'] == 'RED').sum()} red"
    )

    return df


def central_tendency_test(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray
) -> Dict[str, float]:
    """
    Central tendency test: compare mean predicted vs actual default rate.

    Tests whether the model is systematically over- or under-predicting
    defaults on average. A well-calibrated model should have mean predicted
    PD close to actual default rate.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred_proba: Predicted probabilities for positive class

    Returns:
        Dictionary with:
            - mean_predicted: Mean predicted probability
            - mean_actual: Actual default rate
            - difference: mean_predicted - mean_actual
            - relative_error: Percentage error
            - passed: Whether test passed (|difference| < 0.01)

    Example:
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_pred = np.array([0.1, 0.2, 0.6, 0.7])
        >>> result = central_tendency_test(y_true, y_pred)
        >>> print(f"Diff: {result['difference']:.3f}")

    Interpretation:
        difference > 0: Model over-predicting (conservative)
        difference < 0: Model under-predicting (aggressive)
        |difference| < 0.01: Good central tendency
    """
    # Validate inputs
    if len(y_true) != len(y_pred_proba):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} samples, "
            f"y_pred_proba has {len(y_pred_proba)} samples"
        )

    mean_predicted = y_pred_proba.mean()
    mean_actual = y_true.mean()
    difference = mean_predicted - mean_actual

    # Calculate relative error
    if mean_actual > 0:
        relative_error = (difference / mean_actual) * 100
    else:
        relative_error = 0.0 if abs(difference) < 1e-8 else float('inf')

    # Pass if difference is small (< 1 percentage point)
    passed = abs(difference) < 0.01

    logger.debug(
        f"Central tendency: predicted={mean_predicted:.4f}, "
        f"actual={mean_actual:.4f}, diff={difference:.4f}"
    )

    return {
        'mean_predicted': float(mean_predicted),
        'mean_actual': float(mean_actual),
        'difference': float(difference),
        'relative_error': float(relative_error),
        'passed': bool(passed)
    }


def run_all_calibration_tests(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    segment_id: Optional[np.ndarray] = None,
    n_bins: int = 10,
    green_threshold: float = 1.96,
    amber_threshold: float = 2.58
) -> CalibrationResults:
    """
    Run all calibration tests and return consolidated results.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred_proba: Predicted probabilities for positive class
        segment_id: Optional segment assignments for traffic light test
        n_bins: Number of bins for Hosmer-Lemeshow test
        green_threshold: Z-score threshold for green status
        amber_threshold: Z-score threshold for amber status

    Returns:
        CalibrationResults object with all test results

    Example:
        >>> y_true = np.array([0, 0, 1, 1] * 100)
        >>> y_pred = np.array([0.1, 0.2, 0.8, 0.9] * 100)
        >>> segments = np.array([1, 1, 2, 2] * 100)
        >>> results = run_all_calibration_tests(y_true, y_pred, segments)
        >>> print(results)
    """
    logger.info("Running calibration tests for IRB model")

    # Hosmer-Lemeshow test
    hl_chi2, hl_pval = hosmer_lemeshow_test(y_true, y_pred_proba, n_bins)
    hl_passed = hl_pval > 0.05

    # Traffic light test (if segments provided)
    if segment_id is not None:
        tl_results = traffic_light_test(
            segment_id, y_true, y_pred_proba,
            green_threshold, amber_threshold
        )
        n_green = (tl_results['status'] == 'GREEN').sum()
        n_amber = (tl_results['status'] == 'AMBER').sum()
        n_red = (tl_results['status'] == 'RED').sum()
    else:
        # Create dummy segment (all in one segment)
        segment_id_dummy = np.zeros(len(y_true), dtype=int)
        tl_results = traffic_light_test(
            segment_id_dummy, y_true, y_pred_proba,
            green_threshold, amber_threshold
        )
        n_green = (tl_results['status'] == 'GREEN').sum()
        n_amber = (tl_results['status'] == 'AMBER').sum()
        n_red = (tl_results['status'] == 'RED').sum()

    # Central tendency test
    ct_result = central_tendency_test(y_true, y_pred_proba)
    ct_diff = ct_result['difference']

    # Overall pass/fail
    overall_passed = hl_passed and (n_red == 0) and ct_result['passed']

    results = CalibrationResults(
        hosmer_lemeshow_chi2=hl_chi2,
        hosmer_lemeshow_pvalue=hl_pval,
        hosmer_lemeshow_passed=hl_passed,
        traffic_light_results=tl_results,
        n_green=int(n_green),
        n_amber=int(n_amber),
        n_red=int(n_red),
        central_tendency_diff=ct_diff,
        overall_passed=overall_passed
    )

    logger.info(
        f"Calibration tests completed: HL={'PASSED' if hl_passed else 'FAILED'}, "
        f"Traffic Light={n_green}G/{n_amber}A/{n_red}R, "
        f"Overall={'PASSED' if overall_passed else 'FAILED'}"
    )

    return results
