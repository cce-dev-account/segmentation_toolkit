"""
Back-testing Framework for IRB Models

Validates model performance over time by comparing predicted PDs with
observed default rates across multiple time periods. Required for Basel II/III
compliance and regulatory validation.

References:
- Article 185 CRR: Minimum IRB requirements
- EBA Guidelines on PD estimation (EBA/GL/2017/16)
- Basel Committee on Banking Supervision (2005): "Studies on the Validation of Internal Rating Systems"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
from ..logger import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestResults:
    """
    Container for back-testing results.

    Attributes:
        binomial_test_pvalue: P-value from binomial test
        binomial_test_passed: Whether binomial test passed
        chi_squared_stat: Chi-squared statistic for multiple periods
        chi_squared_pvalue: P-value from chi-squared test
        chi_squared_passed: Whether chi-squared test passed
        traffic_light_status: Overall traffic light status (GREEN/AMBER/RED)
        period_results: DataFrame with results per period
        n_periods: Number of time periods tested
        total_observations: Total number of observations across all periods
        total_defaults: Total defaults observed
        mean_predicted_pd: Mean predicted PD across all periods
        mean_actual_pd: Mean actual default rate
        overall_passed: Whether all back-tests passed
    """
    binomial_test_pvalue: float
    binomial_test_passed: bool
    chi_squared_stat: float
    chi_squared_pvalue: float
    chi_squared_passed: bool
    traffic_light_status: str
    period_results: pd.DataFrame
    n_periods: int
    total_observations: int
    total_defaults: int
    mean_predicted_pd: float
    mean_actual_pd: float
    overall_passed: bool

    def __str__(self) -> str:
        """Generate human-readable summary."""
        status_icon = {
            'GREEN': '🟢',
            'AMBER': '🟡',
            'RED': '🔴'
        }.get(self.traffic_light_status, '⚪')

        lines = [
            "=" * 70,
            "BACK-TESTING RESULTS",
            "=" * 70,
            "",
            f"Time Periods Analyzed: {self.n_periods}",
            f"Total Observations: {self.total_observations}",
            f"Total Defaults: {self.total_defaults}",
            "",
            "Overall Performance:",
            f"  Mean Predicted PD: {self.mean_predicted_pd:.4f} ({self.mean_predicted_pd*100:.2f}%)",
            f"  Mean Actual PD: {self.mean_actual_pd:.4f} ({self.mean_actual_pd*100:.2f}%)",
            "",
            "Binomial Test (Aggregate):",
            f"  P-value: {self.binomial_test_pvalue:.4f}",
            f"  Status: {'✓ PASSED' if self.binomial_test_passed else '✗ FAILED'} "
            f"(p {'>' if self.binomial_test_passed else '<'} 0.05)",
            "",
            "Chi-Squared Test (Time Series):",
            f"  Chi-squared: {self.chi_squared_stat:.4f}",
            f"  P-value: {self.chi_squared_pvalue:.4f}",
            f"  Status: {'✓ PASSED' if self.chi_squared_passed else '✗ FAILED'} "
            f"(p {'>' if self.chi_squared_passed else '<'} 0.05)",
            "",
            f"Traffic Light Status: {status_icon} {self.traffic_light_status}",
            "",
            f"Overall Status: {'✓ PASSED' if self.overall_passed else '✗ FAILED'}",
            "=" * 70
        ]

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'binomial_test': {
                'pvalue': float(self.binomial_test_pvalue),
                'passed': bool(self.binomial_test_passed)
            },
            'chi_squared_test': {
                'statistic': float(self.chi_squared_stat),
                'pvalue': float(self.chi_squared_pvalue),
                'passed': bool(self.chi_squared_passed)
            },
            'traffic_light_status': str(self.traffic_light_status),
            'period_results': self.period_results.to_dict('records'),
            'summary': {
                'n_periods': int(self.n_periods),
                'total_observations': int(self.total_observations),
                'total_defaults': int(self.total_defaults),
                'mean_predicted_pd': float(self.mean_predicted_pd),
                'mean_actual_pd': float(self.mean_actual_pd)
            },
            'overall_passed': bool(self.overall_passed)
        }


def binomial_backtest(
    n_observations: int,
    n_defaults: int,
    predicted_pd: float,
    confidence_level: float = 0.95
) -> Tuple[float, bool, float]:
    """
    Binomial test for single period back-testing.

    Tests whether observed defaults are consistent with predicted PD
    under binomial distribution assumption.

    Args:
        n_observations: Number of obligors in the period
        n_defaults: Number of observed defaults
        predicted_pd: Mean predicted PD for the period
        confidence_level: Confidence level for the test (default: 0.95)

    Returns:
        Tuple of (p_value, passed, z_score)

    Example:
        >>> pval, passed, z = binomial_backtest(1000, 25, 0.02)
        >>> print(f"Test {'passed' if passed else 'failed'}: p={pval:.3f}, z={z:.2f}")

    Interpretation:
        p_value > 0.05: Observed defaults consistent with predictions
        p_value < 0.05: Significant deviation (potential model issue)

    Regulatory Context:
        Binomial tests are standard for IRB back-testing. Regulators expect
        observed defaults to fall within reasonable confidence intervals of
        predicted PDs over time.
    """
    if n_observations <= 0:
        raise ValueError(f"n_observations must be positive, got {n_observations}")

    if n_defaults < 0:
        raise ValueError(f"n_defaults cannot be negative, got {n_defaults}")

    if not 0 <= predicted_pd <= 1:
        raise ValueError(f"predicted_pd must be in [0,1], got {predicted_pd}")

    if n_defaults > n_observations:
        raise ValueError(
            f"n_defaults ({n_defaults}) cannot exceed n_observations ({n_observations})"
        )

    # Expected defaults under predicted PD
    expected_defaults = n_observations * predicted_pd

    # Variance under binomial distribution
    variance = n_observations * predicted_pd * (1 - predicted_pd)
    std_dev = np.sqrt(variance) if variance > 0 else 1e-8

    # Z-score
    z_score = (n_defaults - expected_defaults) / std_dev

    # Two-tailed p-value from normal approximation
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    # Determine pass/fail
    alpha = 1 - confidence_level
    passed = p_value > alpha

    logger.debug(
        f"Binomial test: n={n_observations}, defaults={n_defaults}, "
        f"predicted_pd={predicted_pd:.4f}, z={z_score:.2f}, p={p_value:.4f}"
    )

    return float(p_value), bool(passed), float(z_score)


def chi_squared_backtest(
    period_observations: np.ndarray,
    period_defaults: np.ndarray,
    period_predicted_pds: np.ndarray
) -> Tuple[float, float, bool]:
    """
    Chi-squared test for multi-period back-testing.

    Tests whether observed defaults across multiple time periods are
    consistent with predicted PDs using chi-squared goodness-of-fit test.

    Args:
        period_observations: Number of observations per period
        period_defaults: Number of defaults per period
        period_predicted_pds: Mean predicted PD per period

    Returns:
        Tuple of (chi_squared_stat, p_value, passed)

    Example:
        >>> obs = np.array([1000, 1000, 1000])
        >>> defaults = np.array([20, 25, 18])
        >>> pds = np.array([0.02, 0.02, 0.02])
        >>> chi2, pval, passed = chi_squared_backtest(obs, defaults, pds)
        >>> print(f"Chi-squared test: chi2={chi2:.2f}, p={pval:.3f}")

    Interpretation:
        p_value > 0.05: Model predictions consistent across periods
        p_value < 0.05: Significant deviation (systematic bias)

    Raises:
        ValueError: If arrays have different lengths or invalid values

    Regulatory Context:
        Chi-squared tests are used to validate stability of PD estimates
        over time. Consistent deviations indicate model deterioration or
        structural changes requiring recalibration.
    """
    # Validate inputs
    if not (len(period_observations) == len(period_defaults) == len(period_predicted_pds)):
        raise ValueError(
            f"Array length mismatch: observations={len(period_observations)}, "
            f"defaults={len(period_defaults)}, pds={len(period_predicted_pds)}"
        )

    if len(period_observations) < 2:
        raise ValueError(
            f"Need at least 2 periods for chi-squared test, got {len(period_observations)}"
        )

    # Calculate expected defaults per period
    expected_defaults = period_observations * period_predicted_pds

    # Chi-squared statistic: sum((O - E)^2 / V)
    # where V = E * (1 - p) is the binomial variance
    variance = expected_defaults * (1 - period_predicted_pds)
    variance = np.maximum(variance, 1e-8)  # Avoid division by zero

    chi_squared = np.sum((period_defaults - expected_defaults) ** 2 / variance)

    # Degrees of freedom = number of periods - 1
    df = len(period_observations) - 1

    # P-value
    p_value = 1 - stats.chi2.cdf(chi_squared, df)

    # Pass if p > 0.05
    passed = p_value > 0.05

    logger.debug(
        f"Chi-squared test: periods={len(period_observations)}, "
        f"chi2={chi_squared:.4f}, p={p_value:.4f}, df={df}"
    )

    return float(chi_squared), float(p_value), bool(passed)


def traffic_light_backtest(
    z_score: float,
    green_threshold: float = 1.96,
    amber_threshold: float = 2.58
) -> str:
    """
    Classify back-test result using traffic light approach.

    Args:
        z_score: Absolute z-score from binomial test
        green_threshold: Threshold for green status (default: 1.96 for 95% CI)
        amber_threshold: Threshold for amber status (default: 2.58 for 99% CI)

    Returns:
        Traffic light status: 'GREEN', 'AMBER', or 'RED'

    Example:
        >>> status = traffic_light_backtest(1.5)
        >>> print(f"Status: {status}")  # GREEN
        >>> status = traffic_light_backtest(2.2)
        >>> print(f"Status: {status}")  # AMBER
        >>> status = traffic_light_backtest(3.0)
        >>> print(f"Status: {status}")  # RED

    Interpretation:
        GREEN: Within 95% CI - model performing well
        AMBER: Outside 95% but within 99% CI - investigate
        RED: Outside 99% CI - model failed, action required

    Regulatory Context:
        Traffic light approach provides intuitive risk classification for
        supervisors and management. RED status typically triggers mandatory
        model review and potential recalibration.
    """
    abs_z = abs(z_score)

    if abs_z <= green_threshold:
        return 'GREEN'
    elif abs_z <= amber_threshold:
        return 'AMBER'
    else:
        return 'RED'


def run_backtest(
    periods: Union[np.ndarray, pd.Series, List],
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    period_ids: Optional[Union[np.ndarray, pd.Series, List]] = None,
    confidence_level: float = 0.95,
    green_threshold: float = 1.96,
    amber_threshold: float = 2.58
) -> BacktestResults:
    """
    Run comprehensive back-testing analysis across multiple time periods.

    Args:
        periods: Time period identifier for each observation
        y_true: True binary labels (0=non-default, 1=default)
        y_pred_proba: Predicted probabilities of default
        period_ids: Optional explicit list of period IDs to analyze
        confidence_level: Confidence level for binomial test (default: 0.95)
        green_threshold: Z-score threshold for green status
        amber_threshold: Z-score threshold for amber status

    Returns:
        BacktestResults object with comprehensive results

    Example:
        >>> periods = np.array([2020, 2020, 2021, 2021, 2022, 2022])
        >>> y_true = np.array([0, 0, 1, 0, 0, 1])
        >>> y_pred = np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.4])
        >>> results = run_backtest(periods, y_true, y_pred)
        >>> print(results)

    Raises:
        ValueError: If inputs are invalid or have mismatched lengths
    """
    # Convert to numpy arrays
    periods = np.array(periods)
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)

    # Validate inputs
    if not (len(periods) == len(y_true) == len(y_pred_proba)):
        raise ValueError(
            f"Length mismatch: periods={len(periods)}, "
            f"y_true={len(y_true)}, y_pred_proba={len(y_pred_proba)}"
        )

    if len(periods) == 0:
        raise ValueError("Empty arrays provided")

    # Determine period IDs
    if period_ids is None:
        period_ids = np.unique(periods)
    else:
        period_ids = np.array(period_ids)

    logger.info(f"Running back-test across {len(period_ids)} periods")

    # Aggregate results per period
    period_results_list = []

    for period_id in period_ids:
        mask = periods == period_id
        n_obs = mask.sum()

        if n_obs == 0:
            logger.warning(f"Period {period_id} has no observations, skipping")
            continue

        n_defaults = y_true[mask].sum()
        predicted_pd = y_pred_proba[mask].mean()
        actual_pd = n_defaults / n_obs if n_obs > 0 else 0.0

        # Binomial test for this period
        p_value, passed, z_score = binomial_backtest(
            n_obs, n_defaults, predicted_pd, confidence_level
        )

        # Traffic light status
        status = traffic_light_backtest(z_score, green_threshold, amber_threshold)

        period_results_list.append({
            'period': period_id,
            'n_observations': int(n_obs),
            'n_defaults': int(n_defaults),
            'predicted_pd': float(predicted_pd),
            'actual_pd': float(actual_pd),
            'z_score': float(z_score),
            'p_value': float(p_value),
            'binomial_passed': bool(passed),
            'status': status
        })

    if len(period_results_list) == 0:
        raise ValueError("No valid periods found for back-testing")

    period_results_df = pd.DataFrame(period_results_list)

    # Aggregate binomial test (all periods combined)
    total_obs = int(period_results_df['n_observations'].sum())
    total_defaults = int(period_results_df['n_defaults'].sum())
    mean_predicted_pd = float(y_pred_proba.mean())
    mean_actual_pd = float(y_true.mean())

    agg_pvalue, agg_passed, agg_z = binomial_backtest(
        total_obs, total_defaults, mean_predicted_pd, confidence_level
    )

    # Chi-squared test across periods (only if multiple periods)
    if len(period_results_list) >= 2:
        chi2_stat, chi2_pvalue, chi2_passed = chi_squared_backtest(
            period_results_df['n_observations'].values,
            period_results_df['n_defaults'].values,
            period_results_df['predicted_pd'].values
        )
    else:
        # Single period: chi-squared test not applicable
        chi2_stat = 0.0
        chi2_pvalue = 1.0
        chi2_passed = True
        logger.info("Single period detected - chi-squared test skipped")

    # Overall traffic light status (worst case across periods)
    if (period_results_df['status'] == 'RED').any():
        overall_status = 'RED'
    elif (period_results_df['status'] == 'AMBER').any():
        overall_status = 'AMBER'
    else:
        overall_status = 'GREEN'

    # Overall pass/fail
    overall_passed = agg_passed and chi2_passed and (overall_status != 'RED')

    results = BacktestResults(
        binomial_test_pvalue=agg_pvalue,
        binomial_test_passed=agg_passed,
        chi_squared_stat=chi2_stat,
        chi_squared_pvalue=chi2_pvalue,
        chi_squared_passed=chi2_passed,
        traffic_light_status=overall_status,
        period_results=period_results_df,
        n_periods=len(period_results_list),
        total_observations=total_obs,
        total_defaults=total_defaults,
        mean_predicted_pd=mean_predicted_pd,
        mean_actual_pd=mean_actual_pd,
        overall_passed=overall_passed
    )

    logger.info(
        f"Back-test completed: {len(period_ids)} periods, "
        f"status={overall_status}, overall={'PASSED' if overall_passed else 'FAILED'}"
    )

    return results
