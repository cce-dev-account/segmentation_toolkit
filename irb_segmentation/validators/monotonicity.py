"""
Monotonicity Validation for IRB Rating Systems

Validates that PD estimates increase monotonically with risk grades/scores.
Required for Basel II/III compliance under Article 170 CRR.

Monotonicity ensures that:
1. Higher risk grades correspond to higher PDs
2. Risk ranking is consistent with default probability ordering
3. Rating system provides meaningful risk differentiation

References:
- Article 170 CRR: Rating systems must reflect gradations of risk
- EBA Guidelines on PD estimation (EBA/GL/2017/16), Section 5.2
- BCBS 239: Risk Data Aggregation and Risk Reporting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
from ..logger import get_logger

logger = get_logger(__name__)


@dataclass
class MonotonicityResults:
    """
    Container for monotonicity validation results.

    Attributes:
        is_monotonic: Whether PDs are strictly monotonic with risk
        violations: List of monotonicity violations (if any)
        n_violations: Number of monotonicity violations
        spearman_correlation: Spearman rank correlation between risk and PD
        kendall_tau: Kendall's tau correlation
        test_results: DataFrame with detailed test results per segment/grade
        overall_passed: Whether monotonicity validation passed
    """
    is_monotonic: bool
    violations: List[Dict]
    n_violations: int
    spearman_correlation: float
    spearman_pvalue: float
    kendall_tau: float
    kendall_pvalue: float
    test_results: pd.DataFrame
    overall_passed: bool

    def __str__(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 70,
            "MONOTONICITY VALIDATION RESULTS",
            "=" * 70,
            "",
            f"Monotonicity Status: {'✓ PASSED' if self.is_monotonic else '✗ FAILED'}",
            f"Number of Violations: {self.n_violations}",
            "",
            "Rank Correlations:",
            f"  Spearman ρ: {self.spearman_correlation:.4f} (p={self.spearman_pvalue:.4e})",
            f"  Kendall τ: {self.kendall_tau:.4f} (p={self.kendall_pvalue:.4e})",
            "",
        ]

        if self.n_violations > 0:
            lines.append("Violations Detected:")
            for i, violation in enumerate(self.violations[:5], 1):
                lines.append(
                    f"  {i}. Grade {violation['grade_low']} (PD={violation['pd_low']:.4f}) -> "
                    f"Grade {violation['grade_high']} (PD={violation['pd_high']:.4f}) "
                    f"[DECREASE of {violation['decrease']:.4f}]"
                )
            if self.n_violations > 5:
                lines.append(f"  ... and {self.n_violations - 5} more violations")

        lines.extend([
            "",
            f"Overall Status: {'✓ PASSED' if self.overall_passed else '✗ FAILED'}",
            "=" * 70
        ])

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'is_monotonic': bool(self.is_monotonic),
            'n_violations': int(self.n_violations),
            'violations': self.violations,
            'correlations': {
                'spearman': {
                    'rho': float(self.spearman_correlation),
                    'pvalue': float(self.spearman_pvalue)
                },
                'kendall': {
                    'tau': float(self.kendall_tau),
                    'pvalue': float(self.kendall_pvalue)
                }
            },
            'test_results': self.test_results.to_dict('records'),
            'overall_passed': bool(self.overall_passed)
        }


def check_strict_monotonicity(
    risk_grades: np.ndarray,
    predicted_pds: np.ndarray,
    allow_equal: bool = False
) -> Tuple[bool, List[Dict]]:
    """
    Check if PDs are strictly monotonic with respect to risk grades.

    Args:
        risk_grades: Risk grade or score for each segment (higher = riskier)
        predicted_pds: Mean predicted PD for each segment
        allow_equal: If True, allow equal PDs for adjacent grades (default: False)

    Returns:
        Tuple of (is_monotonic, violations_list)
        - is_monotonic: True if strictly monotonic
        - violations_list: List of dictionaries describing violations

    Example:
        >>> grades = np.array([1, 2, 3, 4, 5])
        >>> pds = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        >>> is_mono, violations = check_strict_monotonicity(grades, pds)
        >>> print(f"Monotonic: {is_mono}")
        Monotonic: True

    Interpretation:
        is_monotonic = True: PDs increase with risk grades (valid system)
        is_monotonic = False: Some higher grades have lower PDs (violation)

    Regulatory Context:
        Article 170 CRR requires rating systems to meaningfully differentiate
        risk levels. Monotonicity ensures that grade ordering reflects PD ordering.
    """
    # Validate inputs
    if len(risk_grades) != len(predicted_pds):
        raise ValueError(
            f"Length mismatch: risk_grades={len(risk_grades)}, "
            f"predicted_pds={len(predicted_pds)}"
        )

    if len(risk_grades) < 2:
        raise ValueError(f"Need at least 2 grades for monotonicity check, got {len(risk_grades)}")

    # Sort by risk grades
    sort_idx = np.argsort(risk_grades)
    sorted_grades = risk_grades[sort_idx]
    sorted_pds = predicted_pds[sort_idx]

    violations = []

    # Check for monotonicity violations
    for i in range(len(sorted_grades) - 1):
        grade_low = sorted_grades[i]
        grade_high = sorted_grades[i + 1]
        pd_low = sorted_pds[i]
        pd_high = sorted_pds[i + 1]

        # Check violation
        if allow_equal:
            is_violation = pd_high < pd_low
        else:
            is_violation = pd_high <= pd_low

        if is_violation:
            violations.append({
                'grade_low': float(grade_low),
                'grade_high': float(grade_high),
                'pd_low': float(pd_low),
                'pd_high': float(pd_high),
                'decrease': float(pd_low - pd_high)
            })

    is_monotonic = len(violations) == 0

    logger.debug(
        f"Monotonicity check: {len(sorted_grades)} grades, "
        f"{len(violations)} violations, monotonic={is_monotonic}"
    )

    return is_monotonic, violations


def calculate_rank_correlation(
    risk_grades: np.ndarray,
    predicted_pds: np.ndarray,
    method: str = 'spearman'
) -> Tuple[float, float]:
    """
    Calculate rank correlation between risk grades and PDs.

    Args:
        risk_grades: Risk grade or score for each segment
        predicted_pds: Mean predicted PD for each segment
        method: Correlation method - 'spearman' or 'kendall' (default: 'spearman')

    Returns:
        Tuple of (correlation_coefficient, p_value)

    Example:
        >>> grades = np.array([1, 2, 3, 4, 5])
        >>> pds = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        >>> corr, pval = calculate_rank_correlation(grades, pds)
        >>> print(f"Correlation: {corr:.3f}, p-value: {pval:.3e}")
        Correlation: 1.000, p-value: 0.000e+00

    Interpretation:
        correlation close to +1: Strong positive monotonic relationship (good)
        correlation close to 0: No monotonic relationship (bad)
        correlation close to -1: Inverse relationship (very bad)
        p_value < 0.05: Correlation is statistically significant

    Regulatory Context:
        High rank correlation (ρ > 0.9 or τ > 0.8) indicates strong risk
        differentiation. Low correlation suggests rating system may not
        meaningfully rank order risk.
    """
    # Validate inputs
    if len(risk_grades) != len(predicted_pds):
        raise ValueError(
            f"Length mismatch: risk_grades={len(risk_grades)}, "
            f"predicted_pds={len(predicted_pds)}"
        )

    if len(risk_grades) < 3:
        raise ValueError(
            f"Need at least 3 observations for correlation, got {len(risk_grades)}"
        )

    # Calculate correlation
    if method.lower() == 'spearman':
        corr, pval = stats.spearmanr(risk_grades, predicted_pds)
    elif method.lower() == 'kendall':
        corr, pval = stats.kendalltau(risk_grades, predicted_pds)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'spearman' or 'kendall'.")

    logger.debug(
        f"{method.capitalize()} correlation: {corr:.4f}, p-value: {pval:.4e}"
    )

    return float(corr), float(pval)


def check_monotonic_trend(
    risk_grades: np.ndarray,
    predicted_pds: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, Union[float, bool]]:
    """
    Statistical test for monotonic trend using Mann-Kendall test.

    Tests the null hypothesis that there is no monotonic trend in PDs
    as risk grades increase.

    Args:
        risk_grades: Risk grade or score for each segment
        predicted_pds: Mean predicted PD for each segment
        alpha: Significance level (default: 0.05)

    Returns:
        Dictionary with:
            - trend: 'increasing', 'decreasing', or 'no trend'
            - z_score: Mann-Kendall test statistic
            - p_value: Two-tailed p-value
            - passed: Whether test indicates increasing trend

    Example:
        >>> grades = np.array([1, 2, 3, 4, 5])
        >>> pds = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        >>> result = test_monotonicity_trend(grades, pds)
        >>> print(f"Trend: {result['trend']}, passed: {result['passed']}")

    Interpretation:
        trend = 'increasing' and p < 0.05: Significant increasing trend (good)
        trend = 'no trend': No significant monotonic pattern (bad)
        trend = 'decreasing': PDs decrease with risk (very bad)

    Regulatory Context:
        Statistical evidence of increasing trend supports the claim that
        rating system meaningfully orders risk.
    """
    # Validate inputs
    if len(risk_grades) != len(predicted_pds):
        raise ValueError(
            f"Length mismatch: risk_grades={len(risk_grades)}, "
            f"predicted_pds={len(predicted_pds)}"
        )

    if len(risk_grades) < 3:
        raise ValueError(
            f"Need at least 3 observations for trend test, got {len(risk_grades)}"
        )

    # Sort by risk grades
    sort_idx = np.argsort(risk_grades)
    sorted_pds = predicted_pds[sort_idx]

    # Mann-Kendall test
    n = len(sorted_pds)
    s = 0

    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(sorted_pds[j] - sorted_pds[i])

    # Variance calculation
    var_s = n * (n - 1) * (2 * n + 5) / 18

    # Z-score
    if s > 0:
        z_score = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z_score = (s + 1) / np.sqrt(var_s)
    else:
        z_score = 0.0

    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    # Determine trend
    if p_value < alpha:
        if s > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'
    else:
        trend = 'no trend'

    passed = (trend == 'increasing')

    logger.debug(
        f"Mann-Kendall test: trend={trend}, z={z_score:.2f}, p={p_value:.4e}"
    )

    return {
        'trend': trend,
        'z_score': float(z_score),
        'p_value': float(p_value),
        'passed': bool(passed)
    }


def run_monotonicity_validation(
    risk_grades: Union[np.ndarray, pd.Series, List],
    predicted_pds: Union[np.ndarray, pd.Series, List],
    allow_equal: bool = False,
    min_correlation: float = 0.90,
    alpha: float = 0.05
) -> MonotonicityResults:
    """
    Run comprehensive monotonicity validation.

    Args:
        risk_grades: Risk grade or score for each segment (higher = riskier)
        predicted_pds: Mean predicted PD for each segment
        allow_equal: If True, allow equal PDs for adjacent grades (default: False)
        min_correlation: Minimum required Spearman correlation (default: 0.90)
        alpha: Significance level for statistical tests (default: 0.05)

    Returns:
        MonotonicityResults object with comprehensive validation results

    Example:
        >>> grades = np.array([1, 2, 3, 4, 5, 6, 7])
        >>> pds = np.array([0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.30])
        >>> results = run_monotonicity_validation(grades, pds)
        >>> print(results)

    Raises:
        ValueError: If inputs are invalid or have mismatched lengths
    """
    # Convert to numpy arrays
    risk_grades = np.array(risk_grades)
    predicted_pds = np.array(predicted_pds)

    logger.info(f"Running monotonicity validation for {len(risk_grades)} risk grades")

    # Check strict monotonicity
    is_monotonic, violations = check_strict_monotonicity(
        risk_grades, predicted_pds, allow_equal
    )

    # Calculate rank correlations
    spearman_corr, spearman_pval = calculate_rank_correlation(
        risk_grades, predicted_pds, method='spearman'
    )

    kendall_corr, kendall_pval = calculate_rank_correlation(
        risk_grades, predicted_pds, method='kendall'
    )

    # Test monotonic trend
    trend_result = check_monotonic_trend(risk_grades, predicted_pds, alpha)

    # Create test results DataFrame
    sort_idx = np.argsort(risk_grades)
    test_results_df = pd.DataFrame({
        'risk_grade': risk_grades[sort_idx],
        'predicted_pd': predicted_pds[sort_idx]
    })

    # Overall pass/fail
    overall_passed = (
        is_monotonic and
        spearman_corr >= min_correlation and
        spearman_pval < alpha and
        trend_result['passed']
    )

    results = MonotonicityResults(
        is_monotonic=is_monotonic,
        violations=violations,
        n_violations=len(violations),
        spearman_correlation=spearman_corr,
        spearman_pvalue=spearman_pval,
        kendall_tau=kendall_corr,
        kendall_pvalue=kendall_pval,
        test_results=test_results_df,
        overall_passed=overall_passed
    )

    logger.info(
        f"Monotonicity validation completed: monotonic={is_monotonic}, "
        f"violations={len(violations)}, spearman={spearman_corr:.3f}, "
        f"overall={'PASSED' if overall_passed else 'FAILED'}"
    )

    return results
