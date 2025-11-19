"""
Residual Analysis and Diagnostics for Regression Models

Provides residual diagnostics and heteroscedasticity testing for
LGD and EAD regression models in IRB frameworks.

References:
- Breusch-Pagan Test: Breusch & Pagan (1979)
- White Test: White (1980)
- Durbin-Watson Test: Durbin & Watson (1950, 1951)
- EBA Guidelines on LGD/CCF estimation (EBA/GL/2017/16), Section 5.3
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
from ..logger import get_logger

logger = get_logger(__name__)


@dataclass
class ResidualAnalysisResults:
    """
    Container for residual analysis results.

    Attributes:
        mean_residual: Mean of residuals (should be ~0)
        std_residual: Standard deviation of residuals
        skewness: Skewness of residual distribution
        kurtosis: Excess kurtosis of residual distribution
        normality_test_statistic: Jarque-Bera test statistic
        normality_pvalue: P-value for normality test
        is_normal: Whether residuals are approximately normal
        autocorrelation: First-order autocorrelation coefficient
        durbin_watson: Durbin-Watson statistic
    """
    mean_residual: float
    std_residual: float
    skewness: float
    kurtosis: float
    normality_test_statistic: float
    normality_pvalue: float
    is_normal: bool
    autocorrelation: float
    durbin_watson: float

    def __str__(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 70,
            "RESIDUAL ANALYSIS RESULTS",
            "=" * 70,
            "",
            "Residual Statistics:",
            f"  Mean:                 {self.mean_residual:.6f} (should be ~0)",
            f"  Std Deviation:        {self.std_residual:.4f}",
            f"  Skewness:             {self.skewness:.4f} (normal: ~0)",
            f"  Excess Kurtosis:      {self.kurtosis:.4f} (normal: ~0)",
            "",
            "Normality Test (Jarque-Bera):",
            f"  Test Statistic:       {self.normality_test_statistic:.4f}",
            f"  P-value:              {self.normality_pvalue:.4f}",
            f"  Normality:            {'✓ PASSED' if self.is_normal else '✗ FAILED'} (α=0.05)",
            "",
            "Autocorrelation:",
            f"  First-order:          {self.autocorrelation:.4f}",
            f"  Durbin-Watson:        {self.durbin_watson:.4f} (ideal: ~2.0)",
            "=" * 70
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'mean_residual': float(self.mean_residual),
            'std_residual': float(self.std_residual),
            'skewness': float(self.skewness),
            'kurtosis': float(self.kurtosis),
            'normality': {
                'test_statistic': float(self.normality_test_statistic),
                'pvalue': float(self.normality_pvalue),
                'is_normal': bool(self.is_normal)
            },
            'autocorrelation': {
                'first_order': float(self.autocorrelation),
                'durbin_watson': float(self.durbin_watson)
            }
        }


@dataclass
class HeteroscedasticityTestResults:
    """
    Container for heteroscedasticity test results.

    Attributes:
        test_name: Name of the test ('Breusch-Pagan' or 'White')
        test_statistic: Chi-squared test statistic
        pvalue: P-value for the test
        is_homoscedastic: Whether residuals have constant variance
        degrees_of_freedom: Degrees of freedom for chi-squared test
    """
    test_name: str
    test_statistic: float
    pvalue: float
    is_homoscedastic: bool
    degrees_of_freedom: int

    def __str__(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 70,
            f"{self.test_name.upper()} TEST FOR HETEROSCEDASTICITY",
            "=" * 70,
            "",
            f"Test Statistic (χ²):  {self.test_statistic:.4f}",
            f"Degrees of Freedom:   {self.degrees_of_freedom}",
            f"P-value:              {self.pvalue:.4f}",
            "",
            f"Result: {'✓ Homoscedastic' if self.is_homoscedastic else '✗ Heteroscedastic'} (α=0.05)",
            "",
            "Interpretation:",
            "  H₀: Residuals have constant variance (homoscedastic)",
            "  H₁: Residuals have non-constant variance (heteroscedastic)",
            "",
            f"  {'Fail to reject H₀: No evidence of heteroscedasticity' if self.is_homoscedastic else 'Reject H₀: Evidence of heteroscedasticity detected'}",
            "=" * 70
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'test_name': self.test_name,
            'test_statistic': float(self.test_statistic),
            'pvalue': float(self.pvalue),
            'is_homoscedastic': bool(self.is_homoscedastic),
            'degrees_of_freedom': int(self.degrees_of_freedom)
        }


def analyze_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 0.05
) -> ResidualAnalysisResults:
    """
    Perform comprehensive residual analysis.

    Analyzes residuals for normality, mean zero, and autocorrelation.
    Important for validating regression model assumptions.

    Args:
        y_true: True continuous target values
        y_pred: Predicted continuous values
        alpha: Significance level for normality test (default: 0.05)

    Returns:
        ResidualAnalysisResults object with comprehensive diagnostics

    Example:
        >>> y_true = np.array([0.3, 0.5, 0.2, 0.7, 0.4, 0.6, 0.35, 0.55])
        >>> y_pred = np.array([0.32, 0.48, 0.25, 0.68, 0.42, 0.58, 0.37, 0.53])
        >>> results = analyze_residuals(y_true, y_pred)
        >>> print(results)

    Interpretation:
        mean_residual ≈ 0: Model is unbiased
        is_normal = True: Residuals follow normal distribution (good)
        durbin_watson ≈ 2.0: No autocorrelation (good)
        durbin_watson < 1.5 or > 2.5: Potential autocorrelation issue

    Regulatory Context:
        Residual analysis validates key assumptions for LGD models:
        - Zero mean: Model predictions are unbiased
        - Normality: Enables confidence intervals and hypothesis tests
        - No autocorrelation: Observations are independent
    """
    # Validate inputs
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
        )

    if len(y_true) < 8:
        raise ValueError(
            f"Need at least 8 observations for residual analysis, got {len(y_true)}"
        )

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Calculate residuals
    residuals = y_true - y_pred

    # Basic statistics
    mean_res = np.mean(residuals)
    std_res = np.std(residuals, ddof=1)

    # Skewness and kurtosis
    skew = stats.skew(residuals)
    kurt = stats.kurtosis(residuals)  # Excess kurtosis

    # Jarque-Bera test for normality
    jb_stat, jb_pval = stats.jarque_bera(residuals)
    is_normal = jb_pval >= alpha

    # Autocorrelation
    if len(residuals) > 1:
        autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
    else:
        autocorr = 0.0

    # Durbin-Watson statistic
    # DW = sum((e_t - e_{t-1})^2) / sum(e_t^2)
    diff_residuals = np.diff(residuals)
    dw = np.sum(diff_residuals ** 2) / np.sum(residuals ** 2)

    logger.info(
        f"Residual analysis: mean={mean_res:.6f}, std={std_res:.4f}, "
        f"normal={is_normal}, DW={dw:.4f}"
    )

    results = ResidualAnalysisResults(
        mean_residual=mean_res,
        std_residual=std_res,
        skewness=skew,
        kurtosis=kurt,
        normality_test_statistic=jb_stat,
        normality_pvalue=jb_pval,
        is_normal=is_normal,
        autocorrelation=autocorr,
        durbin_watson=dw
    )

    return results


def breusch_pagan_test(
    residuals: np.ndarray,
    X: np.ndarray,
    alpha: float = 0.05
) -> HeteroscedasticityTestResults:
    """
    Perform Breusch-Pagan test for heteroscedasticity.

    Tests whether residual variance depends on the predictor variables.
    Heteroscedasticity can invalidate standard errors and confidence intervals.

    Args:
        residuals: Regression residuals (y_true - y_pred)
        X: Predictor variables matrix (n_samples, n_features)
        alpha: Significance level (default: 0.05)

    Returns:
        HeteroscedasticityTestResults object

    Example:
        >>> residuals = y_true - y_pred
        >>> X = np.column_stack([x1, x2, x3])
        >>> result = breusch_pagan_test(residuals, X)
        >>> print(result)

    Test Procedure:
        1. Regress squared residuals on X
        2. Calculate explained sum of squares (ESS)
        3. Test statistic = ESS / 2 ~ χ² under H₀
        4. If p-value < α, reject homoscedasticity

    Interpretation:
        is_homoscedastic = True: Residual variance is constant (good)
        is_homoscedastic = False: Residual variance varies with X (problem)

    Regulatory Context:
        Heteroscedasticity in LGD models can lead to:
        - Biased risk estimates for certain exposure sizes
        - Invalid confidence intervals for capital requirements
        - Need for robust standard errors or transformation
    """
    # Validate inputs
    if len(residuals) != len(X):
        raise ValueError(
            f"Length mismatch: residuals={len(residuals)}, X={len(X)}"
        )

    if len(residuals) < 10:
        raise ValueError(
            f"Need at least 10 observations for Breusch-Pagan test, got {len(residuals)}"
        )

    residuals = np.asarray(residuals, dtype=np.float64).flatten()
    X = np.asarray(X, dtype=np.float64)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n, p = X.shape

    # Squared residuals
    u2 = residuals ** 2

    # Add intercept to X if not present
    if not np.allclose(X[:, 0], 1.0):
        X_with_intercept = np.column_stack([np.ones(n), X])
    else:
        X_with_intercept = X

    # Regress u² on X
    # Using normal equations: β = (X'X)^-1 X'u²
    try:
        XtX = X_with_intercept.T @ X_with_intercept
        Xtu2 = X_with_intercept.T @ u2
        beta = np.linalg.solve(XtX, Xtu2)
    except np.linalg.LinAlgError:
        logger.warning("Singular matrix in Breusch-Pagan test, using pseudo-inverse")
        beta = np.linalg.lstsq(X_with_intercept, u2, rcond=None)[0]

    # Fitted values
    u2_fitted = X_with_intercept @ beta

    # Explained sum of squares
    u2_mean = np.mean(u2)
    ess = np.sum((u2_fitted - u2_mean) ** 2)

    # Test statistic: ESS / 2σ⁴ ~ χ²(p) under H₀
    # Using simplified version: ESS / 2 (assuming σ² = 1 after standardization)
    sigma2 = np.var(residuals, ddof=1)
    test_stat = ess / (2 * sigma2 ** 2)

    # Degrees of freedom = number of predictors (excluding intercept)
    df = X.shape[1]

    # P-value from chi-squared distribution
    pval = 1 - stats.chi2.cdf(test_stat, df)

    # Decision
    is_homoscedastic = pval >= alpha

    logger.info(
        f"Breusch-Pagan test: χ²={test_stat:.4f}, df={df}, p={pval:.4f}, "
        f"homoscedastic={is_homoscedastic}"
    )

    results = HeteroscedasticityTestResults(
        test_name="Breusch-Pagan",
        test_statistic=test_stat,
        pvalue=pval,
        is_homoscedastic=is_homoscedastic,
        degrees_of_freedom=df
    )

    return results


def white_test(
    residuals: np.ndarray,
    X: np.ndarray,
    alpha: float = 0.05
) -> HeteroscedasticityTestResults:
    """
    Perform White's test for heteroscedasticity.

    More general than Breusch-Pagan test. Tests for heteroscedasticity
    that depends on X, X², and cross-products of X.

    Args:
        residuals: Regression residuals (y_true - y_pred)
        X: Predictor variables matrix (n_samples, n_features)
        alpha: Significance level (default: 0.05)

    Returns:
        HeteroscedasticityTestResults object

    Example:
        >>> residuals = y_true - y_pred
        >>> X = np.column_stack([x1, x2])
        >>> result = white_test(residuals, X)
        >>> print(result)

    Test Procedure:
        1. Create augmented X with squares and cross-products
        2. Regress squared residuals on augmented X
        3. Calculate n*R² ~ χ² under H₀
        4. If p-value < α, reject homoscedasticity

    Interpretation:
        Similar to Breusch-Pagan but detects more complex patterns
        of heteroscedasticity including non-linear relationships.

    Regulatory Context:
        White's test is useful for LGD models where heteroscedasticity
        may have complex patterns (e.g., variance increasing with both
        exposure size and industry sector).
    """
    # Validate inputs
    if len(residuals) != len(X):
        raise ValueError(
            f"Length mismatch: residuals={len(residuals)}, X={len(X)}"
        )

    if len(residuals) < 10:
        raise ValueError(
            f"Need at least 10 observations for White test, got {len(residuals)}"
        )

    residuals = np.asarray(residuals, dtype=np.float64).flatten()
    X = np.asarray(X, dtype=np.float64)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n, p = X.shape

    # Create augmented X with squares and cross-products
    X_augmented = [np.ones(n)]  # Intercept

    # Original features
    for j in range(p):
        X_augmented.append(X[:, j])

    # Squares
    for j in range(p):
        X_augmented.append(X[:, j] ** 2)

    # Cross-products (only if we have multiple features)
    if p > 1:
        for j in range(p):
            for k in range(j + 1, p):
                X_augmented.append(X[:, j] * X[:, k])

    X_augmented = np.column_stack(X_augmented)

    # Squared residuals
    u2 = residuals ** 2

    # Regress u² on X_augmented
    try:
        XtX = X_augmented.T @ X_augmented
        Xtu2 = X_augmented.T @ u2
        beta = np.linalg.solve(XtX, Xtu2)
    except np.linalg.LinAlgError:
        logger.warning("Singular matrix in White test, using pseudo-inverse")
        beta = np.linalg.lstsq(X_augmented, u2, rcond=None)[0]

    # Fitted values
    u2_fitted = X_augmented @ beta

    # R² from auxiliary regression
    u2_mean = np.mean(u2)
    ss_total = np.sum((u2 - u2_mean) ** 2)
    ss_residual = np.sum((u2 - u2_fitted) ** 2)
    r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0

    # Test statistic: n * R² ~ χ²(df) under H₀
    test_stat = n * r_squared

    # Degrees of freedom = number of regressors - 1 (excluding intercept)
    df = X_augmented.shape[1] - 1

    # P-value from chi-squared distribution
    pval = 1 - stats.chi2.cdf(test_stat, df)

    # Decision
    is_homoscedastic = pval >= alpha

    logger.info(
        f"White test: χ²={test_stat:.4f}, df={df}, p={pval:.4f}, "
        f"homoscedastic={is_homoscedastic}"
    )

    results = HeteroscedasticityTestResults(
        test_name="White",
        test_statistic=test_stat,
        pvalue=pval,
        is_homoscedastic=is_homoscedastic,
        degrees_of_freedom=df
    )

    return results


def run_all_residual_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X: Optional[np.ndarray] = None,
    alpha: float = 0.05
) -> Dict[str, Union[ResidualAnalysisResults, HeteroscedasticityTestResults]]:
    """
    Run comprehensive residual diagnostics.

    Performs residual analysis and heteroscedasticity tests in one call.

    Args:
        y_true: True continuous target values
        y_pred: Predicted continuous values
        X: Predictor variables matrix (optional, for heteroscedasticity tests)
        alpha: Significance level (default: 0.05)

    Returns:
        Dictionary with:
            - 'residual_analysis': ResidualAnalysisResults
            - 'breusch_pagan': HeteroscedasticityTestResults (if X provided)
            - 'white': HeteroscedasticityTestResults (if X provided)

    Example:
        >>> diagnostics = run_all_residual_diagnostics(y_true, y_pred, X)
        >>> print(diagnostics['residual_analysis'])
        >>> print(diagnostics['breusch_pagan'])
        >>> print(diagnostics['white'])
    """
    logger.info("Running comprehensive residual diagnostics")

    results = {}

    # Residual analysis (always performed)
    results['residual_analysis'] = analyze_residuals(y_true, y_pred, alpha)

    # Heteroscedasticity tests (only if X provided)
    if X is not None:
        residuals = y_true - y_pred
        results['breusch_pagan'] = breusch_pagan_test(residuals, X, alpha)
        results['white'] = white_test(residuals, X, alpha)
        logger.info("Heteroscedasticity tests completed")
    else:
        logger.info("Skipping heteroscedasticity tests (X not provided)")

    return results
