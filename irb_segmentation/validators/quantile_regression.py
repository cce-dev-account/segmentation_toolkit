"""
Quantile Regression for LGD Floor Estimation

Implements quantile regression for downturn LGD calibration as required
by Article 181 CRR. Quantile regression estimates conditional quantiles
of the loss distribution, crucial for conservative LGD floors.

References:
- Article 181 CRR: Downturn LGD requirements
- Article 182 CRR: LGD estimates shall reflect economic downturn conditions
- Koenker & Bassett (1978): Regression Quantiles
- EBA Guidelines on LGD/CCF estimation (EBA/GL/2017/16), Section 5.4
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union, List
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize
from ..logger import get_logger

logger = get_logger(__name__)


@dataclass
class QuantileRegressionResults:
    """
    Container for quantile regression results.

    Attributes:
        quantile: Target quantile (e.g., 0.75 for 75th percentile)
        coefficients: Regression coefficients (including intercept)
        predictions: Predicted quantile values
        feature_names: Names of predictor variables
        n_observations: Number of observations
        converged: Whether optimization converged successfully
    """
    quantile: float
    coefficients: np.ndarray
    predictions: np.ndarray
    feature_names: List[str]
    n_observations: int
    converged: bool

    def __str__(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 70,
            f"QUANTILE REGRESSION RESULTS (τ = {self.quantile:.2f})",
            "=" * 70,
            "",
            "Coefficients:",
        ]

        for i, (name, coef) in enumerate(zip(self.feature_names, self.coefficients)):
            lines.append(f"  {name:20s}: {coef:12.6f}")

        lines.extend([
            "",
            f"Observations: {self.n_observations:,}",
            f"Convergence:  {'✓ SUCCESS' if self.converged else '✗ FAILED'}",
            "=" * 70
        ])

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'quantile': float(self.quantile),
            'coefficients': {
                name: float(coef)
                for name, coef in zip(self.feature_names, self.coefficients)
            },
            'n_observations': int(self.n_observations),
            'converged': bool(self.converged)
        }


def quantile_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float
) -> float:
    """
    Calculate quantile loss (pinball loss / check function).

    The quantile loss is asymmetric, penalizing over-predictions and
    under-predictions differently based on the target quantile.

    Args:
        y_true: True values
        y_pred: Predicted values
        quantile: Target quantile in range (0, 1)

    Returns:
        Quantile loss (lower is better)

    Example:
        >>> y_true = np.array([0.5, 0.6, 0.3, 0.8])
        >>> y_pred = np.array([0.55, 0.58, 0.35, 0.75])
        >>> loss = quantile_loss(y_true, y_pred, quantile=0.75)
        >>> print(f"Quantile loss: {loss:.4f}")

    Formula:
        L(y, ŷ) = Σ ρ_τ(y - ŷ)
        where ρ_τ(u) = u(τ - 1{u<0})
              = τ|u|     if u >= 0 (under-prediction)
              = (1-τ)|u| if u < 0  (over-prediction)

    Interpretation:
        For τ=0.75:
        - Under-predictions (y > ŷ) penalized by 0.75
        - Over-predictions (y < ŷ) penalized by 0.25
        This makes the model target the 75th percentile.
    """
    if not (0 < quantile < 1):
        raise ValueError(f"Quantile must be in (0, 1), got {quantile}")

    errors = y_true - y_pred

    loss = np.where(
        errors >= 0,
        quantile * errors,
        (quantile - 1) * errors
    )

    return np.sum(loss)


def fit_quantile_regression(
    X: np.ndarray,
    y: np.ndarray,
    quantile: float = 0.75,
    feature_names: Optional[List[str]] = None,
    method: str = 'SLSQP',
    max_iterations: int = 1000
) -> QuantileRegressionResults:
    """
    Fit quantile regression model using optimization.

    Estimates the conditional quantile of y given X by minimizing
    the quantile loss function.

    Args:
        X: Predictor variables matrix (n_samples, n_features)
        y: Target variable (n_samples,)
        quantile: Target quantile, typically 0.75 or 0.90 for downturn LGD
        feature_names: Names of features (optional)
        method: Optimization method ('SLSQP', 'L-BFGS-B', 'Powell')
        max_iterations: Maximum optimization iterations

    Returns:
        QuantileRegressionResults object

    Example:
        >>> X = np.column_stack([x1, x2, x3])
        >>> y = lgd_values
        >>> results = fit_quantile_regression(X, y, quantile=0.75)
        >>> print(results)
        >>> downturn_lgd = results.predictions

    Regulatory Context:
        Article 181/182 CRR require downturn LGD calibration.
        Quantile regression with τ=0.75 or τ=0.90 provides conservative
        LGD estimates reflecting stressed economic conditions.

    Common Quantiles:
        τ = 0.50: Median regression (robust to outliers)
        τ = 0.75: Conservative estimate (common for downturn LGD)
        τ = 0.90: Very conservative (high-stress scenarios)

    Raises:
        ValueError: If inputs are invalid
    """
    # Validate inputs
    if not (0 < quantile < 1):
        raise ValueError(f"Quantile must be in (0, 1), got {quantile}")

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).flatten()

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n, p = X.shape

    if len(y) != n:
        raise ValueError(f"Length mismatch: X={n}, y={len(y)}")

    if n < p + 1:
        raise ValueError(
            f"Need at least {p+1} observations for {p} features, got {n}"
        )

    # Add intercept
    X_with_intercept = np.column_stack([np.ones(n), X])

    # Feature names
    if feature_names is None:
        feature_names = ['intercept'] + [f'x{i+1}' for i in range(p)]
    else:
        if len(feature_names) != p:
            raise ValueError(
                f"Expected {p} feature names, got {len(feature_names)}"
            )
        feature_names = ['intercept'] + list(feature_names)

    logger.info(
        f"Fitting quantile regression: n={n}, p={p}, τ={quantile:.2f}"
    )

    # Objective function to minimize
    def objective(beta):
        y_pred = X_with_intercept @ beta
        return quantile_loss(y, y_pred, quantile)

    # Initial guess: use OLS coefficients
    try:
        beta_ols = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
        beta_init = beta_ols
    except:
        beta_init = np.zeros(p + 1)

    # Optimize
    result = minimize(
        objective,
        beta_init,
        method=method,
        options={'maxiter': max_iterations}
    )

    coefficients = result.x
    converged = result.success

    if not converged:
        logger.warning(
            f"Quantile regression did not converge: {result.message}"
        )

    # Predictions
    predictions = X_with_intercept @ coefficients

    logger.info(
        f"Quantile regression completed: converged={converged}, "
        f"loss={result.fun:.4f}"
    )

    return QuantileRegressionResults(
        quantile=quantile,
        coefficients=coefficients,
        predictions=predictions,
        feature_names=feature_names,
        n_observations=n,
        converged=converged
    )


def estimate_lgd_floor(
    X: np.ndarray,
    lgd_values: np.ndarray,
    quantile: float = 0.75,
    feature_names: Optional[List[str]] = None
) -> Tuple[float, QuantileRegressionResults]:
    """
    Estimate downturn LGD floor using quantile regression.

    Calculates a conservative LGD floor based on the specified quantile
    of the conditional LGD distribution. Used for Article 181/182 CRR
    downturn calibration.

    Args:
        X: Predictor variables (e.g., industry, seniority, collateral type)
        lgd_values: Observed LGD values (range typically 0-1)
        quantile: Target quantile for floor (default: 0.75)
        feature_names: Names of predictor variables (optional)

    Returns:
        Tuple of (floor_lgd, quantile_regression_results)
        - floor_lgd: Conservative LGD floor estimate
        - quantile_regression_results: Full regression results

    Example:
        >>> X = np.column_stack([
        ...     industry_dummies,
        ...     seniority_indicators,
        ...     collateral_values
        ... ])
        >>> lgd_values = observed_loss_rates
        >>> floor, results = estimate_lgd_floor(X, lgd_values, quantile=0.75)
        >>> print(f"Downturn LGD floor: {floor:.2%}")

    Regulatory Context:
        Article 181 CRR: "LGD estimates shall be based on a period that
        includes a downturn... or adjusted to reflect a downturn."

        Quantile regression provides a data-driven approach to estimating
        downturn LGD by targeting high percentiles of the loss distribution.

    Interpretation:
        quantile=0.75: LGD floor exceeds 75% of historical observations
        quantile=0.90: LGD floor exceeds 90% of historical observations
        Higher quantiles = more conservative = higher capital requirements

    Raises:
        ValueError: If inputs are invalid
    """
    # Validate LGD values
    lgd_values = np.asarray(lgd_values, dtype=np.float64)

    if np.any(lgd_values < 0) or np.any(lgd_values > 1):
        logger.warning(
            f"LGD values outside [0, 1] range detected: "
            f"min={lgd_values.min():.3f}, max={lgd_values.max():.3f}"
        )

    logger.info(
        f"Estimating LGD floor using {quantile:.0%} quantile regression"
    )

    # Fit quantile regression
    results = fit_quantile_regression(
        X, lgd_values, quantile, feature_names
    )

    # Floor estimate: use quantile of predictions or observed values
    # Conservative approach: maximum of quantile regression and empirical quantile
    empirical_quantile = np.quantile(lgd_values, quantile)
    predicted_quantile = np.quantile(results.predictions, quantile)

    floor_lgd = max(empirical_quantile, predicted_quantile)

    logger.info(
        f"LGD floor estimates: empirical={empirical_quantile:.2%}, "
        f"predicted={predicted_quantile:.2%}, floor={floor_lgd:.2%}"
    )

    return floor_lgd, results


def compare_quantile_models(
    X: np.ndarray,
    y: np.ndarray,
    quantiles: List[float] = [0.50, 0.75, 0.90, 0.95],
    feature_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compare quantile regression models at different quantile levels.

    Fits multiple quantile regression models and compares coefficients
    and predictions across quantiles. Useful for sensitivity analysis
    and understanding how LGD floors vary with conservatism.

    Args:
        X: Predictor variables matrix
        y: Target variable (e.g., LGD values)
        quantiles: List of quantiles to estimate (default: [0.50, 0.75, 0.90, 0.95])
        feature_names: Names of predictor variables (optional)

    Returns:
        DataFrame with coefficients for each quantile

    Example:
        >>> X = np.column_stack([x1, x2, x3])
        >>> y = lgd_values
        >>> comparison = compare_quantile_models(X, y)
        >>> print(comparison)

    Interpretation:
        Compare how coefficients change across quantiles:
        - Increasing coefficients: Variable has stronger effect in downturns
        - Decreasing coefficients: Variable has weaker effect in downturns
        - Stable coefficients: Variable effect is consistent across distribution

    Regulatory Use:
        Demonstrates robustness of LGD drivers across economic conditions.
        Shows how downturn calibration differs from TTC (through-the-cycle) estimates.
    """
    logger.info(f"Comparing quantile models: τ = {quantiles}")

    # Fit models at each quantile
    results_list = []

    for q in quantiles:
        result = fit_quantile_regression(X, y, quantile=q, feature_names=feature_names)

        if not result.converged:
            logger.warning(f"Model at τ={q:.2f} did not converge")

        # Store coefficients
        coef_dict = {'quantile': q}
        for name, coef in zip(result.feature_names, result.coefficients):
            coef_dict[name] = coef

        results_list.append(coef_dict)

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results_list)

    logger.info(f"Quantile model comparison completed for {len(quantiles)} quantiles")

    return comparison_df


def validate_downturn_calibration(
    baseline_lgd: np.ndarray,
    downturn_lgd: np.ndarray,
    min_ratio: float = 1.0
) -> Dict[str, Union[float, bool]]:
    """
    Validate downturn LGD calibration meets regulatory requirements.

    Checks that downturn LGD is at least as conservative as baseline LGD,
    as required by Article 181/182 CRR.

    Args:
        baseline_lgd: Through-the-cycle (TTC) LGD estimates
        downturn_lgd: Downturn LGD estimates
        min_ratio: Minimum required downturn/baseline ratio (default: 1.0)

    Returns:
        Dictionary with validation results:
            - mean_baseline: Mean baseline LGD
            - mean_downturn: Mean downturn LGD
            - median_baseline: Median baseline LGD
            - median_downturn: Median downturn LGD
            - downturn_ratio: Mean downturn / Mean baseline
            - passed: Whether downturn >= baseline * min_ratio

    Example:
        >>> baseline = np.array([0.30, 0.35, 0.25, 0.40])
        >>> downturn = np.array([0.38, 0.42, 0.32, 0.48])
        >>> validation = validate_downturn_calibration(baseline, downturn)
        >>> print(f"Downturn ratio: {validation['downturn_ratio']:.2f}")
        >>> print(f"Validation: {'✓ PASSED' if validation['passed'] else '✗ FAILED'}")

    Regulatory Context:
        Article 181 CRR: LGD estimates shall reflect economic downturn conditions.
        Article 182 CRR: Downturn LGD ≥ Long-run average LGD.

        Typical requirements:
        - Downturn LGD ≥ 1.0 × Baseline LGD (minimum)
        - Downturn LGD ≥ 1.2 × Baseline LGD (common practice)

    Raises:
        ValueError: If inputs have mismatched lengths
    """
    # Validate inputs
    baseline_lgd = np.asarray(baseline_lgd, dtype=np.float64)
    downturn_lgd = np.asarray(downturn_lgd, dtype=np.float64)

    if len(baseline_lgd) != len(downturn_lgd):
        raise ValueError(
            f"Length mismatch: baseline={len(baseline_lgd)}, "
            f"downturn={len(downturn_lgd)}"
        )

    # Calculate statistics
    mean_baseline = np.mean(baseline_lgd)
    mean_downturn = np.mean(downturn_lgd)
    median_baseline = np.median(baseline_lgd)
    median_downturn = np.median(downturn_lgd)

    # Downturn ratio
    downturn_ratio = mean_downturn / mean_baseline if mean_baseline > 0 else np.inf

    # Validation
    passed = downturn_ratio >= min_ratio

    logger.info(
        f"Downturn calibration validation: baseline={mean_baseline:.2%}, "
        f"downturn={mean_downturn:.2%}, ratio={downturn_ratio:.2f}, "
        f"passed={passed}"
    )

    return {
        'mean_baseline': float(mean_baseline),
        'mean_downturn': float(mean_downturn),
        'median_baseline': float(median_baseline),
        'median_downturn': float(median_downturn),
        'downturn_ratio': float(downturn_ratio),
        'min_required_ratio': float(min_ratio),
        'passed': bool(passed)
    }
