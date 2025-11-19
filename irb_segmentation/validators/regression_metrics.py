"""
Regression Performance Metrics for Continuous Targets

Implements regression metrics for Loss Given Default (LGD) and
Exposure at Default (EAD) models in IRB frameworks.

This module complements the binary classification metrics with
regression-specific validation for continuous target variables.

References:
- Article 181 CRR: LGD estimation requirements
- Article 182 CRR: Downturn LGD calibration
- EBA Guidelines on LGD/CCF estimation (EBA/GL/2017/16)
- Basel III: A global regulatory framework for more resilient banks
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
from ..logger import get_logger

logger = get_logger(__name__)


@dataclass
class RegressionMetrics:
    """
    Container for regression model performance metrics.

    Attributes:
        r_squared: Coefficient of determination, range [0, 1]
        adjusted_r_squared: R² adjusted for number of predictors
        mae: Mean Absolute Error
        rmse: Root Mean Squared Error
        mape: Mean Absolute Percentage Error (%)
        residual_std: Standard deviation of residuals
        n_observations: Number of observations
        passed_thresholds: Whether metrics meet regulatory thresholds
    """
    r_squared: float
    adjusted_r_squared: float
    mae: float
    rmse: float
    mape: float
    residual_std: float
    n_observations: int
    passed_thresholds: bool

    def __str__(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 70,
            "REGRESSION MODEL PERFORMANCE METRICS",
            "=" * 70,
            "",
            "Goodness of Fit:",
            f"  R²:                   {self.r_squared:.4f} (threshold: > 0.30)",
            f"  Adjusted R²:          {self.adjusted_r_squared:.4f}",
            "",
            "Prediction Error:",
            f"  MAE:                  {self.mae:.4f}",
            f"  RMSE:                 {self.rmse:.4f}",
            f"  MAPE:                 {self.mape:.2f}% (threshold: < 20%)",
            f"  Residual Std Dev:     {self.residual_std:.4f}",
            "",
            f"Observations: {self.n_observations:,}",
            "",
            f"Regulatory Status: {'✓ PASSED' if self.passed_thresholds else '✗ FAILED'}",
            "=" * 70
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization."""
        return {
            'r_squared': float(self.r_squared),
            'adjusted_r_squared': float(self.adjusted_r_squared),
            'mae': float(self.mae),
            'rmse': float(self.rmse),
            'mape': float(self.mape),
            'residual_std': float(self.residual_std),
            'n_observations': int(self.n_observations),
            'passed_thresholds': bool(self.passed_thresholds)
        }


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate coefficient of determination (R²).

    R² measures the proportion of variance in the dependent variable
    that is predictable from the independent variables.

    Args:
        y_true: True continuous target values
        y_pred: Predicted continuous values

    Returns:
        R² in range (-∞, 1], where:
        - 1.0 = perfect predictions
        - 0.0 = predictions are as good as using the mean
        - < 0 = predictions are worse than using the mean

    Raises:
        ValueError: If inputs are invalid or have mismatched lengths

    Example:
        >>> y_true = np.array([0.3, 0.5, 0.2, 0.7])
        >>> y_pred = np.array([0.32, 0.48, 0.25, 0.68])
        >>> r2 = r_squared(y_true, y_pred)
        >>> print(f"R²: {r2:.3f}")
        R²: 0.982

    Interpretation:
        R² > 0.70: Strong predictive power (excellent for LGD models)
        R² > 0.50: Moderate predictive power (acceptable)
        R² > 0.30: Weak predictive power (minimum regulatory threshold)
        R² < 0.30: Insufficient predictive power

    Regulatory Context:
        While not explicitly mandated, R² > 0.30 is commonly used as
        a threshold for LGD model acceptance by regulators.
    """
    # Validate inputs
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
        )

    if len(y_true) == 0:
        raise ValueError("Cannot calculate R² for empty arrays")

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Check for NaN or inf
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Input arrays contain NaN values")

    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        raise ValueError("Input arrays contain infinite values")

    # Calculate R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        # All y_true values are identical
        if ss_res == 0:
            return 1.0  # Perfect prediction of constant value
        else:
            return 0.0  # No variance to explain

    r2 = 1 - (ss_res / ss_tot)

    logger.debug(f"R² = {r2:.4f} (SS_res={ss_res:.2f}, SS_tot={ss_tot:.2f})")

    return float(r2)


def adjusted_r_squared(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_predictors: int
) -> float:
    """
    Calculate adjusted R² accounting for number of predictors.

    Adjusted R² penalizes models with many predictors to prevent
    overfitting and enable fair comparison of models with different
    numbers of features.

    Args:
        y_true: True continuous target values
        y_pred: Predicted continuous values
        n_predictors: Number of predictor variables in the model

    Returns:
        Adjusted R² in range (-∞, 1]

    Raises:
        ValueError: If inputs are invalid or n_predictors >= n_observations

    Example:
        >>> y_true = np.array([0.3, 0.5, 0.2, 0.7, 0.4])
        >>> y_pred = np.array([0.32, 0.48, 0.25, 0.68, 0.42])
        >>> adj_r2 = adjusted_r_squared(y_true, y_pred, n_predictors=2)
        >>> print(f"Adjusted R²: {adj_r2:.3f}")
        Adjusted R²: 0.972

    Interpretation:
        Adjusted R² is always ≤ R² (equal only when n_predictors = 1)
        Use adjusted R² when comparing models with different features

    Formula:
        Adjusted R² = 1 - [(1 - R²) * (n - 1) / (n - p - 1)]
        where n = number of observations, p = number of predictors
    """
    n = len(y_true)

    if n_predictors < 1:
        raise ValueError(f"n_predictors must be >= 1, got {n_predictors}")

    if n_predictors >= n - 1:
        raise ValueError(
            f"n_predictors ({n_predictors}) must be < n_observations - 1 ({n - 1})"
        )

    # Calculate R²
    r2 = r_squared(y_true, y_pred)

    # Adjust for number of predictors
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - n_predictors - 1))

    logger.debug(
        f"Adjusted R² = {adj_r2:.4f} (R²={r2:.4f}, n={n}, p={n_predictors})"
    )

    return float(adj_r2)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error (MAE).

    MAE is the average absolute difference between predictions and
    actual values. It's in the same units as the target variable and
    is less sensitive to outliers than RMSE.

    Args:
        y_true: True continuous target values
        y_pred: Predicted continuous values

    Returns:
        MAE (always >= 0)

    Raises:
        ValueError: If inputs are invalid or have mismatched lengths

    Example:
        >>> y_true = np.array([0.3, 0.5, 0.2, 0.7])
        >>> y_pred = np.array([0.32, 0.48, 0.25, 0.68])
        >>> mae = mean_absolute_error(y_true, y_pred)
        >>> print(f"MAE: {mae:.3f}")
        MAE: 0.023

    Interpretation:
        For LGD models (range 0-1):
        - MAE < 0.05: Excellent accuracy
        - MAE < 0.10: Good accuracy
        - MAE < 0.15: Acceptable accuracy
        - MAE >= 0.15: Poor accuracy

    Regulatory Context:
        MAE is useful for LGD validation as it directly measures average
        prediction error in loss rate units (e.g., percentage points).
    """
    # Validate inputs
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
        )

    if len(y_true) == 0:
        raise ValueError("Cannot calculate MAE for empty arrays")

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Check for NaN or inf
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Input arrays contain NaN values")

    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        raise ValueError("Input arrays contain infinite values")

    # Calculate MAE
    mae = np.mean(np.abs(y_true - y_pred))

    logger.debug(f"MAE = {mae:.4f}")

    return float(mae)


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).

    RMSE is the square root of the average squared difference between
    predictions and actual values. It's more sensitive to large errors
    than MAE and is in the same units as the target variable.

    Args:
        y_true: True continuous target values
        y_pred: Predicted continuous values

    Returns:
        RMSE (always >= 0)

    Raises:
        ValueError: If inputs are invalid or have mismatched lengths

    Example:
        >>> y_true = np.array([0.3, 0.5, 0.2, 0.7])
        >>> y_pred = np.array([0.32, 0.48, 0.25, 0.68])
        >>> rmse = root_mean_squared_error(y_true, y_pred)
        >>> print(f"RMSE: {rmse:.3f}")
        RMSE: 0.025

    Interpretation:
        RMSE is always >= MAE (equal when all errors are identical)
        Large RMSE/MAE ratio indicates presence of large errors/outliers

        For LGD models (range 0-1):
        - RMSE < 0.10: Excellent accuracy
        - RMSE < 0.15: Good accuracy
        - RMSE < 0.20: Acceptable accuracy
        - RMSE >= 0.20: Poor accuracy

    Regulatory Context:
        RMSE is important for LGD validation as it penalizes large
        errors more heavily, which is relevant for downturn scenarios
        where large losses must be accurately predicted.
    """
    # Validate inputs
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
        )

    if len(y_true) == 0:
        raise ValueError("Cannot calculate RMSE for empty arrays")

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Check for NaN or inf
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Input arrays contain NaN values")

    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        raise ValueError("Input arrays contain infinite values")

    # Calculate RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    logger.debug(f"RMSE = {rmse:.4f}")

    return float(rmse)


def mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    MAPE expresses prediction error as a percentage of actual values.
    Useful for LGD/EAD validation where relative errors are important.

    Args:
        y_true: True continuous target values
        y_pred: Predicted continuous values
        epsilon: Small value to avoid division by zero (default: 1e-10)

    Returns:
        MAPE as percentage (e.g., 15.5 means 15.5% error)

    Raises:
        ValueError: If inputs are invalid or have mismatched lengths
        Warning: If y_true contains zeros (MAPE may be unreliable)

    Example:
        >>> y_true = np.array([0.3, 0.5, 0.2, 0.7])
        >>> y_pred = np.array([0.32, 0.48, 0.25, 0.68])
        >>> mape = mean_absolute_percentage_error(y_true, y_pred)
        >>> print(f"MAPE: {mape:.2f}%")
        MAPE: 10.83%

    Interpretation:
        MAPE < 10%: Excellent accuracy
        MAPE < 20%: Good accuracy (regulatory threshold)
        MAPE < 30%: Acceptable accuracy
        MAPE >= 30%: Poor accuracy

    Important Notes:
        - MAPE is undefined when y_true = 0 (uses epsilon to avoid division by zero)
        - MAPE is asymmetric: overestimations and underestimations have different impacts
        - For LGD models, consider using MAE if many zero-loss defaults exist

    Regulatory Context:
        MAPE is commonly used in LGD validation to assess relative
        prediction accuracy. Many regulators expect MAPE < 20% for
        LGD model approval.
    """
    # Validate inputs
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
        )

    if len(y_true) == 0:
        raise ValueError("Cannot calculate MAPE for empty arrays")

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Check for NaN or inf
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Input arrays contain NaN values")

    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        raise ValueError("Input arrays contain infinite values")

    # Warn if zeros present
    n_zeros = np.sum(np.abs(y_true) < epsilon)
    if n_zeros > 0:
        logger.warning(
            f"MAPE calculation: {n_zeros}/{len(y_true)} values are near zero. "
            f"MAPE may be unreliable. Consider using MAE instead."
        )

    # Calculate MAPE with epsilon to avoid division by zero
    abs_percentage_errors = np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))
    mape = np.mean(abs_percentage_errors) * 100

    logger.debug(f"MAPE = {mape:.2f}%")

    return float(mape)


def calculate_all_regression_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    n_predictors: Optional[int] = None,
    r2_threshold: float = 0.30,
    mape_threshold: float = 20.0
) -> RegressionMetrics:
    """
    Calculate all regression performance metrics at once.

    Convenience function that computes R², MAE, RMSE, MAPE and evaluates
    whether the model meets regulatory thresholds.

    Args:
        y_true: True continuous target values
        y_pred: Predicted continuous values
        n_predictors: Number of predictors (optional, for adjusted R²)
        r2_threshold: Minimum acceptable R² (default: 0.30)
        mape_threshold: Maximum acceptable MAPE in % (default: 20.0)

    Returns:
        RegressionMetrics object with all metrics and threshold evaluation

    Example:
        >>> y_true = np.array([0.3, 0.5, 0.2, 0.7, 0.4])
        >>> y_pred = np.array([0.32, 0.48, 0.25, 0.68, 0.42])
        >>> metrics = calculate_all_regression_metrics(y_true, y_pred, n_predictors=3)
        >>> print(metrics)
        >>> print(f"Model passed: {metrics.passed_thresholds}")

    Regulatory Thresholds:
        - R² > 0.30: Minimum threshold for model acceptance
        - MAPE < 20%: Maximum acceptable prediction error

    Raises:
        ValueError: If inputs are invalid
    """
    # Convert to numpy arrays
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    logger.info(f"Calculating regression metrics for {len(y_true)} observations")

    # Calculate all metrics
    r2 = r_squared(y_true, y_pred)

    if n_predictors is not None:
        adj_r2 = adjusted_r_squared(y_true, y_pred, n_predictors)
    else:
        adj_r2 = r2  # No adjustment if n_predictors not provided

    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    # Calculate residual standard deviation
    residuals = y_true - y_pred
    residual_std = np.std(residuals, ddof=1)

    # Check thresholds
    passed = (r2 >= r2_threshold) and (mape <= mape_threshold)

    metrics = RegressionMetrics(
        r_squared=r2,
        adjusted_r_squared=adj_r2,
        mae=mae,
        rmse=rmse,
        mape=mape,
        residual_std=residual_std,
        n_observations=len(y_true),
        passed_thresholds=passed
    )

    logger.info(
        f"Regression metrics: R²={r2:.3f}, MAE={mae:.4f}, RMSE={rmse:.4f}, "
        f"MAPE={mape:.2f}%, passed={passed}"
    )

    return metrics
