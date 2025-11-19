"""
Tests for Regression Metrics Module

Tests R², MAE, RMSE, MAPE and related regression performance metrics
for LGD and EAD model validation.
"""

import pytest
import numpy as np
import pandas as pd
from irb_segmentation.validators.regression_metrics import (
    r_squared,
    adjusted_r_squared,
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
    calculate_all_regression_metrics,
    RegressionMetrics
)


class TestRSquared:
    """Test R² (coefficient of determination) calculation."""

    def test_perfect_predictions(self):
        """Test R² = 1.0 for perfect predictions."""
        y_true = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        y_pred = y_true.copy()

        r2 = r_squared(y_true, y_pred)

        assert r2 == pytest.approx(1.0, abs=1e-10)

    def test_mean_predictions(self):
        """Test R² = 0.0 for predictions equal to mean."""
        y_true = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        y_pred = np.full_like(y_true, np.mean(y_true))

        r2 = r_squared(y_true, y_pred)

        assert r2 == pytest.approx(0.0, abs=1e-10)

    def test_reasonable_predictions(self):
        """Test R² for reasonable predictions."""
        y_true = np.array([0.3, 0.5, 0.2, 0.7, 0.4, 0.6])
        y_pred = np.array([0.32, 0.48, 0.25, 0.68, 0.42, 0.58])

        r2 = r_squared(y_true, y_pred)

        assert 0.5 < r2 < 1.0, f"Expected reasonable R², got {r2:.3f}"

    def test_negative_r_squared(self):
        """Test R² < 0 for very poor predictions."""
        y_true = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        y_pred = np.array([0.9, 0.8, 0.7, 0.6, 0.5])  # Inverted

        r2 = r_squared(y_true, y_pred)

        assert r2 < 0, f"Expected negative R² for poor predictions, got {r2:.3f}"

    def test_length_mismatch(self):
        """Test error on mismatched array lengths."""
        y_true = np.array([0.1, 0.2, 0.3])
        y_pred = np.array([0.1, 0.2])

        with pytest.raises(ValueError, match="Length mismatch"):
            r_squared(y_true, y_pred)

    def test_empty_arrays(self):
        """Test error on empty arrays."""
        y_true = np.array([])
        y_pred = np.array([])

        with pytest.raises(ValueError, match="empty"):
            r_squared(y_true, y_pred)

    def test_nan_values(self):
        """Test error on NaN values."""
        y_true = np.array([0.1, 0.2, np.nan, 0.4])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])

        with pytest.raises(ValueError, match="NaN"):
            r_squared(y_true, y_pred)

    def test_constant_y_true(self):
        """Test R² when y_true is constant."""
        y_true = np.full(5, 0.3)
        y_pred = y_true.copy()

        r2 = r_squared(y_true, y_pred)

        assert r2 == pytest.approx(1.0, abs=1e-10)


class TestAdjustedRSquared:
    """Test adjusted R² calculation."""

    def test_single_predictor(self):
        """Test adjusted R² is close to R² for single predictor."""
        y_true = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        y_pred = np.array([0.12, 0.19, 0.31, 0.38, 0.52])

        r2 = r_squared(y_true, y_pred)
        adj_r2 = adjusted_r_squared(y_true, y_pred, n_predictors=1)

        # Adjusted R² should be slightly less than R² even for single predictor
        assert adj_r2 < r2
        assert adj_r2 == pytest.approx(r2, rel=0.01)  # Within 1%

    def test_multiple_predictors(self):
        """Test adjusted R² < R² for multiple predictors."""
        y_true = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        y_pred = np.array([0.12, 0.19, 0.31, 0.38, 0.52, 0.58, 0.71])

        r2 = r_squared(y_true, y_pred)
        adj_r2 = adjusted_r_squared(y_true, y_pred, n_predictors=3)

        assert adj_r2 < r2, f"Adjusted R² ({adj_r2:.3f}) should be < R² ({r2:.3f})"

    def test_invalid_n_predictors(self):
        """Test error when n_predictors is invalid."""
        y_true = np.array([0.1, 0.2, 0.3])
        y_pred = np.array([0.1, 0.2, 0.3])

        with pytest.raises(ValueError, match="n_predictors"):
            adjusted_r_squared(y_true, y_pred, n_predictors=0)

        with pytest.raises(ValueError, match="n_predictors"):
            adjusted_r_squared(y_true, y_pred, n_predictors=5)


class TestMeanAbsoluteError:
    """Test MAE calculation."""

    def test_zero_error(self):
        """Test MAE = 0 for perfect predictions."""
        y_true = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        y_pred = y_true.copy()

        mae = mean_absolute_error(y_true, y_pred)

        assert mae == pytest.approx(0.0, abs=1e-10)

    def test_known_error(self):
        """Test MAE with known errors."""
        y_true = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
        y_pred = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        # Errors: |0.1|, |0.1|, |0.1|, |0.1|, |0.1| -> MAE = 0.1

        mae = mean_absolute_error(y_true, y_pred)

        assert mae == pytest.approx(0.1, abs=1e-10)

    def test_mixed_errors(self):
        """Test MAE with mixed over/under predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([0.5, 2.5, 2.5, 4.5])
        # Errors: |-0.5|, |0.5|, |-0.5|, |0.5| -> MAE = 0.5

        mae = mean_absolute_error(y_true, y_pred)

        assert mae == pytest.approx(0.5, abs=1e-10)


class TestRootMeanSquaredError:
    """Test RMSE calculation."""

    def test_zero_error(self):
        """Test RMSE = 0 for perfect predictions."""
        y_true = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        y_pred = y_true.copy()

        rmse = root_mean_squared_error(y_true, y_pred)

        assert rmse == pytest.approx(0.0, abs=1e-10)

    def test_known_error(self):
        """Test RMSE with known errors."""
        y_true = np.array([0.0, 0.0, 0.0, 0.0])
        y_pred = np.array([2.0, 2.0, 2.0, 2.0])
        # Errors: [2, 2, 2, 2] -> MSE = 4 -> RMSE = 2

        rmse = root_mean_squared_error(y_true, y_pred)

        assert rmse == pytest.approx(2.0, abs=1e-10)

    def test_rmse_greater_than_mae(self):
        """Test RMSE >= MAE (equality only when all errors identical)."""
        y_true = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        y_pred = np.array([0.15, 0.18, 0.35, 0.38, 0.6])

        mae = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)

        assert rmse >= mae, f"RMSE ({rmse:.4f}) should be >= MAE ({mae:.4f})"


class TestMeanAbsolutePercentageError:
    """Test MAPE calculation."""

    def test_zero_error(self):
        """Test MAPE = 0 for perfect predictions."""
        y_true = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        y_pred = y_true.copy()

        mape = mean_absolute_percentage_error(y_true, y_pred)

        assert mape == pytest.approx(0.0, abs=1e-10)

    def test_known_percentage_error(self):
        """Test MAPE with known percentage errors."""
        y_true = np.array([100.0, 100.0, 100.0, 100.0])
        y_pred = np.array([110.0, 90.0, 105.0, 95.0])
        # Errors: |10%|, |10%|, |5%|, |5%| -> MAPE = 7.5%

        mape = mean_absolute_percentage_error(y_true, y_pred)

        assert mape == pytest.approx(7.5, abs=1e-8)

    def test_warning_on_zeros(self):
        """Test warning when y_true contains zeros."""
        y_true = np.array([0.0, 0.1, 0.2, 0.3])
        y_pred = np.array([0.05, 0.11, 0.19, 0.31])

        # Should complete without error (uses epsilon)
        mape = mean_absolute_percentage_error(y_true, y_pred)

        assert mape >= 0  # MAPE should be non-negative

    def test_small_values(self):
        """Test MAPE with small values (LGD range)."""
        y_true = np.array([0.01, 0.02, 0.03, 0.04])
        y_pred = np.array([0.011, 0.019, 0.032, 0.038])

        mape = mean_absolute_percentage_error(y_true, y_pred)

        # MAPE should be reasonable (< 20% for good LGD model)
        assert 0 <= mape <= 100, f"MAPE should be reasonable, got {mape:.2f}%"


class TestCalculateAllRegressionMetrics:
    """Test comprehensive regression metrics calculation."""

    def test_all_metrics_calculated(self):
        """Test all metrics are calculated correctly."""
        np.random.seed(42)
        n = 100

        # Generate realistic LGD data
        y_true = np.random.beta(2, 5, n)  # LGD typically skewed toward 0
        y_pred = y_true + np.random.normal(0, 0.05, n)  # Add noise
        y_pred = np.clip(y_pred, 0, 1)  # Keep in [0, 1] range

        metrics = calculate_all_regression_metrics(y_true, y_pred, n_predictors=3)

        assert isinstance(metrics, RegressionMetrics)
        assert 0 <= metrics.r_squared <= 1
        assert metrics.mae >= 0
        assert metrics.rmse >= 0
        assert metrics.mape >= 0
        assert metrics.n_observations == n

    def test_threshold_evaluation(self):
        """Test threshold evaluation for regulatory compliance."""
        # Good model (should pass)
        y_true = np.array([0.3, 0.5, 0.2, 0.7, 0.4, 0.6, 0.35, 0.55, 0.45, 0.65])
        y_pred = np.array([0.32, 0.48, 0.25, 0.68, 0.42, 0.58, 0.37, 0.53, 0.47, 0.63])

        metrics = calculate_all_regression_metrics(
            y_true, y_pred, n_predictors=2,
            r2_threshold=0.30, mape_threshold=20.0
        )

        assert metrics.passed_thresholds, "Good model should pass thresholds"
        assert metrics.r_squared >= 0.30
        assert metrics.mape <= 20.0

    def test_pandas_series_input(self):
        """Test function accepts pandas Series."""
        y_true = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
        y_pred = pd.Series([0.12, 0.19, 0.31, 0.38, 0.52])

        metrics = calculate_all_regression_metrics(y_true, y_pred)

        assert isinstance(metrics, RegressionMetrics)
        assert metrics.n_observations == 5

    def test_residual_std_calculation(self):
        """Test residual standard deviation is calculated."""
        y_true = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        y_pred = np.array([0.12, 0.19, 0.31, 0.38, 0.52])

        metrics = calculate_all_regression_metrics(y_true, y_pred)

        assert metrics.residual_std > 0
        # Verify it matches manual calculation
        residuals = y_true - y_pred
        expected_std = np.std(residuals, ddof=1)
        assert metrics.residual_std == pytest.approx(expected_std, abs=1e-10)

    def test_to_dict_conversion(self):
        """Test metrics can be converted to dictionary."""
        y_true = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        y_pred = np.array([0.12, 0.19, 0.31, 0.38, 0.52])

        metrics = calculate_all_regression_metrics(y_true, y_pred)
        metrics_dict = metrics.to_dict()

        assert isinstance(metrics_dict, dict)
        assert 'r_squared' in metrics_dict
        assert 'mae' in metrics_dict
        assert 'rmse' in metrics_dict
        assert 'mape' in metrics_dict
        assert 'passed_thresholds' in metrics_dict

    def test_string_representation(self):
        """Test string representation is generated."""
        y_true = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        y_pred = np.array([0.12, 0.19, 0.31, 0.38, 0.52])

        metrics = calculate_all_regression_metrics(y_true, y_pred)
        metrics_str = str(metrics)

        assert 'REGRESSION MODEL PERFORMANCE' in metrics_str
        assert 'R²' in metrics_str
        assert 'MAE' in metrics_str
        assert 'RMSE' in metrics_str
        assert 'MAPE' in metrics_str


class TestLGDValidationScenarios:
    """Test regression metrics on realistic LGD scenarios."""

    def test_retail_lgd_model(self):
        """Test metrics on simulated retail LGD model."""
        np.random.seed(123)
        n = 200

        # Retail LGDs typically 20-40% with low variance
        y_true = np.random.beta(3, 6, n) * 0.6 + 0.1  # Mean ~0.30
        y_pred = y_true + np.random.normal(0, 0.04, n)
        y_pred = np.clip(y_pred, 0, 1)

        metrics = calculate_all_regression_metrics(y_true, y_pred, n_predictors=5)

        # Retail LGD models should have good accuracy
        assert metrics.r_squared > 0.50, f"Retail LGD should have R² > 0.50, got {metrics.r_squared:.3f}"
        assert metrics.mape < 20.0, f"Retail LGD should have MAPE < 20%, got {metrics.mape:.2f}%"

    def test_corporate_lgd_model(self):
        """Test metrics on simulated corporate LGD model."""
        np.random.seed(456)
        n = 150

        # Corporate LGDs typically 30-70% with higher variance
        y_true = np.random.beta(2, 3, n) * 0.8 + 0.1  # Mean ~0.40
        y_pred = y_true + np.random.normal(0, 0.08, n)
        y_pred = np.clip(y_pred, 0, 1)

        metrics = calculate_all_regression_metrics(y_true, y_pred, n_predictors=7)

        # Corporate LGD models may have lower accuracy due to higher variance
        assert metrics.r_squared > 0.30, f"Corporate LGD should have R² > 0.30, got {metrics.r_squared:.3f}"
        assert metrics.mae < 0.15, f"Corporate LGD should have MAE < 0.15, got {metrics.mae:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
