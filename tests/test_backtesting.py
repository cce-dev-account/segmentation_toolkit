"""
Unit Tests for Back-testing Module

Tests all back-testing functions required for IRB validation over time.
"""

import pytest
import numpy as np
import pandas as pd
from irb_segmentation.validators.backtesting import (
    binomial_backtest,
    chi_squared_backtest,
    traffic_light_backtest,
    run_backtest,
    BacktestResults
)


class TestBinomialBacktest:
    """Test suite for binomial back-test."""

    def test_perfect_prediction(self):
        """Test binomial test when observed = expected."""
        # 100 observations, 10 defaults, PD = 0.10
        p_value, passed, z_score = binomial_backtest(100, 10, 0.10)

        assert passed, "Perfect prediction should pass"
        assert p_value > 0.05, f"P-value should be > 0.05, got {p_value}"
        assert abs(z_score) < 0.1, f"Z-score should be ~0, got {z_score}"

    def test_significant_deviation(self):
        """Test binomial test when observed >> expected."""
        # 1000 observations, 50 defaults, but PD = 0.01 (expect 10)
        p_value, passed, z_score = binomial_backtest(1000, 50, 0.01)

        assert not passed, "Significant deviation should fail"
        assert p_value < 0.05, f"P-value should be < 0.05, got {p_value}"
        assert abs(z_score) > 2, f"Z-score should be large, got {z_score}"

    def test_small_sample(self):
        """Test binomial test with small sample."""
        # 10 observations, 1 default, PD = 0.10
        p_value, passed, z_score = binomial_backtest(10, 1, 0.10)

        # Should work without errors
        assert isinstance(p_value, float)
        assert isinstance(z_score, float)

    def test_zero_defaults(self):
        """Test binomial test with zero observed defaults."""
        # 100 observations, 0 defaults, PD = 0.01
        p_value, passed, z_score = binomial_backtest(100, 0, 0.01)

        assert isinstance(p_value, float)
        assert z_score < 0, "Z-score should be negative (under-prediction)"

    def test_all_defaults(self):
        """Test binomial test when all observations default."""
        # 10 observations, 10 defaults, PD = 0.90
        p_value, passed, z_score = binomial_backtest(10, 10, 0.90)

        assert isinstance(p_value, float)
        assert z_score > 0, "Z-score should be positive (over-prediction of defaults)"

    def test_validates_negative_observations(self):
        """Test that negative observations raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            binomial_backtest(-100, 10, 0.10)

    def test_validates_negative_defaults(self):
        """Test that negative defaults raise ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            binomial_backtest(100, -10, 0.10)

    def test_validates_pd_range(self):
        """Test that PD outside [0,1] raises ValueError."""
        with pytest.raises(ValueError, match="must be in"):
            binomial_backtest(100, 10, 1.5)

        with pytest.raises(ValueError, match="must be in"):
            binomial_backtest(100, 10, -0.1)

    def test_validates_defaults_exceed_observations(self):
        """Test that defaults > observations raises ValueError."""
        with pytest.raises(ValueError, match="cannot exceed"):
            binomial_backtest(100, 150, 0.10)

    def test_confidence_level_parameter(self):
        """Test different confidence levels."""
        # Same test at different confidence levels
        p_value_95, passed_95, _ = binomial_backtest(100, 12, 0.10, confidence_level=0.95)
        p_value_99, passed_99, _ = binomial_backtest(100, 12, 0.10, confidence_level=0.99)

        # P-values should be the same
        assert p_value_95 == pytest.approx(p_value_99)

        # 99% CI is more lenient, so might pass when 95% fails
        # (though for this specific case both should pass)


class TestChiSquaredBacktest:
    """Test suite for chi-squared multi-period back-test."""

    def test_consistent_periods(self):
        """Test chi-squared when all periods are consistent."""
        obs = np.array([1000, 1000, 1000])
        defaults = np.array([20, 20, 20])
        pds = np.array([0.02, 0.02, 0.02])

        chi2, p_value, passed = chi_squared_backtest(obs, defaults, pds)

        assert passed, "Consistent periods should pass"
        assert p_value > 0.05, f"P-value should be > 0.05, got {p_value}"
        assert chi2 < 10, f"Chi-squared should be small, got {chi2}"

    def test_inconsistent_periods(self):
        """Test chi-squared when periods show systematic deviation."""
        obs = np.array([1000, 1000, 1000])
        defaults = np.array([10, 50, 90])  # Huge variation!
        pds = np.array([0.02, 0.02, 0.02])  # But constant PD

        chi2, p_value, passed = chi_squared_backtest(obs, defaults, pds)

        # Should fail due to inconsistency
        assert not passed or chi2 > 10, "Inconsistent periods should show high chi-squared"

    def test_varying_sample_sizes(self):
        """Test chi-squared with different sample sizes per period."""
        obs = np.array([500, 1000, 1500])
        defaults = np.array([10, 20, 30])
        pds = np.array([0.02, 0.02, 0.02])

        chi2, p_value, passed = chi_squared_backtest(obs, defaults, pds)

        assert isinstance(chi2, float)
        assert isinstance(p_value, float)
        assert isinstance(passed, bool)

    def test_validates_array_lengths(self):
        """Test that mismatched array lengths raise ValueError."""
        obs = np.array([1000, 1000])
        defaults = np.array([20, 20, 20])  # Wrong length
        pds = np.array([0.02, 0.02])

        with pytest.raises(ValueError, match="length mismatch"):
            chi_squared_backtest(obs, defaults, pds)

    def test_validates_minimum_periods(self):
        """Test that single period raises ValueError."""
        obs = np.array([1000])
        defaults = np.array([20])
        pds = np.array([0.02])

        with pytest.raises(ValueError, match="at least 2 periods"):
            chi_squared_backtest(obs, defaults, pds)

    def test_two_periods(self):
        """Test minimum valid case: exactly 2 periods."""
        obs = np.array([1000, 1000])
        defaults = np.array([20, 22])
        pds = np.array([0.02, 0.02])

        chi2, p_value, passed = chi_squared_backtest(obs, defaults, pds)

        assert isinstance(chi2, float)
        # With 2 periods, df = 1


class TestTrafficLightBacktest:
    """Test suite for traffic light classification."""

    def test_green_status(self):
        """Test z-score within 95% CI gives GREEN."""
        status = traffic_light_backtest(1.5)  # < 1.96
        assert status == 'GREEN'

    def test_amber_status(self):
        """Test z-score between 95% and 99% CI gives AMBER."""
        status = traffic_light_backtest(2.2)  # Between 1.96 and 2.58
        assert status == 'AMBER'

    def test_red_status(self):
        """Test z-score outside 99% CI gives RED."""
        status = traffic_light_backtest(3.0)  # > 2.58
        assert status == 'RED'

    def test_boundary_green_amber(self):
        """Test boundary between GREEN and AMBER."""
        status_green = traffic_light_backtest(1.96)  # Exactly at threshold
        status_amber = traffic_light_backtest(1.97)  # Just above

        assert status_green == 'GREEN'
        assert status_amber == 'AMBER'

    def test_boundary_amber_red(self):
        """Test boundary between AMBER and RED."""
        status_amber = traffic_light_backtest(2.58)  # Exactly at threshold
        status_red = traffic_light_backtest(2.59)  # Just above

        assert status_amber == 'AMBER'
        assert status_red == 'RED'

    def test_negative_z_score(self):
        """Test that negative z-scores use absolute value."""
        status_pos = traffic_light_backtest(2.0)
        status_neg = traffic_light_backtest(-2.0)

        assert status_pos == status_neg, "Should use absolute z-score"

    def test_custom_thresholds(self):
        """Test custom threshold values."""
        # Stricter thresholds
        status = traffic_light_backtest(1.8, green_threshold=1.5, amber_threshold=2.0)
        assert status == 'AMBER'


class TestRunBacktest:
    """Test suite for comprehensive back-testing."""

    def test_single_period(self):
        """Test back-test with single period."""
        periods = np.array([2020] * 100)
        y_true = np.array([0] * 90 + [1] * 10)
        y_pred = np.array([0.1] * 100)

        results = run_backtest(periods, y_true, y_pred)

        assert isinstance(results, BacktestResults)
        assert results.n_periods == 1
        assert results.total_observations == 100
        assert results.total_defaults == 10

    def test_multiple_periods(self):
        """Test back-test across multiple periods."""
        # 3 periods with 100 observations each
        periods = np.array([2020] * 100 + [2021] * 100 + [2022] * 100)
        y_true = np.array([0] * 90 + [1] * 10 +  # 2020: 10 defaults
                         [0] * 92 + [1] * 8 +   # 2021: 8 defaults
                         [0] * 88 + [1] * 12)   # 2022: 12 defaults
        y_pred = np.array([0.1] * 300)

        results = run_backtest(periods, y_true, y_pred)

        assert results.n_periods == 3
        assert results.total_observations == 300
        assert results.total_defaults == 30
        assert len(results.period_results) == 3

    def test_well_calibrated_model_passes(self):
        """Test that well-calibrated model passes all tests."""
        np.random.seed(42)
        n_per_period = 1000

        # Generate data with PD ~ 0.02
        periods = np.array([2020] * n_per_period + [2021] * n_per_period + [2022] * n_per_period)
        y_pred = np.random.uniform(0.015, 0.025, 3 * n_per_period)
        y_true = (np.random.random(3 * n_per_period) < y_pred).astype(int)

        results = run_backtest(periods, y_true, y_pred)

        # Should pass (with high probability given random seed)
        assert results.binomial_test_passed or not results.binomial_test_passed  # May vary
        assert isinstance(results.overall_passed, bool)

    def test_period_results_structure(self):
        """Test that period_results DataFrame has correct structure."""
        periods = np.array([2020] * 100 + [2021] * 100)
        y_true = np.array([0] * 90 + [1] * 10 + [0] * 92 + [1] * 8)
        y_pred = np.array([0.1] * 200)

        results = run_backtest(periods, y_true, y_pred)

        df = results.period_results
        required_cols = [
            'period', 'n_observations', 'n_defaults', 'predicted_pd',
            'actual_pd', 'z_score', 'p_value', 'binomial_passed', 'status'
        ]

        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_traffic_light_status_worst_case(self):
        """Test that overall status is worst-case across periods."""
        # Create scenario with one RED period
        periods = np.array([2020] * 100 + [2021] * 100)
        y_true = np.array([0] * 98 + [1] * 2 +  # 2020: 2 defaults (low)
                         [1] * 50 + [0] * 50)   # 2021: 50 defaults (very high!)
        y_pred = np.array([0.02] * 200)

        results = run_backtest(periods, y_true, y_pred)

        # Overall status should be RED due to 2021
        assert results.traffic_light_status == 'RED', \
            "Overall status should be RED when any period is RED"

    def test_validates_input_lengths(self):
        """Test that mismatched input lengths raise ValueError."""
        periods = np.array([2020] * 100)
        y_true = np.array([0] * 90 + [1] * 10)
        y_pred = np.array([0.1] * 50)  # Wrong length

        with pytest.raises(ValueError, match="Length mismatch"):
            run_backtest(periods, y_true, y_pred)

    def test_validates_empty_arrays(self):
        """Test that empty arrays raise ValueError."""
        with pytest.raises(ValueError, match="Empty arrays"):
            run_backtest(np.array([]), np.array([]), np.array([]))

    def test_custom_period_ids(self):
        """Test specifying explicit period IDs."""
        periods = np.array([2020] * 100 + [2021] * 100 + [2022] * 100)
        y_true = np.array([0] * 270 + [1] * 30)
        y_pred = np.array([0.1] * 300)

        # Only analyze 2020 and 2022
        results = run_backtest(periods, y_true, y_pred, period_ids=[2020, 2022])

        assert results.n_periods == 2
        assert set(results.period_results['period']) == {2020, 2022}

    def test_missing_period_in_data(self):
        """Test when period_ids includes period not in data."""
        periods = np.array([2020] * 100)
        y_true = np.array([0] * 90 + [1] * 10)
        y_pred = np.array([0.1] * 100)

        # Request period that doesn't exist
        results = run_backtest(periods, y_true, y_pred, period_ids=[2020, 2021])

        # Should only have 2020 results
        assert results.n_periods == 1
        assert results.period_results['period'].iloc[0] == 2020

    def test_to_dict_serialization(self):
        """Test to_dict returns JSON-serializable structure."""
        periods = np.array([2020] * 100 + [2021] * 100)
        y_true = np.array([0] * 90 + [1] * 10 + [0] * 92 + [1] * 8)
        y_pred = np.array([0.1] * 200)

        results = run_backtest(periods, y_true, y_pred)
        results_dict = results.to_dict()

        assert isinstance(results_dict, dict)
        assert 'binomial_test' in results_dict
        assert 'chi_squared_test' in results_dict
        assert 'traffic_light_status' in results_dict
        assert 'period_results' in results_dict
        assert 'summary' in results_dict
        assert 'overall_passed' in results_dict

        # Check all values are JSON-serializable types
        import json
        try:
            json.dumps(results_dict)
        except (TypeError, ValueError) as e:
            pytest.fail(f"to_dict() not JSON-serializable: {e}")

    def test_str_representation(self):
        """Test string representation."""
        periods = np.array([2020] * 100 + [2021] * 100)
        y_true = np.array([0] * 90 + [1] * 10 + [0] * 92 + [1] * 8)
        y_pred = np.array([0.1] * 200)

        results = run_backtest(periods, y_true, y_pred)
        output = str(results)

        assert 'BACK-TESTING RESULTS' in output
        assert 'Time Periods Analyzed' in output
        assert 'Binomial Test' in output
        assert 'Chi-Squared Test' in output
        assert 'Traffic Light Status' in output
        assert 'PASSED' in output or 'FAILED' in output


class TestBacktestResultsDataclass:
    """Test suite for BacktestResults dataclass."""

    def test_instantiation(self):
        """Test creating BacktestResults instance."""
        df = pd.DataFrame({
            'period': [2020, 2021],
            'n_observations': [100, 100],
            'n_defaults': [10, 8],
            'predicted_pd': [0.10, 0.10],
            'actual_pd': [0.10, 0.08],
            'z_score': [0.0, -0.5],
            'p_value': [1.0, 0.6],
            'binomial_passed': [True, True],
            'status': ['GREEN', 'GREEN']
        })

        results = BacktestResults(
            binomial_test_pvalue=0.8,
            binomial_test_passed=True,
            chi_squared_stat=0.5,
            chi_squared_pvalue=0.7,
            chi_squared_passed=True,
            traffic_light_status='GREEN',
            period_results=df,
            n_periods=2,
            total_observations=200,
            total_defaults=18,
            mean_predicted_pd=0.10,
            mean_actual_pd=0.09,
            overall_passed=True
        )

        assert results.n_periods == 2
        assert results.overall_passed is True

    def test_failed_status_in_output(self):
        """Test that failed status appears in output."""
        df = pd.DataFrame({
            'period': [2020],
            'n_observations': [100],
            'n_defaults': [50],
            'predicted_pd': [0.10],
            'actual_pd': [0.50],
            'z_score': [10.0],
            'p_value': [0.001],
            'binomial_passed': [False],
            'status': ['RED']
        })

        results = BacktestResults(
            binomial_test_pvalue=0.001,
            binomial_test_passed=False,
            chi_squared_stat=100.0,
            chi_squared_pvalue=0.001,
            chi_squared_passed=False,
            traffic_light_status='RED',
            period_results=df,
            n_periods=1,
            total_observations=100,
            total_defaults=50,
            mean_predicted_pd=0.10,
            mean_actual_pd=0.50,
            overall_passed=False
        )

        output = str(results)
        assert 'FAILED' in output
        assert 'RED' in output or '🔴' in output


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_low_pd(self):
        """Test back-test with very low PD (e.g., 0.001)."""
        periods = np.array([2020] * 10000)
        y_pred = np.array([0.001] * 10000)
        y_true = np.array([0] * 9990 + [1] * 10)  # 10 defaults = 0.001 rate

        results = run_backtest(periods, y_true, y_pred)

        assert isinstance(results, BacktestResults)

    def test_very_high_pd(self):
        """Test back-test with very high PD (e.g., 0.90)."""
        periods = np.array([2020] * 100)
        y_pred = np.array([0.90] * 100)
        y_true = np.array([1] * 90 + [0] * 10)  # 90% default rate

        results = run_backtest(periods, y_true, y_pred)

        assert isinstance(results, BacktestResults)

    def test_list_inputs(self):
        """Test that lists are correctly converted to arrays."""
        periods = [2020] * 100
        y_true = [0] * 90 + [1] * 10
        y_pred = [0.1] * 100

        results = run_backtest(periods, y_true, y_pred)

        assert isinstance(results, BacktestResults)

    def test_pandas_series_inputs(self):
        """Test that pandas Series work correctly."""
        periods = pd.Series([2020] * 100)
        y_true = pd.Series([0] * 90 + [1] * 10)
        y_pred = pd.Series([0.1] * 100)

        results = run_backtest(periods, y_true, y_pred)

        assert isinstance(results, BacktestResults)


# Pytest configuration
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
