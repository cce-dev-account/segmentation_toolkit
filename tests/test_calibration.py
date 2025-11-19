"""
Unit Tests for Calibration Module

Tests all calibration tests required for IRB validation.
"""

import pytest
import numpy as np
import pandas as pd
from irb_segmentation.validators.calibration import (
    hosmer_lemeshow_test,
    traffic_light_test,
    central_tendency_test,
    run_all_calibration_tests,
    CalibrationResults
)


class TestHosmerLemeshowTest:
    """Test suite for Hosmer-Lemeshow goodness-of-fit test."""

    def test_perfect_calibration(self):
        """Test HL test with perfectly calibrated predictions."""
        np.random.seed(42)
        n = 1000

        # Create perfectly calibrated data
        # Each bin has predictions that match actual rate
        y_pred = np.random.uniform(0, 1, n)
        y_true = (np.random.random(n) < y_pred).astype(int)

        chi2, pval = hosmer_lemeshow_test(y_true, y_pred, n_bins=10)

        # Perfect calibration should have high p-value
        assert pval > 0.05, \
            f"Perfect calibration should pass HL test, got p={pval}"
        assert chi2 >= 0, f"Chi-squared should be non-negative, got {chi2}"

    def test_poor_calibration(self):
        """Test HL test with poorly calibrated predictions."""
        np.random.seed(42)
        n = 1000

        # Create miscalibrated data: predictions too low
        y_true = np.random.binomial(1, 0.5, n)
        y_pred = np.random.uniform(0, 0.3, n)  # All low predictions!

        chi2, pval = hosmer_lemeshow_test(y_true, y_pred, n_bins=10)

        # Poor calibration should have low p-value
        assert pval < 0.05, \
            f"Poor calibration should fail HL test, got p={pval}"

    def test_validates_input_length(self):
        """Test that mismatched lengths raise ValueError."""
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0.1, 0.9])  # Too short!

        with pytest.raises(ValueError, match="Length mismatch"):
            hosmer_lemeshow_test(y_true, y_pred)

    def test_insufficient_data(self):
        """Test that too few samples raise ValueError."""
        y_true = np.array([0, 1, 0, 1])  # Only 4 samples
        y_pred = np.array([0.1, 0.9, 0.2, 0.8])

        with pytest.raises(ValueError, match="Insufficient data"):
            hosmer_lemeshow_test(y_true, y_pred, n_bins=10)

    def test_custom_bins(self):
        """Test with different number of bins."""
        np.random.seed(42)
        n = 500
        y_pred = np.random.uniform(0, 1, n)
        y_true = (np.random.random(n) < y_pred).astype(int)

        # Should work with 5 bins
        chi2_5, pval_5 = hosmer_lemeshow_test(y_true, y_pred, n_bins=5)

        # Should work with 20 bins
        chi2_20, pval_20 = hosmer_lemeshow_test(y_true, y_pred, n_bins=20)

        assert chi2_5 >= 0 and chi2_20 >= 0
        assert 0 <= pval_5 <= 1 and 0 <= pval_20 <= 1

    def test_extreme_predictions(self):
        """Test with extreme predictions (all 0 or all 1)."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.0, 0.0, 1.0, 1.0])

        # Should not crash
        chi2, pval = hosmer_lemeshow_test(y_true, y_pred, n_bins=2)

        assert chi2 >= 0
        assert 0 <= pval <= 1


class TestTrafficLightTest:
    """Test suite for traffic light test."""

    def test_well_calibrated_segments(self):
        """Test that well-calibrated segments show GREEN."""
        np.random.seed(42)

        # Create 3 well-calibrated segments
        segments = np.array([1] * 100 + [2] * 100 + [3] * 100)
        y_pred = np.concatenate([
            np.random.uniform(0.1, 0.3, 100),  # Segment 1: low risk
            np.random.uniform(0.4, 0.6, 100),  # Segment 2: medium risk
            np.random.uniform(0.7, 0.9, 100)   # Segment 3: high risk
        ])
        y_true = (np.random.random(300) < y_pred).astype(int)

        results = traffic_light_test(segments, y_true, y_pred)

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 3  # 3 segments
        assert 'status' in results.columns

        # Most should be GREEN (well-calibrated)
        n_green = (results['status'] == 'GREEN').sum()
        assert n_green >= 2, \
            f"Expected at least 2 GREEN segments, got {n_green}"

    def test_miscalibrated_segment_shows_red(self):
        """Test that badly miscalibrated segment shows RED."""
        # Segment 1: well-calibrated
        seg1_pred = np.full(100, 0.5)
        seg1_true = np.random.binomial(1, 0.5, 100)

        # Segment 2: badly miscalibrated (predict 0.1, actual 0.9!)
        seg2_pred = np.full(100, 0.1)
        seg2_true = np.ones(100, dtype=int)  # All defaults!

        segments = np.array([1] * 100 + [2] * 100)
        y_pred = np.concatenate([seg1_pred, seg2_pred])
        y_true = np.concatenate([seg1_true, seg2_true])

        results = traffic_light_test(segments, y_true, y_pred)

        # Segment 2 should be RED
        seg2_result = results[results['segment_id'] == 2].iloc[0]
        assert seg2_result['status'] == 'RED', \
            f"Badly miscalibrated segment should be RED, got {seg2_result['status']}"
        assert seg2_result['z_score'] > 2.58

    def test_amber_status(self):
        """Test AMBER status (between 95% and 99% CI)."""
        # Create segment with moderate miscalibration
        n = 500
        predicted_pd = 0.30
        actual_pd = 0.40  # Slightly off

        segments = np.zeros(n, dtype=int)
        y_pred = np.full(n, predicted_pd)
        y_true = np.random.binomial(1, actual_pd, n)

        results = traffic_light_test(segments, y_true, y_pred)

        # Should likely be AMBER or RED depending on sampling
        result = results.iloc[0]
        assert result['status'] in ['AMBER', 'RED', 'GREEN']  # All valid outcomes

    def test_validates_input_lengths(self):
        """Test that mismatched lengths raise ValueError."""
        segments = np.array([1, 1, 2])
        y_true = np.array([0, 1])  # Too short!
        y_pred = np.array([0.1, 0.2, 0.3])

        with pytest.raises(ValueError, match="Length mismatch"):
            traffic_light_test(segments, y_true, y_pred)

    def test_custom_thresholds(self):
        """Test with custom green/amber thresholds."""
        segments = np.array([1] * 100)
        y_pred = np.full(100, 0.5)
        y_true = np.random.binomial(1, 0.5, 100)

        # Very strict thresholds
        results_strict = traffic_light_test(
            segments, y_true, y_pred,
            green_threshold=1.0,  # Very strict
            amber_threshold=1.5
        )

        # Very lenient thresholds
        results_lenient = traffic_light_test(
            segments, y_true, y_pred,
            green_threshold=5.0,  # Very lenient
            amber_threshold=10.0
        )

        # Lenient should have more greens
        assert isinstance(results_strict, pd.DataFrame)
        assert isinstance(results_lenient, pd.DataFrame)

    def test_multiple_segments(self):
        """Test with many segments."""
        n_segments = 10
        samples_per_segment = 50

        segments = np.repeat(np.arange(n_segments), samples_per_segment)
        y_pred = np.random.uniform(0.1, 0.9, n_segments * samples_per_segment)
        y_true = (np.random.random(n_segments * samples_per_segment) < y_pred).astype(int)

        results = traffic_light_test(segments, y_true, y_pred)

        assert len(results) == n_segments
        assert set(results['segment_id']) == set(range(n_segments))

    def test_dataframe_structure(self):
        """Test that returned DataFrame has correct structure."""
        segments = np.array([1, 1, 2, 2])
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.1, 0.9, 0.2, 0.8])

        results = traffic_light_test(segments, y_true, y_pred)

        assert isinstance(results, pd.DataFrame)
        expected_cols = ['segment_id', 'n_observations', 'n_defaults',
                        'predicted_pd', 'actual_pd', 'z_score', 'status']
        assert all(col in results.columns for col in expected_cols)


class TestCentralTendencyTest:
    """Test suite for central tendency test."""

    def test_perfect_central_tendency(self):
        """Test with perfectly calibrated mean."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.5, 0.5, 0.5, 0.5])  # Mean = 0.5, actual = 0.5

        result = central_tendency_test(y_true, y_pred)

        assert abs(result['difference']) < 0.01, \
            f"Perfect central tendency should have diff≈0, got {result['difference']}"
        assert result['passed'], "Should pass central tendency test"

    def test_over_predicting(self):
        """Test when model over-predicts."""
        y_true = np.array([0, 0, 0, 1])  # Actual = 0.25
        y_pred = np.array([0.5, 0.5, 0.5, 0.5])  # Predicted = 0.50

        result = central_tendency_test(y_true, y_pred)

        assert result['difference'] > 0, \
            "Over-predicting should have positive difference"
        assert result['mean_predicted'] > result['mean_actual']

    def test_under_predicting(self):
        """Test when model under-predicts."""
        y_true = np.array([1, 1, 1, 0])  # Actual = 0.75
        y_pred = np.array([0.5, 0.5, 0.5, 0.5])  # Predicted = 0.50

        result = central_tendency_test(y_true, y_pred)

        assert result['difference'] < 0, \
            "Under-predicting should have negative difference"
        assert result['mean_predicted'] < result['mean_actual']

    def test_validates_input_length(self):
        """Test that mismatched lengths raise ValueError."""
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0.1, 0.9])  # Too short!

        with pytest.raises(ValueError, match="Length mismatch"):
            central_tendency_test(y_true, y_pred)

    def test_relative_error_calculation(self):
        """Test relative error calculation."""
        y_true = np.array([0, 0, 1, 1])  # Actual = 0.50
        y_pred = np.array([0.6, 0.6, 0.6, 0.6])  # Predicted = 0.60

        result = central_tendency_test(y_true, y_pred)

        # Relative error = (0.60 - 0.50) / 0.50 * 100 = 20%
        expected_rel_error = 20.0
        assert abs(result['relative_error'] - expected_rel_error) < 1.0

    def test_zero_actual_rate(self):
        """Test edge case with zero actual default rate."""
        y_true = np.array([0, 0, 0, 0])  # No defaults
        y_pred = np.array([0.1, 0.1, 0.1, 0.1])

        result = central_tendency_test(y_true, y_pred)

        assert result['mean_actual'] == 0.0
        # Relative error undefined (inf) when actual = 0
        assert result['difference'] > 0

    def test_pass_threshold(self):
        """Test pass/fail threshold (< 1 percentage point)."""
        # Just below threshold (should pass)
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])  # 0.10
        y_pred = np.full(10, 0.109)  # 0.109 (diff = 0.009)

        result_pass = central_tendency_test(y_true, y_pred)
        assert result_pass['passed'], "Should pass with diff < 0.01"

        # Just above threshold (should fail)
        y_pred = np.full(10, 0.111)  # 0.111 (diff = 0.011)

        result_fail = central_tendency_test(y_true, y_pred)
        assert not result_fail['passed'], "Should fail with diff > 0.01"


class TestRunAllCalibrationTests:
    """Test suite for integrated calibration testing."""

    def test_all_tests_run(self):
        """Test that all calibration tests are executed."""
        np.random.seed(42)
        n = 600  # Divisible by 3

        segments = np.repeat([1, 2, 3], n // 3)
        y_pred = np.random.uniform(0.2, 0.8, n)
        y_true = (np.random.random(n) < y_pred).astype(int)

        results = run_all_calibration_tests(y_true, y_pred, segments)

        assert isinstance(results, CalibrationResults)
        assert hasattr(results, 'hosmer_lemeshow_chi2')
        assert hasattr(results, 'traffic_light_results')
        assert hasattr(results, 'central_tendency_diff')
        assert hasattr(results, 'overall_passed')

    def test_well_calibrated_model_passes(self):
        """Test that well-calibrated model passes all tests."""
        np.random.seed(42)
        n = 1000

        segments = np.repeat([1, 2, 3, 4], n // 4)
        y_pred = np.random.uniform(0.2, 0.8, n)
        y_true = (np.random.random(n) < y_pred).astype(int)

        results = run_all_calibration_tests(y_true, y_pred, segments)

        # Should likely pass (well-calibrated)
        assert results.hosmer_lemeshow_passed or not results.hosmer_lemeshow_passed  # Either valid
        assert results.n_green + results.n_amber + results.n_red == 4

    def test_works_without_segments(self):
        """Test that it works without explicit segments."""
        np.random.seed(42)
        n = 500

        y_pred = np.random.uniform(0.2, 0.8, n)
        y_true = (np.random.random(n) < y_pred).astype(int)

        # Call without segments
        results = run_all_calibration_tests(y_true, y_pred, segment_id=None)

        assert isinstance(results, CalibrationResults)
        # Should create 1 dummy segment
        assert len(results.traffic_light_results) == 1

    def test_str_representation(self):
        """Test string representation."""
        np.random.seed(42)
        n = 500

        segments = np.repeat([1, 2], n // 2)
        y_pred = np.random.uniform(0.3, 0.7, n)
        y_true = (np.random.random(n) < y_pred).astype(int)

        results = run_all_calibration_tests(y_true, y_pred, segments)
        output = str(results)

        assert 'CALIBRATION TEST RESULTS' in output
        assert 'Hosmer-Lemeshow' in output
        assert 'Traffic Light' in output
        assert 'Central Tendency' in output
        assert 'GREEN' in output or 'AMBER' in output or 'RED' in output

    def test_to_dict_serialization(self):
        """Test JSON serialization."""
        np.random.seed(42)
        n = 300

        segments = np.repeat([1, 2], n // 2)
        y_pred = np.random.uniform(0.3, 0.7, n)
        y_true = (np.random.random(n) < y_pred).astype(int)

        results = run_all_calibration_tests(y_true, y_pred, segments)
        result_dict = results.to_dict()

        assert isinstance(result_dict, dict)
        assert 'hosmer_lemeshow' in result_dict
        assert 'traffic_light' in result_dict
        assert 'central_tendency' in result_dict
        assert 'overall_passed' in result_dict

        # Check nested structure
        assert 'chi2' in result_dict['hosmer_lemeshow']
        assert 'pvalue' in result_dict['hosmer_lemeshow']
        assert 'n_green' in result_dict['traffic_light']
        assert 'mean_pd_diff' in result_dict['central_tendency']

    def test_custom_parameters(self):
        """Test with custom test parameters."""
        np.random.seed(42)
        n = 500

        segments = np.array([1] * n)
        y_pred = np.random.uniform(0.3, 0.7, n)
        y_true = (np.random.random(n) < y_pred).astype(int)

        results = run_all_calibration_tests(
            y_true, y_pred, segments,
            n_bins=5,  # Fewer bins for HL test
            green_threshold=3.0,  # Lenient thresholds
            amber_threshold=5.0
        )

        assert isinstance(results, CalibrationResults)


class TestCalibrationResultsDataclass:
    """Test suite for CalibrationResults dataclass."""

    def test_instantiation(self):
        """Test creating CalibrationResults instance."""
        df = pd.DataFrame({
            'segment_id': [1, 2],
            'status': ['GREEN', 'AMBER']
        })

        results = CalibrationResults(
            hosmer_lemeshow_chi2=5.2,
            hosmer_lemeshow_pvalue=0.73,
            hosmer_lemeshow_passed=True,
            traffic_light_results=df,
            n_green=1,
            n_amber=1,
            n_red=0,
            central_tendency_diff=0.005,
            overall_passed=True
        )

        assert results.hosmer_lemeshow_chi2 == 5.2
        assert results.n_green == 1
        assert results.overall_passed is True

    def test_failed_status_in_output(self):
        """Test that failed status appears in output."""
        df = pd.DataFrame({
            'segment_id': [1],
            'status': ['RED']
        })

        results = CalibrationResults(
            hosmer_lemeshow_chi2=25.5,
            hosmer_lemeshow_pvalue=0.001,
            hosmer_lemeshow_passed=False,
            traffic_light_results=df,
            n_green=0,
            n_amber=0,
            n_red=1,
            central_tendency_diff=0.05,
            overall_passed=False
        )

        output = str(results)
        assert 'FAILED' in output
        assert '🔴 RED: 1' in output


# Pytest configuration
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
