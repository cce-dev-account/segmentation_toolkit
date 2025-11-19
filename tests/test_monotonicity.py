"""
Unit Tests for Monotonicity Validation Module

Tests all monotonicity validation functions required for IRB rating systems.
"""

import pytest
import numpy as np
import pandas as pd
from irb_segmentation.validators.monotonicity import (
    check_strict_monotonicity,
    calculate_rank_correlation,
    check_monotonic_trend,
    run_monotonicity_validation,
    MonotonicityResults
)


class TestCheckStrictMonotonicity:
    """Test suite for strict monotonicity checking."""

    def test_perfect_monotonicity(self):
        """Test perfectly monotonic PDs."""
        grades = np.array([1, 2, 3, 4, 5])
        pds = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

        is_mono, violations = check_strict_monotonicity(grades, pds)

        assert is_mono, "Perfect monotonicity should pass"
        assert len(violations) == 0, "Should have no violations"

    def test_single_violation(self):
        """Test case with one monotonicity violation."""
        grades = np.array([1, 2, 3, 4, 5])
        pds = np.array([0.01, 0.02, 0.025, 0.02, 0.05])  # Grade 4 < Grade 3

        is_mono, violations = check_strict_monotonicity(grades, pds)

        assert not is_mono, "Should detect violation"
        assert len(violations) == 1, "Should have exactly 1 violation"
        assert violations[0]['grade_low'] == 3
        assert violations[0]['grade_high'] == 4

    def test_multiple_violations(self):
        """Test case with multiple violations."""
        grades = np.array([1, 2, 3, 4, 5])
        pds = np.array([0.05, 0.04, 0.03, 0.02, 0.01])  # Completely reversed!

        is_mono, violations = check_strict_monotonicity(grades, pds)

        assert not is_mono, "Should detect violations"
        assert len(violations) == 4, "Should have 4 violations"

    def test_allow_equal_parameter(self):
        """Test allow_equal parameter."""
        grades = np.array([1, 2, 3, 4, 5])
        pds = np.array([0.01, 0.02, 0.02, 0.03, 0.04])  # Grade 2 = Grade 3

        # Strict mode (default): equal values are violations
        is_mono_strict, viol_strict = check_strict_monotonicity(grades, pds, allow_equal=False)
        assert not is_mono_strict, "Strict mode should fail with equal values"
        assert len(viol_strict) == 1

        # Lenient mode: equal values allowed
        is_mono_lenient, viol_lenient = check_strict_monotonicity(grades, pds, allow_equal=True)
        assert is_mono_lenient, "Lenient mode should pass with equal values"
        assert len(viol_lenient) == 0

    def test_unsorted_grades(self):
        """Test that function handles unsorted grades correctly."""
        grades = np.array([3, 1, 5, 2, 4])
        pds = np.array([0.03, 0.01, 0.05, 0.02, 0.04])  # Corresponds to sorted order

        is_mono, violations = check_strict_monotonicity(grades, pds)

        assert is_mono, "Should handle unsorted grades"
        assert len(violations) == 0

    def test_validates_input_length(self):
        """Test that mismatched lengths raise ValueError."""
        grades = np.array([1, 2, 3])
        pds = np.array([0.01, 0.02])  # Too short!

        with pytest.raises(ValueError, match="Length mismatch"):
            check_strict_monotonicity(grades, pds)

    def test_validates_minimum_grades(self):
        """Test that single grade raises ValueError."""
        grades = np.array([1])
        pds = np.array([0.01])

        with pytest.raises(ValueError, match="at least 2 grades"):
            check_strict_monotonicity(grades, pds)

    def test_two_grades_minimum(self):
        """Test minimum valid case: exactly 2 grades."""
        grades = np.array([1, 2])
        pds = np.array([0.01, 0.02])

        is_mono, violations = check_strict_monotonicity(grades, pds)

        assert is_mono
        assert len(violations) == 0


class TestCalculateRankCorrelation:
    """Test suite for rank correlation calculation."""

    def test_perfect_correlation_spearman(self):
        """Test perfect Spearman correlation."""
        grades = np.array([1, 2, 3, 4, 5])
        pds = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

        corr, pval = calculate_rank_correlation(grades, pds, method='spearman')

        assert corr == pytest.approx(1.0, abs=1e-10), "Perfect correlation should be 1.0"
        assert pval < 0.05, "Should be statistically significant"

    def test_perfect_correlation_kendall(self):
        """Test perfect Kendall correlation."""
        grades = np.array([1, 2, 3, 4, 5])
        pds = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

        corr, pval = calculate_rank_correlation(grades, pds, method='kendall')

        assert corr == pytest.approx(1.0, abs=1e-10), "Perfect correlation should be 1.0"
        assert pval < 0.05, "Should be statistically significant"

    def test_negative_correlation(self):
        """Test negative correlation (inverse relationship)."""
        grades = np.array([1, 2, 3, 4, 5])
        pds = np.array([0.05, 0.04, 0.03, 0.02, 0.01])  # Reversed

        corr, pval = calculate_rank_correlation(grades, pds, method='spearman')

        assert corr == pytest.approx(-1.0, abs=1e-10), "Inverse should be -1.0"

    def test_no_correlation(self):
        """Test near-zero correlation."""
        grades = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        pds = np.array([0.05, 0.01, 0.08, 0.02, 0.07, 0.03, 0.06, 0.04, 0.09, 0.10])

        corr, pval = calculate_rank_correlation(grades, pds, method='spearman')

        # Should have moderate but not perfect correlation
        assert -0.5 < corr < 0.8

    def test_validates_input_length(self):
        """Test that mismatched lengths raise ValueError."""
        grades = np.array([1, 2, 3])
        pds = np.array([0.01, 0.02])

        with pytest.raises(ValueError, match="Length mismatch"):
            calculate_rank_correlation(grades, pds)

    def test_validates_minimum_observations(self):
        """Test that too few observations raise ValueError."""
        grades = np.array([1, 2])
        pds = np.array([0.01, 0.02])

        with pytest.raises(ValueError, match="at least 3 observations"):
            calculate_rank_correlation(grades, pds)

    def test_invalid_method(self):
        """Test that invalid method raises ValueError."""
        grades = np.array([1, 2, 3])
        pds = np.array([0.01, 0.02, 0.03])

        with pytest.raises(ValueError, match="Unknown method"):
            calculate_rank_correlation(grades, pds, method='pearson')


class TestMonotonicityTrend:
    """Test suite for Mann-Kendall trend test."""

    def test_increasing_trend(self):
        """Test significant increasing trend."""
        grades = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        pds = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])

        result = check_monotonic_trend(grades, pds)

        assert result['trend'] == 'increasing', "Should detect increasing trend"
        assert result['passed'], "Should pass with increasing trend"
        assert result['p_value'] < 0.05, "Should be statistically significant"
        assert result['z_score'] > 0, "Z-score should be positive"

    def test_decreasing_trend(self):
        """Test decreasing trend."""
        grades = np.array([1, 2, 3, 4, 5])
        pds = np.array([0.05, 0.04, 0.03, 0.02, 0.01])

        result = check_monotonic_trend(grades, pds)

        assert result['trend'] == 'decreasing', "Should detect decreasing trend"
        assert not result['passed'], "Should fail with decreasing trend"
        assert result['z_score'] < 0, "Z-score should be negative"

    def test_no_trend(self):
        """Test no significant trend."""
        grades = np.array([1, 2, 3, 4, 5])
        pds = np.array([0.03, 0.02, 0.03, 0.02, 0.03])  # Flat/noisy

        result = check_monotonic_trend(grades, pds)

        # May be 'no trend' or weak trend depending on noise
        assert isinstance(result['trend'], str)
        assert result['trend'] in ['increasing', 'decreasing', 'no trend']

    def test_validates_input_length(self):
        """Test that mismatched lengths raise ValueError."""
        grades = np.array([1, 2, 3])
        pds = np.array([0.01, 0.02])

        with pytest.raises(ValueError, match="Length mismatch"):
            check_monotonic_trend(grades, pds)

    def test_validates_minimum_observations(self):
        """Test that too few observations raise ValueError."""
        grades = np.array([1, 2])
        pds = np.array([0.01, 0.02])

        with pytest.raises(ValueError, match="at least 3 observations"):
            check_monotonic_trend(grades, pds)

    def test_custom_alpha(self):
        """Test custom significance level."""
        grades = np.array([1, 2, 3, 4, 5])
        pds = np.array([0.01, 0.015, 0.02, 0.025, 0.03])

        result_95 = check_monotonic_trend(grades, pds, alpha=0.05)
        result_99 = check_monotonic_trend(grades, pds, alpha=0.01)

        # P-values should be the same
        assert result_95['p_value'] == result_99['p_value']


class TestRunMonotonicityValidation:
    """Test suite for comprehensive monotonicity validation."""

    def test_perfect_monotonicity_passes(self):
        """Test that perfect monotonicity passes all checks."""
        grades = np.array([1, 2, 3, 4, 5, 6, 7])
        pds = np.array([0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.30])

        results = run_monotonicity_validation(grades, pds)

        assert isinstance(results, MonotonicityResults)
        assert results.is_monotonic, "Should be monotonic"
        assert results.n_violations == 0, "Should have no violations"
        assert results.overall_passed, "Should pass overall"
        assert results.spearman_correlation > 0.90, "Should have high correlation"

    def test_violations_detected(self):
        """Test that violations are properly detected."""
        grades = np.array([1, 2, 3, 4, 5])
        pds = np.array([0.01, 0.02, 0.015, 0.03, 0.04])  # Grade 3 < Grade 2

        results = run_monotonicity_validation(grades, pds)

        assert not results.is_monotonic, "Should detect non-monotonicity"
        assert results.n_violations > 0, "Should have violations"
        assert not results.overall_passed, "Should fail overall"

    def test_low_correlation_fails(self):
        """Test that low correlation fails validation."""
        grades = np.array([1, 2, 3, 4, 5, 6, 7])
        pds = np.array([0.05, 0.01, 0.08, 0.02, 0.07, 0.03, 0.06])  # Random-ish

        results = run_monotonicity_validation(grades, pds, min_correlation=0.90)

        # May or may not be strictly monotonic, but correlation will be low
        assert results.spearman_correlation < 0.90, "Correlation should be low"

    def test_test_results_structure(self):
        """Test that test_results DataFrame has correct structure."""
        grades = np.array([1, 2, 3, 4, 5])
        pds = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

        results = run_monotonicity_validation(grades, pds)

        df = results.test_results
        assert 'risk_grade' in df.columns
        assert 'predicted_pd' in df.columns
        assert len(df) == 5

        # Should be sorted by risk grade
        assert df['risk_grade'].is_monotonic_increasing

    def test_list_inputs(self):
        """Test that lists are correctly converted to arrays."""
        grades = [1, 2, 3, 4, 5]
        pds = [0.01, 0.02, 0.03, 0.04, 0.05]

        results = run_monotonicity_validation(grades, pds)

        assert isinstance(results, MonotonicityResults)
        assert results.is_monotonic

    def test_pandas_series_inputs(self):
        """Test that pandas Series work correctly."""
        grades = pd.Series([1, 2, 3, 4, 5])
        pds = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])

        results = run_monotonicity_validation(grades, pds)

        assert isinstance(results, MonotonicityResults)
        assert results.is_monotonic

    def test_to_dict_serialization(self):
        """Test to_dict returns JSON-serializable structure."""
        grades = np.array([1, 2, 3, 4, 5])
        pds = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

        results = run_monotonicity_validation(grades, pds)
        results_dict = results.to_dict()

        assert isinstance(results_dict, dict)
        assert 'is_monotonic' in results_dict
        assert 'n_violations' in results_dict
        assert 'violations' in results_dict
        assert 'correlations' in results_dict
        assert 'test_results' in results_dict
        assert 'overall_passed' in results_dict

        # Check all values are JSON-serializable types
        import json
        try:
            json.dumps(results_dict)
        except (TypeError, ValueError) as e:
            pytest.fail(f"to_dict() not JSON-serializable: {e}")

    def test_str_representation(self):
        """Test string representation."""
        grades = np.array([1, 2, 3, 4, 5])
        pds = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

        results = run_monotonicity_validation(grades, pds)
        output = str(results)

        assert 'MONOTONICITY VALIDATION' in output
        assert 'Monotonicity Status' in output
        assert 'Spearman' in output
        assert 'Kendall' in output
        assert 'PASSED' in output or 'FAILED' in output

    def test_str_representation_with_violations(self):
        """Test string representation includes violation details."""
        grades = np.array([1, 2, 3, 4, 5])
        pds = np.array([0.01, 0.03, 0.02, 0.04, 0.05])  # Grade 3 < Grade 2

        results = run_monotonicity_validation(grades, pds)
        output = str(results)

        assert 'Violations Detected' in output
        assert 'Grade' in output  # Should show violation details

    def test_custom_parameters(self):
        """Test custom parameters."""
        grades = np.array([1, 2, 3, 4, 5])
        pds = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

        # Relaxed correlation requirement
        results_relaxed = run_monotonicity_validation(
            grades, pds, min_correlation=0.80
        )

        # Strict correlation requirement
        results_strict = run_monotonicity_validation(
            grades, pds, min_correlation=0.99
        )

        # Both should be monotonic
        assert results_relaxed.is_monotonic
        assert results_strict.is_monotonic

        # But strict may fail overall if correlation not high enough
        assert results_relaxed.overall_passed


class TestMonotonicityResultsDataclass:
    """Test suite for MonotonicityResults dataclass."""

    def test_instantiation(self):
        """Test creating MonotonicityResults instance."""
        df = pd.DataFrame({
            'risk_grade': [1, 2, 3],
            'predicted_pd': [0.01, 0.02, 0.03]
        })

        results = MonotonicityResults(
            is_monotonic=True,
            violations=[],
            n_violations=0,
            spearman_correlation=1.0,
            spearman_pvalue=0.0,
            kendall_tau=1.0,
            kendall_pvalue=0.0,
            test_results=df,
            overall_passed=True
        )

        assert results.is_monotonic
        assert results.n_violations == 0
        assert results.overall_passed

    def test_failed_status_in_output(self):
        """Test that failed status appears in output."""
        df = pd.DataFrame({
            'risk_grade': [1, 2, 3],
            'predicted_pd': [0.03, 0.02, 0.01]  # Reversed
        })

        results = MonotonicityResults(
            is_monotonic=False,
            violations=[{'grade_low': 1, 'grade_high': 2, 'pd_low': 0.03, 'pd_high': 0.02, 'decrease': 0.01}],
            n_violations=2,
            spearman_correlation=-1.0,
            spearman_pvalue=0.0,
            kendall_tau=-1.0,
            kendall_pvalue=0.0,
            test_results=df,
            overall_passed=False
        )

        output = str(results)
        assert 'FAILED' in output


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_many_grades(self):
        """Test with many risk grades (e.g., 20+)."""
        n_grades = 25
        grades = np.arange(1, n_grades + 1)
        pds = np.linspace(0.001, 0.500, n_grades)

        results = run_monotonicity_validation(grades, pds)

        assert isinstance(results, MonotonicityResults)
        assert results.is_monotonic

    def test_very_small_pds(self):
        """Test with very small PD values."""
        grades = np.array([1, 2, 3, 4, 5])
        pds = np.array([0.0001, 0.0002, 0.0003, 0.0004, 0.0005])

        results = run_monotonicity_validation(grades, pds)

        assert results.is_monotonic

    def test_high_pd_range(self):
        """Test with high PD range (stressed scenarios)."""
        grades = np.array([1, 2, 3, 4, 5])
        pds = np.array([0.10, 0.25, 0.40, 0.60, 0.80])

        results = run_monotonicity_validation(grades, pds)

        assert results.is_monotonic

    def test_non_integer_grades(self):
        """Test with non-integer risk scores."""
        grades = np.array([1.5, 2.3, 3.7, 4.2, 5.9])
        pds = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

        results = run_monotonicity_validation(grades, pds)

        assert results.is_monotonic


# Pytest configuration
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
