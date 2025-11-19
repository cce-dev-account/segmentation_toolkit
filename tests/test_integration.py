"""
Integration Tests for IRB Validation Toolkit

Tests end-to-end workflows combining multiple validation modules:
- Performance metrics + Calibration + Back-testing
- Monotonicity + Performance validation
- Complete IRB model validation pipeline

These tests ensure all modules work together correctly in realistic scenarios.
"""

import pytest
import numpy as np
import pandas as pd
from irb_segmentation.validators import (
    # Performance metrics
    calculate_all_metrics,
    # Calibration
    run_all_calibration_tests,
    # Back-testing
    run_backtest,
    # Monotonicity
    run_monotonicity_validation
)


class TestCompleteIRBValidation:
    """Test complete IRB PD model validation workflow."""

    @pytest.fixture
    def realistic_pd_model_data(self):
        """Generate realistic PD model data with 7 risk grades."""
        np.random.seed(42)

        # 7 risk grades with increasing PDs
        risk_grades = np.array([1, 2, 3, 4, 5, 6, 7])
        true_pds = np.array([0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.30])

        # Generate observations per grade (varying sizes)
        n_per_grade = [500, 400, 300, 250, 200, 100, 50]

        # Build dataset
        all_grades = []
        all_predictions = []
        all_actuals = []

        for grade, true_pd, n in zip(risk_grades, true_pds, n_per_grade):
            # Generate predictions with some noise around true PD
            predictions = np.random.normal(true_pd, true_pd * 0.1, n)
            predictions = np.clip(predictions, 0.001, 0.999)

            # Generate actual defaults based on predictions
            actuals = (np.random.random(n) < predictions).astype(int)

            all_grades.extend([grade] * n)
            all_predictions.extend(predictions)
            all_actuals.extend(actuals)

        return {
            'risk_grades': np.array(all_grades),
            'y_pred': np.array(all_predictions),
            'y_true': np.array(all_actuals),
            'grade_pds': true_pds
        }

    def test_full_validation_workflow(self, realistic_pd_model_data):
        """Test complete validation: performance + calibration + monotonicity."""
        data = realistic_pd_model_data

        # 1. Performance Metrics
        perf_metrics = calculate_all_metrics(data['y_true'], data['y_pred'])

        assert perf_metrics.gini > 0.30, "Should have acceptable discriminatory power"
        assert perf_metrics.ks_statistic > 0.20, "Should have acceptable KS"
        assert perf_metrics.passed_thresholds, "Should pass performance thresholds"

        # 2. Calibration Tests
        calib_results = run_all_calibration_tests(
            data['y_true'],
            data['y_pred'],
            segment_id=data['risk_grades']
        )

        assert calib_results.hosmer_lemeshow_passed, "Should pass HL test"
        assert calib_results.n_red == 0, "Should have no RED segments"

        # 3. Monotonicity Validation
        # Calculate mean PD per grade
        grade_mean_pds = []
        for grade in np.unique(data['risk_grades']):
            mask = data['risk_grades'] == grade
            grade_mean_pds.append(data['y_pred'][mask].mean())

        mono_results = run_monotonicity_validation(
            np.unique(data['risk_grades']),
            np.array(grade_mean_pds)
        )

        assert mono_results.is_monotonic, "PDs should be monotonic with risk"
        assert mono_results.spearman_correlation > 0.90, "Should have high correlation"
        assert mono_results.overall_passed, "Should pass monotonicity validation"

        # Overall assessment
        overall_passed = (
            perf_metrics.passed_thresholds and
            calib_results.overall_passed and
            mono_results.overall_passed
        )

        assert overall_passed, "Complete IRB validation should pass"

    def test_performance_and_calibration_consistency(self, realistic_pd_model_data):
        """Test that performance metrics and calibration are consistent."""
        data = realistic_pd_model_data

        # Calculate both
        perf_metrics = calculate_all_metrics(data['y_true'], data['y_pred'])
        calib_results = run_all_calibration_tests(data['y_true'], data['y_pred'])

        # Good performance should correlate with good calibration
        if perf_metrics.gini > 0.50:  # Strong discriminatory power
            # Should also have good calibration (low Brier score)
            assert perf_metrics.brier_score < 0.20, \
                "Strong discrimination should have reasonable calibration"

        # If Hosmer-Lemeshow passes, central tendency should be OK
        if calib_results.hosmer_lemeshow_passed:
            assert abs(calib_results.central_tendency_diff) < 0.05, \
                "Good HL should mean reasonable central tendency"


class TestBacktestingIntegration:
    """Test back-testing integration with other modules."""

    @pytest.fixture
    def time_series_data(self):
        """Generate time-series data for back-testing (3 years)."""
        np.random.seed(42)

        periods = []
        y_true_all = []
        y_pred_all = []

        # 3 years of data
        for year in [2020, 2021, 2022]:
            n_obs = 1000

            # Base PD around 2% with some yearly variation
            base_pd = 0.02 if year == 2020 else (0.025 if year == 2021 else 0.018)

            y_pred = np.random.normal(base_pd, 0.01, n_obs)
            y_pred = np.clip(y_pred, 0.001, 0.999)

            y_true = (np.random.random(n_obs) < y_pred).astype(int)

            periods.extend([year] * n_obs)
            y_pred_all.extend(y_pred)
            y_true_all.extend(y_true)

        return {
            'periods': np.array(periods),
            'y_pred': np.array(y_pred_all),
            'y_true': np.array(y_true_all)
        }

    def test_backtest_with_performance_metrics(self, time_series_data):
        """Test that back-testing results align with performance metrics."""
        data = time_series_data

        # Run back-test
        backtest_results = run_backtest(
            data['periods'],
            data['y_true'],
            data['y_pred']
        )

        # Run performance metrics on full dataset
        perf_metrics = calculate_all_metrics(data['y_true'], data['y_pred'])

        # If back-test passes, performance should be good
        if backtest_results.overall_passed:
            assert perf_metrics.gini > 0, "Passing back-test should have positive Gini"
            assert perf_metrics.brier_score < 0.25, \
                "Passing back-test should have reasonable Brier score"

        # Check consistency of PD predictions
        mean_predicted = backtest_results.mean_predicted_pd
        mean_actual = backtest_results.mean_actual_pd

        # Should be reasonably close for well-calibrated model
        assert abs(mean_predicted - mean_actual) < 0.02, \
            "Aggregate PD should be close to actual"

    def test_period_level_calibration(self, time_series_data):
        """Test that each period passes calibration individually."""
        data = time_series_data

        # Run back-test to get period-level results
        backtest_results = run_backtest(
            data['periods'],
            data['y_true'],
            data['y_pred']
        )

        # For each period, run calibration test
        for period in np.unique(data['periods']):
            mask = data['periods'] == period

            y_true_period = data['y_true'][mask]
            y_pred_period = data['y_pred'][mask]

            if len(y_true_period) >= 100:  # Only if enough data
                calib = run_all_calibration_tests(y_true_period, y_pred_period)

                # Period should pass calibration if back-test passed
                if backtest_results.binomial_test_passed:
                    # At least central tendency should be OK
                    assert abs(calib.central_tendency_diff) < 0.05, \
                        f"Period {period} should have reasonable calibration"


class TestMonotonicityIntegration:
    """Test monotonicity validation with other modules."""

    @pytest.fixture
    def graded_portfolio_data(self):
        """Generate portfolio data with risk grades."""
        np.random.seed(42)

        # 5 risk grades
        grades = [1, 2, 3, 4, 5]
        true_pds = [0.01, 0.03, 0.07, 0.15, 0.30]

        all_data = []

        for grade, true_pd in zip(grades, true_pds):
            n = 200
            y_pred = np.random.normal(true_pd, true_pd * 0.15, n)
            y_pred = np.clip(y_pred, 0.001, 0.999)
            y_true = (np.random.random(n) < y_pred).astype(int)

            for i in range(n):
                all_data.append({
                    'grade': grade,
                    'y_pred': y_pred[i],
                    'y_true': y_true[i]
                })

        df = pd.DataFrame(all_data)
        return df

    def test_monotonicity_with_performance_by_grade(self, graded_portfolio_data):
        """Test that monotonicity aligns with grade-level performance."""
        df = graded_portfolio_data

        # Calculate mean PD per grade
        grade_stats = df.groupby('grade').agg({
            'y_pred': 'mean',
            'y_true': ['sum', 'count']
        }).reset_index()

        grade_stats.columns = ['grade', 'mean_pd', 'n_defaults', 'n_total']

        # Run monotonicity check
        mono_results = run_monotonicity_validation(
            grade_stats['grade'].values,
            grade_stats['mean_pd'].values
        )

        assert mono_results.is_monotonic, "Grades should be monotonic"

        # Calculate performance for each grade
        for grade in df['grade'].unique():
            grade_data = df[df['grade'] == grade]

            if len(grade_data) >= 50:  # Enough for metrics
                # Check if both classes present
                if len(np.unique(grade_data['y_true'])) == 2:
                    metrics = calculate_all_metrics(
                        grade_data['y_true'].values,
                        grade_data['y_pred'].values
                    )

                    # Each grade should have some discriminatory power
                    # (even within a grade, there's variation)
                    assert metrics.gini >= -0.5, \
                        f"Grade {grade} should not have extremely poor performance"
                else:
                    # Single class in this grade - that's OK for low-risk grades
                    # Just verify it's a low-risk grade if all non-defaults
                    if grade_data['y_true'].sum() == 0:
                        assert grade <= 2, \
                            f"Only low-risk grades should have zero defaults"

    def test_monotonicity_violations_detected(self):
        """Test that monotonicity violations are properly detected."""
        # Create data with deliberate violation
        grades = np.array([1, 2, 3, 4, 5])
        pds = np.array([0.01, 0.03, 0.02, 0.05, 0.08])  # Grade 3 < Grade 2!

        mono_results = run_monotonicity_validation(grades, pds)

        assert not mono_results.is_monotonic, "Should detect violation"
        assert mono_results.n_violations > 0, "Should report violations"
        assert not mono_results.overall_passed, "Should fail overall"

        # Violation should be in results
        violations = mono_results.violations
        assert len(violations) > 0

        # Check violation details
        found_violation = False
        for v in violations:
            if v['grade_low'] == 2 and v['grade_high'] == 3:
                found_violation = True
                assert v['pd_low'] > v['pd_high'], "Should show decrease"

        assert found_violation, "Should find specific violation between grade 2 and 3"


class TestCrossModuleValidation:
    """Test scenarios involving multiple modules simultaneously."""

    def test_well_calibrated_monotonic_model(self):
        """Test a model that performs well across all dimensions."""
        np.random.seed(42)

        # Generate excellent model data
        n = 2000
        risk_score = np.random.uniform(0, 10, n)  # Risk score 0-10

        # Perfect monotonic relationship
        true_pd = 1 / (1 + np.exp(-0.5 * (risk_score - 5)))  # Logistic

        # Add small noise
        y_pred = true_pd + np.random.normal(0, 0.02, n)
        y_pred = np.clip(y_pred, 0.001, 0.999)

        # Generate actuals
        y_true = (np.random.random(n) < y_pred).astype(int)

        # Create risk grades (binned risk score)
        grades = pd.cut(risk_score, bins=7, labels=False) + 1

        # 1. Performance
        perf = calculate_all_metrics(y_true, y_pred)
        assert perf.gini > 0.50, "Excellent model should have high Gini"
        assert perf.passed_thresholds, "Should pass performance thresholds"

        # 2. Calibration
        calib = run_all_calibration_tests(y_true, y_pred, segment_id=grades)
        assert calib.hosmer_lemeshow_passed, "Should pass HL test"
        assert calib.n_red == 0, "Should have no RED segments"

        # 3. Monotonicity
        grade_pds = []
        for g in np.unique(grades):
            grade_pds.append(y_pred[grades == g].mean())

        mono = run_monotonicity_validation(np.unique(grades), np.array(grade_pds))
        assert mono.is_monotonic, "Should be monotonic"
        assert mono.spearman_correlation > 0.95, "Should have very high correlation"

    def test_poor_model_fails_multiple_tests(self):
        """Test that a poor model fails appropriate validations."""
        np.random.seed(42)

        # Generate poor model data (random predictions)
        n = 1000
        y_true = np.random.binomial(1, 0.05, n)  # 5% default rate
        y_pred = np.random.uniform(0.01, 0.99, n)  # Random predictions!

        grades = np.repeat([1, 2, 3, 4, 5], n // 5)

        # 1. Performance should fail
        perf = calculate_all_metrics(y_true, y_pred)
        assert perf.gini < 0.30, "Random model should have poor Gini"
        assert not perf.passed_thresholds, "Should fail performance thresholds"

        # 2. Calibration might fail
        calib = run_all_calibration_tests(y_true, y_pred, segment_id=grades)
        # Random model likely won't pass HL test
        # (though it's possible by chance)

        # 3. Monotonicity should fail
        grade_pds = []
        for g in np.unique(grades):
            grade_pds.append(y_pred[grades == g].mean())

        mono = run_monotonicity_validation(np.unique(grades), np.array(grade_pds))
        # Random assignments won't be monotonic
        assert not mono.overall_passed or mono.spearman_correlation < 0.50, \
            "Random model should have poor monotonicity"


class TestRealisticIRBScenarios:
    """Test realistic IRB validation scenarios."""

    def test_retail_mortgage_portfolio(self):
        """Simulate validation of retail mortgage PD model."""
        np.random.seed(42)

        # 10 risk grades typical for retail
        n_grades = 10
        n_per_grade = 500

        # Low PDs for mortgages (0.1% to 5%)
        base_pds = np.linspace(0.001, 0.05, n_grades)

        all_data = {
            'grades': [],
            'y_pred': [],
            'y_true': [],
            'period': []
        }

        # 3 years of data
        for year in [2020, 2021, 2022]:
            for grade, base_pd in enumerate(base_pds, 1):
                # Add economic cycle effect
                cycle_factor = 1.0 if year == 2020 else (1.2 if year == 2021 else 0.9)
                adjusted_pd = base_pd * cycle_factor

                y_pred = np.random.normal(adjusted_pd, adjusted_pd * 0.2, n_per_grade)
                y_pred = np.clip(y_pred, 0.0001, 0.999)
                y_true = (np.random.random(n_per_grade) < y_pred).astype(int)

                all_data['grades'].extend([grade] * n_per_grade)
                all_data['y_pred'].extend(y_pred)
                all_data['y_true'].extend(y_true)
                all_data['period'].extend([year] * n_per_grade)

        grades = np.array(all_data['grades'])
        y_pred = np.array(all_data['y_pred'])
        y_true = np.array(all_data['y_true'])
        periods = np.array(all_data['period'])

        # Run full validation
        perf = calculate_all_metrics(y_true, y_pred)
        calib = run_all_calibration_tests(y_true, y_pred, segment_id=grades)
        backtest = run_backtest(periods, y_true, y_pred)

        # Calculate grade-level PDs
        grade_pds = []
        for g in range(1, n_grades + 1):
            grade_pds.append(y_pred[grades == g].mean())

        mono = run_monotonicity_validation(np.arange(1, n_grades + 1), np.array(grade_pds))

        # Mortgage model should pass all tests
        assert perf.passed_thresholds, "Retail mortgage model should have good performance"
        assert calib.overall_passed, "Should be well-calibrated"
        assert mono.is_monotonic, "Should be monotonic"
        # Backtest might have some variation due to economic cycle
        assert backtest.binomial_test_passed or backtest.chi_squared_passed, \
            "Should pass at least one back-test"

    def test_corporate_default_model(self):
        """Simulate validation of corporate default PD model."""
        np.random.seed(42)

        # 7 risk grades for corporate (fewer, wider spread)
        n_grades = 7
        # Varying sizes (more in good grades, fewer in bad)
        n_per_grade = [300, 250, 200, 150, 100, 50, 30]

        # Corporate PDs range from 0.5% to 40%
        base_pds = np.array([0.005, 0.01, 0.025, 0.06, 0.12, 0.25, 0.40])

        all_data = {'grades': [], 'y_pred': [], 'y_true': []}

        for grade, base_pd, n in zip(range(1, n_grades + 1), base_pds, n_per_grade):
            y_pred = np.random.normal(base_pd, base_pd * 0.25, n)
            y_pred = np.clip(y_pred, 0.001, 0.999)
            y_true = (np.random.random(n) < y_pred).astype(int)

            all_data['grades'].extend([grade] * n)
            all_data['y_pred'].extend(y_pred)
            all_data['y_true'].extend(y_true)

        grades = np.array(all_data['grades'])
        y_pred = np.array(all_data['y_pred'])
        y_true = np.array(all_data['y_true'])

        # Run validations
        perf = calculate_all_metrics(y_true, y_pred)
        calib = run_all_calibration_tests(y_true, y_pred, segment_id=grades)

        grade_pds = []
        for g in range(1, n_grades + 1):
            grade_pds.append(y_pred[grades == g].mean())

        mono = run_monotonicity_validation(np.arange(1, n_grades + 1), np.array(grade_pds))

        # Corporate model validations
        assert perf.gini > 0.30, "Corporate model should have good discrimination"
        assert mono.is_monotonic, "Corporate grades should be monotonic"
        assert mono.spearman_correlation > 0.90, "Should have high rank correlation"


# Pytest configuration
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
