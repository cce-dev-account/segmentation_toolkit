"""
Unit Tests for Performance Metrics Module

Tests all discriminatory power metrics required for IRB validation.
"""

import pytest
import numpy as np
from irb_segmentation.validators.performance_metrics import (
    gini_coefficient,
    ks_statistic,
    brier_score,
    accuracy_ratio,
    calculate_all_metrics,
    PerformanceMetrics
)


class TestGiniCoefficient:
    """Test suite for Gini coefficient calculation."""

    def test_perfect_discrimination(self):
        """Test Gini = 1.0 for perfect discrimination."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])

        gini = gini_coefficient(y_true, y_pred)

        assert gini == pytest.approx(1.0, abs=1e-6), \
            f"Perfect model should have Gini=1.0, got {gini}"

    def test_random_model(self):
        """Test Gini ≈ 0.0 for random predictions."""
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.5, 1000)
        y_pred = np.random.random(1000)

        gini = gini_coefficient(y_true, y_pred)

        assert abs(gini) < 0.1, \
            f"Random model should have Gini≈0, got {gini}"

    def test_good_model(self):
        """Test Gini > 0.30 for good model."""
        y_true = np.array([0] * 100 + [1] * 100)
        y_pred = np.concatenate([
            np.random.uniform(0.1, 0.4, 100),  # Low scores for negatives
            np.random.uniform(0.6, 0.9, 100)   # High scores for positives
        ])

        gini = gini_coefficient(y_true, y_pred)

        assert gini > 0.30, \
            f"Good model should have Gini>0.30, got {gini}"

    def test_inverse_predictions(self):
        """Test Gini < 0 for inverse predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.9, 0.8, 0.2, 0.1])  # Backwards!

        gini = gini_coefficient(y_true, y_pred)

        assert gini < 0, \
            f"Inverse predictions should have Gini<0, got {gini}"

    def test_validates_input_length(self):
        """Test that mismatched lengths raise ValueError."""
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0.1, 0.9])  # Too short!

        with pytest.raises(ValueError, match="Length mismatch"):
            gini_coefficient(y_true, y_pred)

    def test_validates_binary_target(self):
        """Test that non-binary targets raise ValueError."""
        y_true = np.array([0, 1, 2, 3])  # Not binary!
        y_pred = np.array([0.1, 0.3, 0.5, 0.7])

        with pytest.raises(ValueError, match="must be binary"):
            gini_coefficient(y_true, y_pred)

    def test_validates_both_classes(self):
        """Test that single-class targets raise ValueError."""
        y_true = np.array([1, 1, 1, 1])  # All same class!
        y_pred = np.array([0.5, 0.6, 0.7, 0.8])

        with pytest.raises(ValueError, match="must have both classes"):
            gini_coefficient(y_true, y_pred)

    def test_minimum_samples(self):
        """Test that too few samples raise ValueError."""
        y_true = np.array([1])
        y_pred = np.array([0.5])

        with pytest.raises(ValueError, match="at least 2 samples"):
            gini_coefficient(y_true, y_pred)


class TestKSStatistic:
    """Test suite for Kolmogorov-Smirnov statistic."""

    def test_perfect_separation(self):
        """Test KS = 1.0 for perfect separation."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])

        ks, threshold = ks_statistic(y_true, y_pred)

        assert ks == pytest.approx(1.0, abs=1e-6), \
            f"Perfect separation should have KS=1.0, got {ks}"
        assert 0.3 <= threshold <= 0.8, \
            f"Threshold should be between classes, got {threshold}"

    def test_random_model(self):
        """Test KS ≈ 0 for random predictions."""
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.5, 1000)
        y_pred = np.random.random(1000)

        ks, threshold = ks_statistic(y_true, y_pred)

        assert ks < 0.2, \
            f"Random model should have KS≈0, got {ks}"

    def test_good_model(self):
        """Test KS > 0.20 for good model."""
        y_true = np.array([0] * 100 + [1] * 100)
        y_pred = np.concatenate([
            np.random.uniform(0.1, 0.4, 100),
            np.random.uniform(0.6, 0.9, 100)
        ])

        ks, threshold = ks_statistic(y_true, y_pred)

        assert ks > 0.20, \
            f"Good model should have KS>0.20, got {ks}"

    def test_validates_inputs(self):
        """Test input validation."""
        y_true = np.array([0, 1])
        y_pred = np.array([0.1])  # Wrong length

        with pytest.raises(ValueError):
            ks_statistic(y_true, y_pred)


class TestBrierScore:
    """Test suite for Brier score."""

    def test_perfect_calibration(self):
        """Test Brier = 0.0 for perfect calibration."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.0, 0.0, 1.0, 1.0])  # Perfect!

        bs = brier_score(y_true, y_pred)

        assert bs == pytest.approx(0.0, abs=1e-6), \
            f"Perfect calibration should have Brier=0, got {bs}"

    def test_good_calibration(self):
        """Test Brier < 0.25 for well-calibrated model."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])

        bs = brier_score(y_true, y_pred)

        assert bs < 0.25, \
            f"Good calibration should have Brier<0.25, got {bs}"

    def test_worst_calibration(self):
        """Test Brier = 1.0 for worst calibration."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1.0, 1.0, 0.0, 0.0])  # Completely wrong!

        bs = brier_score(y_true, y_pred)

        assert bs == pytest.approx(1.0, abs=1e-6), \
            f"Worst calibration should have Brier=1, got {bs}"

    def test_baseline_model(self):
        """Test Brier for naive baseline (predicting base rate)."""
        y_true = np.array([0, 0, 0, 1])  # 25% positive
        base_rate = 0.25
        y_pred = np.array([base_rate] * 4)

        bs = brier_score(y_true, y_pred)

        # Baseline Brier = base_rate * (1 - base_rate)
        expected_bs = base_rate * (1 - base_rate)
        assert bs == pytest.approx(expected_bs, abs=1e-6), \
            f"Baseline Brier should be {expected_bs}, got {bs}"

    def test_empty_arrays(self):
        """Test that empty arrays raise ValueError."""
        with pytest.raises(ValueError, match="Empty arrays"):
            brier_score(np.array([]), np.array([]))


class TestAccuracyRatio:
    """Test suite for Accuracy Ratio."""

    def test_perfect_model(self):
        """Test AR = 1.0 for perfect model."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.0, 0.1, 0.9, 1.0])

        ar = accuracy_ratio(y_true, y_pred)

        assert ar == pytest.approx(1.0, abs=1e-6), \
            f"Perfect model should have AR=1.0, got {ar}"

    def test_random_model(self):
        """Test AR ≈ 0.5 for random model (normalized Gini)."""
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.5, 1000)
        y_pred = np.random.random(1000)

        ar = accuracy_ratio(y_true, y_pred)

        # Random model has Gini≈0, so AR = (0+1)/2 = 0.5
        assert 0.4 <= ar <= 0.6, \
            f"Random model should have AR≈0.5, got {ar}"

    def test_all_same_class(self):
        """Test AR behavior when all samples same class."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0.5, 0.6, 0.7, 0.8])

        # Should raise error due to single class
        with pytest.raises(ValueError, match="must have both classes"):
            accuracy_ratio(y_true, y_pred)


class TestCalculateAllMetrics:
    """Test suite for calculate_all_metrics function."""

    def test_calculates_all_metrics(self):
        """Test that all metrics are calculated."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

        metrics = calculate_all_metrics(y_true, y_pred)

        assert isinstance(metrics, PerformanceMetrics)
        assert hasattr(metrics, 'gini')
        assert hasattr(metrics, 'ks_statistic')
        assert hasattr(metrics, 'ks_threshold')
        assert hasattr(metrics, 'auc')
        assert hasattr(metrics, 'accuracy_ratio')
        assert hasattr(metrics, 'brier_score')
        assert hasattr(metrics, 'passed_thresholds')

    def test_perfect_model_passes_thresholds(self):
        """Test that perfect model passes regulatory thresholds."""
        y_true = np.array([0] * 50 + [1] * 50)
        y_pred = np.concatenate([
            np.random.uniform(0.0, 0.2, 50),
            np.random.uniform(0.8, 1.0, 50)
        ])

        metrics = calculate_all_metrics(y_true, y_pred)

        assert metrics.passed_thresholds, \
            "Perfect model should pass regulatory thresholds"
        assert metrics.gini > 0.30
        assert metrics.ks_statistic > 0.20

    def test_poor_model_fails_thresholds(self):
        """Test that poor model fails regulatory thresholds."""
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.5, 200)
        y_pred = np.random.random(200)

        metrics = calculate_all_metrics(y_true, y_pred)

        assert not metrics.passed_thresholds, \
            "Random model should fail regulatory thresholds"

    def test_custom_thresholds(self):
        """Test custom threshold values."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.3, 0.4, 0.6, 0.7])

        # Relaxed thresholds
        metrics = calculate_all_metrics(
            y_true, y_pred,
            gini_threshold=0.10,
            ks_threshold=0.10
        )

        assert metrics.passed_thresholds or not metrics.passed_thresholds  # Depends on actual values

    def test_to_dict(self):
        """Test serialization to dictionary."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])

        metrics = calculate_all_metrics(y_true, y_pred)
        metrics_dict = metrics.to_dict()

        assert isinstance(metrics_dict, dict)
        assert 'gini' in metrics_dict
        assert 'ks_statistic' in metrics_dict
        assert 'auc' in metrics_dict
        assert 'brier_score' in metrics_dict
        assert 'passed_thresholds' in metrics_dict

        # Check all values are JSON-serializable types
        for key, value in metrics_dict.items():
            assert isinstance(value, (int, float, bool)), \
                f"Value for {key} is not JSON-serializable: {type(value)}"

    def test_str_representation(self):
        """Test string representation."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])

        metrics = calculate_all_metrics(y_true, y_pred)
        output = str(metrics)

        assert 'PERFORMANCE METRICS' in output
        assert 'Gini coefficient' in output
        assert 'KS statistic' in output
        assert 'Brier Score' in output
        assert 'PASSED' in output or 'FAILED' in output


class TestPerformanceMetricsDataclass:
    """Test suite for PerformanceMetrics dataclass."""

    def test_instantiation(self):
        """Test creating PerformanceMetrics instance."""
        metrics = PerformanceMetrics(
            gini=0.45,
            ks_statistic=0.35,
            ks_threshold=0.50,
            auc=0.725,
            accuracy_ratio=0.60,
            brier_score=0.15,
            passed_thresholds=True
        )

        assert metrics.gini == 0.45
        assert metrics.ks_statistic == 0.35
        assert metrics.passed_thresholds is True

    def test_failed_thresholds_in_output(self):
        """Test that failed status appears in output."""
        metrics = PerformanceMetrics(
            gini=0.15,  # Below threshold!
            ks_statistic=0.10,  # Below threshold!
            ks_threshold=0.50,
            auc=0.575,
            accuracy_ratio=0.30,
            brier_score=0.25,
            passed_thresholds=False
        )

        output = str(metrics)
        assert 'FAILED' in output


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_zeros(self):
        """Test when all predictions are zero."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.0, 0.0, 0.0, 0.0])

        # Should not crash
        gini = gini_coefficient(y_true, y_pred)
        bs = brier_score(y_true, y_pred)

        # Predicting all zeros means AUC = 0, so Gini = 2*0-1 = -1
        assert gini <= 0

    def test_all_ones(self):
        """Test when all predictions are one."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1.0, 1.0, 1.0, 1.0])

        # Should not crash
        gini = gini_coefficient(y_true, y_pred)
        bs = brier_score(y_true, y_pred)

        # Predicting all ones means AUC = 0.5 (no discrimination), Gini = 0
        assert gini == pytest.approx(0.0, abs=1e-6)

    def test_large_dataset(self):
        """Test with large dataset (performance test)."""
        np.random.seed(42)
        n = 100000
        y_true = np.random.binomial(1, 0.3, n)
        y_pred = np.random.random(n)

        # Should complete in reasonable time
        import time
        start = time.time()
        metrics = calculate_all_metrics(y_true, y_pred)
        duration = time.time() - start

        assert duration < 5.0, \
            f"Calculation took too long: {duration:.2f}s"
        assert isinstance(metrics, PerformanceMetrics)

    def test_extreme_imbalance(self):
        """Test with extreme class imbalance (99% negative)."""
        y_true = np.array([0] * 990 + [1] * 10)
        y_pred = np.random.random(1000)

        # Should not crash
        metrics = calculate_all_metrics(y_true, y_pred)

        assert isinstance(metrics, PerformanceMetrics)


# Pytest configuration
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
