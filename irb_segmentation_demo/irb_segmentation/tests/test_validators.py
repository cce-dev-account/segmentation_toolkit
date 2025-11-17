"""
Unit tests for SegmentValidator
"""

import pytest
import numpy as np
from irb_segmentation.validators import SegmentValidator
from irb_segmentation.params import IRBSegmentationParams


class TestSegmentValidator:
    """Test suite for SegmentValidator class"""

    @pytest.fixture
    def sample_data(self):
        """Create sample segmentation data for testing"""
        np.random.seed(42)
        n = 1000

        # Create 3 segments with different default rates
        segments = np.random.choice([0, 1, 2], size=n, p=[0.3, 0.4, 0.3])
        y = np.zeros(n)

        # Assign different default rates to each segment
        y[segments == 0] = np.random.binomial(1, 0.05, np.sum(segments == 0))
        y[segments == 1] = np.random.binomial(1, 0.10, np.sum(segments == 1))
        y[segments == 2] = np.random.binomial(1, 0.20, np.sum(segments == 2))

        return segments, y

    def test_validate_minimum_defaults_pass(self, sample_data):
        """Test validation passes when all segments have sufficient defaults"""
        segments, y = sample_data

        result = SegmentValidator.validate_minimum_defaults(
            segments, y, min_defaults=5
        )

        assert result['passed'] is True
        assert len(result['failed_segments']) == 0
        assert len(result['defaults_per_segment']) == 3

    def test_validate_minimum_defaults_fail(self):
        """Test validation fails when segments lack sufficient defaults"""
        segments = np.array([0, 0, 0, 1, 1, 1])
        y = np.array([1, 0, 0, 1, 0, 0])

        result = SegmentValidator.validate_minimum_defaults(
            segments, y, min_defaults=10
        )

        assert result['passed'] is False
        assert len(result['failed_segments']) == 2

    def test_validate_density_pass(self, sample_data):
        """Test density validation passes with balanced segments"""
        segments, _ = sample_data

        result = SegmentValidator.validate_density(
            segments, min_density=0.1, max_density=0.6
        )

        assert result['passed'] is True
        assert len(result['failed_segments']) == 0

    def test_validate_density_too_small(self):
        """Test density validation fails for segments that are too small"""
        segments = np.array([0] * 950 + [1] * 50)

        result = SegmentValidator.validate_density(
            segments, min_density=0.1, max_density=0.6
        )

        assert result['passed'] is False
        assert len(result['failed_segments']) == 1
        assert result['failed_segments'][0]['violation'] == 'too_small'

    def test_validate_density_too_large(self):
        """Test density validation fails for segments that are too large"""
        segments = np.array([0] * 800 + [1] * 200)

        result = SegmentValidator.validate_density(
            segments, min_density=0.1, max_density=0.5
        )

        assert result['passed'] is False
        failed = [f for f in result['failed_segments'] if f['segment'] == 0]
        assert len(failed) == 1
        assert failed[0]['violation'] == 'too_large'

    def test_validate_statistical_significance_chi_squared(self, sample_data):
        """Test chi-squared significance testing"""
        segments, y = sample_data

        result = SegmentValidator.validate_statistical_significance(
            segments, y, significance_level=0.01, method='chi_squared'
        )

        assert 'p_values' in result
        assert 'adjusted_alpha' in result
        assert result['method'] == 'chi_squared'
        assert result['p_values'].shape == (3, 3)

    def test_calculate_psi_stable(self):
        """Test PSI calculation for stable populations"""
        np.random.seed(42)

        # Create two similar distributions
        ref_segments = np.random.choice([0, 1, 2], size=1000, p=[0.3, 0.4, 0.3])
        cur_segments = np.random.choice([0, 1, 2], size=1000, p=[0.32, 0.38, 0.30])

        result = SegmentValidator.calculate_psi(
            ref_segments, cur_segments, threshold=0.1
        )

        assert result['psi'] < 0.1
        assert result['stability'] == 'stable'
        assert result['passed'] is True

    def test_calculate_psi_unstable(self):
        """Test PSI calculation for unstable populations"""
        # Create very different distributions
        ref_segments = np.array([0] * 500 + [1] * 300 + [2] * 200)
        cur_segments = np.array([0] * 200 + [1] * 300 + [2] * 500)

        result = SegmentValidator.calculate_psi(
            ref_segments, cur_segments, threshold=0.1
        )

        assert result['psi'] > 0.1
        assert result['passed'] is False

    def test_validate_default_rate_differences_pass(self, sample_data):
        """Test default rate difference validation passes"""
        segments, y = sample_data

        result = SegmentValidator.validate_default_rate_differences(
            segments, y, min_diff=0.001
        )

        assert result['passed'] is True
        assert len(result['default_rates']) == 3

    def test_validate_default_rate_differences_fail(self):
        """Test default rate difference validation fails for similar rates"""
        segments = np.array([0] * 100 + [1] * 100)
        y = np.array([1] * 10 + [0] * 90 + [1] * 11 + [0] * 89)

        result = SegmentValidator.validate_default_rate_differences(
            segments, y, min_diff=0.05
        )

        assert result['passed'] is False
        assert len(result['failed_pairs']) > 0

    def test_validate_binomial_confidence(self, sample_data):
        """Test binomial confidence interval calculation"""
        segments, y = sample_data

        result = SegmentValidator.validate_binomial_confidence(
            segments, y, confidence_level=0.95
        )

        assert 'confidence_intervals' in result
        assert len(result['confidence_intervals']) == 3

        for seg, ci in result['confidence_intervals'].items():
            assert 'default_rate' in ci
            assert 'lower_bound' in ci
            assert 'upper_bound' in ci
            assert ci['lower_bound'] <= ci['default_rate'] <= ci['upper_bound']

    def test_run_all_validations(self, sample_data):
        """Test running all configured validations"""
        segments, y = sample_data
        params = IRBSegmentationParams(
            min_defaults_per_leaf=5,
            min_segment_density=0.1,
            max_segment_density=0.6,
            validation_tests=['chi_squared', 'binomial']
        )

        result = SegmentValidator.run_all_validations(
            segments, y, params
        )

        assert 'all_passed' in result
        assert 'validations' in result
        assert 'chi_squared' in result['validations']
        assert 'binomial' in result['validations']
        assert 'min_defaults' in result['validations']
        assert 'density' in result['validations']

    def test_psi_with_missing_segments(self):
        """Test PSI handles segments that appear in one period but not another"""
        ref_segments = np.array([0] * 100 + [1] * 100)
        cur_segments = np.array([0] * 100 + [2] * 100)  # Segment 2 is new

        result = SegmentValidator.calculate_psi(ref_segments, cur_segments)

        # Should not raise error and should calculate PSI
        assert 'psi' in result
        assert result['psi'] > 0  # Should indicate a shift

    def test_statistical_significance_bonferroni_correction(self):
        """Test that Bonferroni correction is applied correctly"""
        segments = np.array([0] * 100 + [1] * 100 + [2] * 100)
        y = np.random.binomial(1, 0.1, 300)

        result = SegmentValidator.validate_statistical_significance(
            segments, y, significance_level=0.05
        )

        # With 3 segments, there are 3 pairwise comparisons
        # Adjusted alpha should be 0.05 / 3
        assert result['n_comparisons'] == 3
        assert abs(result['adjusted_alpha'] - 0.05 / 3) < 1e-10
