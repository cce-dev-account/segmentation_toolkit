"""
Unit tests for IRBSegmentationParams
"""

import pytest
import numpy as np
from irb_segmentation.params import IRBSegmentationParams


class TestIRBSegmentationParams:
    """Test suite for IRBSegmentationParams class"""

    def test_default_initialization(self):
        """Test that default parameters are valid"""
        params = IRBSegmentationParams()
        assert params.max_depth == 3
        assert params.min_samples_split == 1000
        assert params.min_samples_leaf == 500
        assert params.criterion == 'gini'

    def test_to_sklearn_params(self):
        """Test extraction of sklearn-compatible parameters"""
        params = IRBSegmentationParams(
            max_depth=5,
            min_samples_split=500,
            min_samples_leaf=250,
            criterion='entropy',
            random_state=123
        )

        sklearn_params = params.to_sklearn_params()

        assert sklearn_params['max_depth'] == 5
        assert sklearn_params['min_samples_split'] == 500
        assert sklearn_params['min_samples_leaf'] == 250
        assert sklearn_params['criterion'] == 'entropy'
        assert sklearn_params['random_state'] == 123

        # Should not include IRB-specific parameters
        assert 'min_defaults_per_leaf' not in sklearn_params
        assert 'min_segment_density' not in sklearn_params

    def test_invalid_max_depth(self):
        """Test that invalid max_depth raises error"""
        with pytest.raises(ValueError, match="max_depth must be positive"):
            IRBSegmentationParams(max_depth=0)

    def test_invalid_criterion(self):
        """Test that invalid criterion raises error"""
        with pytest.raises(ValueError, match="criterion must be"):
            IRBSegmentationParams(criterion='invalid')

    def test_invalid_density_constraints(self):
        """Test that invalid density constraints raise error"""
        with pytest.raises(ValueError, match="min_segment_density"):
            IRBSegmentationParams(min_segment_density=0.8, max_segment_density=0.5)

    def test_invalid_significance_level(self):
        """Test that invalid significance level raises error"""
        with pytest.raises(ValueError, match="significance_level"):
            IRBSegmentationParams(significance_level=1.5)

    def test_monotone_constraints_validation(self):
        """Test that invalid monotone constraints raise error"""
        with pytest.raises(ValueError, match="monotone_constraints"):
            IRBSegmentationParams(monotone_constraints={'feature1': 2})

    def test_invalid_validation_tests(self):
        """Test that invalid validation tests raise error"""
        with pytest.raises(ValueError, match="Invalid validation tests"):
            IRBSegmentationParams(validation_tests=['invalid_test'])

    def test_samples_split_leaf_consistency(self):
        """Test that min_samples_split and min_samples_leaf are consistent"""
        with pytest.raises(ValueError, match="min_samples_split.*min_samples_leaf"):
            IRBSegmentationParams(min_samples_split=100, min_samples_leaf=100)

    def test_basel_warning(self):
        """Test warning when min_defaults_per_leaf is below Basel minimum"""
        with pytest.warns(UserWarning, match="Basel"):
            IRBSegmentationParams(min_defaults_per_leaf=10)

    def test_get_summary(self):
        """Test that get_summary returns formatted string"""
        params = IRBSegmentationParams()
        summary = params.get_summary()

        assert isinstance(summary, str)
        assert "IRB Segmentation Parameters" in summary
        assert "max_depth" in summary
        assert "min_defaults_per_leaf" in summary

    def test_custom_constraints(self):
        """Test initialization with custom business constraints"""
        params = IRBSegmentationParams(
            monotone_constraints={'credit_score': 1, 'ltv': -1},
            forced_splits={'ltv': 80.0},
            validation_tests=['chi_squared', 'psi']
        )

        assert params.monotone_constraints == {'credit_score': 1, 'ltv': -1}
        assert params.forced_splits == {'ltv': 80.0}
        assert params.validation_tests == ['chi_squared', 'psi']

    def test_validate_params_returns_empty_list_when_valid(self):
        """Test that validate_params returns empty list for valid params"""
        params = IRBSegmentationParams()
        issues = params.validate_params()
        assert issues == []
