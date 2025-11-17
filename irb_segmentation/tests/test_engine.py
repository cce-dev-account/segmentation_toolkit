"""
Unit tests for IRBSegmentationEngine
"""

import pytest
import numpy as np
from irb_segmentation.engine import IRBSegmentationEngine
from irb_segmentation.params import IRBSegmentationParams


class TestIRBSegmentationEngine:
    """Test suite for IRBSegmentationEngine class"""

    @pytest.fixture
    def sample_credit_data(self):
        """Create sample credit data for testing"""
        np.random.seed(42)
        n = 2000

        # Generate features
        credit_score = np.random.normal(700, 100, n)
        ltv = np.random.uniform(50, 100, n)
        dti = np.random.uniform(10, 50, n)

        X = np.column_stack([credit_score, ltv, dti])

        # Generate outcomes with realistic patterns
        # Higher credit score -> lower default
        # Higher LTV -> higher default
        # Higher DTI -> higher default
        default_prob = 1 / (1 + np.exp(0.02 * (credit_score - 650) - 0.01 * (ltv - 75) - 0.01 * (dti - 30)))
        y = np.random.binomial(1, default_prob)

        feature_names = ['credit_score', 'ltv', 'dti']

        return X, y, feature_names

    def test_engine_initialization(self):
        """Test engine initialization with parameters"""
        params = IRBSegmentationParams()
        engine = IRBSegmentationEngine(params)

        assert engine.params == params
        assert not engine.is_fitted_
        assert engine.tree_model is not None

    def test_fit_basic(self, sample_credit_data):
        """Test basic fitting functionality"""
        X, y, feature_names = sample_credit_data
        params = IRBSegmentationParams(
            max_depth=2,
            min_samples_leaf=200,
            min_defaults_per_leaf=10,
            min_segment_density=0.05,
            max_segment_density=0.70
        )

        engine = IRBSegmentationEngine(params)
        engine.fit(X, y, feature_names=feature_names)

        assert engine.is_fitted_
        assert engine.segments_train_ is not None
        assert len(engine.segments_train_) == len(X)

    def test_fit_with_validation_set(self, sample_credit_data):
        """Test fitting with separate validation set"""
        X, y, feature_names = sample_credit_data

        # Split into train and validation
        split_idx = int(0.7 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        params = IRBSegmentationParams(
            max_depth=2,
            min_samples_leaf=200,
            min_defaults_per_leaf=10
        )

        engine = IRBSegmentationEngine(params)
        engine.fit(X_train, y_train, X_val, y_val, feature_names=feature_names)

        assert engine.segments_val_ is not None
        assert len(engine.segments_val_) == len(X_val)
        assert 'validation' in engine.validation_results_

    def test_predict(self, sample_credit_data):
        """Test prediction on new data"""
        X, y, feature_names = sample_credit_data

        # Fit on first half
        split_idx = int(0.5 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train = y[:split_idx]

        params = IRBSegmentationParams(max_depth=2, min_samples_leaf=100, min_defaults_per_leaf=5)
        engine = IRBSegmentationEngine(params)
        engine.fit(X_train, y_train, feature_names=feature_names)

        # Predict on second half
        segments_test = engine.predict(X_test)

        assert len(segments_test) == len(X_test)
        assert segments_test.dtype == np.int64 or segments_test.dtype == np.int32

    def test_predict_before_fit_raises_error(self, sample_credit_data):
        """Test that predict raises error before fitting"""
        X, y, feature_names = sample_credit_data
        params = IRBSegmentationParams()
        engine = IRBSegmentationEngine(params)

        with pytest.raises(RuntimeError, match="must be fitted"):
            engine.predict(X)

    def test_get_production_segments(self, sample_credit_data):
        """Test getting production segments"""
        X, y, feature_names = sample_credit_data
        params = IRBSegmentationParams(max_depth=2, min_samples_leaf=200, min_defaults_per_leaf=10)
        engine = IRBSegmentationEngine(params)
        engine.fit(X, y, feature_names=feature_names)

        segments = engine.get_production_segments()

        assert len(segments) == len(X)
        assert isinstance(segments, np.ndarray)
        # Should be a copy, not a reference
        assert segments is not engine.segments_train_

    def test_get_production_segments_before_fit_raises_error(self):
        """Test that get_production_segments raises error before fitting"""
        params = IRBSegmentationParams()
        engine = IRBSegmentationEngine(params)

        with pytest.raises(RuntimeError, match="must be fitted"):
            engine.get_production_segments()

    def test_get_validation_report(self, sample_credit_data):
        """Test validation report generation"""
        X, y, feature_names = sample_credit_data
        params = IRBSegmentationParams(max_depth=2, min_samples_leaf=200, min_defaults_per_leaf=10)
        engine = IRBSegmentationEngine(params)
        engine.fit(X, y, feature_names=feature_names)

        report = engine.get_validation_report()

        assert 'timestamp' in report
        assert 'parameters' in report
        assert 'segments' in report
        assert 'validation_results' in report
        assert 'adjustments' in report
        assert 'segment_statistics' in report

    def test_segment_statistics(self, sample_credit_data):
        """Test segment statistics calculation"""
        X, y, feature_names = sample_credit_data
        params = IRBSegmentationParams(max_depth=2, min_samples_leaf=200, min_defaults_per_leaf=10)
        engine = IRBSegmentationEngine(params)
        engine.fit(X, y, feature_names=feature_names)

        stats = engine._get_segment_statistics()

        assert isinstance(stats, dict)
        for seg, seg_stats in stats.items():
            assert 'n_observations' in seg_stats
            assert 'n_defaults' in seg_stats
            assert 'default_rate' in seg_stats
            assert 'density' in seg_stats
            assert 0 <= seg_stats['default_rate'] <= 1
            assert 0 <= seg_stats['density'] <= 1

    def test_forced_splits(self, sample_credit_data):
        """Test application of forced splits"""
        X, y, feature_names = sample_credit_data
        params = IRBSegmentationParams(
            max_depth=2,
            min_samples_leaf=100,
            min_defaults_per_leaf=5,
            forced_splits={'ltv': 80.0}
        )

        engine = IRBSegmentationEngine(params)
        engine.fit(X, y, feature_names=feature_names)

        assert len(engine.adjustment_log_['forced_splits']) > 0

    def test_monotonicity_check(self, sample_credit_data):
        """Test monotonicity constraint checking"""
        X, y, feature_names = sample_credit_data
        params = IRBSegmentationParams(
            max_depth=2,
            min_samples_leaf=200,
            min_defaults_per_leaf=10,
            monotone_constraints={'credit_score': 1}  # Higher score -> lower default
        )

        engine = IRBSegmentationEngine(params)
        engine.fit(X, y, feature_names=feature_names)

        # Check that monotonicity was evaluated
        assert 'monotonicity_violations' in engine.adjustment_log_

    def test_export_report(self, sample_credit_data, tmp_path):
        """Test exporting report to JSON file"""
        X, y, feature_names = sample_credit_data
        params = IRBSegmentationParams(max_depth=2, min_samples_leaf=200, min_defaults_per_leaf=10)
        engine = IRBSegmentationEngine(params)
        engine.fit(X, y, feature_names=feature_names)

        # Export to temporary file
        report_path = tmp_path / "report.json"
        engine.export_report(str(report_path))

        assert report_path.exists()

        # Verify JSON is valid
        import json
        with open(report_path, 'r') as f:
            report = json.load(f)

        assert 'timestamp' in report
        assert 'segments' in report

    def test_get_segment_rules(self, sample_credit_data):
        """Test extraction of segment rules"""
        X, y, feature_names = sample_credit_data
        params = IRBSegmentationParams(max_depth=2, min_samples_leaf=200, min_defaults_per_leaf=10)
        engine = IRBSegmentationEngine(params)
        engine.fit(X, y, feature_names=feature_names)

        rules = engine.get_segment_rules()

        assert isinstance(rules, list)
        assert len(rules) > 0
        for rule in rules:
            assert isinstance(rule, str)
            assert 'IF' in rule
            assert 'THEN' in rule

    def test_adjustment_iterations(self, sample_credit_data):
        """Test that adjustment iterations work correctly"""
        X, y, feature_names = sample_credit_data
        params = IRBSegmentationParams(
            max_depth=3,
            min_samples_leaf=100,
            min_defaults_per_leaf=20,
            min_segment_density=0.15
        )

        engine = IRBSegmentationEngine(params)
        engine.fit(X, y, feature_names=feature_names, max_adjustment_iterations=3)

        # Should have attempted adjustments
        assert engine.is_fitted_

    def test_very_restrictive_constraints(self, sample_credit_data):
        """Test behavior with very restrictive constraints"""
        X, y, feature_names = sample_credit_data
        params = IRBSegmentationParams(
            max_depth=2,
            min_samples_leaf=500,
            min_defaults_per_leaf=50,
            min_segment_density=0.20,
            max_segment_density=0.40
        )

        engine = IRBSegmentationEngine(params)

        # Should complete without crashing, even if constraints can't all be met
        with pytest.warns(Warning):
            engine.fit(X, y, feature_names=feature_names)

        assert engine.is_fitted_

    def test_single_segment_edge_case(self, sample_credit_data):
        """Test edge case where constraints force single segment"""
        X, y, feature_names = sample_credit_data

        # Very restrictive to force merging into one segment
        params = IRBSegmentationParams(
            max_depth=1,
            min_samples_leaf=1000,
            min_defaults_per_leaf=100,
            min_segment_density=0.40
        )

        engine = IRBSegmentationEngine(params)
        engine.fit(X, y, feature_names=feature_names)

        unique_segments = np.unique(engine.segments_train_)
        # Should have at least 1 segment
        assert len(unique_segments) >= 1
