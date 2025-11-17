"""
Unit Tests for IRB Segmentation Input Validation

Tests cover:
- Array validation (shape, dtype, NaN/inf handling)
- Binary target validation (value range, default counts, rates)
- Feature names validation
- Train/validation set compatibility
- ValidationError formatting and error messages
"""

import pytest
import numpy as np
from irb_segmentation.validators import (
    validate_array,
    validate_binary_target,
    validate_feature_names,
    validate_train_val_compatibility,
    ValidationError
)


@pytest.mark.unit
class TestValidationError:
    """Test ValidationError exception class."""

    def test_basic_validation_error(self):
        """Test creating a basic ValidationError."""
        error = ValidationError("Test error message")
        assert "Test error message" in str(error)

    def test_validation_error_with_field(self):
        """Test ValidationError with field parameter."""
        error = ValidationError(
            "Invalid input",
            field="X_train"
        )
        assert "X_train" in str(error)

    def test_validation_error_with_expected_actual(self):
        """Test ValidationError with expected and actual."""
        error = ValidationError(
            "Shape mismatch",
            field="X",
            expected="(100, 10)",
            actual="(50, 10)"
        )

        error_str = str(error)
        assert "Expected: (100, 10)" in error_str
        assert "Actual: (50, 10)" in error_str

    def test_validation_error_with_fix(self):
        """Test ValidationError includes fix suggestion."""
        error = ValidationError(
            "Array has NaN values",
            fix="Remove NaN: X = X[~np.isnan(X).any(axis=1)]"
        )

        error_str = str(error)
        assert "Fix:" in error_str
        assert "Remove NaN" in error_str

    def test_validation_error_full_details(self):
        """Test ValidationError with all parameters."""
        error = ValidationError(
            "Invalid array",
            field="X_train",
            expected="numpy.ndarray",
            actual="pandas.DataFrame",
            fix="Convert: X = np.array(df)"
        )

        error_str = str(error)
        assert "[VALIDATION ERROR]" in error_str
        assert "Field: X_train" in error_str
        assert "Expected: numpy.ndarray" in error_str
        assert "Actual: pandas.DataFrame" in error_str
        assert "Fix: Convert" in error_str


@pytest.mark.unit
class TestArrayValidation:
    """Test validate_array function."""

    def test_valid_2d_array(self):
        """Test valid 2D array passes validation."""
        X = np.random.randn(500, 10)
        # Should not raise exception
        validate_array(X, name="X", min_samples=100)

    def test_array_not_numpy(self):
        """Test validation fails for non-numpy array."""
        X = [[1, 2, 3], [4, 5, 6]]  # Python list

        with pytest.raises(ValidationError) as exc_info:
            validate_array(X)

        assert "must be a NumPy array" in str(exc_info.value)
        assert "numpy.ndarray" in str(exc_info.value)

    def test_array_not_2d(self):
        """Test validation fails for 1D array."""
        X = np.array([1, 2, 3, 4, 5])

        with pytest.raises(ValidationError) as exc_info:
            validate_array(X, name="X")

        assert "must be 2-dimensional" in str(exc_info.value)

    def test_array_too_few_samples(self):
        """Test validation fails for too few samples."""
        X = np.random.randn(50, 10)

        with pytest.raises(ValidationError) as exc_info:
            validate_array(X, min_samples=100)

        assert "too few samples" in str(exc_info.value).lower()
        assert "50" in str(exc_info.value)
        assert "100" in str(exc_info.value)

    def test_array_too_few_features(self):
        """Test validation fails for too few features."""
        X = np.random.randn(1000, 0)  # No features

        with pytest.raises(ValidationError) as exc_info:
            validate_array(X, min_features=1)

        assert "too few features" in str(exc_info.value).lower()

    def test_array_with_nan_not_allowed(self):
        """Test validation fails when NaN not allowed."""
        X = np.random.randn(500, 10)
        X[50, 3] = np.nan

        with pytest.raises(ValidationError) as exc_info:
            validate_array(X, allow_nan=False)

        assert "NaN" in str(exc_info.value) or "nan" in str(exc_info.value).lower()

    def test_array_with_nan_allowed(self):
        """Test validation passes when NaN allowed."""
        X = np.random.randn(500, 10)
        X[50, 3] = np.nan

        # Should not raise exception
        validate_array(X, allow_nan=True)

    def test_array_with_inf_not_allowed(self):
        """Test validation fails when inf not allowed."""
        X = np.random.randn(500, 10)
        X[50, 3] = np.inf

        with pytest.raises(ValidationError) as exc_info:
            validate_array(X, allow_inf=False)

        assert "infinity" in str(exc_info.value).lower()

    def test_array_with_inf_allowed(self):
        """Test validation passes when inf allowed."""
        X = np.random.randn(500, 10)
        X[50, 3] = np.inf

        # Should not raise exception
        validate_array(X, allow_inf=True)

    def test_array_wrong_dtype(self):
        """Test validation fails for unexpected dtype."""
        X = np.array([[1, 2, 3]] * 200, dtype=np.int32)  # Enough samples

        with pytest.raises(ValidationError) as exc_info:
            validate_array(X, expected_dtype=np.float64, min_samples=100)

        assert "dtype" in str(exc_info.value).lower() or "data type" in str(exc_info.value).lower()

    def test_array_correct_dtype(self):
        """Test validation passes for expected dtype."""
        X = np.array([[1.0, 2.0]] * 200, dtype=np.float64)  # Enough samples

        # Should not raise exception
        validate_array(X, expected_dtype=np.float64, min_samples=100)

    def test_array_custom_name_in_error(self):
        """Test custom array name appears in error message."""
        X = np.array([1, 2, 3])  # 1D array

        with pytest.raises(ValidationError) as exc_info:
            validate_array(X, name="my_custom_array")

        assert "my_custom_array" in str(exc_info.value)


@pytest.mark.unit
class TestBinaryTargetValidation:
    """Test validate_binary_target function."""

    def test_valid_binary_target(self):
        """Test valid binary target passes validation."""
        y = np.zeros(1000, dtype=int)
        y[:100] = 1  # 10% default rate

        n_samples, n_defaults, default_rate = validate_binary_target(
            y, min_samples=500, min_defaults=50
        )

        assert n_samples == 1000
        assert n_defaults == 100
        assert np.isclose(default_rate, 0.10)

    def test_target_not_numpy(self):
        """Test validation fails for non-numpy target."""
        y = [0, 1, 0, 1, 0]

        with pytest.raises(ValidationError) as exc_info:
            validate_binary_target(y)

        assert "must be a NumPy array" in str(exc_info.value)

    def test_target_not_1d(self):
        """Test validation fails for 2D target."""
        y = np.array([[0, 1], [1, 0]])

        with pytest.raises(ValidationError) as exc_info:
            validate_binary_target(y)

        assert "must be 1-dimensional" in str(exc_info.value)

    def test_target_too_few_samples(self):
        """Test validation fails for too few samples."""
        y = np.array([0, 1, 0, 1, 0])

        with pytest.raises(ValidationError) as exc_info:
            validate_binary_target(y, min_samples=100)

        assert "too few samples" in str(exc_info.value).lower()

    def test_target_not_binary(self):
        """Test validation fails for non-binary values."""
        y = np.array([0, 1, 2, 1, 0] * 50)  # Has value 2, enough samples

        with pytest.raises(ValidationError) as exc_info:
            validate_binary_target(y, min_samples=100)

        assert "binary" in str(exc_info.value).lower()

    def test_target_too_few_defaults(self):
        """Test validation fails for too few defaults."""
        y = np.zeros(1000, dtype=int)
        y[:5] = 1  # Only 5 defaults

        with pytest.raises(ValidationError) as exc_info:
            validate_binary_target(y, min_defaults=50)

        assert "too few defaults" in str(exc_info.value).lower() or \
               "positive" in str(exc_info.value).lower()

    def test_target_all_zeros(self):
        """Test validation fails for all zeros."""
        y = np.zeros(1000, dtype=int)

        with pytest.raises(ValidationError) as exc_info:
            validate_binary_target(y, min_defaults=1)

        error_str = str(exc_info.value).lower()
        assert any(phrase in error_str for phrase in
                  ["only zeros", "no positive", "no defaults", "too few defaults"])

    def test_target_all_ones(self):
        """Test validation handles all ones (100% default rate)."""
        y = np.ones(1000, dtype=int)

        with pytest.raises(ValidationError) as exc_info:
            validate_binary_target(y, max_default_rate=0.50)

        assert "default rate" in str(exc_info.value).lower()

    def test_target_excessive_default_rate(self):
        """Test validation fails for excessive default rate."""
        y = np.zeros(1000, dtype=int)
        y[:600] = 1  # 60% default rate

        with pytest.raises(ValidationError) as exc_info:
            validate_binary_target(y, max_default_rate=0.50)

        assert "default rate" in str(exc_info.value).lower()
        assert "50" in str(exc_info.value) or "0.5" in str(exc_info.value)

    def test_target_has_nan(self):
        """Test validation fails for NaN in target."""
        y = np.array([0.0, 1.0] * 100)  # Create enough samples
        y[50] = np.nan  # Add NaN

        with pytest.raises(ValidationError) as exc_info:
            validate_binary_target(y, min_samples=100)

        assert "nan" in str(exc_info.value).lower()

    def test_target_custom_name_in_error(self):
        """Test custom target name appears in error message."""
        y = np.array([0, 1, 2])  # Invalid

        with pytest.raises(ValidationError) as exc_info:
            validate_binary_target(y, name="my_target")

        assert "my_target" in str(exc_info.value)

    def test_target_returns_correct_values(self):
        """Test function returns correct statistics."""
        y = np.zeros(2000, dtype=int)
        y[:300] = 1  # 15% default rate

        n_samples, n_defaults, default_rate = validate_binary_target(y)

        assert n_samples == 2000
        assert n_defaults == 300
        assert np.isclose(default_rate, 0.15)


@pytest.mark.unit
class TestFeatureNamesValidation:
    """Test validate_feature_names function."""

    def test_valid_feature_names(self):
        """Test valid feature names pass validation."""
        feature_names = ['age', 'income', 'credit_score', 'debt_ratio', 'employment']

        validated_names = validate_feature_names(feature_names, n_features=5)

        assert validated_names == feature_names
        assert len(validated_names) == 5

    def test_generate_default_names(self):
        """Test generating default feature names."""
        feature_names = validate_feature_names(None, n_features=10)

        assert len(feature_names) == 10
        assert feature_names[0] == 'feature_0'
        assert feature_names[9] == 'feature_9'

    def test_wrong_number_of_names(self):
        """Test validation fails for wrong number of names."""
        feature_names = ['age', 'income', 'credit_score']  # Only 3 names for 5 features

        with pytest.raises(ValidationError) as exc_info:
            validate_feature_names(feature_names, n_features=5)

        assert "number of feature names" in str(exc_info.value).lower()
        assert "3" in str(exc_info.value)
        assert "5" in str(exc_info.value)

    def test_duplicate_feature_names(self):
        """Test validation fails for duplicate names."""
        feature_names = ['age', 'income', 'age', 'debt_ratio', 'income']

        with pytest.raises(ValidationError) as exc_info:
            validate_feature_names(feature_names, n_features=5)

        assert "duplicate" in str(exc_info.value).lower()

    def test_empty_feature_names(self):
        """Test validation fails for empty names."""
        feature_names = ['age', '', 'income', 'debt', 'employment']

        with pytest.raises(ValidationError) as exc_info:
            validate_feature_names(feature_names, n_features=5)

        assert "empty" in str(exc_info.value).lower()

    def test_none_names_in_list(self):
        """Test that None in names list passes through unchanged."""
        # Note: Current implementation allows None to pass through
        # because str(None).strip() = "None" which is not empty
        # This documents current behavior
        feature_names = ['age', None, 'income']

        validated = validate_feature_names(feature_names, n_features=3)
        # None passes through unchanged in current implementation
        assert validated[1] is None

    def test_whitespace_only_names(self):
        """Test validation fails for whitespace-only names."""
        feature_names = ['age', '   ', 'income']

        with pytest.raises(ValidationError) as exc_info:
            validate_feature_names(feature_names, n_features=3)

        assert "empty" in str(exc_info.value).lower()


@pytest.mark.unit
class TestTrainValCompatibility:
    """Test validate_train_val_compatibility function."""

    def test_compatible_sets(self):
        """Test compatible train/val sets pass validation."""
        X_train = np.random.randn(1000, 10)
        y_train = np.zeros(1000, dtype=int)
        y_train[:100] = 1

        X_val = np.random.randn(300, 10)
        y_val = np.zeros(300, dtype=int)
        y_val[:30] = 1

        # Should not raise exception
        validate_train_val_compatibility(X_train, y_train, X_val, y_val)

    def test_incompatible_feature_count(self):
        """Test validation fails for different feature counts."""
        X_train = np.random.randn(1000, 10)
        y_train = np.zeros(1000, dtype=int)

        X_val = np.random.randn(300, 5)  # Different number of features
        y_val = np.zeros(300, dtype=int)

        with pytest.raises(ValidationError) as exc_info:
            validate_train_val_compatibility(X_train, y_train, X_val, y_val)

        assert "different number of features" in str(exc_info.value).lower()
        assert "10" in str(exc_info.value)
        assert "5" in str(exc_info.value)

    def test_incompatible_X_y_lengths_train(self):
        """Test validation fails when X_train and y_train have different lengths."""
        X_train = np.random.randn(1000, 10)
        y_train = np.zeros(800, dtype=int)  # Different length

        X_val = np.random.randn(300, 10)
        y_val = np.zeros(300, dtype=int)

        with pytest.raises(ValidationError) as exc_info:
            validate_train_val_compatibility(X_train, y_train, X_val, y_val)

        assert "mismatched" in str(exc_info.value).lower()

    def test_incompatible_X_y_lengths_val(self):
        """Test validation fails when X_val and y_val have different lengths."""
        X_train = np.random.randn(1000, 10)
        y_train = np.zeros(1000, dtype=int)

        X_val = np.random.randn(300, 10)
        y_val = np.zeros(250, dtype=int)  # Different length

        with pytest.raises(ValidationError) as exc_info:
            validate_train_val_compatibility(X_train, y_train, X_val, y_val)

        assert "mismatched" in str(exc_info.value).lower()

    def test_validation_set_too_small(self):
        """Test warning for very small validation set."""
        X_train = np.random.randn(1000, 10)
        y_train = np.zeros(1000, dtype=int)
        y_train[:100] = 1

        X_val = np.random.randn(10, 10)  # Very small
        y_val = np.zeros(10, dtype=int)
        y_val[:1] = 1

        # Should not raise exception (just log warning)
        validate_train_val_compatibility(X_train, y_train, X_val, y_val)


@pytest.mark.unit
class TestValidationIntegration:
    """Integration tests for validation functions."""

    def test_full_workflow_validation(self):
        """Test full validation workflow."""
        # Create valid data
        X_train = np.random.randn(1000, 5)
        y_train = np.zeros(1000, dtype=int)
        y_train[:150] = 1

        X_val = np.random.randn(300, 5)
        y_val = np.zeros(300, dtype=int)
        y_val[:45] = 1

        feature_names = ['f1', 'f2', 'f3', 'f4', 'f5']

        # All validations should pass
        validate_array(X_train, name="X_train", min_samples=500)
        validate_binary_target(y_train, name="y_train", min_defaults=50)
        validate_array(X_val, name="X_val", min_samples=100)
        validate_binary_target(y_val, name="y_val", min_defaults=10)
        validate_train_val_compatibility(X_train, y_train, X_val, y_val)
        validated_names = validate_feature_names(feature_names, n_features=5)

        assert validated_names == feature_names

    def test_validation_catches_multiple_issues(self):
        """Test that validation catches multiple issues."""
        # Create data with multiple issues
        X_train = np.random.randn(50, 5)  # Too few samples
        X_train[10, 2] = np.nan  # Has NaN
        y_train = np.array([0, 1, 2, 0, 1] * 10)  # Not binary

        # Should catch multiple issues
        issues_found = []

        try:
            validate_array(X_train, min_samples=100, allow_nan=False)
        except ValidationError as e:
            issues_found.append(str(e))

        try:
            validate_binary_target(y_train)
        except ValidationError as e:
            issues_found.append(str(e))

        assert len(issues_found) >= 2


@pytest.mark.unit
class TestValidationEdgeCases:
    """Test edge cases for validation functions."""

    def test_array_exactly_min_samples(self):
        """Test array with exactly minimum samples."""
        X = np.random.randn(100, 5)
        # Should pass
        validate_array(X, min_samples=100)

    def test_target_exactly_min_defaults(self):
        """Test target with exactly minimum defaults."""
        y = np.zeros(500, dtype=int)
        y[:50] = 1

        n_samples, n_defaults, default_rate = validate_binary_target(
            y, min_defaults=50
        )
        assert n_defaults == 50

    def test_array_single_feature(self):
        """Test array with single feature."""
        X = np.random.randn(1000, 1)
        validate_array(X, min_features=1)

    def test_very_large_array(self):
        """Test validation on very large array."""
        X = np.random.randn(100000, 50)
        validate_array(X, min_samples=10000)

    def test_zero_variance_feature(self):
        """Test array with zero-variance feature (constant)."""
        X = np.random.randn(1000, 5)
        X[:, 2] = 5.0  # Constant feature

        # Should still pass validation (not our job to check variance)
        validate_array(X)

    def test_extreme_default_rate_near_boundary(self):
        """Test default rate at boundary."""
        y = np.zeros(1000, dtype=int)
        y[:500] = 1  # Exactly 50%

        n_samples, n_defaults, default_rate = validate_binary_target(
            y, max_default_rate=0.50
        )
        assert np.isclose(default_rate, 0.50)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
