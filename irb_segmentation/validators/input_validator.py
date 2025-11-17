"""
Input Validation for NumPy Arrays

Provides validation functions for arrays used in IRB segmentation:
- Shape and dimensionality checks
- Data type validation
- NaN and infinity handling
- Binary target validation
- Train/validation set compatibility

All errors include actionable guidance for fixing issues.
"""

import numpy as np
from typing import List, Optional, Tuple
from ..logger import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """
    Custom exception for validation errors with actionable messages.

    Attributes:
        message: Human-readable error description
        field: Field or parameter that failed validation
        expected: Expected value or condition
        actual: Actual value that caused the error
        fix: Suggested fix for the issue
    """

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        expected: Optional[str] = None,
        actual: Optional[str] = None,
        fix: Optional[str] = None
    ):
        self.message = message
        self.field = field
        self.expected = expected
        self.actual = actual
        self.fix = fix

        # Build detailed message
        parts = [f"[VALIDATION ERROR] {message}"]
        if field:
            parts.append(f"  Field: {field}")
        if expected:
            parts.append(f"  Expected: {expected}")
        if actual:
            parts.append(f"  Actual: {actual}")
        if fix:
            parts.append(f"  Fix: {fix}")

        super().__init__("\n".join(parts))


def validate_array(
    X: np.ndarray,
    name: str = "X",
    min_samples: int = 100,
    min_features: int = 1,
    allow_nan: bool = False,
    allow_inf: bool = False,
    expected_dtype: Optional[type] = None
) -> None:
    """
    Validate a NumPy array for IRB segmentation.

    Args:
        X: Array to validate
        name: Name of the array (for error messages)
        min_samples: Minimum number of samples required
        min_features: Minimum number of features required
        allow_nan: Whether NaN values are allowed
        allow_inf: Whether infinity values are allowed
        expected_dtype: Expected data type (optional)

    Raises:
        ValidationError: If validation fails with actionable guidance

    Example:
        >>> X = np.random.rand(1000, 10)
        >>> validate_array(X, name="X_train", min_samples=500)
        >>> # Passes validation
    """
    # Check if input is a NumPy array
    if not isinstance(X, np.ndarray):
        raise ValidationError(
            f"{name} must be a NumPy array",
            field=name,
            expected="numpy.ndarray",
            actual=str(type(X).__name__),
            fix=f"Convert to NumPy array: X = np.array({name})"
        )

    # Check dimensionality
    if X.ndim != 2:
        raise ValidationError(
            f"{name} must be 2-dimensional",
            field=name,
            expected="2D array with shape (n_samples, n_features)",
            actual=f"{X.ndim}D array with shape {X.shape}",
            fix=f"Reshape to 2D: {name} = {name}.reshape(-1, 1) for 1 feature, or {name} = {name}.reshape(1, -1) for 1 sample"
        )

    n_samples, n_features = X.shape

    # Check minimum samples
    if n_samples < min_samples:
        raise ValidationError(
            f"{name} has too few samples",
            field=name,
            expected=f"At least {min_samples} samples",
            actual=f"{n_samples} samples",
            fix=f"Provide more data or reduce min_samples parameter to {n_samples}"
        )

    # Check minimum features
    if n_features < min_features:
        raise ValidationError(
            f"{name} has too few features",
            field=name,
            expected=f"At least {min_features} features",
            actual=f"{n_features} features",
            fix="Add more features or check if data was loaded correctly"
        )

    # Check data type
    if expected_dtype is not None and not np.issubdtype(X.dtype, expected_dtype):
        raise ValidationError(
            f"{name} has incorrect data type",
            field=name,
            expected=str(expected_dtype),
            actual=str(X.dtype),
            fix=f"Convert data type: {name} = {name}.astype({expected_dtype.__name__})"
        )

    # Check for NaN values
    if not allow_nan and np.any(np.isnan(X)):
        n_nan = np.sum(np.isnan(X))
        pct_nan = 100 * n_nan / X.size
        raise ValidationError(
            f"{name} contains NaN values",
            field=name,
            expected="No NaN values",
            actual=f"{n_nan} NaN values ({pct_nan:.2f}% of data)",
            fix=f"Handle missing values: {name} = SimpleImputer().fit_transform({name}) or {name}.fillna(0)"
        )

    # Check for infinity values
    if not allow_inf and np.any(np.isinf(X)):
        n_inf = np.sum(np.isinf(X))
        pct_inf = 100 * n_inf / X.size
        raise ValidationError(
            f"{name} contains infinity values",
            field=name,
            expected="No infinity values",
            actual=f"{n_inf} infinity values ({pct_inf:.2f}% of data)",
            fix=f"Replace infinities: {name}[np.isinf({name})] = np.nan, then impute or use np.clip()"
        )

    logger.info(f"{name} validation passed: shape {X.shape}, dtype {X.dtype}")


def validate_binary_target(
    y: np.ndarray,
    name: str = "y",
    min_samples: int = 100,
    min_defaults: int = 10,
    max_default_rate: float = 0.50
) -> Tuple[int, int, float]:
    """
    Validate a binary target array.

    Args:
        y: Binary target array (0/1)
        name: Name of the target (for error messages)
        min_samples: Minimum total samples
        min_defaults: Minimum number of defaults (1s)
        max_default_rate: Maximum allowed default rate

    Returns:
        Tuple of (n_samples, n_defaults, default_rate)

    Raises:
        ValidationError: If validation fails

    Example:
        >>> y = np.array([0, 0, 1, 0, 1, 1])
        >>> n, n_def, rate = validate_binary_target(y, min_samples=5)
        >>> print(f"Default rate: {rate:.2%}")
        Default rate: 50.00%
    """
    # Check if array
    if not isinstance(y, np.ndarray):
        raise ValidationError(
            f"{name} must be a NumPy array",
            field=name,
            expected="numpy.ndarray",
            actual=str(type(y).__name__),
            fix=f"Convert to NumPy array: y = np.array({name})"
        )

    # Check 1D
    if y.ndim != 1:
        raise ValidationError(
            f"{name} must be 1-dimensional",
            field=name,
            expected="1D array with shape (n_samples,)",
            actual=f"{y.ndim}D array with shape {y.shape}",
            fix=f"Flatten to 1D: {name} = {name}.ravel() or {name} = {name}.flatten()"
        )

    n_samples = len(y)

    # Check minimum samples
    if n_samples < min_samples:
        raise ValidationError(
            f"{name} has too few samples",
            field=name,
            expected=f"At least {min_samples} samples",
            actual=f"{n_samples} samples",
            fix=f"Provide more data or reduce min_samples to {n_samples}"
        )

    # Check for NaN
    if np.any(np.isnan(y)):
        n_nan = np.sum(np.isnan(y))
        raise ValidationError(
            f"{name} contains NaN values",
            field=name,
            expected="No NaN values in target",
            actual=f"{n_nan} NaN values",
            fix=f"Remove NaN rows: mask = ~np.isnan({name}); X, y = X[mask], y[mask]"
        )

    # Check binary values
    unique_values = np.unique(y)
    if not np.all(np.isin(unique_values, [0, 1])):
        raise ValidationError(
            f"{name} must contain only binary values (0 and 1)",
            field=name,
            expected="Values in {{0, 1}}",
            actual=f"Unique values: {unique_values}",
            fix=f"Convert to binary: {name} = ({name} > threshold).astype(int)"
        )

    # Check number of defaults
    n_defaults = int(np.sum(y))
    if n_defaults < min_defaults:
        raise ValidationError(
            f"{name} has too few defaults (positive cases)",
            field=name,
            expected=f"At least {min_defaults} defaults",
            actual=f"{n_defaults} defaults",
            fix="Collect more default cases or reduce min_defaults_per_leaf parameter"
        )

    # Check default rate
    default_rate = n_defaults / n_samples
    if default_rate > max_default_rate:
        raise ValidationError(
            f"{name} has excessive default rate",
            field=name,
            expected=f"Default rate <= {max_default_rate:.1%}",
            actual=f"{default_rate:.2%} default rate",
            fix="Check if target is inverted or data is imbalanced. Consider resampling."
        )

    # Check if all zeros or all ones
    if n_defaults == 0:
        raise ValidationError(
            f"{name} contains only zeros (no positive cases)",
            field=name,
            expected="At least some positive cases",
            actual="All zeros",
            fix="Check if target column is correct or if data filtering removed all defaults"
        )

    if n_defaults == n_samples:
        raise ValidationError(
            f"{name} contains only ones (no negative cases)",
            field=name,
            expected="At least some negative cases",
            actual="All ones",
            fix="Check if target column is correct or if data is mislabeled"
        )

    logger.info(f"{name} validation passed: {n_samples} samples, {n_defaults} defaults ({default_rate:.2%})")

    return n_samples, n_defaults, default_rate


def validate_feature_names(
    feature_names: Optional[List[str]],
    n_features: int
) -> List[str]:
    """
    Validate and generate feature names if not provided.

    Args:
        feature_names: Optional list of feature names
        n_features: Expected number of features

    Returns:
        Validated or generated feature names

    Raises:
        ValidationError: If feature names are invalid

    Example:
        >>> names = validate_feature_names(["age", "income"], 2)
        >>> print(names)
        ['age', 'income']
    """
    if feature_names is None:
        # Generate default names
        feature_names = [f"feature_{i}" for i in range(n_features)]
        logger.info(f"Generated {n_features} default feature names")
        return feature_names

    if not isinstance(feature_names, (list, tuple)):
        raise ValidationError(
            "feature_names must be a list or tuple",
            field="feature_names",
            expected="list or tuple",
            actual=str(type(feature_names).__name__),
            fix="Convert to list: feature_names = list(feature_names)"
        )

    if len(feature_names) != n_features:
        raise ValidationError(
            "Number of feature names doesn't match number of features",
            field="feature_names",
            expected=f"{n_features} feature names",
            actual=f"{len(feature_names)} feature names",
            fix=f"Provide exactly {n_features} feature names or set feature_names=None to auto-generate"
        )

    # Check for duplicates
    if len(feature_names) != len(set(feature_names)):
        duplicates = [name for name in feature_names if feature_names.count(name) > 1]
        raise ValidationError(
            "Feature names contain duplicates",
            field="feature_names",
            expected="Unique feature names",
            actual=f"Duplicates: {list(set(duplicates))}",
            fix="Rename duplicates: feature_names = [f'{name}_{i}' if duplicate else name]"
        )

    # Check for empty strings
    if any(not str(name).strip() for name in feature_names):
        raise ValidationError(
            "Feature names contain empty strings",
            field="feature_names",
            expected="Non-empty feature names",
            actual="Some feature names are empty",
            fix="Replace empty names: feature_names = [f'feature_{i}' if not name else name for i, name in enumerate(feature_names)]"
        )

    logger.info(f"Feature names validation passed: {len(feature_names)} features")

    return list(feature_names)


def validate_train_val_compatibility(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> None:
    """
    Validate that training and validation sets are compatible.

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target

    Raises:
        ValidationError: If sets are incompatible

    Example:
        >>> X_train = np.random.rand(1000, 10)
        >>> y_train = np.random.randint(0, 2, 1000)
        >>> X_val = np.random.rand(300, 10)
        >>> y_val = np.random.randint(0, 2, 300)
        >>> validate_train_val_compatibility(X_train, y_train, X_val, y_val)
    """
    # Check feature count
    if X_train.shape[1] != X_val.shape[1]:
        raise ValidationError(
            "Training and validation sets have different number of features",
            field="n_features",
            expected=f"{X_train.shape[1]} features (from training set)",
            actual=f"{X_val.shape[1]} features (in validation set)",
            fix="Ensure both datasets have same features. Check preprocessing pipeline."
        )

    # Check sample-target alignment
    if len(X_train) != len(y_train):
        raise ValidationError(
            "Training features and target have mismatched lengths",
            field="X_train / y_train",
            expected=f"{len(X_train)} samples in X_train",
            actual=f"{len(y_train)} samples in y_train",
            fix="Ensure X_train and y_train have same number of rows"
        )

    if len(X_val) != len(y_val):
        raise ValidationError(
            "Validation features and target have mismatched lengths",
            field="X_val / y_val",
            expected=f"{len(X_val)} samples in X_val",
            actual=f"{len(y_val)} samples in y_val",
            fix="Ensure X_val and y_val have same number of rows"
        )

    # Check data types
    if X_train.dtype != X_val.dtype:
        logger.warning(
            f"Training and validation sets have different dtypes: "
            f"X_train={X_train.dtype}, X_val={X_val.dtype}"
        )

    # Check for data leakage indicators (identical rows)
    if X_train.shape[0] < 10000 and X_val.shape[0] < 10000:  # Only for small datasets
        # Sample check for identical rows
        sample_size = min(100, X_train.shape[0], X_val.shape[0])
        train_sample = X_train[:sample_size]
        val_sample = X_val[:sample_size]

        # Check if any validation samples appear in training set
        identical_count = 0
        for val_row in val_sample:
            if np.any(np.all(np.isclose(train_sample, val_row, rtol=1e-10), axis=1)):
                identical_count += 1

        if identical_count > sample_size * 0.1:  # More than 10% overlap
            logger.warning(
                f"Possible data leakage detected: {identical_count}/{sample_size} "
                f"validation samples found in training set"
            )

    logger.info(
        f"Train/val compatibility validated: "
        f"X_train={X_train.shape}, y_train={y_train.shape}, "
        f"X_val={X_val.shape}, y_val={y_val.shape}"
    )
