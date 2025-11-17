"""
Test Logging and Validation Functionality

Comprehensive tests for:
- Structured logging
- Input validation with actionable errors
- DataFrame validation
- Integration with engine
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from irb_segmentation.logger import get_logger
from irb_segmentation.validators import (
    ValidationError,
    validate_array,
    validate_binary_target,
    validate_feature_names,
    validate_train_val_compatibility,
    validate_dataframe,
    validate_target_column,
    check_data_quality
)
from irb_segmentation.params import IRBSegmentationParams
from irb_segmentation.engine import IRBSegmentationEngine

# Setup logger for tests
logger = get_logger(__name__)


def test_logging():
    """Test logging functionality."""
    print("=" * 80)
    print("TEST 1: LOGGING FUNCTIONALITY")
    print("=" * 80)

    # Test basic logging
    logger.info("Testing INFO level logging")
    logger.warning("Testing WARNING level logging")
    logger.error("Testing ERROR level logging")

    print("[OK] Basic logging works\n")


def test_validation_errors():
    """Test ValidationError with actionable messages."""
    print("=" * 80)
    print("TEST 2: VALIDATION ERRORS")
    print("=" * 80)

    try:
        raise ValidationError(
            "Test validation error",
            field="test_field",
            expected="Expected value",
            actual="Actual value",
            fix="Fix: Do this to resolve"
        )
    except ValidationError as e:
        print(f"ValidationError message:\n{str(e)}\n")
        assert "test_field" in str(e)
        assert "Expected value" in str(e)
        assert "Actual value" in str(e)
        assert "Fix: Do this" in str(e)

    print("[OK] ValidationError provides actionable messages\n")


def test_array_validation():
    """Test array validation with various failure modes."""
    print("=" * 80)
    print("TEST 3: ARRAY VALIDATION")
    print("=" * 80)

    # Test 1: Valid array
    X_valid = np.random.rand(1000, 10)
    try:
        validate_array(X_valid, name="X_valid", min_samples=100, min_features=5)
        print("[OK] Valid array passed validation")
    except ValidationError as e:
        print(f"[FAIL] Valid array failed: {e}")
        return False

    # Test 2: Too few samples
    X_small = np.random.rand(50, 10)
    try:
        validate_array(X_small, name="X_small", min_samples=100)
        print("[FAIL] Small array should have failed")
        return False
    except ValidationError as e:
        print(f"[OK] Caught small array error")
        assert "too few samples" in str(e).lower()
        assert "50 samples" in str(e)
        print(f"     Error message includes sample count and fix suggestion")

    # Test 3: Wrong dimensions
    X_1d = np.random.rand(1000)
    try:
        validate_array(X_1d, name="X_1d")
        print("[FAIL] 1D array should have failed")
        return False
    except ValidationError as e:
        print(f"[OK] Caught 1D array error")
        assert "2-dimensional" in str(e).lower()
        assert "reshape" in str(e).lower()

    # Test 4: Contains NaN
    X_nan = np.random.rand(1000, 10)
    X_nan[0, 0] = np.nan
    try:
        validate_array(X_nan, name="X_nan", allow_nan=False)
        print("[FAIL] Array with NaN should have failed")
        return False
    except ValidationError as e:
        print(f"[OK] Caught NaN error")
        assert "nan" in str(e).lower()
        assert "fillna" in str(e).lower() or "imputer" in str(e).lower()

    # Test 5: Contains infinity
    X_inf = np.random.rand(1000, 10)
    X_inf[0, 0] = np.inf
    try:
        validate_array(X_inf, name="X_inf", allow_inf=False)
        print("[FAIL] Array with infinity should have failed")
        return False
    except ValidationError as e:
        print(f"[OK] Caught infinity error")
        assert "infinity" in str(e).lower()

    print("\n[OK] All array validation tests passed\n")
    return True


def test_binary_target_validation():
    """Test binary target validation."""
    print("=" * 80)
    print("TEST 4: BINARY TARGET VALIDATION")
    print("=" * 80)

    # Test 1: Valid binary target
    y_valid = np.array([0, 0, 1, 1, 0, 1] * 100)  # 600 samples
    try:
        n_samples, n_defaults, rate = validate_binary_target(
            y_valid, name="y_valid", min_samples=100, min_defaults=10
        )
        print(f"[OK] Valid target passed: {n_samples} samples, {n_defaults} defaults ({rate:.2%})")
    except ValidationError as e:
        print(f"[FAIL] Valid target failed: {e}")
        return False

    # Test 2: Too few defaults
    y_few_defaults = np.array([0, 0, 0, 0, 1] * 50)  # Only 50 defaults in 250 samples
    try:
        validate_binary_target(y_few_defaults, name="y_few_defaults", min_samples=100, min_defaults=100)
        print("[FAIL] Target with few defaults should have failed")
        return False
    except ValidationError as e:
        print(f"[OK] Caught few defaults error")
        assert ("too few defaults" in str(e).lower() or "positive" in str(e).lower())
        assert "50 defaults" in str(e)

    # Test 3: All zeros
    y_all_zeros = np.zeros(100)
    try:
        validate_binary_target(y_all_zeros, name="y_all_zeros")
        print("[FAIL] All-zeros target should have failed")
        return False
    except ValidationError as e:
        print(f"[OK] Caught all-zeros error")
        assert ("only zeros" in str(e).lower() or "no positive" in str(e).lower() or
                "too few defaults" in str(e).lower())

    # Test 4: Non-binary values
    y_multiclass = np.array([0, 1, 2, 0, 1, 2] * 20)
    try:
        validate_binary_target(y_multiclass, name="y_multiclass")
        print("[FAIL] Multi-class target should have failed")
        return False
    except ValidationError as e:
        print(f"[OK] Caught non-binary error")
        assert "binary" in str(e).lower()

    # Test 5: Contains NaN
    y_nan = np.array([0.0, 1.0, np.nan, 0.0, 1.0] * 20)
    try:
        validate_binary_target(y_nan, name="y_nan")
        print("[FAIL] Target with NaN should have failed")
        return False
    except ValidationError as e:
        print(f"[OK] Caught NaN in target error")
        assert "nan" in str(e).lower()

    print("\n[OK] All binary target validation tests passed\n")
    return True


def test_feature_names_validation():
    """Test feature names validation."""
    print("=" * 80)
    print("TEST 5: FEATURE NAMES VALIDATION")
    print("=" * 80)

    # Test 1: Valid feature names
    names = ["age", "income", "score"]
    validated = validate_feature_names(names, n_features=3)
    assert validated == names
    print(f"[OK] Valid feature names passed: {validated}")

    # Test 2: None (should generate defaults)
    validated = validate_feature_names(None, n_features=5)
    assert len(validated) == 5
    assert validated[0] == "feature_0"
    print(f"[OK] Generated default names: {validated}")

    # Test 3: Wrong count
    try:
        validate_feature_names(["a", "b"], n_features=3)
        print("[FAIL] Wrong count should have failed")
        return False
    except ValidationError as e:
        print(f"[OK] Caught wrong count error")
        assert "doesn't match" in str(e).lower()

    # Test 4: Duplicate names
    try:
        validate_feature_names(["age", "income", "age"], n_features=3)
        print("[FAIL] Duplicate names should have failed")
        return False
    except ValidationError as e:
        print(f"[OK] Caught duplicate names error")
        assert "duplicate" in str(e).lower()

    # Test 5: Empty names
    try:
        validate_feature_names(["age", "", "income"], n_features=3)
        print("[FAIL] Empty names should have failed")
        return False
    except ValidationError as e:
        print(f"[OK] Caught empty names error")
        assert "empty" in str(e).lower()

    print("\n[OK] All feature names validation tests passed\n")
    return True


def test_train_val_compatibility():
    """Test train/validation compatibility checks."""
    print("=" * 80)
    print("TEST 6: TRAIN/VAL COMPATIBILITY")
    print("=" * 80)

    # Test 1: Compatible sets
    X_train = np.random.rand(1000, 10)
    y_train = np.random.randint(0, 2, 1000)
    X_val = np.random.rand(300, 10)
    y_val = np.random.randint(0, 2, 300)

    try:
        validate_train_val_compatibility(X_train, y_train, X_val, y_val)
        print("[OK] Compatible train/val sets passed")
    except ValidationError as e:
        print(f"[FAIL] Compatible sets failed: {e}")
        return False

    # Test 2: Different feature counts
    X_val_wrong = np.random.rand(300, 8)  # Wrong number of features
    try:
        validate_train_val_compatibility(X_train, y_train, X_val_wrong, y_val)
        print("[FAIL] Different feature counts should have failed")
        return False
    except ValidationError as e:
        print(f"[OK] Caught feature count mismatch")
        assert "different number of features" in str(e).lower()

    # Test 3: Mismatched X/y lengths
    y_val_wrong = np.random.randint(0, 2, 200)  # Wrong length
    try:
        validate_train_val_compatibility(X_train, y_train, X_val, y_val_wrong)
        print("[FAIL] Mismatched X/y lengths should have failed")
        return False
    except ValidationError as e:
        print(f"[OK] Caught length mismatch")
        assert "mismatched lengths" in str(e).lower()

    print("\n[OK] All train/val compatibility tests passed\n")
    return True


def test_dataframe_validation():
    """Test DataFrame validation."""
    print("=" * 80)
    print("TEST 7: DATAFRAME VALIDATION")
    print("=" * 80)

    # Test 1: Valid DataFrame
    df_valid = pd.DataFrame({
        'age': np.random.randint(18, 80, 1000),
        'income': np.random.randint(20000, 150000, 1000),
        'score': np.random.rand(1000),
        'target': np.random.randint(0, 2, 1000)
    })

    report = validate_dataframe(df_valid, min_rows=100)
    print(f"[OK] Valid DataFrame passed")
    print(f"     Rows: {report.n_rows}, Cols: {report.n_cols}")
    print(f"     Warnings: {len(report.warnings)}")

    # Test 2: DataFrame with missing values
    df_missing = df_valid.copy()
    df_missing.loc[0:50, 'age'] = np.nan

    report = validate_dataframe(df_missing, max_missing_pct=0.10)
    print(f"\n[OK] DataFrame with missing values detected")
    print(f"     Missing columns: {len(report.missing_values)}")
    if report.warnings:
        print(f"     First warning: {report.warnings[0][:80]}...")

    # Test 3: Too few rows
    df_small = df_valid.head(50)
    try:
        validate_dataframe(df_small, min_rows=100)
        print("[FAIL] Small DataFrame should have failed")
        return False
    except ValidationError as e:
        print(f"\n[OK] Caught too few rows error")
        assert "too few rows" in str(e).lower()

    # Test 4: Target column validation
    target, n_pos, rate = validate_target_column(df_valid, 'target', expected_values=[0, 1])
    print(f"\n[OK] Target validation passed: {n_pos} positives ({rate:.2%})")

    # Test 5: Missing target column
    try:
        validate_target_column(df_valid, 'nonexistent_target')
        print("[FAIL] Missing target column should have failed")
        return False
    except ValidationError as e:
        print(f"[OK] Caught missing target column error")
        assert "not found" in str(e).lower()

    print("\n[OK] All DataFrame validation tests passed\n")
    return True


def test_engine_integration():
    """Test validation integration with engine."""
    print("=" * 80)
    print("TEST 8: ENGINE INTEGRATION")
    print("=" * 80)

    # Create valid data with realistic default rate (15%)
    np.random.seed(42)
    X_train = np.random.rand(1000, 5)
    y_train = np.zeros(1000, dtype=int)
    y_train[:150] = 1  # 15% default rate

    X_val = np.random.rand(300, 5)
    y_val = np.zeros(300, dtype=int)
    y_val[:45] = 1  # 15% default rate

    # Test 1: Valid data should work
    params = IRBSegmentationParams(
        max_depth=3,
        min_samples_leaf=50,
        min_defaults_per_leaf=10
    )
    engine = IRBSegmentationEngine(params)

    try:
        logger.info("Fitting engine with valid data...")
        engine.fit(X_train, y_train, X_val, y_val, feature_names=['f1', 'f2', 'f3', 'f4', 'f5'])
        print("[OK] Engine fit with valid data succeeded")
        print(f"     Created {len(np.unique(engine.segments_train_))} segments")
    except Exception as e:
        print(f"[FAIL] Engine fit failed: {e}")
        return False

    # Test 2: Predict with valid data
    try:
        segments = engine.predict(X_val)
        print(f"[OK] Prediction succeeded: {len(segments)} predictions")
    except Exception as e:
        print(f"[FAIL] Prediction failed: {e}")
        return False

    # Test 3: Invalid input to fit (wrong shape)
    X_wrong = np.random.rand(1000, 3)  # Wrong number of features
    try:
        engine2 = IRBSegmentationEngine(params)
        engine2.fit(X_wrong, y_train, X_val, y_val)
        print("[FAIL] Wrong shape should have been caught")
        return False
    except ValidationError as e:
        print(f"[OK] Engine validation caught wrong shape")
        print(f"     Error: {str(e)[:100]}...")

    # Test 4: Invalid input to predict (wrong shape)
    X_pred_wrong = np.random.rand(100, 3)  # Wrong number of features
    try:
        engine.predict(X_pred_wrong)
        print("[FAIL] Wrong prediction shape should have been caught")
        return False
    except ValidationError as e:
        print(f"[OK] Prediction validation caught wrong shape")

    # Test 5: Too few samples
    X_tiny = np.random.rand(50, 5)
    y_tiny = np.random.randint(0, 2, 50)
    try:
        engine3 = IRBSegmentationEngine(params)
        engine3.fit(X_tiny, y_tiny)
        print("[FAIL] Too few samples should have been caught")
        return False
    except ValidationError as e:
        print(f"[OK] Engine validation caught too few samples")
        assert "too few samples" in str(e).lower()

    print("\n[OK] All engine integration tests passed\n")
    return True


def test_data_quality_check():
    """Test comprehensive data quality checking."""
    print("=" * 80)
    print("TEST 9: DATA QUALITY CHECK")
    print("=" * 80)

    # Create DataFrame with various issues
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, 1000),
        'income': np.random.randint(20000, 150000, 1000),
        'constant_col': [1] * 1000,  # Constant column
        'high_cardinality': range(1000),  # High cardinality (likely an ID)
        'missing_col': [np.nan if i % 3 == 0 else i for i in range(1000)],  # Missing values
        'target': np.random.randint(0, 2, 1000)
    })

    # Add some duplicates
    df = pd.concat([df, df.head(10)])

    # Test with auto-fix
    cleaned_df, report = check_data_quality(df, target_col='target', auto_fix=True)

    print(f"[OK] Data quality check completed")
    print(f"     Original shape: {df.shape}")
    print(f"     Cleaned shape: {cleaned_df.shape}")
    print(f"     Constant columns removed: {len(report.constant_columns)}")
    print(f"     High cardinality columns removed: {len(report.high_cardinality_columns)}")
    print(f"     Duplicates removed: {report.duplicated_rows}")
    print(f"     Total warnings: {len(report.warnings)}")

    # Verify constant column was removed
    assert 'constant_col' not in cleaned_df.columns
    print(f"\n[OK] Auto-fix removed problematic columns")

    # Print full report
    print(f"\nFull Data Quality Report:")
    print(str(report))

    print("\n[OK] Data quality check test passed\n")
    return True


def main():
    """Run all tests."""
    print("\n")
    print("=" * 80)
    print("COMPREHENSIVE VALIDATION AND LOGGING TESTS")
    print("=" * 80)
    print("\n")

    tests = [
        ("Logging", test_logging),
        ("Validation Errors", test_validation_errors),
        ("Array Validation", test_array_validation),
        ("Binary Target Validation", test_binary_target_validation),
        ("Feature Names Validation", test_feature_names_validation),
        ("Train/Val Compatibility", test_train_val_compatibility),
        ("DataFrame Validation", test_dataframe_validation),
        ("Engine Integration", test_engine_integration),
        ("Data Quality Check", test_data_quality_check),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            result = test_func()
            if result is False:
                failed += 1
                print(f"[FAIL] {test_name} test failed\n")
            else:
                passed += 1
        except Exception as e:
            failed += 1
            print(f"[FAIL] {test_name} test crashed: {e}\n")
            import traceback
            traceback.print_exc()

    print("\n")
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\n[SUCCESS] All tests passed!")
    else:
        print(f"\n[WARNING] {failed} test(s) failed")

    print("=" * 80)
    print("\n")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
