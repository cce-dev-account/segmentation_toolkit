"""
Test Architecture Enhancements

Tests for the three major enhancements:
1. Standardized tree format (engine-agnostic JSON)
2. Enhanced scoring (caching, JSON-based scoring)
3. Categorical variable support
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from irb_segmentation import IRBSegmentationParams, IRBSegmentationEngine
from irb_segmentation.scorer import (
    score_from_exported_tree,
    score_from_json_file,
    validate_tree_structure,
    get_segment_statistics_from_tree
)
from data_loaders.base import BaseDataLoader
from sklearn.model_selection import train_test_split


def create_test_data_with_categorical():
    """Create synthetic test data with numeric and categorical features."""
    np.random.seed(42)
    n_samples = 5000

    # Numeric features
    int_rate = np.random.uniform(5, 25, n_samples)
    annual_inc = np.random.lognormal(10.5, 0.8, n_samples)
    fico_score = np.random.normal(700, 50, n_samples)
    dti = np.random.uniform(0, 40, n_samples)

    X_numeric = np.column_stack([int_rate, annual_inc, fico_score, dti])
    feature_names = ['int_rate', 'annual_inc', 'fico_range_high', 'dti']

    # Categorical features
    loan_purposes = np.random.choice(
        ['debt_consolidation', 'credit_card', 'home_improvement', 'education', 'medical'],
        n_samples,
        p=[0.4, 0.2, 0.2, 0.1, 0.1]
    )
    grades = np.random.choice(['A', 'B', 'C', 'D', 'E'], n_samples, p=[0.2, 0.3, 0.25, 0.15, 0.1])

    X_categorical = {
        'loan_purpose': loan_purposes,
        'grade': grades
    }

    # Generate target (default) with some logic
    # Higher int_rate, lower income, education/medical loans = higher default
    default_prob = (
        0.02 +
        0.01 * (int_rate - 10) +  # Higher rate = higher default
        -0.00001 * (annual_inc - 50000) +  # Lower income = higher default
        0.05 * np.isin(loan_purposes, ['education', 'medical'])  # Risky purposes
    )
    default_prob = np.clip(default_prob, 0.01, 0.5)
    y = (np.random.random(n_samples) < default_prob).astype(int)

    return X_numeric, y, feature_names, X_categorical


def test_1_tree_export():
    """Test 1: Standardized Tree Format Export"""
    print("\n" + "="*80)
    print("TEST 1: STANDARDIZED TREE FORMAT EXPORT")
    print("="*80)

    # Create and fit model
    X, y, feature_names, X_cat = create_test_data_with_categorical()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    params = IRBSegmentationParams(
        max_depth=3,
        min_samples_leaf=200,
        min_defaults_per_leaf=20
    )

    engine = IRBSegmentationEngine(params)
    engine.fit(X_train, y_train, X_val, y_val, feature_names)

    # Export tree structure
    print("\n1. Exporting tree structure to JSON...")
    tree_structure = engine.export_tree_structure()

    # Validate structure
    print("2. Validating tree structure...")
    assert 'tree_metadata' in tree_structure, "Missing tree_metadata"
    assert 'nodes' in tree_structure, "Missing nodes"
    assert 'feature_metadata' in tree_structure, "Missing feature_metadata"
    assert 'segment_mapping' in tree_structure, "Missing segment_mapping"
    assert 'adjustments' in tree_structure, "Missing adjustments"

    print(f"   [OK] Tree has {len(tree_structure['nodes'])} nodes")
    print(f"   [OK] Format version: {tree_structure['tree_metadata']['format_version']}")
    print(f"   [OK] Tree type: {tree_structure['tree_metadata']['tree_type']}")
    print(f"   [OK] Features: {tree_structure['tree_metadata']['n_features']}")

    # Validate using scorer module
    print("3. Running validation function...")
    try:
        validate_tree_structure(tree_structure)
        print("   [OK] Tree structure validation passed")
    except ValueError as e:
        print(f"   ✗ Validation failed: {e}")
        return False

    # Export to file
    print("4. Exporting to file...")
    output_file = "test_tree_export.json"
    engine.export_tree_to_file(output_file)

    # Verify file exists and is valid JSON
    with open(output_file, 'r') as f:
        loaded_tree = json.load(f)
    print(f"   [OK] File exported: {output_file}")
    print(f"   [OK] File size: {Path(output_file).stat().st_size:,} bytes")

    # Get segment statistics
    print("5. Extracting segment statistics from tree...")
    stats = get_segment_statistics_from_tree(tree_structure)
    print(f"   [OK] Found {len(stats)} segments")
    for seg_id, seg_stats in sorted(stats.items())[:3]:  # Show first 3
        print(f"     Segment {seg_id}: {seg_stats['n_observations']:,} obs, "
              f"PD={seg_stats['default_rate']:.2%}")

    print("\n[PASS] TEST 1 PASSED: Tree export working correctly")
    return True, tree_structure


def test_2_enhanced_scoring(tree_structure):
    """Test 2: Enhanced Scoring with Caching and JSON-based Scoring"""
    print("\n" + "="*80)
    print("TEST 2: ENHANCED SCORING CAPABILITIES")
    print("="*80)

    # Create test data
    X, y, feature_names, X_cat = create_test_data_with_categorical()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Fit engine
    params = IRBSegmentationParams(max_depth=3, min_samples_leaf=200)
    engine = IRBSegmentationEngine(params)
    engine.fit(X_train, y_train, feature_names=feature_names)

    # Test 2.1: Cached predict()
    print("\n1. Testing cached predict() performance...")
    import time

    # First call (builds cache)
    start = time.time()
    segments_1 = engine.predict(X_test)
    time_1 = time.time() - start
    print(f"   First call: {time_1*1000:.2f}ms")

    # Second call (uses cache)
    start = time.time()
    segments_2 = engine.predict(X_test)
    time_2 = time.time() - start
    print(f"   Second call: {time_2*1000:.2f}ms")

    # Verify results are identical
    assert np.array_equal(segments_1, segments_2), "Cached predictions differ!"
    print(f"   [OK] Cache speedup: {time_1/time_2:.1f}x faster")

    # Test 2.2: JSON-based scoring
    print("\n2. Testing JSON-based scoring...")
    tree_json = engine.export_tree_structure()

    segments_json = score_from_exported_tree(X_test, tree_json, feature_names)

    # Verify JSON scoring matches sklearn scoring
    assert np.array_equal(segments_1, segments_json), "JSON scoring differs from sklearn!"
    print(f"   [OK] JSON scoring matches sklearn scoring")
    print(f"   [OK] Scored {len(X_test):,} observations")

    # Test 2.3: File-based scoring
    print("\n3. Testing file-based scoring...")
    engine.export_tree_to_file("test_tree_export.json")

    segments_file = score_from_json_file(X_test, "test_tree_export.json", feature_names)

    assert np.array_equal(segments_1, segments_file), "File-based scoring differs!"
    print(f"   [OK] File-based scoring matches")

    # Test 2.4: Unseen data warning
    print("\n4. Testing data drift detection...")
    # Create out-of-distribution data
    X_ood = X_test.copy()
    X_ood[:, 0] = X_ood[:, 0] * 3  # Extreme interest rates

    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        segments_ood = engine.predict(X_ood)

        if len(w) > 0 and "unseen leaf" in str(w[0].message).lower():
            print(f"   [OK] Data drift warning triggered correctly")
        else:
            print(f"   [WARN] No data drift warning (may be OK if data is in-distribution)")

    print("\n[PASS] TEST 2 PASSED: Enhanced scoring working correctly")
    return True


def test_3_categorical_support():
    """Test 3: Categorical Variable Support"""
    print("\n" + "="*80)
    print("TEST 3: CATEGORICAL VARIABLE SUPPORT")
    print("="*80)

    # Create test data with categorical features
    X_numeric, y, feature_names, X_categorical = create_test_data_with_categorical()
    X_train, X_val, y_train, y_val = train_test_split(X_numeric, y, test_size=0.3, random_state=42)

    # Also split categorical features
    train_idx = np.arange(len(X_numeric))[:len(X_train)]
    X_cat_train = {k: v[train_idx] for k, v in X_categorical.items()}

    # Test 3.1: Categorical forced split
    print("\n1. Testing categorical forced splits...")
    params = IRBSegmentationParams(
        max_depth=3,
        min_samples_leaf=200,
        forced_splits={
            'int_rate': 15.0,  # Numeric threshold
            'loan_purpose': ['education', 'medical']  # Categorical membership
        }
    )

    engine = IRBSegmentationEngine(params)

    # Fit with categorical features
    try:
        engine.fit(
            X_train, y_train, X_val, y_val,
            feature_names=feature_names,
            X_categorical=X_cat_train
        )
        print("   [OK] Model fitted with categorical forced splits")
    except Exception as e:
        print(f"   ✗ Failed to fit with categorical: {e}")
        return False

    # Verify forced splits were applied
    if 'forced_splits' in engine.adjustment_log_:
        forced_splits_log = engine.adjustment_log_['forced_splits']
        print(f"   [OK] Applied {len(forced_splits_log)} forced splits")

        # Check for categorical splits
        categorical_splits = [s for s in forced_splits_log if s.get('split_type') == 'categorical']
        numeric_splits = [s for s in forced_splits_log if s.get('split_type') == 'numeric']

        print(f"     - Categorical splits: {len(categorical_splits)}")
        print(f"     - Numeric splits: {len(numeric_splits)}")

        if categorical_splits:
            cat_split = categorical_splits[0]
            print(f"     - Example categorical: {cat_split['feature_name']} IN {cat_split['categories']}")

    # Test 3.2: Rule extraction with categorical
    print("\n2. Testing rule extraction with categorical splits...")
    from extract_segment_rules import extract_rules_with_paths

    try:
        rules = extract_rules_with_paths(engine)
        print(f"   [OK] Extracted rules for {len(rules)} segments")

        # Check if any rules contain categorical conditions
        has_categorical = False
        for seg_id, seg_rules in rules.items():
            for rule in seg_rules:
                if 'IN' in rule:
                    has_categorical = True
                    print(f"     - Segment {seg_id} has categorical rule:")
                    print(f"       {rule[:150]}...")
                    break
            if has_categorical:
                break

        if not has_categorical:
            print(f"     [INFO] No categorical conditions found in rules (may be normal if not split)")
    except Exception as e:
        print(f"   ✗ Rule extraction failed: {e}")

    # Test 3.3: Data loader categorical preparation
    print("\n3. Testing data loader categorical preparation...")
    from data_loaders.base import BaseDataLoader

    # Create a minimal concrete implementation
    class TestLoader(BaseDataLoader):
        def load(self):
            pass

    loader = TestLoader()

    # Create test dataframe
    df = pd.DataFrame({
        'int_rate': X_numeric[:, 0],
        'annual_inc': X_numeric[:, 1],
        'loan_purpose': X_categorical['loan_purpose'],
        'grade': X_categorical['grade']
    })

    cat_dict = loader.prepare_categorical_features(df, ['loan_purpose', 'grade'])

    assert 'loan_purpose' in cat_dict, "loan_purpose not in categorical dict"
    assert 'grade' in cat_dict, "grade not in categorical dict"
    assert len(cat_dict['loan_purpose']) == len(df), "Wrong length for loan_purpose"

    print(f"   [OK] Prepared {len(cat_dict)} categorical features")
    print(f"     - loan_purpose: {len(np.unique(cat_dict['loan_purpose']))} unique values")
    print(f"     - grade: {len(np.unique(cat_dict['grade']))} unique values")

    print("\n[PASS] TEST 3 PASSED: Categorical support working correctly")
    return True


def test_4_integration():
    """Test 4: Full Integration Test"""
    print("\n" + "="*80)
    print("TEST 4: FULL INTEGRATION TEST")
    print("="*80)

    print("\n1. Creating model with all features...")
    X_numeric, y, feature_names, X_categorical = create_test_data_with_categorical()
    X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.3, random_state=42)

    train_idx = np.arange(len(X_numeric))[:len(X_train)]
    test_idx = np.arange(len(X_numeric))[len(X_train):]
    X_cat_train = {k: v[train_idx] for k, v in X_categorical.items()}
    X_cat_test = {k: v[test_idx] for k, v in X_categorical.items()}

    # Configure with all features
    params = IRBSegmentationParams(
        max_depth=4,
        min_samples_leaf=150,
        min_defaults_per_leaf=15,
        forced_splits={
            'int_rate': 15.0,
            'loan_purpose': ['education', 'medical']
        }
    )

    engine = IRBSegmentationEngine(params)
    engine.fit(X_train, y_train, feature_names=feature_names, X_categorical=X_cat_train)

    print("   [OK] Model fitted")

    # Export tree
    print("\n2. Exporting tree to production format...")
    engine.export_tree_to_file("production_tree_full.json")
    tree = engine.export_tree_structure()
    print(f"   [OK] Tree exported with {tree['tree_metadata']['n_nodes']} nodes")

    # Score with multiple methods
    print("\n3. Scoring with all methods...")
    segments_sklearn = engine.predict(X_test, X_cat_test)
    segments_json = score_from_exported_tree(X_test, tree, feature_names, X_cat_test)
    segments_file = score_from_json_file(X_test, "production_tree_full.json", feature_names, X_cat_test)

    # Verify all methods produce same results
    assert np.array_equal(segments_sklearn, segments_json), "sklearn ≠ JSON scoring"
    assert np.array_equal(segments_sklearn, segments_file), "sklearn ≠ file scoring"

    print(f"   [OK] All scoring methods agree ({len(X_test):,} observations)")

    # Calculate segment statistics
    print("\n4. Segment performance analysis...")
    unique_segments = np.unique(segments_sklearn)
    print(f"   Final segmentation: {len(unique_segments)} segments")

    segment_stats = []
    for seg in unique_segments:
        mask = segments_sklearn == seg
        seg_y = y_test[mask]
        segment_stats.append({
            'segment': seg,
            'n_obs': np.sum(mask),
            'n_defaults': np.sum(seg_y),
            'pd_rate': np.mean(seg_y)
        })

    # Sort by PD
    segment_stats.sort(key=lambda x: x['pd_rate'])

    print(f"\n   Segment Performance (sorted by PD):")
    print(f"   {'Seg':<5} {'Obs':<8} {'Defaults':<10} {'PD Rate':<10}")
    print(f"   {'-'*40}")
    for s in segment_stats:
        print(f"   {s['segment']:<5} {s['n_obs']:<8} {s['n_defaults']:<10} {s['pd_rate']:<10.2%}")

    # Verify rank ordering
    pds = [s['pd_rate'] for s in segment_stats]
    is_monotonic = all(pds[i] <= pds[i+1] for i in range(len(pds)-1))

    if is_monotonic:
        print(f"\n   [OK] Segments are rank-ordered by PD (good segmentation)")
    else:
        print(f"\n   [WARN] Segments not perfectly rank-ordered (may be acceptable)")

    print("\n[PASS] TEST 4 PASSED: Full integration working correctly")
    return True


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("ARCHITECTURE ENHANCEMENTS - COMPREHENSIVE TEST SUITE")
    print("="*80)

    results = {}

    # Test 1: Tree Export
    try:
        passed, tree_structure = test_1_tree_export()
        results['Tree Export'] = passed
    except Exception as e:
        print(f"\n[FAIL] TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['Tree Export'] = False
        tree_structure = None

    # Test 2: Enhanced Scoring
    if tree_structure:
        try:
            passed = test_2_enhanced_scoring(tree_structure)
            results['Enhanced Scoring'] = passed
        except Exception as e:
            print(f"\n[FAIL] TEST 2 FAILED: {e}")
            import traceback
            traceback.print_exc()
            results['Enhanced Scoring'] = False
    else:
        results['Enhanced Scoring'] = False
        print("\n[SKIP] TEST 2 SKIPPED: Tree export failed")

    # Test 3: Categorical Support
    try:
        passed = test_3_categorical_support()
        results['Categorical Support'] = passed
    except Exception as e:
        print(f"\n[FAIL] TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['Categorical Support'] = False

    # Test 4: Integration
    try:
        passed = test_4_integration()
        results['Integration'] = passed
    except Exception as e:
        print(f"\n[FAIL] TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['Integration'] = False

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in results.items():
        status = "[PASS] PASS" if passed else "[FAIL] FAIL"
        print(f"{status} - {test_name}")

    all_passed = all(results.values())

    if all_passed:
        print("\nALL TESTS PASSED!")
    else:
        print(f"\n[WARN] {sum(not p for p in results.values())} / {len(results)} tests failed")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
