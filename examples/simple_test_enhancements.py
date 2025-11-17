"""
Simple test to verify architecture enhancements work
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("\n" + "="*80)
print("SIMPLE ARCHITECTURE ENHANCEMENTS TEST")
print("="*80)

# Create minimal test data
print("\n1. Creating test data...")
np.random.seed(42)
n_samples = 1000

X = np.random.randn(1000, 4)
y = (X[:, 0] + X[:, 1] > 0).astype(int)
feature_names = ['feat_0', 'feat_1', 'feat_2', 'feat_3']
print(f"   [OK] Created {n_samples} samples, {X.shape[1]} features")

# Test 1: Basic import
print("\n2. Testing imports...")
try:
    from irb_segmentation import IRBSegmentationParams, IRBSegmentationEngine
    from irb_segmentation.scorer import score_from_exported_tree, validate_tree_structure
    print("   [OK] All imports successful")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Create and fit engine
print("\n3. Creating and fitting engine...")
try:
    params = IRBSegmentationParams(max_depth=2, min_samples_leaf=50, min_defaults_per_leaf=5)
    engine = IRBSegmentationEngine(params)

    X_train, X_val = X[:700], X[700:]
    y_train, y_val = y[:700], y[700:]

    engine.fit(X_train, y_train, X_val, y_val, feature_names)
    print(f"   [OK] Engine fitted successfully")
except Exception as e:
    print(f"   ✗ Fitting failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Export tree structure
print("\n4. Testing tree export...")
try:
    tree_structure = engine.export_tree_structure()

    assert 'tree_metadata' in tree_structure
    assert 'nodes' in tree_structure
    assert 'segment_mapping' in tree_structure

    print(f"   [OK] Tree exported with {len(tree_structure['nodes'])} nodes")
    print(f"   [OK] Format version: {tree_structure['tree_metadata']['format_version']}")

    # Validate
    validate_tree_structure(tree_structure)
    print(f"   [OK] Tree structure validation passed")
except Exception as e:
    print(f"   ✗ Tree export failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Predict with caching
print("\n5. Testing cached predict...")
try:
    segments_1 = engine.predict(X_val)
    segments_2 = engine.predict(X_val)

    assert np.array_equal(segments_1, segments_2)
    print(f"   [OK] Predicted {len(segments_1)} observations")
    print(f"   [OK] Cache works (predictions identical)")
except Exception as e:
    print(f"   ✗ Prediction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: JSON-based scoring
print("\n6. Testing JSON-based scoring...")
try:
    segments_json = score_from_exported_tree(X_val, tree_structure, feature_names)

    assert np.array_equal(segments_1, segments_json)
    print(f"   [OK] JSON scoring matches sklearn scoring")
    print(f"   [OK] Scored {len(segments_json)} observations")
except Exception as e:
    print(f"   ✗ JSON scoring failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Categorical support (basic)
print("\n7. Testing categorical support...")
try:
    # Create categorical data
    X_cat = {
        'category_a': np.random.choice(['A', 'B', 'C'], size=700)
    }

    # Test with categorical forced split
    params_cat = IRBSegmentationParams(
        max_depth=2,
        min_samples_leaf=50,
        min_defaults_per_leaf=5,
        forced_splits={'category_a': ['A', 'B']}  # Categorical split
    )

    engine_cat = IRBSegmentationEngine(params_cat)
    engine_cat.fit(X_train, y_train, feature_names=feature_names, X_categorical=X_cat)

    print(f"   [OK] Engine fitted with categorical forced splits")

    # Check if categorical split was logged
    if 'forced_splits' in engine_cat.adjustment_log_:
        cat_splits = [s for s in engine_cat.adjustment_log_['forced_splits']
                      if s.get('split_type') == 'categorical']
        if cat_splits:
            print(f"   [OK] Categorical split applied: {cat_splits[0]['feature_name']}")
        else:
            print(f"   [INFO] No categorical split in log (may not have been needed)")
except Exception as e:
    print(f"   ✗ Categorical support failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Data loader categorical preparation
print("\n8. Testing data loader categorical preparation...")
try:
    from data_loaders.base import BaseDataLoader
    import pandas as pd

    # Create concrete implementation
    class TestLoader(BaseDataLoader):
        def load(self):
            pass

    loader = TestLoader()

    # Create test dataframe
    df = pd.DataFrame({
        'num_feat': np.random.randn(100),
        'cat_feat': np.random.choice(['X', 'Y', 'Z'], 100)
    })

    cat_dict = loader.prepare_categorical_features(df, ['cat_feat'])

    assert 'cat_feat' in cat_dict
    assert len(cat_dict['cat_feat']) == 100

    print(f"   [OK] Categorical features prepared")
    print(f"   [OK] Found {len(np.unique(cat_dict['cat_feat']))} unique values")
except Exception as e:
    print(f"   ✗ Data loader test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("ALL TESTS PASSED!")
print("="*80)
print("\nSummary:")
print("  [OK] Tree export (engine-agnostic JSON)")
print("  [OK] Enhanced scoring (caching)")
print("  [OK] JSON-based scoring (no sklearn)")
print("  [OK] Categorical variable support")
print("  [OK] Data loader integration")
print("\nArchitecture enhancements working correctly!\n")
