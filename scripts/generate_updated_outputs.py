"""
Generate Updated Outputs with Architecture Enhancements

This script demonstrates the new features:
1. Tree export in engine-agnostic JSON format
2. Simplified segment rules (no redundant conditions)
3. Updated Excel templates
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from irb_segmentation import IRBSegmentationParams, IRBSegmentationEngine
from sklearn.model_selection import train_test_split

print("\n" + "="*80)
print("GENERATING UPDATED OUTPUTS WITH ARCHITECTURE ENHANCEMENTS")
print("="*80)

# Load Lending Club data
print("\n1. Loading Lending Club data...")
df = pd.read_csv("data/lending_club_test.csv", low_memory=False)
print(f"   Loaded {len(df):,} rows")

# Create target
df['default'] = df['loan_status'].apply(
    lambda x: 1 if 'Charged Off' in str(x) or 'Default' in str(x) else 0
)

# Select numerical features
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != 'default'][:14]

print(f"   Using {len(numeric_cols)} numeric features")

X = df[numeric_cols].fillna(0).values
y = df['default'].values

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Fit model
print("\n2. Fitting IRB segmentation model...")
params = IRBSegmentationParams(
    max_depth=5,
    min_samples_split=20000,
    min_samples_leaf=10000,
    min_defaults_per_leaf=500,
    min_segment_density=0.05,
    max_segment_density=0.40
)

engine = IRBSegmentationEngine(params)
engine.fit(X_train, y_train, X_val, y_val, feature_names=numeric_cols)

print(f"   Model fitted with {len(np.unique(engine.segments_train_))} segments")

# Generate Output 1: Engine-Agnostic Tree Export
print("\n3. Exporting tree in engine-agnostic JSON format...")
engine.export_tree_to_file("tree_structure_v2.json")
print("   [OK] Exported: tree_structure_v2.json")

tree = engine.export_tree_structure()
print(f"   - Format version: {tree['tree_metadata']['format_version']}")
print(f"   - Tree type: {tree['tree_metadata']['tree_type']}")
print(f"   - Nodes: {tree['tree_metadata']['n_nodes']}")
print(f"   - Max depth: {tree['tree_metadata']['max_depth']}")
print(f"   - Segments: {tree['segment_mapping']['n_segments']}")

# Generate Output 2: Simplified Segment Rules
print("\n4. Extracting simplified segment rules...")
from extract_segment_rules import extract_rules_with_paths, get_segment_thresholds
import json

rules = extract_rules_with_paths(engine)
thresholds = get_segment_thresholds(engine)
stats = engine._get_segment_statistics()

output = {
    "segment_rules": {},
    "feature_thresholds": thresholds,
    "segment_statistics": stats,
    "feature_names": numeric_cols
}

# Format rules (these are now simplified by extract_segment_rules.py)
for seg_id in sorted(rules.keys()):
    seg_stats = stats.get(seg_id) or stats.get(str(seg_id))
    output["segment_rules"][str(seg_id)] = {
        "rules": rules[seg_id],
        "statistics": seg_stats
    }

with open("segment_rules_v2_simplified.json", 'w') as f:
    json.dump(output, f, indent=2)

print("   [OK] Exported: segment_rules_v2_simplified.json")

# Show example of simplification
print("\n5. Demonstrating rule simplification:")
if rules:
    first_seg = sorted(rules.keys())[0]
    first_rule = rules[first_seg][0] if rules[first_seg] else "No rules"
    print(f"   Segment {first_seg} - First path:")
    print(f"   {first_rule[:200]}...")

# Generate Output 3: Updated Excel Template
print("\n6. Generating updated Excel template with simplified rules...")
try:
    from interfaces.create_excel_template import create_excel_template

    create_excel_template(
        segment_stats=stats,
        feature_thresholds=thresholds,
        rules_data=output,
        output_file="modification_template_v5_enhanced.xlsx"
    )
    print("   [OK] Exported: modification_template_v5_enhanced.xlsx")
    print("   - Contains simplified segment rules (no redundant conditions)")
    print("   - Ready for categorical split display")
except Exception as e:
    print(f"   [WARN] Excel template generation failed: {e}")
    print("   (This may be due to openpyxl not being installed)")

# Generate Output 4: Validation Report
print("\n7. Exporting validation report...")
engine.export_report("validation_report_v2.json")
print("   [OK] Exported: validation_report_v2.json")

# Generate Output 5: Test JSON Scoring
print("\n8. Testing JSON-based scoring...")
from irb_segmentation.scorer import score_from_json_file, validate_tree_structure

# Validate tree structure
try:
    validate_tree_structure(tree)
    print("   [OK] Tree structure validation passed")
except Exception as e:
    print(f"   [FAIL] Tree validation failed: {e}")

# Score using JSON (no sklearn)
X_test = X_val[:1000]  # Score first 1000 observations
segments_json = score_from_json_file(X_test, "tree_structure_v2.json", numeric_cols)
segments_sklearn = engine.predict(X_test)

# Verify they match
if np.array_equal(segments_json, segments_sklearn):
    print(f"   [OK] JSON scoring matches sklearn ({len(X_test):,} observations)")
else:
    print(f"   [FAIL] JSON scoring differs from sklearn!")

# Summary
print("\n" + "="*80)
print("OUTPUTS GENERATED SUCCESSFULLY")
print("="*80)
print("\nNew Output Files:")
print("  1. tree_structure_v2.json              - Engine-agnostic tree format")
print("  2. segment_rules_v2_simplified.json    - Simplified segment rules")
print("  3. modification_template_v5_enhanced.xlsx - Excel with simplified conditions")
print("  4. validation_report_v2.json           - Full validation report")
print("\nKey Improvements:")
print("  - Rules are simplified (no redundant conditions)")
print("  - Tree can be scored without sklearn")
print("  - Excel template ready for categorical features")
print("  - Full audit trail in JSON format")
print("\n" + "="*80)
