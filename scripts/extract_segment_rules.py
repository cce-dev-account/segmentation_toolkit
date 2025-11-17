"""
Extract Segment Rules from Fitted Model

Shows the actual decision tree rules that define each segment.
"""

import numpy as np
import pandas as pd
import json
import sys
from pathlib import Path
from sklearn.tree import _tree

sys.path.insert(0, str(Path(__file__).parent))

from irb_segmentation import IRBSegmentationParams, IRBSegmentationEngine
from sklearn.model_selection import train_test_split


def load_and_fit_model():
    """Load data and fit model to extract rules."""
    print("\nLoading Lending Club data to extract segment rules...")

    df = pd.read_csv("data/lending_club_test.csv", low_memory=False)
    print(f"Loaded {len(df):,} rows")

    # Create target
    df['default'] = df['loan_status'].apply(
        lambda x: 1 if 'Charged Off' in str(x) or 'Default' in str(x) else 0
    )

    # Select numerical features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'default'][:14]

    X = df[numeric_cols].fillna(0).values
    y = df['default'].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Fit model with same parameters
    params = IRBSegmentationParams(
        max_depth=5,
        min_samples_split=20000,
        min_samples_leaf=10000,
        min_defaults_per_leaf=500,
        min_segment_density=0.05,
        max_segment_density=0.40
    )

    print("\nFitting model...")
    engine = IRBSegmentationEngine(params)
    engine.fit(X_train, y_train, X_val, y_val, feature_names=numeric_cols)

    return engine


def extract_rules_with_paths(engine):
    """
    Extract complete path rules for each segment.

    Returns:
        Dict mapping segment_id -> list of rules (path from root to leaf)
    """
    tree = engine.tree_model.tree_
    feature_names = engine.feature_names_

    # Map leaf nodes to final segments
    train_leaves = engine.tree_model.apply(engine.X_train_)
    leaf_to_segment = {}
    for leaf, segment in zip(train_leaves, engine.segments_train_):
        leaf_to_segment[leaf] = segment

    # Extract paths
    segment_rules = {}

    def recurse(node, path=[]):
        if tree.feature[node] != _tree.TREE_UNDEFINED:
            # Internal node
            feature = feature_names[tree.feature[node]]
            threshold = tree.threshold[node]

            # Left child (<=)
            left_path = path + [(feature, "<=", threshold, "numeric")]
            recurse(tree.children_left[node], left_path)

            # Right child (>)
            right_path = path + [(feature, ">", threshold, "numeric")]
            recurse(tree.children_right[node], right_path)
        else:
            # Leaf node
            segment = leaf_to_segment.get(node, -1)
            if segment not in segment_rules:
                segment_rules[segment] = []
            segment_rules[segment].append(path)

    recurse(0)

    # Add categorical splits from forced_splits log
    categorical_splits = {}
    if hasattr(engine, 'adjustment_log_') and 'forced_splits' in engine.adjustment_log_:
        for split in engine.adjustment_log_['forced_splits']:
            if split.get('split_type') == 'categorical':
                seg = split['segment']
                new_seg = split['new_segment']
                feature = split['feature_name']
                categories = split['categories']

                # Add categorical condition to the new segment
                if new_seg not in categorical_splits:
                    categorical_splits[new_seg] = []
                categorical_splits[new_seg].append((feature, "IN", categories, "categorical"))

    # Convert to readable format
    readable_rules = {}
    for seg_id, paths in segment_rules.items():
        readable_rules[seg_id] = []
        for path in paths:
            rule_parts = []
            for item in path:
                if len(item) == 4:
                    feature, op, value, rule_type = item
                    if rule_type == "categorical":
                        # Format categorical rule
                        rule_parts.append(f"{feature} IN {value}")
                    else:
                        # Format numeric rule
                        rule_parts.append(f"{feature} {op} {value:.2f}")
                else:
                    # Legacy format (3-tuple)
                    feature, op, threshold = item
                    rule_parts.append(f"{feature} {op} {threshold:.2f}")

            # Add categorical splits if any
            if seg_id in categorical_splits:
                for feature, op, categories, rule_type in categorical_splits[seg_id]:
                    rule_parts.append(f"{feature} IN {categories}")

            readable_rules[seg_id].append(" AND ".join(rule_parts) if rule_parts else "All observations")

    return readable_rules


def get_segment_thresholds(engine):
    """
    Extract unique thresholds for each feature used in the tree.

    Returns:
        Dict mapping feature_name -> list of thresholds
    """
    tree = engine.tree_model.tree_
    feature_names = engine.feature_names_

    thresholds = {}

    def recurse(node):
        if tree.feature[node] != _tree.TREE_UNDEFINED:
            feature_idx = tree.feature[node]
            feature = feature_names[feature_idx]
            threshold = tree.threshold[node]

            if feature not in thresholds:
                thresholds[feature] = set()
            thresholds[feature].add(threshold)

            recurse(tree.children_left[node])
            recurse(tree.children_right[node])

    recurse(0)

    # Convert sets to sorted lists
    return {k: sorted(list(v)) for k, v in thresholds.items()}


def main():
    """Extract and save segment rules."""
    print("\n" + "=" * 80)
    print("SEGMENT RULE EXTRACTION")
    print("=" * 80)

    # Fit model
    engine = load_and_fit_model()

    # Extract rules
    print("\nExtracting segment rules...")
    rules = extract_rules_with_paths(engine)

    # Extract thresholds
    print("Extracting feature thresholds...")
    thresholds = get_segment_thresholds(engine)

    # Get segment statistics
    stats = engine._get_segment_statistics()

    # Combine everything
    output = {
        "segment_rules": {},
        "feature_thresholds": thresholds,
        "segment_statistics": stats,
        "feature_names": engine.feature_names_
    }

    # Format rules nicely
    for seg_id in sorted(rules.keys()):
        seg_stats = stats.get(seg_id) or stats.get(str(seg_id))
        output["segment_rules"][str(seg_id)] = {
            "rules": rules[seg_id],
            "statistics": seg_stats,
            "description": format_segment_description(seg_id, rules[seg_id], seg_stats)
        }

    # Save to JSON
    with open("segment_rules_detailed.json", 'w') as f:
        json.dump(output, f, indent=2)

    print("\n[OK] Rules saved to: segment_rules_detailed.json")

    # Print summary
    print("\n" + "=" * 80)
    print("SEGMENT RULES SUMMARY")
    print("=" * 80)

    for seg_id in sorted(rules.keys()):
        s = stats[seg_id]
        print(f"\n{'='*80}")
        print(f"SEGMENT {seg_id}: {s['n_observations']:,} obs, {s['default_rate']:.2%} PD")
        print(f"{'='*80}")

        for i, rule in enumerate(rules[seg_id][:3], 1):  # Show first 3 paths
            if rule:
                print(f"{i}. IF {rule}")
            else:
                print(f"{i}. All observations")

        if len(rules[seg_id]) > 3:
            print(f"   ... and {len(rules[seg_id]) - 3} more paths")

    print("\n" + "=" * 80)
    print("FEATURE THRESHOLDS")
    print("=" * 80)

    for feature, values in sorted(thresholds.items()):
        print(f"\n{feature}:")
        for val in values:
            print(f"  - {val:.2f}")

    return output


def format_segment_description(seg_id, rules, stats):
    """Create human-readable segment description."""
    dr = stats['default_rate']

    if dr < 0.05:
        risk = "Very Low Risk"
    elif dr < 0.10:
        risk = "Low Risk"
    elif dr < 0.15:
        risk = "Medium Risk"
    elif dr < 0.20:
        risk = "High Risk"
    else:
        risk = "Very High Risk"

    # Get most common conditions
    all_conditions = []
    for rule in rules:
        if rule:
            all_conditions.extend(rule.split(" AND "))

    # Count feature mentions
    feature_counts = {}
    for cond in all_conditions:
        for part in cond.split():
            if part in ['<=', '>', '<', '>=']:
                continue
            feature_counts[part] = feature_counts.get(part, 0) + 1

    top_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:3]

    desc = f"{risk} segment with {stats['n_observations']:,} observations ({stats['default_rate']:.2%} default rate). "
    if top_features:
        desc += f"Key features: {', '.join([f[0] for f in top_features])}."

    return desc


if __name__ == "__main__":
    main()
