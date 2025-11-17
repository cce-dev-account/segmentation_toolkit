"""
IRB Segmentation Engine

Main engine for creating and validating IRB PD model segments.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from typing import Dict, List, Optional, Tuple
import warnings
import json
from datetime import datetime

from .params import IRBSegmentationParams
from .validators import SegmentValidator
from .adjustments import SegmentAdjuster


class IRBSegmentationEngine:
    """
    Main engine for IRB PD model segmentation.

    Uses sklearn DecisionTreeClassifier as the core implementation, with
    post-processing to enforce IRB regulatory and business constraints.

    Example:
        >>> params = IRBSegmentationParams(
        ...     max_depth=3,
        ...     min_defaults_per_leaf=30,
        ...     min_segment_density=0.10
        ... )
        >>> engine = IRBSegmentationEngine(params)
        >>> engine.fit(X_train, y_train, X_val, y_val)
        >>> segments = engine.get_production_segments()
    """

    def __init__(self, params: IRBSegmentationParams):
        """
        Initialize the segmentation engine.

        Args:
            params: IRBSegmentationParams object with all configuration
        """
        self.params = params
        self.tree_model = DecisionTreeClassifier(**params.to_sklearn_params())

        # Storage for results and logs
        self.segments_train_ = None
        self.segments_val_ = None
        self.X_train_ = None
        self.y_train_ = None
        self.X_val_ = None
        self.y_val_ = None
        self.feature_names_ = None
        self.X_categorical_ = None  # Store categorical features separately
        self._leaf_to_segment_cache = None  # Cache for predict() optimization
        self.is_fitted_ = False

        # Logs for audit trail
        self.adjustment_log_ = {
            'merges': [],
            'splits': [],
            'forced_splits': [],
            'monotonicity_violations': []
        }
        self.validation_results_ = {}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        X_categorical: Optional[Dict[str, np.ndarray]] = None,
        max_adjustment_iterations: int = 5
    ):
        """
        Fit the segmentation model and apply IRB constraints.

        Args:
            X: Training feature matrix (numeric features only)
            y: Training binary outcomes (0/1)
            X_val: Optional validation feature matrix
            y_val: Optional validation binary outcomes
            feature_names: Optional list of feature names
            X_categorical: Optional dict mapping feature names to categorical arrays
            max_adjustment_iterations: Maximum iterations for constraint enforcement

        Returns:
            self
        """
        # Store data
        self.X_train_ = X
        self.y_train_ = y
        self.X_val_ = X_val
        self.y_val_ = y_val
        self.feature_names_ = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        self.X_categorical_ = X_categorical

        print(f"Training segmentation model on {len(X)} observations...")
        print(f"  Default rate: {np.mean(y):.4f}")
        print(f"  Features: {X.shape[1]}")

        # Step 1: Train sklearn tree
        self.tree_model.fit(X, y)
        self.segments_train_ = self.tree_model.apply(X)  # Get leaf node indices
        segments = self.segments_train_.copy()

        print(f"\nInitial tree created {len(np.unique(self.segments_train_))} segments")

        # Step 2: Apply business constraints
        segments = self._apply_business_constraints(X, self.segments_train_.copy(), y)

        # Step 3: Iteratively adjust segments to meet IRB requirements
        segments = self._enforce_irb_constraints(
            X, segments, y, max_iterations=max_adjustment_iterations
        )

        # Step 4: Final validation
        print("\nRunning final validation...")
        self.segments_train_ = segments
        self.is_fitted_ = True  # Mark as fitted before calling predict

        self.validation_results_['train'] = SegmentValidator.run_all_validations(
            self.segments_train_, y, self.params
        )

        # Validate on validation set if provided
        if X_val is not None and y_val is not None:
            self.segments_val_ = self.predict(X_val)
            self.validation_results_['validation'] = SegmentValidator.run_all_validations(
                self.segments_val_, y_val, self.params, reference_segments=self.segments_train_
            )

        # Print summary
        self._print_fit_summary()

        return self

    def _apply_business_constraints(
        self,
        X: np.ndarray,
        segments: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """Apply forced splits and other business rules."""
        if not self.params.forced_splits:
            return segments

        print("\nApplying forced splits...")
        # Map feature names to indices if needed
        forced_splits_indexed = {}
        for feature_name, threshold in self.params.forced_splits.items():
            if isinstance(feature_name, str):
                try:
                    feature_idx = self.feature_names_.index(feature_name)
                except ValueError:
                    warnings.warn(f"Feature '{feature_name}' not found, skipping forced split")
                    continue
            else:
                feature_idx = feature_name
            forced_splits_indexed[feature_idx] = threshold

        if forced_splits_indexed:
            segments, split_log = SegmentAdjuster.apply_forced_splits(
                X, segments, forced_splits_indexed, self.feature_names_, self.X_categorical_
            )
            self.adjustment_log_['forced_splits'] = split_log
            print(f"  Applied {len(split_log)} forced splits")

        return segments

    def _enforce_irb_constraints(
        self,
        X: np.ndarray,
        segments: np.ndarray,
        y: np.ndarray,
        max_iterations: int = 5
    ) -> np.ndarray:
        """
        Iteratively adjust segments to meet IRB requirements.

        Args:
            X: Feature matrix
            segments: Current segment labels
            y: Binary outcomes
            max_iterations: Maximum adjustment iterations

        Returns:
            Adjusted segments
        """
        print("\nEnforcing IRB constraints...")

        for iteration in range(max_iterations):
            print(f"\n  Iteration {iteration + 1}/{max_iterations}")

            # Check current state
            validation = SegmentValidator.run_all_validations(
                segments, y, self.params
            )

            if validation['all_passed']:
                print("  [PASS] All constraints satisfied")
                break

            # Merge small segments
            density_issues = validation['validations'].get('density', {})
            defaults_issues = validation['validations'].get('min_defaults', {})

            needs_merge = (
                not density_issues.get('passed', True) or
                not defaults_issues.get('passed', True)
            )

            if needs_merge:
                print("  Merging small segments...")
                segments, merge_log = SegmentAdjuster.merge_small_segments(
                    X, segments, y,
                    self.params.min_segment_density,
                    self.params.min_defaults_per_leaf
                )
                self.adjustment_log_['merges'].extend(merge_log)
                print(f"    Merged {len(merge_log)} segments")

            # Split large segments if needed
            if not density_issues.get('passed', True):
                large_segments = [
                    s for s in density_issues.get('failed_segments', [])
                    if s['violation'] == 'too_large'
                ]
                if large_segments:
                    print("  Splitting large segments...")
                    segments, split_log = SegmentAdjuster.split_large_segments(
                        X, segments, y,
                        self.params.max_segment_density,
                        self.feature_names_
                    )
                    self.adjustment_log_['splits'].extend(split_log)
                    print(f"    Split {len(split_log)} segments")

        else:
            warnings.warn(
                f"Could not satisfy all IRB constraints after {max_iterations} iterations"
            )

        # Check monotonicity (log violations but don't auto-fix)
        if self.params.monotone_constraints:
            print("\nChecking monotonicity constraints...")
            monotone_constraints_indexed = {}
            for feature_name, direction in self.params.monotone_constraints.items():
                if isinstance(feature_name, str):
                    try:
                        feature_idx = self.feature_names_.index(feature_name)
                    except ValueError:
                        warnings.warn(f"Feature '{feature_name}' not found")
                        continue
                else:
                    feature_idx = feature_name
                monotone_constraints_indexed[feature_idx] = direction

            _, violation_log = SegmentAdjuster.enforce_monotonicity(
                X, segments, y, monotone_constraints_indexed, self.feature_names_
            )
            self.adjustment_log_['monotonicity_violations'] = violation_log

            if violation_log:
                print(f"  [WARN] Found {len(violation_log)} monotonicity violations")
            else:
                print("  [PASS] All monotonicity constraints satisfied")

        return segments

    def predict(self, X: np.ndarray, X_categorical: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """
        Predict segment labels for new data.

        Args:
            X: Feature matrix (numeric features)
            X_categorical: Optional dict of categorical features (for categorical forced splits)

        Returns:
            Segment labels
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction")

        # Use cached mapping if available
        if self._leaf_to_segment_cache is None:
            # Build and cache mapping from training leaf nodes to final segments
            self._leaf_to_segment_cache = {}
            train_leaves = self.tree_model.apply(self.X_train_)

            for leaf, segment in zip(train_leaves, self.segments_train_):
                self._leaf_to_segment_cache[int(leaf)] = int(segment)

        # Use tree to get leaf nodes
        leaf_nodes = self.tree_model.apply(X)

        # Apply mapping with fallback
        segments = np.array([
            self._leaf_to_segment_cache.get(int(leaf), 0)
            for leaf in leaf_nodes
        ])

        # Warn if any unseen leaves were encountered
        unseen_leaves = set(leaf_nodes) - set(self._leaf_to_segment_cache.keys())
        if unseen_leaves:
            warnings.warn(
                f"Encountered {len(unseen_leaves)} unseen leaf nodes in prediction. "
                f"These were assigned to segment 0. This may indicate data distribution shift."
            )

        return segments

    def get_production_segments(self) -> np.ndarray:
        """
        Get validated, production-ready segments for training data.

        Returns:
            Segment labels for training data

        Raises:
            RuntimeError: If model is not fitted
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before getting production segments")

        return self.segments_train_.copy()

    def get_validation_report(self) -> Dict:
        """
        Generate comprehensive validation report.

        Returns:
            Dictionary with validation results, adjustment logs, and segment statistics
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before generating report")

        report = {
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'max_depth': self.params.max_depth,
                'min_samples_leaf': self.params.min_samples_leaf,
                'min_defaults_per_leaf': self.params.min_defaults_per_leaf,
                'min_segment_density': self.params.min_segment_density,
                'max_segment_density': self.params.max_segment_density,
            },
            'segments': {
                'n_segments': len(np.unique(self.segments_train_)),
                'train_size': len(self.segments_train_),
            },
            'validation_results': self.validation_results_,
            'adjustments': self.adjustment_log_,
            'segment_statistics': self._get_segment_statistics()
        }

        if self.X_val_ is not None:
            report['segments']['validation_size'] = len(self.segments_val_)

        return report

    def _get_segment_statistics(self) -> Dict:
        """Calculate detailed statistics for each segment."""
        unique_segments = np.unique(self.segments_train_)
        stats = {}

        for seg in unique_segments:
            mask = self.segments_train_ == seg
            seg_y = self.y_train_[mask]

            stats[int(seg)] = {
                'n_observations': int(np.sum(mask)),
                'n_defaults': int(np.sum(seg_y)),
                'default_rate': float(np.mean(seg_y)),
                'density': float(np.sum(mask) / len(self.segments_train_))
            }

        return stats

    def _print_fit_summary(self):
        """Print a summary of the fitting process."""
        print("\n" + "=" * 60)
        print("SEGMENTATION SUMMARY")
        print("=" * 60)

        print(f"\nFinal segments: {len(np.unique(self.segments_train_))}")

        # Print segment statistics
        stats = self._get_segment_statistics()
        print("\nSegment Statistics:")
        print(f"{'Segment':<10}{'Count':<10}{'Defaults':<12}{'PD Rate':<12}{'Density':<10}")
        print("-" * 60)

        for seg in sorted(stats.keys()):
            s = stats[seg]
            print(
                f"{seg:<10}{s['n_observations']:<10}{s['n_defaults']:<12}"
                f"{s['default_rate']:<12.4f}{s['density']:<10.2%}"
            )

        # Print validation summary
        print("\nValidation Results:")
        val_train = self.validation_results_.get('train', {})
        if val_train.get('all_passed'):
            print("  [PASS] All training validations passed")
        else:
            print("  [FAIL] Some training validations failed:")
            for test_name, result in val_train.get('validations', {}).items():
                if not result.get('passed', True):
                    print(f"    - {test_name}")

        if 'validation' in self.validation_results_:
            val_val = self.validation_results_['validation']
            if val_val.get('all_passed'):
                print("  [PASS] All validation set checks passed")
            else:
                print("  [FAIL] Some validation set checks failed")

        # Print adjustment summary
        print("\nAdjustments Applied:")
        print(f"  Merges: {len(self.adjustment_log_['merges'])}")
        print(f"  Splits: {len(self.adjustment_log_['splits'])}")
        print(f"  Forced splits: {len(self.adjustment_log_['forced_splits'])}")
        print(f"  Monotonicity violations: {len(self.adjustment_log_['monotonicity_violations'])}")

        print("\n" + "=" * 60)

    def export_report(self, filepath: str):
        """
        Export validation report to JSON file.

        Args:
            filepath: Path to save the JSON report
        """
        report = self.get_validation_report()

        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj

        report = convert_numpy(report)

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Report exported to: {filepath}")

    def get_segment_rules(self) -> List[str]:
        """
        Extract human-readable rules for each segment.

        Returns:
            List of rule strings describing segment definitions
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before extracting rules")

        from sklearn.tree import _tree

        tree = self.tree_model.tree_
        feature_names = self.feature_names_
        rules = []

        def recurse(node, rule_str=""):
            if tree.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_names[tree.feature[node]]
                threshold = tree.threshold[node]

                # Left child (<=)
                left_rule = f"{rule_str} AND {name} <= {threshold:.4f}" if rule_str else f"{name} <= {threshold:.4f}"
                recurse(tree.children_left[node], left_rule)

                # Right child (>)
                right_rule = f"{rule_str} AND {name} > {threshold:.4f}" if rule_str else f"{name} > {threshold:.4f}"
                recurse(tree.children_right[node], right_rule)
            else:
                # Leaf node
                rules.append(f"IF {rule_str} THEN Segment {node}")

        recurse(0)
        return rules

    def export_tree_structure(self) -> Dict:
        """
        Export tree structure in engine-agnostic JSON format.

        This format allows the tree to be reconstructed without sklearn dependencies,
        enabling engine swapping and platform-independent scoring.

        Returns:
            Dictionary with standardized tree representation including:
            - nodes: List of split and leaf nodes
            - feature_metadata: Feature names and types
            - segment_mapping: Maps tree leaves to final segments (after IRB adjustments)
            - adjustments: Log of all post-tree modifications
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before exporting tree structure")

        from sklearn.tree import _tree

        tree = self.tree_model.tree_
        feature_names = self.feature_names_

        # Build leaf-to-segment mapping (accounts for merges/splits)
        train_leaves = self.tree_model.apply(self.X_train_)
        leaf_to_segment = {}
        for leaf, segment in zip(train_leaves, self.segments_train_):
            leaf_to_segment[int(leaf)] = int(segment)

        nodes = []

        def recurse(node_id):
            if tree.feature[node_id] != _tree.TREE_UNDEFINED:
                # Internal node (split)
                feature_idx = tree.feature[node_id]
                nodes.append({
                    "id": int(node_id),
                    "type": "split",
                    "feature": feature_names[feature_idx],
                    "feature_index": int(feature_idx),
                    "threshold": float(tree.threshold[node_id]),
                    "left_child": int(tree.children_left[node_id]),
                    "right_child": int(tree.children_right[node_id]),
                    "n_samples": int(tree.n_node_samples[node_id]),
                    "impurity": float(tree.impurity[node_id])
                })

                # Recurse on children
                recurse(tree.children_left[node_id])
                recurse(tree.children_right[node_id])
            else:
                # Leaf node
                final_segment = leaf_to_segment.get(int(node_id), 0)
                nodes.append({
                    "id": int(node_id),
                    "type": "leaf",
                    "segment": final_segment,
                    "n_samples": int(tree.n_node_samples[node_id]),
                    "impurity": float(tree.impurity[node_id]),
                    "value": tree.value[node_id].tolist()
                })

        recurse(0)

        # Build complete structure
        structure = {
            "tree_metadata": {
                "format_version": "1.0",
                "tree_type": "decision_tree",
                "task": "irb_segmentation",
                "n_features": len(feature_names),
                "n_nodes": len(nodes),
                "max_depth": int(tree.max_depth)
            },
            "nodes": nodes,
            "feature_metadata": {
                "names": feature_names,
                "n_features": len(feature_names)
            },
            "segment_mapping": {
                "leaf_to_segment": leaf_to_segment,
                "n_segments": len(np.unique(self.segments_train_))
            },
            "adjustments": self.adjustment_log_,
            "parameters": {
                "max_depth": self.params.max_depth,
                "min_samples_leaf": self.params.min_samples_leaf,
                "min_defaults_per_leaf": self.params.min_defaults_per_leaf,
                "min_segment_density": self.params.min_segment_density,
                "max_segment_density": self.params.max_segment_density,
                "forced_splits": self.params.forced_splits,
                "monotone_constraints": self.params.monotone_constraints
            }
        }

        return structure

    def export_tree_to_file(self, filepath: str):
        """
        Export tree structure to JSON file.

        Args:
            filepath: Path to save JSON file
        """
        structure = self.export_tree_structure()

        with open(filepath, 'w') as f:
            json.dump(structure, f, indent=2)

        print(f"Tree structure exported to: {filepath}")
