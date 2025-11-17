"""
Standalone Scoring Module

Scores observations using exported tree structure (JSON format).
No sklearn dependency required for production scoring.
"""

import numpy as np
import json
from typing import Dict, Optional


def score_from_exported_tree(
    X: np.ndarray,
    tree_structure: Dict,
    feature_names: Optional[list] = None,
    X_categorical: Optional[Dict[str, np.ndarray]] = None
) -> np.ndarray:
    """
    Score observations using exported tree structure.

    This function provides production scoring without requiring sklearn.
    It uses the standardized JSON tree format exported by IRBSegmentationEngine.

    Args:
        X: Feature matrix (numeric features only, shape: [n_samples, n_features])
        tree_structure: Dictionary from export_tree_structure() or loaded from JSON
        feature_names: Optional list of feature names (must match tree structure)
        X_categorical: Optional dict of categorical features for categorical splits

    Returns:
        Array of segment labels (shape: [n_samples])

    Example:
        >>> # Export tree during model development
        >>> engine.export_tree_to_file("production_tree.json")
        >>>
        >>> # Later, in production environment
        >>> with open("production_tree.json") as f:
        ...     tree = json.load(f)
        >>> segments = score_from_exported_tree(X_new, tree)
    """
    # Extract feature names from tree if not provided
    if feature_names is None:
        feature_names = tree_structure['feature_metadata']['names']

    # Build index for fast node lookup
    nodes_by_id = {node['id']: node for node in tree_structure['nodes']}

    # Get leaf-to-segment mapping
    leaf_to_segment = tree_structure['segment_mapping']['leaf_to_segment']
    # Convert string keys to int (JSON serialization converts int keys to strings)
    leaf_to_segment = {int(k): int(v) for k, v in leaf_to_segment.items()}

    # Create feature name to index mapping
    feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}

    def traverse_tree(observation_idx: int) -> int:
        """
        Traverse tree for a single observation to find its leaf node.

        Args:
            observation_idx: Index of observation in X

        Returns:
            Leaf node ID
        """
        node_id = 0  # Start at root

        while True:
            node = nodes_by_id[node_id]

            if node['type'] == 'leaf':
                return node_id

            # Split node - determine which child to follow
            feature = node['feature']
            threshold = node['threshold']
            feature_idx = feature_to_idx[feature]

            # Get observation's value for this feature
            value = X[observation_idx, feature_idx]

            # Follow left (<=) or right (>) child
            if value <= threshold:
                node_id = node['left_child']
            else:
                node_id = node['right_child']

    # Score all observations
    n_samples = X.shape[0]
    segments = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        leaf_id = traverse_tree(i)
        # Map leaf to final segment (accounting for post-tree adjustments)
        segments[i] = leaf_to_segment.get(leaf_id, 0)

    return segments


def score_from_json_file(
    X: np.ndarray,
    json_filepath: str,
    feature_names: Optional[list] = None,
    X_categorical: Optional[Dict[str, np.ndarray]] = None
) -> np.ndarray:
    """
    Score observations using exported tree JSON file.

    Convenience wrapper around score_from_exported_tree().

    Args:
        X: Feature matrix
        json_filepath: Path to JSON file from export_tree_to_file()
        feature_names: Optional feature names
        X_categorical: Optional categorical features

    Returns:
        Array of segment labels
    """
    with open(json_filepath, 'r') as f:
        tree_structure = json.load(f)

    return score_from_exported_tree(X, tree_structure, feature_names, X_categorical)


def get_segment_statistics_from_tree(tree_structure: Dict) -> Dict:
    """
    Extract segment statistics from exported tree structure.

    Args:
        tree_structure: Dictionary from export_tree_structure()

    Returns:
        Dictionary mapping segment ID to statistics
    """
    # Get statistics from adjustments metadata
    if 'segment_statistics' in tree_structure.get('parameters', {}):
        return tree_structure['parameters']['segment_statistics']

    # Otherwise, compute from leaf nodes
    leaf_to_segment = tree_structure['segment_mapping']['leaf_to_segment']
    leaf_to_segment = {int(k): int(v) for k, v in leaf_to_segment.items()}

    nodes_by_id = {node['id']: node for node in tree_structure['nodes']}

    segment_stats = {}
    for node_id, segment in leaf_to_segment.items():
        node = nodes_by_id[node_id]
        if node['type'] == 'leaf':
            n_samples = node['n_samples']
            # value is [[n_class_0, n_class_1]]
            value = np.array(node['value'])
            n_defaults = int(value[0, 1])
            default_rate = n_defaults / n_samples if n_samples > 0 else 0.0

            if segment not in segment_stats:
                segment_stats[segment] = {
                    'n_observations': 0,
                    'n_defaults': 0,
                    'default_rate': 0.0
                }

            segment_stats[segment]['n_observations'] += n_samples
            segment_stats[segment]['n_defaults'] += n_defaults

    # Recalculate default rates
    for segment, stats in segment_stats.items():
        n_obs = stats['n_observations']
        n_def = stats['n_defaults']
        stats['default_rate'] = n_def / n_obs if n_obs > 0 else 0.0

    return segment_stats


def validate_tree_structure(tree_structure: Dict) -> bool:
    """
    Validate that tree structure has required fields.

    Args:
        tree_structure: Dictionary from export_tree_structure()

    Returns:
        True if valid, raises ValueError otherwise
    """
    required_keys = ['tree_metadata', 'nodes', 'feature_metadata', 'segment_mapping']

    for key in required_keys:
        if key not in tree_structure:
            raise ValueError(f"Missing required key: {key}")

    # Check tree metadata
    metadata = tree_structure['tree_metadata']
    if metadata.get('format_version') != '1.0':
        raise ValueError(f"Unsupported format version: {metadata.get('format_version')}")

    # Check nodes is non-empty
    if not tree_structure['nodes']:
        raise ValueError("Tree has no nodes")

    # Check root node exists (id=0)
    nodes_by_id = {node['id']: node for node in tree_structure['nodes']}
    if 0 not in nodes_by_id:
        raise ValueError("Missing root node (id=0)")

    return True
