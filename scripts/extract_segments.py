"""
Segment Sub-sampling Extractor

Extract sub-samples of the original dataset based on segment assignments
from IRB segmentation. Supports both direct segment filtering and
tree-based segment assignment.

Usage:
    # Extract single segment
    python scripts/extract_segments.py data.csv tree_structure.json --segments 2 --output seg2.csv

    # Extract multiple segments
    python scripts/extract_segments.py data.csv tree_structure.json --segments 1 2 3 --output low_risk.csv

    # Python API
    from scripts.extract_segments import SegmentExtractor
    extractor = SegmentExtractor('tree_structure.json')
    data = extractor.extract_from_csv('data.csv', segment_ids=[1, 2])
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import List, Optional, Union, Dict, Tuple
import argparse


class SegmentExtractor:
    """
    Extract data sub-samples by segment ID using tree structure.
    """

    def __init__(self, tree_path: str):
        """
        Initialize extractor with tree structure.

        Args:
            tree_path: Path to tree_structure.json
        """
        self.tree_path = Path(tree_path)

        # Load tree structure
        with open(self.tree_path, 'r') as f:
            self.tree_data = json.load(f)

        # Parse tree
        self.nodes = {node['id']: node for node in self.tree_data['nodes']}
        self.leaf_to_segment = {
            int(k): v for k, v in self.tree_data['segment_mapping']['leaf_to_segment'].items()
        }
        self.feature_names = self.tree_data['feature_metadata']['names']
        self.n_segments = self.tree_data['segment_mapping']['n_segments']

        print(f"Loaded tree: {self.n_segments} segments, {len(self.feature_names)} features")

    def assign_segments(self, X: np.ndarray) -> np.ndarray:
        """
        Assign segments to observations using tree structure.

        Args:
            X: Feature matrix (n_samples x n_features)

        Returns:
            Array of segment assignments (n_samples,)
        """
        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Feature count mismatch: expected {len(self.feature_names)}, got {X.shape[1]}"
            )

        segments = np.zeros(len(X), dtype=int)

        # Traverse tree for each observation
        for i in range(len(X)):
            node_id = 0  # Start at root

            # Traverse until leaf
            while self.nodes[node_id]['type'] != 'leaf':
                node = self.nodes[node_id]
                feature_idx = node['feature_index']
                threshold = node['threshold']

                # Navigate tree
                if X[i, feature_idx] <= threshold:
                    node_id = node['left_child']
                else:
                    node_id = node['right_child']

            # Assign segment from leaf mapping
            segments[i] = self.leaf_to_segment.get(node_id, -1)

        return segments

    def extract_from_dataframe(
        self,
        df: pd.DataFrame,
        segment_ids: Union[int, List[int]],
        feature_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Extract rows belonging to specified segment(s) from DataFrame.

        Args:
            df: Input DataFrame
            segment_ids: Segment ID(s) to extract
            feature_columns: Column names for features (if None, uses all numeric columns)

        Returns:
            DataFrame with only observations from specified segments
        """
        # Normalize segment_ids to list
        if isinstance(segment_ids, int):
            segment_ids = [segment_ids]

        # Extract features
        if feature_columns is None:
            # Try to match features from tree
            available_features = [col for col in self.feature_names if col in df.columns]
            if len(available_features) != len(self.feature_names):
                # Fall back to numeric columns
                feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                print(f"Warning: Using all numeric columns ({len(feature_columns)} features)")
            else:
                feature_columns = available_features

        # Ensure correct feature order
        if set(feature_columns) == set(self.feature_names):
            # Reorder to match tree
            feature_columns = self.feature_names

        # Extract feature matrix
        X = df[feature_columns].fillna(0).values

        # Assign segments
        segments = self.assign_segments(X)

        # Filter by segment IDs
        mask = np.isin(segments, segment_ids)

        # Add segment column to output
        df_output = df[mask].copy()
        df_output['segment_id'] = segments[mask]

        print(f"Extracted {len(df_output):,} rows from segments {segment_ids}")
        print(f"  Original: {len(df):,} rows")
        print(f"  Filtered: {len(df_output):,} rows ({len(df_output)/len(df)*100:.1f}%)")

        return df_output

    def extract_from_csv(
        self,
        csv_path: str,
        segment_ids: Union[int, List[int]],
        output_path: Optional[str] = None,
        feature_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Extract segments from CSV file.

        Args:
            csv_path: Path to input CSV
            segment_ids: Segment ID(s) to extract
            output_path: Optional path to save output CSV
            feature_columns: Column names for features

        Returns:
            DataFrame with extracted segments
        """
        # Load CSV
        print(f"\nLoading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

        # Extract segments
        df_output = self.extract_from_dataframe(df, segment_ids, feature_columns)

        # Save if requested
        if output_path:
            df_output.to_csv(output_path, index=False)
            print(f"\nSaved to: {output_path}")

        return df_output

    def extract_from_arrays(
        self,
        X: np.ndarray,
        segment_ids: Union[int, List[int]],
        y: Optional[np.ndarray] = None,
        return_indices: bool = False
    ) -> Union[Tuple[np.ndarray, ...], Dict]:
        """
        Extract segments from numpy arrays.

        Args:
            X: Feature matrix
            segment_ids: Segment ID(s) to extract
            y: Optional target array
            return_indices: If True, return indices instead of data

        Returns:
            If return_indices=False: (X_filtered, y_filtered) or X_filtered
            If return_indices=True: dict with 'indices', 'X', 'y', 'segments'
        """
        # Normalize segment_ids
        if isinstance(segment_ids, int):
            segment_ids = [segment_ids]

        # Assign segments
        segments = self.assign_segments(X)

        # Filter
        mask = np.isin(segments, segment_ids)
        indices = np.where(mask)[0]

        print(f"Extracted {len(indices):,} / {len(X):,} observations ({len(indices)/len(X)*100:.1f}%)")

        if return_indices:
            result = {
                'indices': indices,
                'X': X[mask],
                'segments': segments[mask]
            }
            if y is not None:
                result['y'] = y[mask]
            return result
        else:
            if y is not None:
                return X[mask], y[mask]
            else:
                return X[mask]

    def get_segment_statistics(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Get statistics for all segments in the data.

        Args:
            X: Feature matrix
            y: Optional binary target (for default rate calculation)

        Returns:
            DataFrame with segment statistics
        """
        segments = self.assign_segments(X)
        unique_segments = np.unique(segments)

        stats = []
        for seg_id in sorted(unique_segments):
            mask = segments == seg_id
            n_obs = np.sum(mask)

            stat = {
                'segment_id': int(seg_id),
                'n_observations': int(n_obs),
                'density': float(n_obs / len(X))
            }

            if y is not None:
                n_defaults = np.sum(y[mask])
                stat['n_defaults'] = int(n_defaults)
                stat['default_rate'] = float(n_defaults / n_obs if n_obs > 0 else 0)

            stats.append(stat)

        df_stats = pd.DataFrame(stats)
        return df_stats

    def split_by_segments(
        self,
        csv_path: str,
        output_dir: str,
        prefix: str = "segment"
    ) -> Dict[int, str]:
        """
        Split CSV into separate files by segment.

        Args:
            csv_path: Path to input CSV
            output_dir: Directory to save segment files
            prefix: Prefix for output files

        Returns:
            Dictionary mapping segment ID to output file path
        """
        # Load data
        print(f"\nLoading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df):,} rows")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Extract features
        available_features = [col for col in self.feature_names if col in df.columns]
        if len(available_features) != len(self.feature_names):
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            feature_columns = self.feature_names

        X = df[feature_columns].fillna(0).values

        # Assign segments
        segments = self.assign_segments(X)
        unique_segments = np.unique(segments)

        # Split and save
        output_files = {}
        print(f"\nSplitting into {len(unique_segments)} segment files...")

        for seg_id in sorted(unique_segments):
            mask = segments == seg_id
            df_segment = df[mask].copy()
            df_segment['segment_id'] = seg_id

            output_file = output_path / f"{prefix}_{seg_id}.csv"
            df_segment.to_csv(output_file, index=False)

            output_files[int(seg_id)] = str(output_file)
            print(f"  Segment {seg_id}: {len(df_segment):,} rows -> {output_file}")

        return output_files

    def filter_by_segment_criteria(
        self,
        csv_path: str,
        criteria: Dict[str, any],
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Filter segments by criteria (e.g., default_rate, density).

        Args:
            csv_path: Path to input CSV
            criteria: Dictionary with filtering criteria, e.g.:
                     {'min_default_rate': 0.10, 'max_default_rate': 0.20}
                     {'min_density': 0.05, 'max_observations': 10000}
            output_path: Optional path to save filtered data

        Returns:
            DataFrame with observations from segments matching criteria
        """
        # Load data
        df = pd.read_csv(csv_path)

        # Check if target column exists
        target_col = None
        for col in ['default', 'target', 'y', 'label', 'loan_status']:
            if col in df.columns:
                target_col = col
                break

        if target_col is None:
            print("Warning: No target column found. Cannot filter by default_rate.")
            y = None
        else:
            y = df[target_col].values

        # Extract features
        available_features = [col for col in self.feature_names if col in df.columns]
        if len(available_features) != len(self.feature_names):
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            feature_columns = self.feature_names

        X = df[feature_columns].fillna(0).values

        # Get segment statistics
        stats_df = self.get_segment_statistics(X, y)

        # Apply criteria
        mask = np.ones(len(stats_df), dtype=bool)

        if 'min_default_rate' in criteria and y is not None:
            mask &= stats_df['default_rate'] >= criteria['min_default_rate']

        if 'max_default_rate' in criteria and y is not None:
            mask &= stats_df['default_rate'] <= criteria['max_default_rate']

        if 'min_density' in criteria:
            mask &= stats_df['density'] >= criteria['min_density']

        if 'max_density' in criteria:
            mask &= stats_df['density'] <= criteria['max_density']

        if 'min_observations' in criteria:
            mask &= stats_df['n_observations'] >= criteria['min_observations']

        if 'max_observations' in criteria:
            mask &= stats_df['n_observations'] <= criteria['max_observations']

        # Get matching segment IDs
        matching_segments = stats_df[mask]['segment_id'].tolist()

        print(f"\nSegments matching criteria: {matching_segments}")
        print(f"  Criteria: {criteria}")
        print(f"  Matched: {len(matching_segments)} / {len(stats_df)} segments")

        # Extract data
        df_output = self.extract_from_dataframe(df, matching_segments, feature_columns)

        if output_path:
            df_output.to_csv(output_path, index=False)
            print(f"\nSaved to: {output_path}")

        return df_output


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Extract segment sub-samples from dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract single segment
  python extract_segments.py data.csv tree.json --segments 2 -o seg2.csv

  # Extract multiple segments (low risk)
  python extract_segments.py data.csv tree.json --segments 1 2 -o low_risk.csv

  # Split into separate files by segment
  python extract_segments.py data.csv tree.json --split-all -o output_dir/

  # Filter by criteria
  python extract_segments.py data.csv tree.json --min-dr 0.10 --max-dr 0.20
        """
    )

    parser.add_argument('csv_path', help='Path to input CSV file')
    parser.add_argument('tree_path', help='Path to tree_structure.json')
    parser.add_argument('--segments', '-s', nargs='+', type=int,
                        help='Segment ID(s) to extract')
    parser.add_argument('--output', '-o', help='Output file path')
    parser.add_argument('--split-all', action='store_true',
                        help='Split into separate files by segment')
    parser.add_argument('--stats', action='store_true',
                        help='Show segment statistics only')

    # Criteria filtering
    parser.add_argument('--min-dr', type=float, help='Minimum default rate')
    parser.add_argument('--max-dr', type=float, help='Maximum default rate')
    parser.add_argument('--min-density', type=float, help='Minimum segment density')
    parser.add_argument('--max-density', type=float, help='Maximum segment density')

    args = parser.parse_args()

    # Create extractor
    extractor = SegmentExtractor(args.tree_path)

    print("\n" + "=" * 70)
    print("SEGMENT EXTRACTOR")
    print("=" * 70)

    # Show statistics
    if args.stats:
        df = pd.read_csv(args.csv_path)

        # Try to find target column
        target_col = None
        for col in ['default', 'target', 'y', 'label', 'loan_status']:
            if col in df.columns:
                target_col = col
                break

        available_features = [col for col in extractor.feature_names if col in df.columns]
        if len(available_features) != len(extractor.feature_names):
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            feature_columns = extractor.feature_names

        X = df[feature_columns].fillna(0).values
        y = df[target_col].values if target_col else None

        stats_df = extractor.get_segment_statistics(X, y)
        print("\nSegment Statistics:")
        print(stats_df.to_string(index=False))
        return

    # Split into separate files
    if args.split_all:
        output_dir = args.output or "segment_splits"
        output_files = extractor.split_by_segments(args.csv_path, output_dir)
        print(f"\n{len(output_files)} files created in: {output_dir}")
        return

    # Filter by criteria
    if any([args.min_dr, args.max_dr, args.min_density, args.max_density]):
        criteria = {}
        if args.min_dr:
            criteria['min_default_rate'] = args.min_dr
        if args.max_dr:
            criteria['max_default_rate'] = args.max_dr
        if args.min_density:
            criteria['min_density'] = args.min_density
        if args.max_density:
            criteria['max_density'] = args.max_density

        df = extractor.filter_by_segment_criteria(args.csv_path, criteria, args.output)
        return

    # Extract specific segments
    if args.segments:
        df = extractor.extract_from_csv(args.csv_path, args.segments, args.output)
    else:
        print("\nError: Must specify --segments, --split-all, --stats, or criteria filters")
        parser.print_help()


if __name__ == '__main__':
    main()
