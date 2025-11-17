"""
Extract segments from raw Lending Club CSV data.

This script loads raw data, processes it using the lending club loader,
then assigns segments and exports filtered data.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Import data loader and extractor
from data_loaders.lending_club import LendingClubLoader
from scripts.extract_segments import SegmentExtractor

def extract_segments_from_raw(
    csv_path: str,
    tree_path: str,
    segment_ids,
    output_path: str = None
):
    """
    Extract segments from raw Lending Club CSV.

    Args:
        csv_path: Path to raw lending club CSV
        tree_path: Path to tree_structure.json
        segment_ids: Segment ID(s) to extract (int or list)
        output_path: Where to save extracted data
    """
    print("\n" + "=" * 70)
    print("EXTRACTING SEGMENTS FROM RAW LENDING CLUB DATA")
    print("=" * 70)

    # Step 1: Load raw data
    print(f"\n1. Loading raw data: {csv_path}")
    df_raw = pd.read_csv(csv_path, low_memory=False)
    print(f"   Loaded {len(df_raw):,} rows, {len(df_raw.columns)} columns")

    # Step 2: Process features using lending club loader
    print("\n2. Processing features (same as segmentation training)...")
    loader = LendingClubLoader()

    # Apply the same feature engineering
    df_processed = loader._preprocess(df_raw)

    # Extract features and target
    feature_names = [col for col in df_processed.columns if col not in ['default', 'issue_year', 'issue_month']]
    X = df_processed[feature_names].values
    y = df_processed['default'].values
    print(f"   Processed to {X.shape[1]} features")
    print(f"   Features: {feature_names[:5]}...")

    # Step 3: Load tree and assign segments
    print(f"\n3. Loading segmentation tree: {tree_path}")
    extractor = SegmentExtractor(tree_path)
    print(f"   Tree has {extractor.n_segments} segments")

    print("\n4. Assigning segments to data...")
    segments = extractor.assign_segments(X)

    # Show distribution
    unique, counts = np.unique(segments, return_counts=True)
    print("\n   Segment distribution:")
    for seg, count in zip(unique, counts):
        pct = count / len(segments) * 100
        defaults = np.sum(y[segments == seg])
        dr = defaults / count * 100 if count > 0 else 0
        print(f"     Segment {seg}: {count:,} obs ({pct:.1f}%), {defaults:,} defaults ({dr:.1f}% DR)")

    # Step 5: Filter by requested segments
    if isinstance(segment_ids, int):
        segment_ids = [segment_ids]

    print(f"\n5. Filtering for segments: {segment_ids}")
    mask = np.isin(segments, segment_ids)

    # Use processed dataframe (cleaner data with segment assignments)
    df_output = df_processed[mask].copy()
    df_output['segment_id'] = segments[mask]

    print(f"   Extracted {len(df_output):,} rows ({len(df_output)/len(df_processed)*100:.1f}% of processed data)")

    # Step 6: Save
    if output_path:
        df_output.to_csv(output_path, index=False)
        print(f"\n6. Saved to: {output_path}")
        print(f"   Columns: {len(df_output.columns)} (includes 'segment_id' and 'predicted_default')")

    return df_output

def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract segments from raw Lending Club CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract low-risk segments (1, 2)
  python extract_segments_from_raw.py data/lending_club_test.csv \\
      output/lending_club_categorical/tree_structure.json \\
      --segments 1 2 --output output/low_risk.csv

  # Extract high-risk segment (4)
  python extract_segments_from_raw.py data/lending_club_test.csv \\
      output/lending_club_categorical/tree_structure.json \\
      --segments 4 --output output/high_risk.csv
        """
    )

    parser.add_argument('csv_path', help='Path to raw lending club CSV')
    parser.add_argument('tree_path', help='Path to tree_structure.json')
    parser.add_argument('--segments', '-s', nargs='+', type=int, required=True,
                        help='Segment ID(s) to extract')
    parser.add_argument('--output', '-o', required=True,
                        help='Output CSV path')

    args = parser.parse_args()

    # Extract
    df = extract_segments_from_raw(
        args.csv_path,
        args.tree_path,
        args.segments,
        args.output
    )

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"\nExtracted segments {args.segments}")
    print(f"Output saved to: {args.output}")
    print(f"\nYou can now analyze this subset in Excel, Python, or your preferred tool.")


if __name__ == '__main__':
    main()
