"""
Run cluster visualization on existing segmentation results.

This script loads the data using the lending club loader (same as training),
assigns segments using the saved tree, then creates visualizations.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

from data_loaders.lending_club import load_lending_club
from scripts.visualize_clusters import ClusterVisualizer
from scripts.extract_segments import SegmentExtractor

def main():
    print("\n" + "=" * 70)
    print("CLUSTER VISUALIZATION GENERATOR")
    print("=" * 70)

    # Configuration
    tree_path = "output/lending_club_categorical/tree_structure.json"
    report_path = "output/lending_club_categorical/baseline_report.json"
    output_prefix = "output/lending_club_categorical/viz"

    # Step 1: Load data (use same parameters as original segmentation)
    print("\n1. Loading Lending Club data...")
    print("   Using sample for faster visualization (100k rows)")

    X_train, y_train, X_val, y_val, X_oot, y_oot, feature_names, X_categorical = load_lending_club(
        data_dir="./data",
        sample_size=100000,  # Use sample for speed
        use_oot=True,
        random_state=42
    )

    print(f"   Loaded {len(X_train):,} training observations")

    # Step 2: Assign segments using saved tree
    print("\n2. Assigning segments using saved tree...")
    extractor = SegmentExtractor(tree_path)
    segments_train = extractor.assign_segments(X_train)

    # Show segment distribution
    unique, counts = np.unique(segments_train, return_counts=True)
    print("\n   Segment distribution:")
    for seg, count in zip(unique, counts):
        pct = count / len(segments_train) * 100
        defaults = np.sum(y_train[segments_train == seg])
        dr = defaults / count * 100 if count > 0 else 0
        print(f"     Segment {seg}: {count:,} obs ({pct:.1f}%), {dr:.1f}% default rate")

    # Step 3: Initialize visualizer
    print(f"\n3. Initializing visualizer...")
    viz = ClusterVisualizer(report_path)

    # Step 4: Load data into visualizer
    print(f"\n4. Loading data into visualizer...")
    viz.load_data_from_arrays(
        X_train=X_train,
        y_train=y_train,
        segments_train=segments_train,
        feature_names=feature_names
    )

    # Step 5: Generate all visualizations
    print(f"\n5. Generating visualizations...")
    print("   This may take a few minutes...")

    viz.create_all_visualizations(output_prefix=output_prefix)

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE!")
    print("=" * 70)
    print(f"\nGenerated files:")
    print(f"  {output_prefix}_pca_scatter.png")
    print(f"  {output_prefix}_pca_heatmap.png")
    print(f"  {output_prefix}_tsne_scatter.png")
    print(f"  {output_prefix}_feature_distributions.png")
    print("\nOpen these PNG files to view your cluster visualizations!")


if __name__ == '__main__':
    main()
