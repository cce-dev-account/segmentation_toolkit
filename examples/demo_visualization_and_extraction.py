"""
Demo: Cluster Visualization and Segment Extraction

This example demonstrates how to:
1. Visualize segmentation clusters in 2D
2. Extract sub-samples by segment ID

Run after executing the segmentation pipeline to generate outputs.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scripts.visualize_clusters import ClusterVisualizer
from scripts.extract_segments import SegmentExtractor


def demo_visualization(output_dir: str = "output/lending_club_categorical"):
    """
    Demonstrate cluster visualization.

    Args:
        output_dir: Directory with segmentation outputs
    """
    print("\n" + "=" * 70)
    print("DEMO: CLUSTER VISUALIZATION")
    print("=" * 70)

    # Initialize visualizer
    report_path = Path(output_dir) / "baseline_report.json"
    viz = ClusterVisualizer(str(report_path))

    # Note: In a real scenario, you would load data from the pipeline
    # For this demo, we'll show the pattern

    print("\nTo use the visualizer, you need to provide the data:")
    print("""
    # Option 1: Load from arrays (after running pipeline)
    from irb_segmentation import SegmentationPipeline
    from irb_segmentation.config import SegmentationConfig

    config = SegmentationConfig.from_yaml('config.yaml')
    pipeline = SegmentationPipeline(config)
    pipeline.load_data()
    pipeline.fit_baseline()

    # Now visualize
    viz = ClusterVisualizer('output/baseline_report.json')
    viz.load_data_from_arrays(
        X_train=pipeline.X_train,
        y_train=pipeline.y_train,
        segments_train=pipeline.baseline_engine.segments_train_,
        feature_names=pipeline.feature_names
    )

    # Generate visualizations
    viz.reduce_dimensions('pca')
    viz.plot_clusters(save_path='clusters_pca.png')
    viz.plot_default_rate_heatmap(save_path='heatmap_pca.png')
    viz.plot_feature_distributions(save_path='features.png')

    # Try t-SNE for different perspective
    viz.reduce_dimensions('tsne')
    viz.plot_clusters(save_path='clusters_tsne.png')

    # Or generate all at once
    viz.create_all_visualizations(output_prefix='output/viz')
    """)


def demo_extraction(
    tree_path: str = "output/lending_club_categorical/tree_structure.json",
    data_path: str = "data/lending_club.csv"
):
    """
    Demonstrate segment extraction.

    Args:
        tree_path: Path to tree_structure.json
        data_path: Path to original data CSV
    """
    print("\n" + "=" * 70)
    print("DEMO: SEGMENT EXTRACTION")
    print("=" * 70)

    # Check if files exist
    if not Path(tree_path).exists():
        print(f"\nError: Tree structure not found at {tree_path}")
        print("Please run the segmentation pipeline first to generate outputs.")
        return

    # Initialize extractor
    extractor = SegmentExtractor(tree_path)

    print("\nExtractor initialized successfully!")
    print(f"  Segments: {extractor.n_segments}")
    print(f"  Features: {len(extractor.feature_names)}")

    # Example 1: Extract specific segments
    print("\n" + "-" * 70)
    print("Example 1: Extract Low-Risk Segments (1 and 2)")
    print("-" * 70)

    if Path(data_path).exists():
        print(f"\nWould extract from: {data_path}")
        print("Usage:")
        print(f"  df = extractor.extract_from_csv('{data_path}', segment_ids=[1, 2])")
        print(f"  df.to_csv('low_risk_subset.csv', index=False)")
    else:
        print(f"\nData file not found: {data_path}")
        print("Update the path or use your own data file")

    # Example 2: Split by all segments
    print("\n" + "-" * 70)
    print("Example 2: Split Into Separate Files by Segment")
    print("-" * 70)
    print("Usage:")
    print(f"  output_files = extractor.split_by_segments(")
    print(f"      csv_path='{data_path}',")
    print(f"      output_dir='segment_splits',")
    print(f"      prefix='segment'")
    print(f"  )")

    # Example 3: Filter by criteria
    print("\n" + "-" * 70)
    print("Example 3: Filter by Default Rate Range")
    print("-" * 70)
    print("Usage:")
    print(f"  df_medium_risk = extractor.filter_by_segment_criteria(")
    print(f"      csv_path='{data_path}',")
    print(f"      criteria={{'min_default_rate': 0.10, 'max_default_rate': 0.20}},")
    print(f"      output_path='medium_risk.csv'")
    print(f"  )")

    # Example 4: Get statistics
    print("\n" + "-" * 70)
    print("Example 4: Get Segment Statistics")
    print("-" * 70)
    print("Usage:")
    print("""
    # Load your data
    df = pd.read_csv('data.csv')
    X = df[extractor.feature_names].fillna(0).values
    y = df['default'].values  # if you have target

    # Get stats
    stats = extractor.get_segment_statistics(X, y)
    print(stats)
    """)

    # Command-line usage examples
    print("\n" + "-" * 70)
    print("Command-Line Usage Examples")
    print("-" * 70)
    print("\n# Extract single segment:")
    print(f"python scripts/extract_segments.py {data_path} {tree_path} \\")
    print(f"    --segments 2 --output segment_2.csv")

    print("\n# Extract multiple segments:")
    print(f"python scripts/extract_segments.py {data_path} {tree_path} \\")
    print(f"    --segments 1 2 3 --output low_risk.csv")

    print("\n# Split all segments:")
    print(f"python scripts/extract_segments.py {data_path} {tree_path} \\")
    print(f"    --split-all --output segment_splits/")

    print("\n# Show statistics:")
    print(f"python scripts/extract_segments.py {data_path} {tree_path} --stats")

    print("\n# Filter by default rate:")
    print(f"python scripts/extract_segments.py {data_path} {tree_path} \\")
    print(f"    --min-dr 0.10 --max-dr 0.20 --output medium_risk.csv")


def demo_complete_workflow():
    """
    Show complete workflow from segmentation to visualization and extraction.
    """
    print("\n" + "=" * 70)
    print("COMPLETE WORKFLOW DEMO")
    print("=" * 70)

    print("""
This demonstrates the full workflow:

1. Run Segmentation
   -----------------
   from irb_segmentation import SegmentationPipeline
   from irb_segmentation.config import SegmentationConfig

   config = SegmentationConfig.from_yaml('config_examples/lending_club_categorical.yaml')
   pipeline = SegmentationPipeline(config)
   pipeline.run_all(pause_for_edits=False)

   # Outputs generated:
   # - output/baseline_report.json
   # - output/tree_structure.json
   # - output/dashboard.html
   # - output/segment_rules.txt


2. Visualize Clusters
   -------------------
   from scripts.visualize_clusters import ClusterVisualizer

   viz = ClusterVisualizer('output/baseline_report.json')
   viz.load_data_from_arrays(
       X_train=pipeline.X_train,
       y_train=pipeline.y_train,
       segments_train=pipeline.baseline_engine.segments_train_,
       feature_names=pipeline.feature_names
   )

   # Generate all visualizations
   viz.create_all_visualizations(output_prefix='output/viz')

   # Results:
   # - output/viz_pca_scatter.png
   # - output/viz_pca_heatmap.png
   # - output/viz_tsne_scatter.png (if dataset < 50k obs)
   # - output/viz_feature_distributions.png


3. Extract Segments
   ----------------
   from scripts.extract_segments import SegmentExtractor

   extractor = SegmentExtractor('output/tree_structure.json')

   # Extract low-risk segments (1, 2)
   df_low_risk = extractor.extract_from_csv(
       csv_path='data/lending_club.csv',
       segment_ids=[1, 2],
       output_path='output/low_risk_subset.csv'
   )

   # Extract high-risk segment (4)
   df_high_risk = extractor.extract_from_csv(
       csv_path='data/lending_club.csv',
       segment_ids=[4],
       output_path='output/high_risk_subset.csv'
   )

   # Split into separate files
   output_files = extractor.split_by_segments(
       csv_path='data/lending_club.csv',
       output_dir='output/segments',
       prefix='segment'
   )


4. Analysis on Sub-samples
   ------------------------
   # Now you can analyze each segment separately
   import pandas as pd

   df_low = pd.read_csv('output/low_risk_subset.csv')
   df_high = pd.read_csv('output/high_risk_subset.csv')

   # Compare characteristics
   print("Low Risk:")
   print(df_low.describe())

   print("High Risk:")
   print(df_high.describe())

   # Build segment-specific models
   # Perform targeted analysis
   # Generate segment reports
    """)


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("VISUALIZATION AND EXTRACTION DEMO")
    print("=" * 70)

    # Show complete workflow
    demo_complete_workflow()

    # Demo visualization
    demo_visualization()

    # Demo extraction
    demo_extraction()

    print("\n" + "=" * 70)
    print("For working examples, ensure you have:")
    print("  1. Run the segmentation pipeline")
    print("  2. Generated baseline_report.json and tree_structure.json")
    print("  3. Have the original data file available")
    print("=" * 70)


if __name__ == '__main__':
    main()
