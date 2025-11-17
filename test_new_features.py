"""
Test script for new visualization and extraction features.
"""

import sys
from pathlib import Path

# Test imports
try:
    from scripts.visualize_clusters import ClusterVisualizer
    print("[OK] ClusterVisualizer imported successfully")
except ImportError as e:
    print(f"[FAIL] ClusterVisualizer import failed: {e}")
    sys.exit(1)

try:
    from scripts.extract_segments import SegmentExtractor
    print("[OK] SegmentExtractor imported successfully")
except ImportError as e:
    print(f"[FAIL] SegmentExtractor import failed: {e}")
    sys.exit(1)

# Test with existing outputs
output_dir = Path("output/lending_club_categorical")
tree_path = output_dir / "tree_structure.json"
report_path = output_dir / "baseline_report.json"

print("\n" + "=" * 70)
print("TESTING SEGMENT EXTRACTOR")
print("=" * 70)

if tree_path.exists():
    try:
        extractor = SegmentExtractor(str(tree_path))
        print(f"\n[OK] SegmentExtractor initialized")
        print(f"  Segments: {extractor.n_segments}")
        print(f"  Features: {len(extractor.feature_names)}")
        print(f"  Feature names: {extractor.feature_names[:5]}...")

        # Test segment assignment with dummy data
        import numpy as np
        X_test = np.random.randn(10, len(extractor.feature_names))
        segments = extractor.assign_segments(X_test)
        print(f"\n[OK] Segment assignment works")
        print(f"  Test data: {len(X_test)} rows")
        print(f"  Assigned segments: {segments}")

    except Exception as e:
        print(f"\n[FAIL] SegmentExtractor test failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"\n[INFO] Tree structure not found at {tree_path}")
    print("  Run the segmentation pipeline first to generate outputs")

print("\n" + "=" * 70)
print("TESTING CLUSTER VISUALIZER")
print("=" * 70)

if report_path.exists():
    try:
        viz = ClusterVisualizer(str(report_path))
        print(f"\n[OK] ClusterVisualizer initialized")
        print(f"  Report loaded: {viz.report_path}")
        print(f"  Segments in report: {len(viz.report['segment_statistics'])}")

        # Show segment statistics
        print("\n  Segment statistics:")
        for seg_id, stats in sorted(viz.report['segment_statistics'].items()):
            print(f"    Segment {seg_id}: {stats['n_observations']:,} obs, "
                  f"{stats['default_rate']:.2%} default rate")

    except Exception as e:
        print(f"\n[FAIL] ClusterVisualizer test failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"\n[INFO] Baseline report not found at {report_path}")
    print("  Run the segmentation pipeline first to generate outputs")

print("\n" + "=" * 70)
print("ALL TESTS COMPLETED")
print("=" * 70)
print("\nNext steps:")
print("1. Run the segmentation pipeline if outputs don't exist")
print("2. Use demo_visualization_and_extraction.py for usage examples")
print("3. See scripts/visualize_clusters.py and scripts/extract_segments.py for full API")
