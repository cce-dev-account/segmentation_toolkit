"""
IRB Segmentation Demo Runner

Simple script to run segmentation with a config file.

Usage:
    python run_demo.py demo_german.yaml           # Quick start (auto-downloads)
    python run_demo.py demo_taiwan.yaml           # Medium scale
    python run_demo.py demo_lending_club.yaml     # Production scale
"""

import sys
from pathlib import Path
from irb_segmentation.pipeline import SegmentationPipeline
from irb_segmentation.config import SegmentationConfig

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_demo.py <config_file>")
        print("\nAvailable demos:")
        print("  python run_demo.py demo_german.yaml         # Quick start (1K rows, auto-downloads)")
        print("  python run_demo.py demo_taiwan.yaml         # Medium (30K rows, requires download)")
        print("  python run_demo.py demo_lending_club.yaml   # Full scale (1.3M rows, requires download)")
        sys.exit(1)

    config_path = Path(sys.argv[1])

    if not config_path.exists():
        print(f"Error: Config file '{config_path}' not found")
        print("\nMake sure you're running from the irb_segmentation_demo/ directory")
        sys.exit(1)

    print("=" * 80)
    print("IRB SEGMENTATION DEMO")
    print("=" * 80)
    print(f"Config: {config_path}")
    print()

    # Load config and run pipeline
    config = SegmentationConfig.from_yaml(str(config_path))
    pipeline = SegmentationPipeline(config)
    pipeline.run_all(pause_for_edits=False)

    print()
    print("=" * 80)
    print("DEMO COMPLETE!")
    print("=" * 80)
    print("\nOutput files generated:")
    print(f"  - {pipeline.config.output.output_dir}/segment_rules.txt       (human-readable rules)")
    print(f"  - {pipeline.config.output.output_dir}/baseline_report.json    (complete validation report)")
    print(f"  - {pipeline.config.output.output_dir}/segment_summary.xlsx    (Excel summary)")
    print(f"  - {pipeline.config.output.output_dir}/dashboard.html          (interactive visualization)")
    print()
    print("See OUTPUT_GUIDE.md for help interpreting results.")
    print()

if __name__ == "__main__":
    main()
