"""
Simple Script to Run IRB Segmentation Analysis

Usage:
    python run_segmentation.py                              # Uses categorical_example.yaml
    python run_segmentation.py config.yaml                  # Uses your YAML config
    python run_segmentation.py config.xlsx                  # Uses your Excel config
    python run_segmentation.py config.csv                   # Uses your CSV config
"""

import sys
from pathlib import Path
from irb_segmentation import SegmentationPipeline, SegmentationConfig


def main():
    # Get config path from command line or use default
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = 'config_examples/categorical_example.yaml'
        print(f"No config specified, using default: {config_path}")

    # Check if config exists
    if not Path(config_path).exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    # Determine config type from extension
    suffix = Path(config_path).suffix.lower()

    print(f"\n{'='*60}")
    print(f"IRB Segmentation Analysis")
    print(f"{'='*60}")
    print(f"Config: {config_path}")
    print(f"{'='*60}\n")

    # Load and run pipeline
    try:
        if suffix == '.yaml' or suffix == '.yml':
            config = SegmentationConfig.from_yaml(config_path)
            pipeline = SegmentationPipeline(config)
        elif suffix == '.xlsx':
            config = SegmentationConfig.from_excel(config_path)
            pipeline = SegmentationPipeline(config)
        elif suffix == '.csv':
            from interfaces.config_csv import ConfigCSV
            config = ConfigCSV.from_csv(config_path)
            pipeline = SegmentationPipeline(config)
        else:
            print(f"ERROR: Unsupported config format: {suffix}")
            print("Supported formats: .yaml, .yml, .xlsx, .csv")
            sys.exit(1)

        # Run segmentation (complete workflow without pausing for edits)
        pipeline.run_all(pause_for_edits=False)

        print(f"\n{'='*60}")
        print(f"✓ Segmentation Complete!")
        print(f"{'='*60}")
        print(f"Output directory: {pipeline.config.output.output_dir}")
        print(f"\nGenerated files:")

        output_dir = Path(pipeline.config.output.output_dir)
        if output_dir.exists():
            for file in sorted(output_dir.iterdir()):
                print(f"  - {file.name}")

        print(f"\n{'='*60}\n")

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: Segmentation failed")
        print(f"{'='*60}")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
