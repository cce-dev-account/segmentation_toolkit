"""
Quick Script to Run Lending Club Segmentation

Uses the built-in Lending Club data loader.
No config file needed - just run it!

Usage:
    python run_lending_club.py
"""

from irb_segmentation import SegmentationPipeline, create_default_config


def main():
    print(f"\n{'='*60}")
    print(f"IRB Segmentation Analysis - Lending Club")
    print(f"{'='*60}\n")

    # Create config for lending club
    config = create_default_config(
        data_source='lending_club',
        data_type='lending_club'
    )

    # Customize config
    config.name = "Lending Club Segmentation"
    config.description = "IRB segmentation analysis of Lending Club loan data"
    config.output.output_dir = './output/lending_club'
    config.data.sample_size = 50000  # Limit to 50k for faster run

    # Optional: Add categorical features if your data has them
    # config.data.categorical_columns = ['loan_purpose', 'grade']

    # Optional: Add forced splits
    # config.irb_params.forced_splits = {
    #     'int_rate': 15.0,
    #     'fico_range_low': 680
    # }

    print("Configuration:")
    print(f"  Data source: {config.data.source}")
    print(f"  Sample size: {config.data.sample_size or 'Full dataset'}")
    print(f"  Output dir: {config.output.output_dir}")
    print(f"\nRunning segmentation...\n")

    # Run pipeline
    try:
        pipeline = SegmentationPipeline(config)
        # Run complete workflow without pausing for edits
        pipeline.run_all(pause_for_edits=False)

        print(f"\n{'='*60}")
        print(f"✓ Segmentation Complete!")
        print(f"{'='*60}")
        print(f"Output directory: {config.output.output_dir}")
        print(f"\nOpen the dashboard:")
        print(f"  {config.output.output_dir}/dashboard.html")
        print(f"\n{'='*60}\n")

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: Segmentation failed")
        print(f"{'='*60}")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
