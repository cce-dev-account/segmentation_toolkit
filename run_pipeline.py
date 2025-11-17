#!/usr/bin/env python
"""
Segmentation Pipeline CLI

Command-line interface for running the segmentation pipeline.

Usage:
    # Run entire pipeline
    python run_pipeline.py --config config.yaml

    # Run specific stage
    python run_pipeline.py --config config.yaml --stage fit
    python run_pipeline.py --config config.yaml --stage apply --template output/template.xlsx

    # Create config interactively
    python run_pipeline.py --build-config

    # Analyze dataset and create config
    python run_pipeline.py --analyze data/my_data.csv --output my_config.yaml
"""

import argparse
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from irb_segmentation import (
    SegmentationConfig,
    SegmentationPipeline,
    ConfigBuilder
)


def load_config(config_path: str) -> SegmentationConfig:
    """
    Load configuration from file (YAML, JSON, Excel, or CSV).

    Args:
        config_path: Path to configuration file

    Returns:
        SegmentationConfig instance
    """
    path = Path(config_path)
    suffix = path.suffix.lower()

    print(f"\nLoading configuration from: {config_path}")

    if suffix == '.yaml' or suffix == '.yml':
        return SegmentationConfig.from_yaml(config_path)
    elif suffix == '.json':
        return SegmentationConfig.from_json(config_path)
    elif suffix == '.xlsx':
        print("  (Excel format detected)")
        return SegmentationConfig.from_excel(config_path)
    elif suffix == '.csv':
        print("  (CSV format detected)")
        return SegmentationConfig.from_csv(config_path)
    else:
        # Default to YAML
        print(f"  Warning: Unknown file type '{suffix}', attempting YAML")
        return SegmentationConfig.from_yaml(config_path)


def run_all(config_path: str, pause: bool = True):
    """Run entire pipeline."""
    config = load_config(config_path)

    print("\nStarting complete pipeline...")
    pipeline = SegmentationPipeline(config)
    pipeline.run_all(pause_for_edits=pause)

    if not pause:
        print("\n✓ Pipeline completed successfully!")
        print(f"  Results saved to: {config.output.output_dir}")


def run_stage(config_path: str, stage: str, template_path: str = None):
    """Run specific pipeline stage."""
    config = load_config(config_path)

    pipeline = SegmentationPipeline(config)

    valid_stages = ['load', 'fit', 'export', 'apply', 'compare', 'export_all']

    if stage not in valid_stages:
        print(f"Error: Invalid stage '{stage}'. Valid stages: {', '.join(valid_stages)}")
        sys.exit(1)

    try:
        if stage == 'load':
            pipeline.load_data()

        elif stage == 'fit':
            pipeline.load_data()
            pipeline.fit_baseline()

        elif stage == 'export':
            pipeline.load_data()
            pipeline.fit_baseline()
            template_path = pipeline.export_template()
            print(f"\n✓ Template exported: {template_path}")

        elif stage == 'apply':
            if not template_path:
                print("Error: --template is required for 'apply' stage")
                sys.exit(1)
            pipeline.load_data()
            pipeline.fit_baseline()
            pipeline.apply_modifications(template_path)

        elif stage == 'compare':
            if not template_path:
                print("Error: --template is required for 'compare' stage")
                sys.exit(1)
            pipeline.load_data()
            pipeline.fit_baseline()
            pipeline.apply_modifications(template_path)
            pipeline.compare_results()

        elif stage == 'export_all':
            pipeline.load_data()
            pipeline.fit_baseline()
            if template_path:
                pipeline.apply_modifications(template_path)
                pipeline.compare_results()
            pipeline.export_all()

        print(f"\n✓ Stage '{stage}' completed successfully!")

    except RuntimeError as e:
        print(f"\nError: {e}")
        print(f"Make sure to run previous stages first.")
        sys.exit(1)


def build_config_interactive():
    """Build configuration interactively."""
    config = ConfigBuilder.interactive_build()
    print("\n✓ Configuration created successfully!")
    return config


def analyze_dataset(data_path: str, output_path: str, target_col: str = None):
    """Analyze dataset and create configuration."""
    print(f"\nAnalyzing dataset: {data_path}")

    if not Path(data_path).exists():
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)

    config = ConfigBuilder.from_dataset(
        data_path=data_path,
        target_column=target_col,
        analyze=True
    )

    # Validate
    ConfigBuilder.validate_and_warn(config)

    # Save
    config.to_yaml(output_path)
    print(f"\n✓ Configuration saved to: {output_path}")

    # Show estimated runtime
    runtime = ConfigBuilder.estimate_runtime(config)
    print(f"  Estimated runtime: {runtime}")

    print(f"\nNext steps:")
    print(f"  1. Review/edit configuration: {output_path}")
    print(f"  2. Run pipeline: python run_pipeline.py --config {output_path}")


def validate_config(config_path: str):
    """Validate configuration file."""
    print(f"\nValidating configuration: {config_path}")

    try:
        config = SegmentationConfig.from_yaml(config_path)
        issues = config.validate()

        if issues:
            print("\n⚠ Configuration issues found:")
            for issue in issues:
                print(f"  - {issue}")
            sys.exit(1)
        else:
            print("\n✓ Configuration is valid!")
            print(config.summary())

            # Estimate runtime
            runtime = ConfigBuilder.estimate_runtime(config)
            print(f"\nEstimated runtime: {runtime}")

    except Exception as e:
        print(f"\n✗ Error loading configuration: {e}")
        sys.exit(1)


def export_excel(config_path: str, output_path: str, template_type: str = 'standard'):
    """Export configuration to Excel format."""
    print(f"\nLoading configuration: {config_path}")
    config = load_config(config_path)

    print(f"Exporting to Excel: {output_path}")
    config.to_excel(output_path, template=template_type)
    print(f"\n✓ Configuration exported to Excel: {output_path}")


def export_csv(config_path: str, output_path: str):
    """Export configuration to CSV format."""
    print(f"\nLoading configuration: {config_path}")
    config = load_config(config_path)

    print(f"Exporting to CSV: {output_path}")
    config.to_csv(output_path)
    print(f"\n✓ Configuration exported to CSV: {output_path}")


def create_excel_template(output_path: str, template_type: str = 'simple'):
    """Create blank Excel configuration template."""
    print(f"\nCreating Excel template: {output_path}")
    ConfigBuilder.create_excel_template(output_path, template_type)
    print(f"\n✓ Template created: {output_path}")
    print(f"  Template type: {template_type}")
    print(f"\nNext steps:")
    print(f"  1. Edit template in Excel")
    print(f"  2. Run pipeline: python run_pipeline.py --config {output_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='IRB Segmentation Pipeline CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run entire pipeline
  python run_pipeline.py --config config.yaml

  # Run without pausing (skip template editing)
  python run_pipeline.py --config config.yaml --no-pause

  # Run specific stages
  python run_pipeline.py --config config.yaml --stage fit
  python run_pipeline.py --config config.yaml --stage apply --template output/template.xlsx

  # Build configuration interactively
  python run_pipeline.py --build-config

  # Analyze dataset and create config
  python run_pipeline.py --analyze data/lending_club.csv --output lending_club_config.yaml

  # Validate configuration
  python run_pipeline.py --validate config.yaml

  # Use example configs
  python run_pipeline.py --config config_examples/german_credit.yaml
  python run_pipeline.py --config config_examples/lending_club.yaml

  # Excel/CSV configuration management
  python run_pipeline.py --create-template --excel-output my_config.xlsx --template-type standard
  python run_pipeline.py --export-excel config.yaml --excel-output config.xlsx
  python run_pipeline.py --export-csv config.yaml --csv-output config.csv
  python run_pipeline.py --config my_config.xlsx
        """
    )

    # Main command modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--config',
        type=str,
        help='Path to configuration YAML file'
    )
    mode_group.add_argument(
        '--build-config',
        action='store_true',
        help='Build configuration interactively'
    )
    mode_group.add_argument(
        '--analyze',
        type=str,
        metavar='DATA_PATH',
        help='Analyze dataset and create configuration'
    )
    mode_group.add_argument(
        '--validate',
        type=str,
        metavar='CONFIG_PATH',
        help='Validate configuration file'
    )
    mode_group.add_argument(
        '--export-excel',
        type=str,
        metavar='CONFIG_PATH',
        help='Export configuration to Excel format'
    )
    mode_group.add_argument(
        '--export-csv',
        type=str,
        metavar='CONFIG_PATH',
        help='Export configuration to CSV format'
    )
    mode_group.add_argument(
        '--create-template',
        action='store_true',
        help='Create blank Excel configuration template'
    )

    # Options for --config mode
    parser.add_argument(
        '--stage',
        type=str,
        choices=['load', 'fit', 'export', 'apply', 'compare', 'export_all'],
        help='Run specific pipeline stage'
    )
    parser.add_argument(
        '--template',
        type=str,
        help='Path to template file (required for apply/compare stages)'
    )
    parser.add_argument(
        '--no-pause',
        action='store_true',
        help='Run without pausing for template editing'
    )

    # Options for --analyze mode
    parser.add_argument(
        '--output',
        type=str,
        default='config.yaml',
        help='Output path for generated configuration (default: config.yaml)'
    )
    parser.add_argument(
        '--target-column',
        type=str,
        help='Name of target column (auto-detected if not specified)'
    )

    # Options for Excel/CSV export
    parser.add_argument(
        '--template-type',
        type=str,
        choices=['simple', 'standard', 'advanced'],
        default='simple',
        help='Type of Excel template (default: simple)'
    )
    parser.add_argument(
        '--excel-output',
        type=str,
        help='Output path for Excel export (default: auto-generated)'
    )
    parser.add_argument(
        '--csv-output',
        type=str,
        help='Output path for CSV export (default: auto-generated)'
    )

    args = parser.parse_args()

    # Dispatch to appropriate function
    try:
        if args.config:
            if args.stage:
                run_stage(args.config, args.stage, args.template)
            else:
                run_all(args.config, pause=not args.no_pause)

        elif args.build_config:
            build_config_interactive()

        elif args.analyze:
            analyze_dataset(args.analyze, args.output, args.target_column)

        elif args.validate:
            validate_config(args.validate)

        elif args.export_excel:
            output_path = args.excel_output or args.export_excel.replace('.yaml', '.xlsx').replace('.json', '.xlsx')
            export_excel(args.export_excel, output_path, args.template_type)

        elif args.export_csv:
            output_path = args.csv_output or args.export_csv.replace('.yaml', '.csv').replace('.json', '.csv').replace('.xlsx', '.csv')
            export_csv(args.export_csv, output_path)

        elif args.create_template:
            output_path = args.excel_output or 'config_template.xlsx'
            create_excel_template(output_path, args.template_type)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
