"""
Configuration Builder

Interactive helper to generate SegmentationConfig based on dataset analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from pathlib import Path
import warnings

from .config import SegmentationConfig, DataConfig, OutputConfig
from .params import IRBSegmentationParams


class ConfigBuilder:
    """
    Helper class to build SegmentationConfig from dataset analysis.

    Provides smart defaults based on dataset characteristics like size,
    default rate, and number of features.
    """

    @staticmethod
    def from_dataset(
        data_path: str,
        target_column: Optional[str] = None,
        analyze: bool = True,
        data_type: str = 'csv',
        output_dir: str = './output',
        name: Optional[str] = None
    ) -> SegmentationConfig:
        """
        Create configuration by analyzing a dataset.

        Args:
            data_path: Path to data file or loader name
            target_column: Name of target column (auto-detected if None)
            analyze: Whether to analyze dataset and suggest parameters
            data_type: Type of data source
            output_dir: Output directory
            name: Name for this configuration

        Returns:
            SegmentationConfig with suggested parameters

        Example:
            >>> config = ConfigBuilder.from_dataset(
            ...     'data/my_data.csv',
            ...     target_column='default',
            ...     analyze=True
            ... )
            >>> config.to_yaml('my_config.yaml')
        """
        if analyze and data_type == 'csv':
            # Analyze CSV file
            df = pd.read_csv(data_path)

            # Find target column
            if target_column is None:
                for col_name in ['default', 'target', 'y', 'label']:
                    if col_name in df.columns:
                        target_column = col_name
                        break
                if target_column is None:
                    target_column = df.columns[-1]
                    warnings.warn(f"Target column not specified, using last column: {target_column}")

            # Extract characteristics
            n_rows = len(df)
            n_features = len(df.select_dtypes(include=[np.number]).columns) - 1  # Exclude target
            default_rate = df[target_column].mean()

            print(f"\nDataset Analysis:")
            print(f"  Rows: {n_rows:,}")
            print(f"  Numeric features: {n_features}")
            print(f"  Default rate: {default_rate:.4f}")

            # Suggest parameters
            irb_params = ConfigBuilder.suggest_parameters(
                n_rows=n_rows,
                default_rate=default_rate,
                n_features=n_features
            )

        else:
            # Use defaults for non-CSV or when not analyzing
            irb_params = IRBSegmentationParams()

        # Build config
        config = SegmentationConfig(
            data=DataConfig(
                source=data_path,
                data_type=data_type
            ),
            irb_params=irb_params,
            output=OutputConfig(
                output_dir=output_dir
            ),
            name=name or f"Segmentation Analysis - {Path(data_path).stem}"
        )

        return config

    @staticmethod
    def suggest_parameters(
        n_rows: int,
        default_rate: float,
        n_features: int
    ) -> IRBSegmentationParams:
        """
        Suggest IRB parameters based on dataset characteristics.

        Uses heuristics based on:
        - Dataset size → min_samples_leaf, min_defaults_per_leaf
        - Default rate → adjustments for rare events
        - Number of features → max_depth

        Args:
            n_rows: Number of observations
            default_rate: Proportion of defaults
            n_features: Number of features

        Returns:
            IRBSegmentationParams with suggested values
        """
        print(f"\nSuggesting parameters...")

        # Determine max_depth based on dataset size and features
        if n_rows < 5000:
            max_depth = 2
        elif n_rows < 50000:
            max_depth = 3
        elif n_rows < 500000:
            max_depth = 4
        else:
            max_depth = 5

        # Limit depth by features (rule of thumb: depth ≤ log2(features) + 2)
        max_depth_by_features = int(np.log2(max(n_features, 2))) + 2
        max_depth = min(max_depth, max_depth_by_features)

        # Determine min_samples_leaf based on size
        if n_rows < 5000:
            min_samples_leaf = max(50, int(n_rows * 0.05))
        elif n_rows < 50000:
            min_samples_leaf = max(500, int(n_rows * 0.02))
        elif n_rows < 500000:
            min_samples_leaf = max(2000, int(n_rows * 0.01))
        else:
            min_samples_leaf = max(10000, int(n_rows * 0.005))

        # Determine min_defaults_per_leaf based on default rate and size
        expected_defaults = n_rows * default_rate
        min_segments = 3  # Minimum useful segmentation

        if default_rate < 0.02:
            # Very low default rate - need larger samples
            min_defaults_per_leaf = max(5, int(expected_defaults / (min_segments * 3)))
            warnings.warn(
                f"Low default rate ({default_rate:.2%}) - consider using more data "
                f"or relaxing min_defaults_per_leaf"
            )
        elif default_rate < 0.05:
            min_defaults_per_leaf = max(10, int(expected_defaults / (min_segments * 4)))
        elif default_rate < 0.10:
            min_defaults_per_leaf = max(20, int(expected_defaults / (min_segments * 5)))
        else:
            min_defaults_per_leaf = max(30, int(expected_defaults / (min_segments * 6)))

        # Basel II/III recommends at least 20
        if min_defaults_per_leaf < 20:
            warnings.warn(
                f"Suggested min_defaults_per_leaf ({min_defaults_per_leaf}) is below "
                f"Basel II/III recommended minimum of 20"
            )

        # Determine segment density based on size
        if n_rows < 10000:
            min_segment_density = 0.10  # 10%
            max_segment_density = 0.50  # 50%
        elif n_rows < 100000:
            min_segment_density = 0.08
            max_segment_density = 0.45
        else:
            min_segment_density = 0.05  # 5%
            max_segment_density = 0.40  # 40%

        # Create parameters
        params = IRBSegmentationParams(
            max_depth=max_depth,
            min_samples_split=min_samples_leaf * 2,
            min_samples_leaf=min_samples_leaf,
            min_defaults_per_leaf=min_defaults_per_leaf,
            min_segment_density=min_segment_density,
            max_segment_density=max_segment_density
        )

        # Print suggestions
        print(f"\n  Suggested Parameters:")
        print(f"    max_depth: {max_depth} (based on {n_rows:,} rows, {n_features} features)")
        print(f"    min_samples_leaf: {min_samples_leaf:,}")
        print(f"    min_defaults_per_leaf: {min_defaults_per_leaf}")
        print(f"    min_segment_density: {min_segment_density:.1%}")
        print(f"    max_segment_density: {max_segment_density:.1%}")
        print(f"\n  Rationale:")
        print(f"    - Dataset size: {n_rows:,} rows → depth {max_depth}, leaf size {min_samples_leaf:,}")
        print(f"    - Default rate: {default_rate:.2%} → {min_defaults_per_leaf} min defaults")
        print(f"    - Expected segments: ~{2**max_depth} (max tree leaves)")

        return params

    @staticmethod
    def estimate_runtime(config: SegmentationConfig) -> str:
        """
        Estimate runtime based on configuration.

        Args:
            config: SegmentationConfig to estimate

        Returns:
            Human-readable runtime estimate
        """
        # Rough heuristics based on experience
        if config.data.data_type == 'german_credit':
            return "~30 seconds"
        elif config.data.sample_size:
            if config.data.sample_size < 10000:
                return "~1 minute"
            elif config.data.sample_size < 100000:
                return "~3-5 minutes"
            else:
                return "~10-15 minutes"
        else:
            # Based on data type defaults
            type_estimates = {
                'german_credit': "~30 seconds",
                'taiwan_credit': "~2-3 minutes",
                'lending_club': "~10-15 minutes",
                'home_credit': "~5-10 minutes"
            }
            return type_estimates.get(config.data.data_type, "~5-10 minutes")

    @staticmethod
    def validate_and_warn(config: SegmentationConfig) -> None:
        """
        Validate configuration and print warnings for potential issues.

        Args:
            config: Configuration to validate
        """
        issues = config.validate()

        if issues:
            print("\n⚠ Configuration Issues Found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("\n✓ Configuration is valid")

        # Additional warnings
        irb = config.irb_params

        # Check if parameters are too restrictive
        expected_segments = 2 ** irb.max_depth
        min_obs_per_segment = irb.min_samples_leaf * expected_segments

        if config.data.sample_size and config.data.sample_size < min_obs_per_segment:
            warnings.warn(
                f"Sample size ({config.data.sample_size:,}) may be too small for "
                f"{expected_segments} segments with {irb.min_samples_leaf:,} samples each"
            )

        # Check density constraints
        if irb.min_segment_density * expected_segments > 1.0:
            warnings.warn(
                f"min_segment_density ({irb.min_segment_density:.1%}) may be too high "
                f"for {expected_segments} segments"
            )

    @staticmethod
    def to_excel(config: SegmentationConfig, output_path: str, template: str = 'standard'):
        """
        Export configuration to Excel workbook.

        Args:
            config: Configuration to export
            output_path: Path to save Excel file
            template: Template type - 'simple', 'standard', or 'advanced'
        """
        from interfaces.config_to_excel import ConfigToExcel
        ConfigToExcel.export_config(config, output_path, template)

    @staticmethod
    def from_excel(excel_path: str) -> SegmentationConfig:
        """
        Load configuration from Excel workbook.

        Args:
            excel_path: Path to Excel file

        Returns:
            SegmentationConfig instance
        """
        from interfaces.excel_to_config import ConfigFromExcel
        return ConfigFromExcel.import_config(excel_path)

    @staticmethod
    def validate_excel(excel_path: str) -> None:
        """
        Validate Excel configuration and print issues.

        Args:
            excel_path: Path to Excel file
        """
        from interfaces.excel_to_config import ConfigFromExcel
        issues = ConfigFromExcel.validate_excel(excel_path)

        if issues:
            print("\n⚠ Excel Validation Issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("\n✓ Excel configuration is valid")

    @staticmethod
    def create_excel_template(output_path: str, template_type: str = 'simple'):
        """
        Create blank Excel configuration template.

        Args:
            output_path: Path to save template
            template_type: 'simple', 'standard', or 'advanced'
        """
        from interfaces.config_to_excel import ConfigToExcel
        ConfigToExcel.create_template(output_path, template_type)
        print(f"\nExcel template created: {output_path}")
        print(f"Template type: {template_type}")

    @staticmethod
    def interactive_build() -> SegmentationConfig:
        """
        Build configuration interactively via command line prompts.

        Returns:
            SegmentationConfig based on user inputs
        """
        print("\n" + "=" * 70)
        print("INTERACTIVE CONFIGURATION BUILDER")
        print("=" * 70)

        # Data source
        print("\n1. DATA SOURCE")
        data_source = input("   Enter path to CSV file or loader name (german_credit/lending_club/etc): ").strip()

        data_type = 'csv'
        if data_source in ['german_credit', 'lending_club', 'taiwan_credit', 'home_credit']:
            data_type = data_source

        # Sample size
        print("\n2. SAMPLE SIZE (optional)")
        sample_input = input("   Enter sample size (leave blank for full dataset): ").strip()
        sample_size = int(sample_input) if sample_input else None

        # Analyze dataset if CSV
        if data_type == 'csv' and Path(data_source).exists():
            analyze = input("\n3. Analyze dataset and suggest parameters? (y/n): ").strip().lower() == 'y'
            if analyze:
                return ConfigBuilder.from_dataset(
                    data_path=data_source,
                    analyze=True,
                    data_type=data_type
                )

        # Manual parameter entry
        print("\n3. IRB PARAMETERS")
        print("   (Press Enter to use defaults)")

        max_depth = input(f"   max_depth [{3}]: ").strip()
        max_depth = int(max_depth) if max_depth else 3

        min_samples_leaf = input(f"   min_samples_leaf [{500}]: ").strip()
        min_samples_leaf = int(min_samples_leaf) if min_samples_leaf else 500

        min_defaults = input(f"   min_defaults_per_leaf [{20}]: ").strip()
        min_defaults = int(min_defaults) if min_defaults else 20

        # Build config
        config = SegmentationConfig(
            data=DataConfig(
                source=data_source,
                data_type=data_type,
                sample_size=sample_size
            ),
            irb_params=IRBSegmentationParams(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                min_defaults_per_leaf=min_defaults
            )
        )

        # Validate
        ConfigBuilder.validate_and_warn(config)

        # Save option
        print("\n4. SAVE CONFIGURATION")
        save = input("   Save configuration? (y/n): ").strip().lower() == 'y'
        if save:
            filename = input("   Filename [config.yaml]: ").strip() or "config.yaml"
            config.to_yaml(filename)

        return config


def quick_config(
    dataset_size: str,
    default_rate: str = 'medium',
    output_dir: str = './output'
) -> SegmentationConfig:
    """
    Create quick configuration based on simple categories.

    Args:
        dataset_size: 'small' (<10K), 'medium' (10K-100K), 'large' (>100K)
        default_rate: 'low' (<5%), 'medium' (5-15%), 'high' (>15%)
        output_dir: Output directory

    Returns:
        SegmentationConfig with preset parameters

    Example:
        >>> config = quick_config('large', 'medium')
        >>> config.to_yaml('quick_config.yaml')
    """
    # Parameter presets
    presets = {
        'small': {
            'low': IRBSegmentationParams(max_depth=2, min_samples_leaf=100, min_defaults_per_leaf=5),
            'medium': IRBSegmentationParams(max_depth=3, min_samples_leaf=100, min_defaults_per_leaf=10),
            'high': IRBSegmentationParams(max_depth=3, min_samples_leaf=100, min_defaults_per_leaf=15)
        },
        'medium': {
            'low': IRBSegmentationParams(max_depth=3, min_samples_leaf=1000, min_defaults_per_leaf=15),
            'medium': IRBSegmentationParams(max_depth=4, min_samples_leaf=1000, min_defaults_per_leaf=25),
            'high': IRBSegmentationParams(max_depth=4, min_samples_leaf=1000, min_defaults_per_leaf=35)
        },
        'large': {
            'low': IRBSegmentationParams(max_depth=5, min_samples_leaf=10000, min_defaults_per_leaf=200),
            'medium': IRBSegmentationParams(max_depth=5, min_samples_leaf=10000, min_defaults_per_leaf=500),
            'high': IRBSegmentationParams(max_depth=5, min_samples_leaf=10000, min_defaults_per_leaf=800)
        }
    }

    params = presets[dataset_size][default_rate]

    config = SegmentationConfig(
        data=DataConfig(
            source='',  # User must set
            data_type='csv'
        ),
        irb_params=params,
        output=OutputConfig(output_dir=output_dir),
        name=f"Quick Config: {dataset_size} dataset, {default_rate} default rate"
    )

    print(f"\nCreated quick configuration:")
    print(f"  Dataset size: {dataset_size}")
    print(f"  Default rate: {default_rate}")
    print(f"  max_depth: {params.max_depth}")
    print(f"  min_samples_leaf: {params.min_samples_leaf}")
    print(f"  min_defaults_per_leaf: {params.min_defaults_per_leaf}")
    print(f"\n⚠ Remember to set config.data.source before running!")

    return config
