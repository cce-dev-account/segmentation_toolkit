"""
Segmentation Configuration Management

Unified configuration class for the entire segmentation workflow.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union
from pathlib import Path
import json
import yaml
from .params import IRBSegmentationParams


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    source: str  # Path to file or loader name ('german_credit', 'lending_club', etc.)
    data_type: str = 'csv'  # 'csv', 'german_credit', 'lending_club', 'taiwan_credit', 'home_credit'
    sample_size: Optional[int] = None  # For large datasets
    use_oot: bool = False  # Use out-of-time validation split
    random_state: int = 42
    categorical_columns: Optional[List[str]] = None  # Columns to treat as categorical (not one-hot encoded)
    target_column: Optional[str] = None  # For CSV: explicitly specify target column name

    def validate(self) -> List[str]:
        """Validate data configuration."""
        issues = []

        valid_types = ['csv', 'german_credit', 'lending_club', 'taiwan_credit', 'home_credit']
        if self.data_type not in valid_types:
            issues.append(f"Invalid data_type '{self.data_type}'. Valid types: {valid_types}")

        if self.sample_size is not None and self.sample_size <= 0:
            issues.append("sample_size must be positive")

        # Check if source exists for csv type
        if self.data_type == 'csv':
            source_path = Path(self.source)
            if not source_path.exists():
                issues.append(f"Data source not found: {self.source}")

        return issues


@dataclass
class OutputConfig:
    """Configuration for output formats and paths."""

    output_dir: str = "./output"
    output_formats: List[str] = field(default_factory=lambda: ["json", "excel", "html"])

    # Specific output file names (optional, auto-generated if not provided)
    report_name: Optional[str] = None
    template_name: Optional[str] = None
    dashboard_name: Optional[str] = None

    # What to generate
    create_dashboard: bool = True
    create_excel_template: bool = True
    extract_rules: bool = True

    def validate(self) -> List[str]:
        """Validate output configuration."""
        issues = []

        valid_formats = ['json', 'excel', 'html', 'yaml']
        invalid_formats = set(self.output_formats) - set(valid_formats)
        if invalid_formats:
            issues.append(f"Invalid output formats: {invalid_formats}. Valid: {valid_formats}")

        return issues


@dataclass
class SegmentationConfig:
    """
    Master configuration for segmentation workflow.

    This class unifies all configuration needed for the entire pipeline:
    - Data loading
    - IRB segmentation parameters
    - Output generation
    - Workflow control

    Example:
        >>> config = SegmentationConfig(
        ...     data=DataConfig(
        ...         source='data/lending_club.csv',
        ...         data_type='csv',
        ...         sample_size=50000
        ...     ),
        ...     irb_params=IRBSegmentationParams(
        ...         max_depth=5,
        ...         min_defaults_per_leaf=500
        ...     ),
        ...     output=OutputConfig(
        ...         output_dir='./results',
        ...         output_formats=['json', 'html']
        ...     )
        ... )
        >>> config.to_yaml('my_config.yaml')
    """

    # Core configurations
    data: DataConfig
    irb_params: IRBSegmentationParams
    output: OutputConfig = field(default_factory=OutputConfig)

    # Workflow settings
    run_validation: bool = True
    verbose: bool = True

    # Metadata
    name: Optional[str] = None
    description: Optional[str] = None

    def validate(self) -> List[str]:
        """
        Validate entire configuration.

        Returns:
            List of validation error messages (empty if all valid)
        """
        issues = []

        # Validate sub-configs
        issues.extend(self.data.validate())
        issues.extend(self.irb_params.validate_params())
        issues.extend(self.output.validate())

        # Cross-validation checks
        if self.data.sample_size is not None:
            # Check if sample size makes sense with min_samples_leaf
            if self.data.sample_size < self.irb_params.min_samples_leaf * 3:
                issues.append(
                    f"sample_size ({self.data.sample_size}) should be at least 3x "
                    f"min_samples_leaf ({self.irb_params.min_samples_leaf})"
                )

        return issues

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'data': asdict(self.data),
            'irb_params': {
                'max_depth': self.irb_params.max_depth,
                'min_samples_split': self.irb_params.min_samples_split,
                'min_samples_leaf': self.irb_params.min_samples_leaf,
                'criterion': self.irb_params.criterion,
                'random_state': self.irb_params.random_state,
                'min_defaults_per_leaf': self.irb_params.min_defaults_per_leaf,
                'min_default_rate_diff': self.irb_params.min_default_rate_diff,
                'significance_level': self.irb_params.significance_level,
                'min_segment_density': self.irb_params.min_segment_density,
                'max_segment_density': self.irb_params.max_segment_density,
                'monotone_constraints': self.irb_params.monotone_constraints,
                'forced_splits': self.irb_params.forced_splits,
                'validation_tests': self.irb_params.validation_tests,
            },
            'output': asdict(self.output),
            'run_validation': self.run_validation,
            'verbose': self.verbose,
            'name': self.name,
            'description': self.description
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'SegmentationConfig':
        """Create from dictionary."""
        return cls(
            data=DataConfig(**data['data']),
            irb_params=IRBSegmentationParams(**data['irb_params']),
            output=OutputConfig(**data['output']),
            run_validation=data.get('run_validation', True),
            verbose=data.get('verbose', True),
            name=data.get('name'),
            description=data.get('description')
        )

    def to_json(self, filepath: str) -> None:
        """
        Export configuration to JSON file.

        Args:
            filepath: Path to save JSON file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        if self.verbose:
            print(f"Configuration saved to: {filepath}")

    @classmethod
    def from_json(cls, filepath: str) -> 'SegmentationConfig':
        """
        Load configuration from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            SegmentationConfig instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        return cls.from_dict(data)

    def to_yaml(self, filepath: str) -> None:
        """
        Export configuration to YAML file.

        Args:
            filepath: Path to save YAML file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

        if self.verbose:
            print(f"Configuration saved to: {filepath}")

    @classmethod
    def from_yaml(cls, filepath: str) -> 'SegmentationConfig':
        """
        Load configuration from YAML file.

        Args:
            filepath: Path to YAML file

        Returns:
            SegmentationConfig instance
        """
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    def to_excel(self, filepath: str, template: str = 'standard') -> None:
        """
        Export configuration to Excel workbook.

        Args:
            filepath: Path to save Excel file
            template: Template type - 'simple', 'standard', or 'advanced'
        """
        from interfaces.config_to_excel import ConfigToExcel
        ConfigToExcel.export_config(self, filepath, template)

        if self.verbose:
            print(f"Configuration exported to Excel: {filepath}")

    @classmethod
    def from_excel(cls, filepath: str) -> 'SegmentationConfig':
        """
        Load configuration from Excel workbook.

        Args:
            filepath: Path to Excel file

        Returns:
            SegmentationConfig instance
        """
        from interfaces.excel_to_config import ConfigFromExcel
        return ConfigFromExcel.import_config(filepath)

    def to_csv(self, filepath: str) -> None:
        """
        Export configuration to CSV file.

        Args:
            filepath: Path to save CSV file
        """
        from interfaces.config_csv import ConfigCSV
        ConfigCSV.to_csv(self, filepath)

        if self.verbose:
            print(f"Configuration exported to CSV: {filepath}")

    @classmethod
    def from_csv(cls, filepath: str) -> 'SegmentationConfig':
        """
        Load configuration from CSV file.

        Args:
            filepath: Path to CSV file

        Returns:
            SegmentationConfig instance
        """
        from interfaces.config_csv import ConfigCSV
        return ConfigCSV.from_csv(filepath)

    def summary(self) -> str:
        """
        Get human-readable summary of configuration.

        Returns:
            Formatted string describing the configuration
        """
        lines = [
            "=" * 70,
            "SEGMENTATION CONFIGURATION SUMMARY",
            "=" * 70,
        ]

        if self.name:
            lines.extend(["", f"Name: {self.name}"])
        if self.description:
            lines.extend(["", f"Description: {self.description}"])

        lines.extend([
            "",
            "DATA CONFIGURATION:",
            f"  Source: {self.data.source}",
            f"  Type: {self.data.data_type}",
            f"  Sample size: {self.data.sample_size or 'Full dataset'}",
            f"  Use OOT: {self.data.use_oot}",
            "",
            "IRB PARAMETERS:",
            f"  Tree depth: {self.irb_params.max_depth}",
            f"  Min samples per leaf: {self.irb_params.min_samples_leaf}",
            f"  Min defaults per leaf: {self.irb_params.min_defaults_per_leaf}",
            f"  Segment density range: {self.irb_params.min_segment_density:.1%} - {self.irb_params.max_segment_density:.1%}",
            f"  Forced splits: {len(self.irb_params.forced_splits)} features",
            f"  Monotone constraints: {len(self.irb_params.monotone_constraints)} features",
            "",
            "OUTPUT CONFIGURATION:",
            f"  Output directory: {self.output.output_dir}",
            f"  Formats: {', '.join(self.output.output_formats)}",
            f"  Create dashboard: {self.output.create_dashboard}",
            f"  Create Excel template: {self.output.create_excel_template}",
            "",
            "WORKFLOW SETTINGS:",
            f"  Run validation: {self.run_validation}",
            f"  Verbose output: {self.verbose}",
            "=" * 70
        ])

        return "\n".join(lines)


def create_default_config(
    data_source: str,
    data_type: str = 'csv',
    output_dir: str = './output'
) -> SegmentationConfig:
    """
    Create a default configuration with reasonable parameters.

    Args:
        data_source: Path to data file or loader name
        data_type: Type of data source
        output_dir: Directory for output files

    Returns:
        SegmentationConfig with default parameters
    """
    return SegmentationConfig(
        data=DataConfig(
            source=data_source,
            data_type=data_type
        ),
        irb_params=IRBSegmentationParams(),  # Uses defaults from params.py
        output=OutputConfig(
            output_dir=output_dir
        )
    )
