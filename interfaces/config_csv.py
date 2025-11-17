"""
CSV Configuration Import/Export

Simple flat format for configuration - good for version control and programmatic use.
"""

import csv
from typing import Dict, List, Any
import sys
from pathlib import Path
import json

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from irb_segmentation.config import SegmentationConfig, DataConfig, OutputConfig
from irb_segmentation.params import IRBSegmentationParams


class ConfigCSV:
    """Import/export SegmentationConfig to/from CSV."""

    @classmethod
    def to_csv(cls, config: SegmentationConfig, output_path: str):
        """
        Export SegmentationConfig to CSV file.

        Args:
            config: Configuration to export
            output_path: Path to save CSV file
        """
        rows = []

        # Metadata
        rows.append(['metadata', 'name', config.name or '', 'Configuration name'])
        rows.append(['metadata', 'description', config.description or '', 'Configuration description'])
        rows.append(['metadata', 'run_validation', str(config.run_validation), 'Run validation tests'])
        rows.append(['metadata', 'verbose', str(config.verbose), 'Verbose output'])

        # Data config
        rows.append(['data', 'source', config.data.source, 'Data file path or loader name'])
        rows.append(['data', 'data_type', config.data.data_type, 'Type of data source'])
        rows.append(['data', 'sample_size', str(config.data.sample_size) if config.data.sample_size else '', 'Sample size (blank for full dataset)'])
        rows.append(['data', 'use_oot', str(config.data.use_oot), 'Use out-of-time validation'])
        rows.append(['data', 'random_state', str(config.data.random_state), 'Random seed'])
        categorical_cols = ','.join(config.data.categorical_columns) if config.data.categorical_columns else ''
        rows.append(['data', 'categorical_columns', categorical_cols, 'Comma-separated categorical column names'])
        rows.append(['data', 'target_column', config.data.target_column or '', 'Target column name (for CSV files)'])

        # IRB parameters
        irb = config.irb_params
        rows.append(['irb_params', 'max_depth', str(irb.max_depth), 'Maximum tree depth'])
        rows.append(['irb_params', 'min_samples_split', str(irb.min_samples_split), 'Min samples to split node'])
        rows.append(['irb_params', 'min_samples_leaf', str(irb.min_samples_leaf), 'Min samples per leaf'])
        rows.append(['irb_params', 'criterion', irb.criterion, 'Split criterion (gini/entropy)'])
        rows.append(['irb_params', 'random_state', str(irb.random_state), 'Random seed'])
        rows.append(['irb_params', 'min_defaults_per_leaf', str(irb.min_defaults_per_leaf), 'Min defaults per segment'])
        rows.append(['irb_params', 'min_default_rate_diff', str(irb.min_default_rate_diff), 'Min PD difference'])
        rows.append(['irb_params', 'significance_level', str(irb.significance_level), 'Statistical significance level'])
        rows.append(['irb_params', 'min_segment_density', str(irb.min_segment_density), 'Min segment density'])
        rows.append(['irb_params', 'max_segment_density', str(irb.max_segment_density), 'Max segment density'])

        # Forced splits
        for feature, value in irb.forced_splits.items():
            if isinstance(value, list):
                value_str = json.dumps(value)  # Serialize list as JSON
            else:
                value_str = str(value)
            rows.append(['forced_splits', feature, value_str, 'Forced split threshold/values'])

        # Monotone constraints
        for feature, direction in irb.monotone_constraints.items():
            rows.append(['monotone_constraints', feature, str(direction), 'Monotone direction (1/-1/0)'])

        # Validation tests
        for test in irb.validation_tests:
            rows.append(['validation_tests', test, 'true', 'Include this validation test'])

        # Output config
        rows.append(['output', 'output_dir', config.output.output_dir, 'Output directory'])
        rows.append(['output', 'output_formats', ','.join(config.output.output_formats), 'Comma-separated output formats'])
        rows.append(['output', 'report_name', config.output.report_name or '', 'Custom report name'])
        rows.append(['output', 'template_name', config.output.template_name or '', 'Custom template name'])
        rows.append(['output', 'dashboard_name', config.output.dashboard_name or '', 'Custom dashboard name'])
        rows.append(['output', 'create_dashboard', str(config.output.create_dashboard), 'Create HTML dashboard'])
        rows.append(['output', 'create_excel_template', str(config.output.create_excel_template), 'Create Excel template'])
        rows.append(['output', 'extract_rules', str(config.output.extract_rules), 'Extract segment rules'])

        # Write CSV
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['section', 'parameter', 'value', 'description'])
            writer.writerows(rows)

        print(f"Configuration exported to CSV: {output_path}")

    @classmethod
    def from_csv(cls, csv_path: str) -> SegmentationConfig:
        """
        Import SegmentationConfig from CSV file.

        Args:
            csv_path: Path to CSV file

        Returns:
            SegmentationConfig instance
        """
        # Parse CSV into sections
        sections = {
            'metadata': {},
            'data': {},
            'irb_params': {},
            'forced_splits': {},
            'monotone_constraints': {},
            'validation_tests': [],
            'output': {}
        }

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                section = row['section']
                parameter = row['parameter']
                value = row['value']

                if section == 'validation_tests':
                    if cls._parse_bool(value):
                        sections['validation_tests'].append(parameter)
                elif section == 'forced_splits':
                    # Check if value is JSON (list)
                    if value.startswith('['):
                        sections['forced_splits'][parameter] = json.loads(value)
                    else:
                        # Try to parse as float, fallback to string
                        try:
                            sections['forced_splits'][parameter] = float(value)
                        except ValueError:
                            sections['forced_splits'][parameter] = value
                elif section == 'monotone_constraints':
                    sections['monotone_constraints'][parameter] = int(value)
                else:
                    sections[section][parameter] = value

        # Build config objects
        metadata = sections['metadata']
        data_dict = sections['data']
        irb_dict = sections['irb_params']
        output_dict = sections['output']

        # Parse data config
        categorical_cols_raw = data_dict.get('categorical_columns', '')
        if categorical_cols_raw and categorical_cols_raw.strip():
            categorical_columns = [col.strip() for col in categorical_cols_raw.split(',')]
        else:
            categorical_columns = None

        target_column = data_dict.get('target_column', '') or None

        data_config = DataConfig(
            source=data_dict.get('source', ''),
            data_type=data_dict.get('data_type', 'csv'),
            sample_size=int(data_dict['sample_size']) if data_dict.get('sample_size') else None,
            use_oot=cls._parse_bool(data_dict.get('use_oot', 'False')),
            random_state=int(data_dict.get('random_state', 42)),
            categorical_columns=categorical_columns,
            target_column=target_column
        )

        # Parse IRB params
        irb_params = IRBSegmentationParams(
            max_depth=int(irb_dict.get('max_depth', 3)),
            min_samples_split=int(irb_dict.get('min_samples_split', 1000)),
            min_samples_leaf=int(irb_dict.get('min_samples_leaf', 500)),
            criterion=irb_dict.get('criterion', 'gini'),
            random_state=int(irb_dict.get('random_state', 42)),
            min_defaults_per_leaf=int(irb_dict.get('min_defaults_per_leaf', 20)),
            min_default_rate_diff=float(irb_dict.get('min_default_rate_diff', 0.001)),
            significance_level=float(irb_dict.get('significance_level', 0.01)),
            min_segment_density=float(irb_dict.get('min_segment_density', 0.10)),
            max_segment_density=float(irb_dict.get('max_segment_density', 0.50)),
            forced_splits=sections['forced_splits'],
            monotone_constraints=sections['monotone_constraints'],
            validation_tests=sections['validation_tests'] or ['chi_squared', 'psi', 'binomial']
        )

        # Parse output config
        output_formats = output_dict.get('output_formats', 'json,html').split(',')
        output_config = OutputConfig(
            output_dir=output_dict.get('output_dir', './output'),
            output_formats=[f.strip() for f in output_formats],
            report_name=output_dict.get('report_name') or None,
            template_name=output_dict.get('template_name') or None,
            dashboard_name=output_dict.get('dashboard_name') or None,
            create_dashboard=cls._parse_bool(output_dict.get('create_dashboard', 'True')),
            create_excel_template=cls._parse_bool(output_dict.get('create_excel_template', 'True')),
            extract_rules=cls._parse_bool(output_dict.get('extract_rules', 'True'))
        )

        # Create config
        config = SegmentationConfig(
            data=data_config,
            irb_params=irb_params,
            output=output_config,
            run_validation=cls._parse_bool(metadata.get('run_validation', 'True')),
            verbose=cls._parse_bool(metadata.get('verbose', 'True')),
            name=metadata.get('name') or None,
            description=metadata.get('description') or None
        )

        return config

    @staticmethod
    def _parse_bool(value: Any) -> bool:
        """Parse boolean from CSV string."""
        if isinstance(value, bool):
            return value
        if not value:
            return False
        return str(value).strip().lower() in ['true', 'yes', '1', 'y']


if __name__ == '__main__':
    # Test CSV export/import
    from irb_segmentation import create_default_config

    config = create_default_config(
        data_source='data/test.csv',
        data_type='csv'
    )
    config.name = "Test CSV Config"

    # Export
    ConfigCSV.to_csv(config, 'test_config.csv')

    # Import
    config_imported = ConfigCSV.from_csv('test_config.csv')

    print("\nOriginal:")
    print(config.summary())

    print("\nImported:")
    print(config_imported.summary())

    print("\nRound-trip test complete!")
