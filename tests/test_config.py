"""
Unit Tests for IRB Segmentation Configuration

Tests cover:
- DataConfig validation and properties
- OutputConfig validation and properties
- LoggingConfig validation
- SegmentationConfig validation and cross-validation
- Configuration serialization (JSON, YAML, dict)
- Configuration deserialization
- Default configuration creation
- Edge cases and error handling
"""

import pytest
import json
import yaml
import tempfile
from pathlib import Path

from irb_segmentation.config import (
    DataConfig,
    OutputConfig,
    LoggingConfig,
    SegmentationConfig,
    create_default_config
)
from irb_segmentation.params import IRBSegmentationParams


@pytest.mark.unit
class TestDataConfig:
    """Test DataConfig validation and properties."""

    def test_create_valid_data_config(self, temp_data_file):
        """Test creating a valid DataConfig."""
        config = DataConfig(
            source=str(temp_data_file),
            data_type='csv',
            target_column='default'
        )

        assert config.source == str(temp_data_file)
        assert config.data_type == 'csv'
        assert config.target_column == 'default'

    def test_data_config_defaults(self):
        """Test DataConfig default values."""
        config = DataConfig(source='test.csv')

        assert config.data_type == 'csv'
        assert config.sample_size is None
        assert config.use_oot is False
        assert config.random_state == 42
        assert config.categorical_columns is None
        assert config.target_column is None

    def test_data_config_with_sample_size(self):
        """Test DataConfig with sample size."""
        config = DataConfig(
            source='test.csv',
            sample_size=10000
        )

        assert config.sample_size == 10000

    def test_data_config_validation_invalid_type(self):
        """Test validation catches invalid data_type."""
        config = DataConfig(
            source='test.csv',
            data_type='invalid_type'
        )

        issues = config.validate()
        assert len(issues) > 0
        assert any('Invalid data_type' in issue for issue in issues)

    def test_data_config_validation_negative_sample_size(self):
        """Test validation catches negative sample_size."""
        config = DataConfig(
            source='test.csv',
            sample_size=-100
        )

        issues = config.validate()
        assert len(issues) > 0
        assert any('positive' in issue.lower() for issue in issues)

    def test_data_config_validation_missing_csv_file(self):
        """Test validation catches missing CSV file."""
        config = DataConfig(
            source='nonexistent_file.csv',
            data_type='csv'
        )

        issues = config.validate()
        assert len(issues) > 0
        assert any('not found' in issue.lower() for issue in issues)

    def test_data_config_validation_valid_config(self, temp_data_file):
        """Test validation passes for valid config."""
        config = DataConfig(
            source=str(temp_data_file),
            data_type='csv',
            sample_size=1000
        )

        issues = config.validate()
        assert len(issues) == 0

    def test_data_config_with_categorical_columns(self):
        """Test DataConfig with categorical columns."""
        config = DataConfig(
            source='test.csv',
            categorical_columns=['col1', 'col2', 'col3']
        )

        assert len(config.categorical_columns) == 3
        assert 'col1' in config.categorical_columns


@pytest.mark.unit
class TestOutputConfig:
    """Test OutputConfig validation and properties."""

    def test_create_valid_output_config(self):
        """Test creating a valid OutputConfig."""
        config = OutputConfig(
            output_dir='./results',
            output_formats=['json', 'excel', 'html']
        )

        assert config.output_dir == './results'
        assert len(config.output_formats) == 3

    def test_output_config_defaults(self):
        """Test OutputConfig default values."""
        config = OutputConfig()

        assert config.output_dir == "./output"
        assert 'json' in config.output_formats
        assert 'excel' in config.output_formats
        assert 'html' in config.output_formats
        assert config.create_dashboard is True
        assert config.create_excel_template is True
        assert config.extract_rules is True

    def test_output_config_custom_names(self):
        """Test OutputConfig with custom file names."""
        config = OutputConfig(
            report_name='my_report.html',
            template_name='my_template.xlsx',
            dashboard_name='my_dashboard.html'
        )

        assert config.report_name == 'my_report.html'
        assert config.template_name == 'my_template.xlsx'
        assert config.dashboard_name == 'my_dashboard.html'

    def test_output_config_validation_invalid_format(self):
        """Test validation catches invalid output format."""
        config = OutputConfig(
            output_formats=['json', 'invalid_format', 'xml']
        )

        issues = config.validate()
        assert len(issues) > 0
        assert any('Invalid output formats' in issue for issue in issues)

    def test_output_config_validation_valid_formats(self):
        """Test validation passes for valid formats."""
        config = OutputConfig(
            output_formats=['json', 'excel', 'html', 'yaml']
        )

        issues = config.validate()
        assert len(issues) == 0

    def test_output_config_disable_features(self):
        """Test disabling output features."""
        config = OutputConfig(
            create_dashboard=False,
            create_excel_template=False,
            extract_rules=False
        )

        assert config.create_dashboard is False
        assert config.create_excel_template is False
        assert config.extract_rules is False


@pytest.mark.unit
class TestLoggingConfig:
    """Test LoggingConfig validation and properties."""

    def test_create_valid_logging_config(self):
        """Test creating a valid LoggingConfig."""
        config = LoggingConfig(
            level='DEBUG',
            log_file='output/training.log'
        )

        assert config.level == 'DEBUG'
        assert config.log_file == 'output/training.log'

    def test_logging_config_defaults(self):
        """Test LoggingConfig default values."""
        config = LoggingConfig()

        assert config.level == 'INFO'
        assert config.log_file is None
        assert config.log_format is None

    def test_logging_config_all_levels(self):
        """Test all valid log levels."""
        levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

        for level in levels:
            config = LoggingConfig(level=level)
            issues = config.validate()
            assert len(issues) == 0

    def test_logging_config_validation_invalid_level(self):
        """Test validation catches invalid log level."""
        config = LoggingConfig(level='INVALID')

        issues = config.validate()
        assert len(issues) > 0
        assert any('Invalid log level' in issue for issue in issues)

    def test_logging_config_case_insensitive_validation(self):
        """Test that validation handles lowercase levels."""
        config = LoggingConfig(level='info')

        # Should still validate (upper() is applied in validation)
        issues = config.validate()
        assert len(issues) == 0

    def test_logging_config_custom_format(self):
        """Test LoggingConfig with custom format."""
        custom_format = '%(asctime)s - %(name)s - %(message)s'
        config = LoggingConfig(log_format=custom_format)

        assert config.log_format == custom_format


@pytest.mark.unit
class TestSegmentationConfig:
    """Test SegmentationConfig validation and integration."""

    def test_create_valid_segmentation_config(self, temp_data_file, tmp_path):
        """Test creating a valid SegmentationConfig."""
        config = SegmentationConfig(
            data=DataConfig(
                source=str(temp_data_file),
                data_type='csv'
            ),
            irb_params=IRBSegmentationParams(),
            output=OutputConfig(
                output_dir=str(tmp_path / 'output')
            )
        )

        assert config.data.source == str(temp_data_file)
        assert config.irb_params is not None
        assert config.output.output_dir == str(tmp_path / 'output')

    def test_segmentation_config_defaults(self, temp_data_file):
        """Test SegmentationConfig default values."""
        config = SegmentationConfig(
            data=DataConfig(source=str(temp_data_file)),
            irb_params=IRBSegmentationParams()
        )

        # Output uses defaults
        assert config.output.output_dir == "./output"

        # Workflow settings
        assert config.run_validation is True
        assert config.verbose is True

        # Metadata
        assert config.name is None
        assert config.description is None
        assert config.logging is None

    def test_segmentation_config_with_logging(self, temp_data_file):
        """Test SegmentationConfig with logging."""
        config = SegmentationConfig(
            data=DataConfig(source=str(temp_data_file)),
            irb_params=IRBSegmentationParams(),
            logging=LoggingConfig(
                level='DEBUG',
                log_file='output/debug.log'
            )
        )

        assert config.logging is not None
        assert config.logging.level == 'DEBUG'

    def test_segmentation_config_with_metadata(self, temp_data_file):
        """Test SegmentationConfig with name and description."""
        config = SegmentationConfig(
            data=DataConfig(source=str(temp_data_file)),
            irb_params=IRBSegmentationParams(),
            name='Test Segmentation',
            description='A test configuration for unit testing'
        )

        assert config.name == 'Test Segmentation'
        assert config.description == 'A test configuration for unit testing'

    def test_segmentation_config_validation_propagates(self, temp_data_file):
        """Test that validation checks all sub-configs."""
        # Create config with issues that validation (not constructor) will catch
        config = SegmentationConfig(
            data=DataConfig(
                source='nonexistent.csv',  # Invalid - file doesn't exist
                data_type='csv'
            ),
            irb_params=IRBSegmentationParams(),  # Valid params
            output=OutputConfig(
                output_formats=['invalid_format']  # Invalid format
            ),
            logging=LoggingConfig(
                level='INVALID'  # Invalid level
            )
        )

        issues = config.validate()
        # Should catch multiple issues from different sub-configs
        assert len(issues) >= 3  # At least data, output, and logging issues

    def test_segmentation_config_cross_validation_sample_size(self, temp_data_file):
        """Test cross-validation between sample_size and min_samples_leaf."""
        config = SegmentationConfig(
            data=DataConfig(
                source=str(temp_data_file),
                data_type='csv',
                sample_size=100  # Too small for min_samples_leaf
            ),
            irb_params=IRBSegmentationParams(
                min_samples_leaf=50  # 100 < 3*50
            )
        )

        issues = config.validate()
        assert len(issues) > 0
        assert any('sample_size' in issue and '3x' in issue for issue in issues)

    def test_segmentation_config_valid_cross_validation(self, temp_data_file):
        """Test that valid cross-validation passes."""
        config = SegmentationConfig(
            data=DataConfig(
                source=str(temp_data_file),
                data_type='csv',
                sample_size=10000
            ),
            irb_params=IRBSegmentationParams(
                min_samples_leaf=50
            )
        )

        issues = config.validate()
        # Should not have cross-validation issues
        cross_val_issues = [i for i in issues if 'sample_size' in i and '3x' in i]
        assert len(cross_val_issues) == 0


@pytest.mark.unit
class TestConfigSerialization:
    """Test configuration serialization and deserialization."""

    def test_config_to_dict(self, temp_data_file):
        """Test converting config to dictionary."""
        config = SegmentationConfig(
            data=DataConfig(source=str(temp_data_file)),
            irb_params=IRBSegmentationParams(),
            name='Test Config'
        )

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert 'data' in config_dict
        assert 'irb_params' in config_dict
        assert 'output' in config_dict
        assert config_dict['name'] == 'Test Config'

    def test_config_from_dict(self, temp_data_file):
        """Test creating config from dictionary."""
        config_dict = {
            'data': {
                'source': str(temp_data_file),
                'data_type': 'csv',
                'sample_size': None,
                'use_oot': False,
                'random_state': 42,
                'categorical_columns': None,
                'target_column': 'default'
            },
            'irb_params': {
                'max_depth': 5,
                'min_samples_split': 100,
                'min_samples_leaf': 50,
                'criterion': 'gini',
                'random_state': 42,
                'min_defaults_per_leaf': 10,
                'min_default_rate_diff': 0.02,
                'significance_level': 0.05,
                'min_segment_density': 0.05,
                'max_segment_density': 0.50,
                'monotone_constraints': {},
                'forced_splits': {},
                'validation_tests': []
            },
            'output': {
                'output_dir': './output',
                'output_formats': ['json', 'excel'],
                'report_name': None,
                'template_name': None,
                'dashboard_name': None,
                'create_dashboard': True,
                'create_excel_template': True,
                'extract_rules': True
            },
            'run_validation': True,
            'verbose': True,
            'name': 'From Dict',
            'description': None
        }

        config = SegmentationConfig.from_dict(config_dict)

        assert config.data.source == str(temp_data_file)
        assert config.irb_params.max_depth == 5
        assert config.name == 'From Dict'

    def test_config_to_json(self, temp_data_file, tmp_path):
        """Test exporting config to JSON."""
        config = SegmentationConfig(
            data=DataConfig(source=str(temp_data_file)),
            irb_params=IRBSegmentationParams(),
            verbose=False  # Suppress print
        )

        json_path = tmp_path / "config.json"
        config.to_json(str(json_path))

        assert json_path.exists()

        # Verify it's valid JSON
        with open(json_path) as f:
            loaded = json.load(f)
        assert 'data' in loaded

    def test_config_from_json(self, temp_data_file, tmp_path):
        """Test loading config from JSON."""
        config_dict = {
            'data': {
                'source': str(temp_data_file),
                'data_type': 'csv',
                'sample_size': None,
                'use_oot': False,
                'random_state': 42,
                'categorical_columns': None,
                'target_column': 'default'
            },
            'irb_params': {
                'max_depth': 5,
                'min_samples_split': 100,
                'min_samples_leaf': 50,
                'criterion': 'gini',
                'random_state': 42,
                'min_defaults_per_leaf': 10,
                'min_default_rate_diff': 0.02,
                'significance_level': 0.05,
                'min_segment_density': 0.05,
                'max_segment_density': 0.50,
                'monotone_constraints': {},
                'forced_splits': {},
                'validation_tests': []
            },
            'output': {
                'output_dir': './output',
                'output_formats': ['json'],
                'report_name': None,
                'template_name': None,
                'dashboard_name': None,
                'create_dashboard': True,
                'create_excel_template': True,
                'extract_rules': True
            },
            'run_validation': True,
            'verbose': False,
            'name': 'JSON Test',
            'description': None
        }

        json_path = tmp_path / "config.json"
        with open(json_path, 'w') as f:
            json.dump(config_dict, f)

        config = SegmentationConfig.from_json(str(json_path))

        assert config.name == 'JSON Test'
        assert config.data.source == str(temp_data_file)

    def test_config_to_yaml(self, temp_data_file, tmp_path):
        """Test exporting config to YAML."""
        config = SegmentationConfig(
            data=DataConfig(source=str(temp_data_file)),
            irb_params=IRBSegmentationParams(),
            name='YAML Test',
            verbose=False
        )

        yaml_path = tmp_path / "config.yaml"
        config.to_yaml(str(yaml_path))

        assert yaml_path.exists()

        # Verify it's valid YAML
        with open(yaml_path) as f:
            loaded = yaml.safe_load(f)
        assert 'data' in loaded
        assert loaded['name'] == 'YAML Test'

    def test_config_from_yaml(self, temp_data_file, tmp_path):
        """Test loading config from YAML."""
        config_dict = {
            'data': {
                'source': str(temp_data_file),
                'data_type': 'csv',
                'sample_size': None,
                'use_oot': False,
                'random_state': 42,
                'categorical_columns': None,
                'target_column': 'default'
            },
            'irb_params': {
                'max_depth': 5,
                'min_samples_split': 100,
                'min_samples_leaf': 50,
                'criterion': 'gini',
                'random_state': 42,
                'min_defaults_per_leaf': 10,
                'min_default_rate_diff': 0.02,
                'significance_level': 0.05,
                'min_segment_density': 0.05,
                'max_segment_density': 0.50,
                'monotone_constraints': {},
                'forced_splits': {},
                'validation_tests': []
            },
            'output': {
                'output_dir': './output',
                'output_formats': ['yaml'],
                'report_name': None,
                'template_name': None,
                'dashboard_name': None,
                'create_dashboard': True,
                'create_excel_template': True,
                'extract_rules': True
            },
            'run_validation': True,
            'verbose': False,
            'name': 'YAML Test',
            'description': 'A YAML test configuration'
        }

        yaml_path = tmp_path / "config.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f)

        config = SegmentationConfig.from_yaml(str(yaml_path))

        assert config.name == 'YAML Test'
        assert config.description == 'A YAML test configuration'

    def test_config_roundtrip_dict(self, temp_data_file):
        """Test config -> dict -> config roundtrip."""
        original = SegmentationConfig(
            data=DataConfig(
                source=str(temp_data_file),
                sample_size=5000
            ),
            irb_params=IRBSegmentationParams(min_samples_leaf=75),
            name='Roundtrip Test'
        )

        config_dict = original.to_dict()
        reconstructed = SegmentationConfig.from_dict(config_dict)

        assert reconstructed.data.source == original.data.source
        assert reconstructed.data.sample_size == original.data.sample_size
        assert reconstructed.irb_params.min_samples_leaf == original.irb_params.min_samples_leaf
        assert reconstructed.name == original.name

    def test_config_roundtrip_json(self, temp_data_file, tmp_path):
        """Test config -> JSON -> config roundtrip."""
        original = SegmentationConfig(
            data=DataConfig(source=str(temp_data_file)),
            irb_params=IRBSegmentationParams(),
            name='JSON Roundtrip',
            verbose=False
        )

        json_path = tmp_path / "roundtrip.json"
        original.to_json(str(json_path))
        reconstructed = SegmentationConfig.from_json(str(json_path))

        assert reconstructed.name == original.name
        assert reconstructed.data.source == original.data.source


@pytest.mark.unit
class TestConfigSummary:
    """Test configuration summary generation."""

    def test_summary_basic(self, temp_data_file):
        """Test basic summary generation."""
        config = SegmentationConfig(
            data=DataConfig(source=str(temp_data_file)),
            irb_params=IRBSegmentationParams()
        )

        summary = config.summary()

        assert isinstance(summary, str)
        assert 'SEGMENTATION CONFIGURATION SUMMARY' in summary
        assert str(temp_data_file) in summary

    def test_summary_with_name_and_description(self, temp_data_file):
        """Test summary includes name and description."""
        config = SegmentationConfig(
            data=DataConfig(source=str(temp_data_file)),
            irb_params=IRBSegmentationParams(),
            name='Test Config',
            description='A test configuration'
        )

        summary = config.summary()

        assert 'Name: Test Config' in summary
        assert 'Description: A test configuration' in summary

    def test_summary_with_logging(self, temp_data_file):
        """Test summary includes logging config."""
        config = SegmentationConfig(
            data=DataConfig(source=str(temp_data_file)),
            irb_params=IRBSegmentationParams(),
            logging=LoggingConfig(
                level='DEBUG',
                log_file='output/test.log'
            )
        )

        summary = config.summary()

        assert 'LOGGING CONFIGURATION' in summary
        assert 'DEBUG' in summary
        assert 'output/test.log' in summary

    def test_summary_without_logging(self, temp_data_file):
        """Test summary when logging is None."""
        config = SegmentationConfig(
            data=DataConfig(source=str(temp_data_file)),
            irb_params=IRBSegmentationParams(),
            logging=None
        )

        summary = config.summary()

        # Should not have logging section
        assert 'LOGGING CONFIGURATION' not in summary


@pytest.mark.unit
class TestCreateDefaultConfig:
    """Test default config creation helper."""

    def test_create_default_config_minimal(self, temp_data_file):
        """Test creating default config with minimal args."""
        config = create_default_config(str(temp_data_file))

        assert config.data.source == str(temp_data_file)
        assert config.data.data_type == 'csv'
        assert config.irb_params is not None
        assert config.output.output_dir == './output'

    def test_create_default_config_with_type(self, temp_data_file):
        """Test creating default config with data type."""
        config = create_default_config(
            str(temp_data_file),
            data_type='csv'
        )

        assert config.data.data_type == 'csv'

    def test_create_default_config_with_output_dir(self, temp_data_file):
        """Test creating default config with custom output dir."""
        config = create_default_config(
            str(temp_data_file),
            output_dir='./custom_output'
        )

        assert config.output.output_dir == './custom_output'

    def test_default_config_is_valid(self, temp_data_file):
        """Test that default config passes validation."""
        config = create_default_config(str(temp_data_file))

        issues = config.validate()
        # Should have no critical issues (may have non-existent file warning)
        assert len(issues) <= 1


@pytest.mark.unit
class TestConfigEdgeCases:
    """Test configuration edge cases and error handling."""

    def test_config_with_empty_output_formats(self):
        """Test config with empty output formats list."""
        config = OutputConfig(output_formats=[])

        assert len(config.output_formats) == 0

    def test_config_to_dict_with_all_none_optional(self, temp_data_file):
        """Test to_dict handles None optional fields."""
        config = SegmentationConfig(
            data=DataConfig(source=str(temp_data_file)),
            irb_params=IRBSegmentationParams(),
            logging=None
        )

        config_dict = config.to_dict()
        # Should not have logging key or it should be None
        assert config_dict.get('logging') is None or 'logging' not in config_dict

    def test_config_from_dict_missing_optional_logging(self, temp_data_file):
        """Test from_dict handles missing logging config."""
        config_dict = {
            'data': {
                'source': str(temp_data_file),
                'data_type': 'csv',
                'sample_size': None,
                'use_oot': False,
                'random_state': 42,
                'categorical_columns': None,
                'target_column': None
            },
            'irb_params': {
                'max_depth': 5,
                'min_samples_split': 100,
                'min_samples_leaf': 50,
                'criterion': 'gini',
                'random_state': 42,
                'min_defaults_per_leaf': 10,
                'min_default_rate_diff': 0.02,
                'significance_level': 0.05,
                'min_segment_density': 0.05,
                'max_segment_density': 0.50,
                'monotone_constraints': {},
                'forced_splits': {},
                'validation_tests': []
            },
            'output': {
                'output_dir': './output',
                'output_formats': ['json'],
                'report_name': None,
                'template_name': None,
                'dashboard_name': None,
                'create_dashboard': True,
                'create_excel_template': True,
                'extract_rules': True
            },
            'run_validation': True,
            'verbose': True
        }

        config = SegmentationConfig.from_dict(config_dict)
        assert config.logging is None

    def test_config_verbose_false_suppresses_print(self, temp_data_file, tmp_path, capsys):
        """Test that verbose=False suppresses print statements."""
        config = SegmentationConfig(
            data=DataConfig(source=str(temp_data_file)),
            irb_params=IRBSegmentationParams(),
            verbose=False
        )

        json_path = tmp_path / "quiet.json"
        config.to_json(str(json_path))

        captured = capsys.readouterr()
        assert "Configuration saved" not in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
