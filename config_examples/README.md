# Configuration Examples

This directory contains example configuration files for different use cases.

## Available Configurations

### `template.yaml`
Generic template with all available options documented. Use this as a starting point for custom configurations.

### `german_credit.yaml`
- **Dataset**: German Credit (1K rows, auto-downloads)
- **Use case**: Quick testing and development
- **Parameters**: Relaxed for small dataset
- **Runtime**: ~30 seconds

### `taiwan_credit.yaml`
- **Dataset**: Taiwan Credit Card (30K rows)
- **Use case**: IRB validation testing
- **Parameters**: Standard IRB requirements
- **Runtime**: ~2-3 minutes

### `lending_club.yaml`
- **Dataset**: Lending Club (2.26M rows)
- **Use case**: Production simulation
- **Parameters**: Large-scale parameters
- **Runtime**: ~10-15 minutes

## Usage

### Option 1: Using the Pipeline Orchestrator

```python
from irb_segmentation import SegmentationConfig
from irb_segmentation.pipeline import SegmentationPipeline

# Load configuration
config = SegmentationConfig.from_yaml('config_examples/german_credit.yaml')

# Run pipeline
pipeline = SegmentationPipeline(config)
pipeline.run_all()
```

### Option 2: Using CLI

```bash
# Run with example config
python run_pipeline.py --config config_examples/german_credit.yaml

# Run specific stages
python run_pipeline.py --config config_examples/lending_club.yaml --stage fit
python run_pipeline.py --config config_examples/lending_club.yaml --stage apply
```

### Option 3: Customizing Configs

```bash
# 1. Copy template
cp config_examples/template.yaml my_config.yaml

# 2. Edit my_config.yaml with your settings

# 3. Run
python run_pipeline.py --config my_config.yaml
```

## Configuration Sections

### `data`
- `source`: Path to data file or loader name
- `data_type`: Type of data (csv, german_credit, lending_club, etc.)
- `sample_size`: Optional sampling for large datasets
- `use_oot`: Enable out-of-time validation split
- `random_state`: Random seed for reproducibility

### `irb_params`
- **Tree structure**: `max_depth`, `min_samples_split`, `min_samples_leaf`
- **IRB requirements**: `min_defaults_per_leaf`, `min_default_rate_diff`
- **Density controls**: `min_segment_density`, `max_segment_density`
- **Business rules**: `forced_splits`, `monotone_constraints`
- **Validation**: `validation_tests` (chi_squared, psi, binomial)

### `output`
- `output_dir`: Where to save results
- `output_formats`: Which formats to generate (json, excel, html)
- `create_dashboard`: Generate HTML dashboard
- `create_excel_template`: Generate Excel template for editing
- `extract_rules`: Extract segment rules

### `workflow`
- `run_validation`: Run all validation tests
- `verbose`: Print detailed progress

## Parameter Guidelines

### Dataset Size Recommendations

| Dataset Size | max_depth | min_samples_leaf | min_defaults_per_leaf |
|--------------|-----------|------------------|------------------------|
| < 5K rows    | 2-3       | 50-100          | 5-10                  |
| 5K-50K rows  | 3-4       | 500-1000        | 20-30                 |
| 50K-500K rows| 4-5       | 2000-5000       | 100-300               |
| > 500K rows  | 5-6       | 10000+          | 500+                  |

### Default Rate Considerations

- **High default rate (>20%)**: Can use smaller `min_defaults_per_leaf`
- **Low default rate (<5%)**: Need larger sample sizes
- **Very low (<2%)**: May need to relax IRB requirements or use larger dataset

## Validation

Check if your config is valid:

```python
from irb_segmentation import SegmentationConfig

config = SegmentationConfig.from_yaml('my_config.yaml')
issues = config.validate()

if issues:
    print("Configuration issues found:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("Configuration is valid!")
    print(config.summary())
```

## Tips

1. **Start small**: Test with `german_credit.yaml` first
2. **Sample large datasets**: Use `sample_size` for initial testing
3. **Save configs**: Always save working configs for reproducibility
4. **Version control**: Track config changes with git
5. **Document changes**: Use the `description` field to explain modifications
