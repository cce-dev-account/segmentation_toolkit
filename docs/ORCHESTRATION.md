# IRB Segmentation - Orchestration Guide

**Unified workflow for segmentation analysis from data loading through iterative threshold editing**

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Three Ways to Run](#three-ways-to-run)
4. [Configuration System](#configuration-system)
5. [Excel & CSV Configuration](#excel--csv-configuration)
6. [Categorical Features Support](#categorical-features-support)
7. [Pipeline Stages](#pipeline-stages)
8. [Examples](#examples)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The orchestration layer provides a unified interface to the segmentation toolkit with:

✅ **Single configuration file** - All settings in one place (YAML/JSON/Excel/CSV)
✅ **Data loaded once** - Kept in memory across all stages
✅ **Stage-based execution** - Run all at once OR step-by-step
✅ **Interactive config builder** - Smart parameter suggestions
✅ **Multiple interfaces** - CLI, Python API, or Jupyter notebooks
✅ **Business-friendly formats** - Edit configs in Excel or CSV

### Architecture

```
SegmentationConfig
├── DataConfig (where & how to load)
├── IRBSegmentationParams (model parameters)
└── OutputConfig (what to generate)

SegmentationPipeline
├── load_data() → loads once, keeps in memory
├── fit_baseline() → initial model
├── export_template() → Excel/JSON for editing
├── apply_modifications() → refit with changes
├── compare_results() → before/after analysis
└── export_all() → all outputs
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Create Configuration

**Option A: Analyze your dataset (recommended)**
```bash
python run_pipeline.py --analyze data/my_data.csv --output my_config.yaml
```

**Option B: Use example configuration**
```bash
cp config_examples/template.yaml my_config.yaml
# Edit my_config.yaml with your settings
```

**Option C: Interactive builder**
```bash
python run_pipeline.py --build-config
```

### 3. Run Pipeline

```bash
python run_pipeline.py --config my_config.yaml
```

Done! Results will be in `./output/`

---

## Three Ways to Run

### Method 1: Command Line (Recommended for production)

```bash
# Full pipeline
python run_pipeline.py --config config.yaml

# Stage-by-stage
python run_pipeline.py --config config.yaml --stage fit
python run_pipeline.py --config config.yaml --stage export
# [Edit template]
python run_pipeline.py --config config.yaml --stage apply --template output/template.xlsx
python run_pipeline.py --config config.yaml --stage compare
python run_pipeline.py --config config.yaml --stage export_all
```

### Method 2: Python API (Recommended for scripting)

```python
from irb_segmentation import SegmentationConfig, SegmentationPipeline

# Load configuration
config = SegmentationConfig.from_yaml('config.yaml')

# Create and run pipeline
pipeline = SegmentationPipeline(config)
pipeline.run_all(pause_for_edits=True)

# After editing template
pipeline.apply_modifications('output/template.xlsx')
pipeline.compare_results()
pipeline.export_all()
```

### Method 3: Jupyter Notebooks (Recommended for exploration)

**Setup notebook**: `notebooks/setup_segmentation.ipynb`
- Analyze dataset
- Get parameter suggestions
- Create configuration

**Workflow notebook**: `notebooks/segmentation_workflow.ipynb`
- Run full pipeline interactively
- View results inline
- Compare baseline vs modified

---

## Configuration System

### Configuration File Structure

```yaml
name: "My Analysis"
description: "Production segmentation for loan portfolio"

# Data configuration
data:
  source: "data/loans.csv"
  data_type: "csv"
  sample_size: null  # Use full dataset
  use_oot: false
  random_state: 42

# IRB parameters
irb_params:
  max_depth: 5
  min_samples_leaf: 10000
  min_defaults_per_leaf: 500
  min_segment_density: 0.05
  max_segment_density: 0.40

  # Business constraints
  forced_splits:
    fico_score: 680
    dti: 35.0
  monotone_constraints:
    credit_score: 1  # Higher = lower risk
    interest_rate: -1  # Higher = higher risk

  validation_tests:
    - chi_squared
    - psi
    - binomial

# Output configuration
output:
  output_dir: "./output"
  output_formats: [json, excel, html]
  create_dashboard: true
  create_excel_template: true
  extract_rules: true

# Workflow settings
run_validation: true
verbose: true
```

### Creating Configurations

#### 1. From Dataset Analysis (Smart Suggestions)

```python
from irb_segmentation import ConfigBuilder

config = ConfigBuilder.from_dataset(
    data_path='data/my_data.csv',
    analyze=True  # Analyzes and suggests parameters
)

config.to_yaml('my_config.yaml')
```

Output:
```
Dataset Analysis:
  Rows: 100,000
  Numeric features: 15
  Default rate: 0.1250

Suggesting parameters...
  Suggested Parameters:
    max_depth: 4 (based on 100,000 rows, 15 features)
    min_samples_leaf: 2,000
    min_defaults_per_leaf: 312
    min_segment_density: 8.0%
    max_segment_density: 45.0%

  Rationale:
    - Dataset size: 100,000 rows → depth 4, leaf size 2,000
    - Default rate: 12.50% → 312 min defaults
    - Expected segments: ~16 (max tree leaves)

✓ Configuration is valid
  Estimated runtime: ~3-5 minutes
```

#### 2. Quick Presets

```python
from irb_segmentation.config_builder import quick_config

config = quick_config(
    dataset_size='large',    # small/medium/large
    default_rate='medium',   # low/medium/high
    output_dir='./output'
)

config.data.source = 'data/my_data.csv'  # Set your data path
config.to_yaml('quick_config.yaml')
```

#### 3. From Template

```bash
cp config_examples/template.yaml my_config.yaml
# Edit my_config.yaml
```

---

## Excel & CSV Configuration

**Business-friendly configuration formats** - Edit configs in familiar spreadsheet tools

### Why Use Excel/CSV?

✅ **Business User Friendly** - No need to learn YAML syntax
✅ **Data Validation** - Dropdowns and input validation prevent errors
✅ **Visual Formatting** - Color-coded Basel compliance indicators
✅ **Familiar Interface** - Use Excel, Google Sheets, or any spreadsheet tool
✅ **Version Control** - CSV format works great with git
✅ **Round-trip Compatibility** - Convert freely between YAML ↔ Excel ↔ CSV ↔ JSON

### Excel Configuration Features

**Multi-sheet workbook with:**
- **Overview** - Name, description, workflow settings
- **Data Configuration** - Source, type, sampling options
- **IRB Parameters** - Tree structure, Basel requirements, density controls
- **Business Constraints** - Forced splits, monotone constraints
- **Output Configuration** - Directory, formats, what to generate
- **Validation Tests** - Checkboxes for which tests to run
- **Help** - Parameter guide with examples and recommendations

**Data Validation:**
- Dropdowns for enums (data types, output formats, etc.)
- Number range validation (e.g., max_depth: 1-10)
- Required field highlighting

**Conditional Formatting:**
- 🟢 Green: Meets Basel recommendations (≥20 defaults per leaf)
- 🟡 Yellow: Below Basel recommendations (warning)
- 🔴 Red: Invalid values

### Creating Excel Templates

#### Option 1: From CLI

```bash
# Create blank template
python run_pipeline.py --create-template --excel-output my_config.xlsx --template-type standard

# Template types:
# - simple: Basic parameters only
# - standard: All common options (recommended)
# - advanced: All options including advanced constraints
```

#### Option 2: From Python

```python
from irb_segmentation import ConfigBuilder

# Create blank template
ConfigBuilder.create_excel_template(
    output_path='my_config.xlsx',
    template_type='standard'
)
```

#### Option 3: Convert Existing YAML

```bash
# Convert YAML to Excel
python run_pipeline.py --export-excel config.yaml --excel-output config.xlsx

# Specify template type
python run_pipeline.py --export-excel config.yaml --excel-output config.xlsx --template-type advanced
```

### Editing Excel Configurations

1. **Open in Excel** (or Google Sheets, LibreOffice, etc.)
2. **Navigate sheets** using tabs at bottom
3. **Edit values** in cells with light blue background
4. **Use dropdowns** where available
5. **Check validation** - Invalid values will be highlighted
6. **Save** when done

**Example edits:**
- Change `max_depth` from 3 to 5
- Add forced splits: `fico_score = 680`
- Set monotone constraint: `credit_score = Increasing`
- Enable/disable validation tests with checkboxes

### Loading Excel Configurations

#### From CLI

```bash
# Run pipeline with Excel config
python run_pipeline.py --config my_config.xlsx

# Stage-by-stage with Excel
python run_pipeline.py --config my_config.xlsx --stage fit
```

#### From Python

```python
from irb_segmentation import SegmentationConfig, ConfigBuilder

# Method 1: Direct load
config = SegmentationConfig.from_excel('my_config.xlsx')

# Method 2: Via ConfigBuilder (with validation)
config = ConfigBuilder.from_excel('my_config.xlsx')

# Validate before use
ConfigBuilder.validate_and_warn(config)
```

### CSV Configuration

**Simplified flat format** - Great for version control and programmatic editing

#### CSV Structure

```csv
section,parameter,value,description
metadata,name,My Analysis,Configuration name
data,source,data/loans.csv,Data file path
data,data_type,csv,Type of data source
irb_params,max_depth,5,Maximum tree depth
irb_params,min_samples_leaf,10000,Min samples per leaf
forced_splits,fico_score,680,Forced split threshold
monotone_constraints,credit_score,1,Monotone direction (1/-1/0)
```

#### Export to CSV

```bash
# CLI
python run_pipeline.py --export-csv config.yaml --csv-output config.csv

# Python
config.to_csv('config.csv')
```

#### Load from CSV

```bash
# CLI
python run_pipeline.py --config my_config.csv

# Python
config = SegmentationConfig.from_csv('my_config.csv')
```

### Converting Between Formats

**All formats are interchangeable:**

```python
from irb_segmentation import SegmentationConfig

# Load from any format
config = SegmentationConfig.from_yaml('config.yaml')
# config = SegmentationConfig.from_json('config.json')
# config = SegmentationConfig.from_excel('config.xlsx')
# config = SegmentationConfig.from_csv('config.csv')

# Export to any format
config.to_yaml('output.yaml')
config.to_json('output.json')
config.to_excel('output.xlsx', template='standard')
config.to_csv('output.csv')
```

**From CLI:**

```bash
# YAML → Excel
python run_pipeline.py --export-excel config.yaml --excel-output config.xlsx

# Excel → CSV
python run_pipeline.py --export-csv config.xlsx --csv-output config.csv

# CSV → YAML (via pipeline run and export)
python run_pipeline.py --config config.csv --stage load
# Then export from pipeline
```

### Best Practices for Excel/CSV

✅ **Excel for collaboration** - Business users can edit constraints
✅ **CSV for git** - Better diff/merge in version control
✅ **YAML for readability** - Human-readable, well-commented
✅ **JSON for APIs** - Machine-readable, no dependencies

**Recommended workflow:**
1. **Create**: Start with Excel template or YAML
2. **Collaborate**: Share Excel file with business users
3. **Version**: Export to CSV or YAML for git
4. **Deploy**: Use any format (all work the same)

### CLI Reference for Excel/CSV

```bash
# Create Excel template
python run_pipeline.py --create-template --excel-output my_config.xlsx
python run_pipeline.py --create-template --excel-output my_config.xlsx --template-type advanced

# Export existing config to Excel
python run_pipeline.py --export-excel config.yaml --excel-output config.xlsx
python run_pipeline.py --export-excel config.yaml  # Auto-generates: config.xlsx

# Export existing config to CSV
python run_pipeline.py --export-csv config.yaml --csv-output config.csv
python run_pipeline.py --export-csv config.yaml  # Auto-generates: config.csv

# Run pipeline with Excel/CSV config
python run_pipeline.py --config my_config.xlsx
python run_pipeline.py --config my_config.csv
python run_pipeline.py --config my_config.xlsx --stage fit
```

---

## Categorical Features Support

**Use categorical variables natively without one-hot encoding**

### Why Use Categorical Features?

✅ **Business-Interpretable Rules** - "loan_purpose = education" instead of "loan_purpose_education = 1"
✅ **No Feature Explosion** - Preserve original categories instead of creating hundreds of binary columns
✅ **Regulatory Documentation** - Clearer audit trails for Basel II/III compliance
✅ **Mixed Segmentors** - Combine numeric and categorical splits in the same model

### How It Works

1. **Specify categorical columns** in your config
2. **Use list syntax** for categorical forced splits
3. **Get membership-based rules** like "loan_purpose IN ['education', 'medical']"

### Configuration

**YAML:**
```yaml
data:
  source: "data/loans.csv"
  data_type: "csv"
  categorical_columns:
    - loan_purpose
    - grade
    - home_ownership

irb_params:
  forced_splits:
    int_rate: 15.0  # Numeric threshold
    loan_purpose:  # Categorical membership
      - education
      - medical
      - small_business
```

**Excel:**
- **Data Configuration sheet** → "Categorical Columns" field
- Enter: `loan_purpose, grade, home_ownership` (comma-separated)

**CSV:**
```csv
section,parameter,value,description
data,categorical_columns,"loan_purpose,grade",Categorical column names
forced_splits,loan_purpose,"[""education"", ""medical""]",Categorical split values
```

### Example Usage

**Numeric-Only Segmentation:**
```yaml
# Traditional approach - all features numeric
irb_params:
  forced_splits:
    fico_score: 680
    dti: 35.0
```

**With Categorical Features:**
```yaml
# Mixed numeric + categorical
data:
  categorical_columns: [loan_purpose, grade]

irb_params:
  forced_splits:
    fico_score: 680  # Numeric
    loan_purpose: [education, medical]  # Categorical
  monotone_constraints:
    credit_score: 1  # Numeric only
```

### Categorical Split Interpretation

**Input:**
```yaml
forced_splits:
  loan_purpose: [education, medical, small_business]
```

**Generated Rule:**
```
IF loan_purpose IN ['education', 'medical', 'small_business'] THEN Segment 2
```

**Benefits:**
- Clear business logic
- Easy to explain to stakeholders
- Regulatory-compliant documentation
- No need to track one-hot encoding schemes

### Supported Data Types

**Categorical columns work with:**
- ✅ CSV files (via `categorical_columns` config)
- ✅ Custom data loaders (return `X_categorical` dict)
- ✅ Forced splits (membership-based)
- ⚠️  Tree splits (auto-converted to numeric during training)
- ❌ Monotone constraints (numeric features only)

### Example: Full Workflow

**1. Create config with categoricals:**
```yaml
# config_categorical.yaml
data:
  source: "data/lending_club_test.csv"
  data_type: "csv"
  sample_size: 50000
  categorical_columns:
    - loan_purpose
    - grade

irb_params:
  max_depth: 4
  min_samples_leaf: 1000
  min_defaults_per_leaf: 100
  forced_splits:
    int_rate: 15.0
    loan_purpose: [education, medical]
```

**2. Run pipeline:**
```bash
python run_pipeline.py --config config_categorical.yaml
```

**3. View rules:**
```
Segment 1: IF int_rate <= 15.0 AND loan_purpose NOT IN ['education', 'medical'] THEN PD = 3.2%
Segment 2: IF int_rate <= 15.0 AND loan_purpose IN ['education', 'medical'] THEN PD = 8.5%
Segment 3: IF int_rate > 15.0 THEN PD = 12.1%
```

### Best Practices

✅ **Use for low-cardinality features** - Best for 3-20 unique values
✅ **Document category meanings** - Add descriptions in config
✅ **Combine with numeric splits** - Mix both types for optimal segmentation
✅ **Validate category stability** - Ensure categories are consistent over time

**When to use categoricals:**
- Loan purpose, product type, grade/rating
- Geographic regions (states, countries)
- Employment status, home ownership
- Industry codes, occupation types

**When NOT to use:**
- High-cardinality features (>50 categories)
- Free-text fields
- Continuous variables
- Time-based features (use numeric instead)

### See Also

- Example config: `config_examples/categorical_example.yaml`
- Architecture docs: `ARCHITECTURE_ENHANCEMENTS.md` (Section 3)
- Test file: `test_architecture_enhancements.py`

---

## Pipeline Stages

### Stage 1: Load Data

**What it does:**
- Loads data from CSV or built-in loader
- Applies sampling if configured
- Creates train/validation split
- **Keeps data in memory for all subsequent stages**

```python
pipeline.load_data()

# Data is now available:
print(f"Training: {len(pipeline.X_train)} observations")
print(f"Features: {pipeline.feature_names}")
```

### Stage 2: Fit Baseline

**What it does:**
- Trains initial decision tree
- Applies IRB constraints (merging/splitting)
- Validates all requirements
- Stores baseline results

```python
pipeline.fit_baseline()

# View results:
stats = pipeline.baseline_engine._get_segment_statistics()
print(f"Created {len(stats)} segments")
```

### Stage 3: Export Template

**What it does:**
- Generates Excel or JSON template
- Shows current segment rules and thresholds
- Ready for user editing

```python
template_path = pipeline.export_template(format='excel')
print(f"Edit: {template_path}")
```

**Template contains:**
- Current segments and statistics
- Editable forced_splits
- Merge_segments options
- Parameter adjustments

### Stage 4: Apply Modifications

**What it does:**
- Reads edited template
- Refits model with new constraints
- Applies manual merges if specified
- Re-validates requirements

```python
pipeline.apply_modifications('output/template.xlsx')

# View modified results:
modified_stats = pipeline.modified_engine._get_segment_statistics()
```

### Stage 5: Compare Results

**What it does:**
- Compares baseline vs modified
- Shows segment count changes
- Reports validation status

```python
comparison = pipeline.compare_results()
print(f"Segments: {comparison['baseline']['n_segments']} → {comparison['modified']['n_segments']}")
```

### Stage 6: Export All

**What it does:**
- Exports validation reports (JSON)
- Generates dashboards (HTML)
- Exports tree structure for production
- Creates comparison reports

```python
exported = pipeline.export_all()
for output_type, path in exported.items():
    print(f"{output_type}: {path}")
```

---

## Examples

### Example 1: Quick Test with German Credit

```bash
# Use built-in dataset (auto-downloads)
python run_pipeline.py --config config_examples/german_credit.yaml
```

**Runtime:** ~30 seconds
**Output:** `output/german_credit/`

### Example 2: Production Run with Lending Club

```bash
# Step 1: Create configuration
python run_pipeline.py --analyze data/lending_club_test.csv --output lending_config.yaml

# Step 2: Run pipeline
python run_pipeline.py --config lending_config.yaml

# Step 3: Edit template
# (Edit output/modification_template.xlsx)

# Step 4: Apply modifications
python run_pipeline.py --config lending_config.yaml --stage apply --template output/modification_template.xlsx

# Step 5: Compare and export
python run_pipeline.py --config lending_config.yaml --stage export_all --template output/modification_template.xlsx
```

### Example 3: Iterative Development (Notebook)

```python
# In notebook: notebooks/segmentation_workflow.ipynb

# 1. Load config
config = SegmentationConfig.from_yaml('my_config.yaml')
pipeline = SegmentationPipeline(config)

# 2. Run baseline
pipeline.load_data()
pipeline.fit_baseline()

# View results inline
display(pd.DataFrame(pipeline.baseline_engine._get_segment_statistics()).T)

# 3. Export and iterate
template = pipeline.export_template()
# Edit template...
pipeline.apply_modifications(template)

# 4. Compare
pipeline.compare_results()
```

### Example 4: Custom Configuration

```python
from irb_segmentation import (
    SegmentationConfig,
    DataConfig,
    IRBSegmentationParams,
    OutputConfig
)

config = SegmentationConfig(
    name="Custom Loan Segmentation",
    data=DataConfig(
        source='data/my_loans.csv',
        data_type='csv',
        sample_size=50000  # Sample for testing
    ),
    irb_params=IRBSegmentationParams(
        max_depth=5,
        min_samples_leaf=5000,
        min_defaults_per_leaf=250,
        forced_splits={
            'fico_score': 680,
            'ltv': 80.0,
            'dti': 35.0
        },
        monotone_constraints={
            'credit_score': 1,
            'interest_rate': -1
        }
    ),
    output=OutputConfig(
        output_dir='./results/custom',
        output_formats=['json', 'html']
    )
)

config.to_yaml('custom_config.yaml')
```

---

## Best Practices

### 1. Configuration Management

✅ **Version control configs** - Track with git
✅ **Name meaningfully** - e.g., `lending_v1_fico680.yaml`
✅ **Document changes** - Use `description` field
✅ **Validate before running** - `python run_pipeline.py --validate config.yaml`

### 2. Data Management

✅ **Start with sampling** - Test with `sample_size: 10000`
✅ **Use built-in loaders** - When available (german_credit, lending_club, etc.)
✅ **Check data quality** - Review default rate and feature distributions

### 3. Parameter Selection

✅ **Use ConfigBuilder** - Get smart suggestions
✅ **Start conservative** - Use suggested parameters first
✅ **Test incrementally** - Adjust one parameter at a time
✅ **Monitor validation** - All tests should pass

### 4. Workflow

✅ **Save configs** - Before each run
✅ **Document modifications** - In template files
✅ **Compare versions** - Keep baseline and modified results
✅ **Export early, export often** - Save intermediate results

### 5. Production Deployment

✅ **Use tree structure JSON** - Platform-independent
✅ **Test with OOT data** - Use `use_oot: true`
✅ **Monitor PSI** - Check temporal stability
✅ **Document assumptions** - Business rules and constraints

---

## Troubleshooting

### Issue: Configuration validation fails

**Problem:** `ValueError: Invalid parameters`

**Solution:**
```bash
python run_pipeline.py --validate config.yaml
```
Review and fix reported issues.

### Issue: Insufficient defaults per segment

**Problem:** `Some segments have <20 defaults`

**Solutions:**
1. Lower `min_defaults_per_leaf`
2. Reduce `max_depth`
3. Use larger dataset or sample

### Issue: Memory error with large dataset

**Problem:** `MemoryError` when loading data

**Solutions:**
1. Add sampling: `sample_size: 100000`
2. Use smaller `max_depth` and `min_samples_leaf`
3. Process in stages (load → fit → clear → reload for modifications)

### Issue: Template editing not working

**Problem:** Changes in Excel don't apply

**Solutions:**
1. Ensure Excel file is saved
2. Check file path is correct
3. Verify JSON structure if using JSON template
4. Try exporting as JSON: `pipeline.export_template(format='json')`

### Issue: Pipeline takes too long

**Problem:** Runtime exceeds expectations

**Solutions:**
1. Start with sampling: `sample_size: 10000`
2. Reduce `max_depth`
3. Increase `min_samples_leaf`
4. Use simpler validation tests

### Issue: Segments don't make business sense

**Problem:** Risk ordering seems wrong

**Solutions:**
1. Add `forced_splits` for regulatory thresholds
2. Specify `monotone_constraints`
3. Review feature engineering
4. Check for data quality issues

---

## CLI Reference

```bash
# Run entire pipeline
python run_pipeline.py --config config.yaml

# Run without pausing
python run_pipeline.py --config config.yaml --no-pause

# Run specific stage
python run_pipeline.py --config config.yaml --stage <stage>
# Stages: load, fit, export, apply, compare, export_all

# Apply modifications
python run_pipeline.py --config config.yaml --stage apply --template output/template.xlsx

# Build configuration
python run_pipeline.py --build-config

# Analyze dataset
python run_pipeline.py --analyze data/file.csv --output config.yaml --target-column default

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
```

---

## API Reference

### SegmentationConfig

```python
from irb_segmentation import SegmentationConfig

# Load from any format
config = SegmentationConfig.from_yaml('config.yaml')
config = SegmentationConfig.from_json('config.json')
config = SegmentationConfig.from_excel('config.xlsx')
config = SegmentationConfig.from_csv('config.csv')

# Save to any format
config.to_yaml('config.yaml')
config.to_json('config.json')
config.to_excel('config.xlsx', template='standard')  # template: simple/standard/advanced
config.to_csv('config.csv')

# Validate
issues = config.validate()
print(config.summary())
```

### SegmentationPipeline

```python
from irb_segmentation import SegmentationPipeline

pipeline = SegmentationPipeline(config)

# Stage methods
pipeline.load_data()
pipeline.fit_baseline()
template = pipeline.export_template(format='excel')  # or 'json'
pipeline.apply_modifications(template_path)
comparison = pipeline.compare_results()
files = pipeline.export_all()

# Convenience
pipeline.run_all(pause_for_edits=True)

# Inspection
state = pipeline.get_state()
```

### ConfigBuilder

```python
from irb_segmentation import ConfigBuilder

# From dataset
config = ConfigBuilder.from_dataset('data/file.csv', analyze=True)

# Suggest parameters
params = ConfigBuilder.suggest_parameters(
    n_rows=100000,
    default_rate=0.12,
    n_features=15
)

# Interactive
config = ConfigBuilder.interactive_build()

# Validate
ConfigBuilder.validate_and_warn(config)

# Estimate runtime
runtime = ConfigBuilder.estimate_runtime(config)

# Excel/CSV operations
ConfigBuilder.to_excel(config, 'config.xlsx', template='standard')
config = ConfigBuilder.from_excel('config.xlsx')
ConfigBuilder.validate_excel('config.xlsx')
ConfigBuilder.create_excel_template('blank_template.xlsx', template_type='simple')
```

---

## What's Next?

### Completed ✓
- Unified configuration system (YAML/JSON/Excel/CSV)
- Pipeline orchestrator with stage control
- Smart parameter suggestions
- Multiple interfaces (CLI, API, notebooks)
- Example configurations
- Excel/CSV configuration support with validation and formatting
- Business-friendly config templates

### Future Enhancements
- LightGBM/CatBoost backend support
- Real-time monitoring dashboard
- A/B testing framework for segment modifications
- Ensemble segmentation
- Automated parameter tuning

---

## Support

- **Documentation**: See README.md for framework details
- **Examples**: Check `config_examples/` directory
- **Notebooks**: Interactive guides in `notebooks/`
- **Issues**: Report problems via GitHub issues

---

**Ready to start? Try the quick start above or open `notebooks/setup_segmentation.ipynb`!**
