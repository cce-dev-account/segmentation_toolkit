# IRB PD Model Segmentation Framework

Production-ready segmentation framework for Internal Ratings-Based (IRB) Probability of Default models with YAML configuration, categorical feature support, and multiple output formats.

## Features

- **YAML Configuration**: Define entire segmentation pipeline in config files
- **Categorical Features**: Full support for categorical variables (loan purpose, grade, property type)
- **Multiple Output Formats**: JSON, Excel, HTML dashboard, and human-readable text rules
- **Cluster Visualization**: 2D scatter plots using PCA/t-SNE/UMAP with default rate heatmaps
- **Segment Extraction**: Filter and extract data sub-samples by segment ID
- **Two Split Strategies**: Temporal split (regulatory) or full dataset (maximum power)
- **Regulatory Compliance**: Basel II/III requirements (minimum defaults, statistical significance)
- **Business Constraints**: Forced splits, monotonicity constraints, density controls
- **Comprehensive Validation**: Chi-squared tests, PSI, binomial confidence intervals
- **Audit-Ready**: Full logging of adjustments and validation results

## Installation

```bash
pip install numpy pandas scikit-learn scipy openpyxl pyyaml
```

## Quick Start

```bash
# Run with Lending Club dataset (temporal split - regulatory approach)
python run_segmentation.py config_examples/lending_club_categorical.yaml

# Or use full dataset for maximum statistical power
python run_segmentation.py config_examples/lending_club_full_data.yaml
```

**Output files generated:**
- `segment_rules.txt` - Human-readable IF...THEN rules
- `baseline_report.json` - Complete validation report
- `segment_summary.xlsx` - Excel summary table
- `dashboard.html` - Interactive visualization
- `config_template.xlsx` - Editable configuration template

## Project Structure

```
segmentation_analysis/
├── irb_segmentation/           # Core framework
│   ├── params.py               # Configuration parameters
│   ├── engine.py               # Segmentation engine
│   ├── validators.py           # Validation functions
│   ├── adjustments.py          # Post-processing
│   ├── pipeline.py             # Orchestration layer
│   └── config_to_excel.py      # Excel template export
├── data_loaders/               # Dataset loaders
│   ├── base.py                 # Base loader class
│   ├── german_credit.py        # German Credit (1K obs)
│   ├── taiwan_credit.py        # Taiwan Credit (30K obs)
│   ├── lending_club.py         # Lending Club (1.3M obs)
│   └── home_credit.py          # Home Credit (300K obs)
├── config_examples/            # Configuration files
│   ├── lending_club_categorical.yaml    # Temporal split (regulatory)
│   └── lending_club_full_data.yaml      # Full dataset (max power)
├── docs/                       # Documentation
│   ├── DATA_SPLITS.md          # Temporal vs full data strategies
│   ├── ORCHESTRATION.md        # Pipeline architecture
│   └── VISUALIZATION.md        # Output formats guide
├── scripts/                    # Utility scripts
│   ├── download_datasets.py    # Dataset downloaders
│   ├── convert_to_excel.py     # JSON to Excel converter
│   ├── visualize_tree.py       # Tree visualization
│   ├── visualize_clusters.py   # 2D cluster visualization (PCA/t-SNE/UMAP)
│   └── extract_segments.py     # Segment data extraction
├── examples/                   # Example scripts
│   └── test_lending_club_simple.py
├── archive/                    # Old documentation
├── run_segmentation.py         # Main entry point
└── README.md                   # This file
```

## Configuration-Based Workflow

Define your segmentation in YAML:

```yaml
# config_examples/lending_club_categorical.yaml
name: "Lending Club Categorical Segmentation"

data:
  source: "lending_club"
  use_oot: true              # Temporal split (2007-2012 train, 2013-2014 val, 2015+ OOT)
  categorical_columns:       # Full categorical support
    - grade                  # A-G loan grade
    - home_ownership         # RENT, OWN, MORTGAGE
    - purpose                # debt_consolidation, credit_card, etc.

irb_params:
  max_depth: 5
  min_samples_leaf: 10000
  min_defaults_per_leaf: 500

  forced_splits:             # Business constraints
    interest_rate: 15.0
    credit_score: 680
    dti: 25.0

  monotone_constraints:      # Risk direction
    credit_score: 1          # Higher = lower risk
    interest_rate: -1        # Higher = higher risk

output:
  output_dir: "./output/lending_club_categorical"
  output_formats:
    - json
    - excel
    - html
  extract_rules: true        # Generate human-readable rules
```

## Two Data Split Strategies

### Temporal Split (Regulatory Approach)
**Config:** `lending_club_categorical.yaml` (`use_oot: true`)

- **Training:** 2007-2012 → 95,902 obs (7%)
- **Validation:** 2013-2014 → 357,907 obs (27%)
- **OOT Test:** 2015+ → 894,290 obs (66%)
- **Best for:** Production deployments, regulatory submissions, temporal stability testing

### Full Dataset (Maximum Power)
**Config:** `lending_club_full_data.yaml` (`use_oot: false`)

- **Training:** Random 70% → ~944,000 obs
- **Validation:** Random 30% → ~404,000 obs
- **Best for:** Research, feature engineering, maximum accuracy

See [docs/DATA_SPLITS.md](docs/DATA_SPLITS.md) for detailed comparison.

## Understanding the Output

### Segment Rules (Human-Readable)
```
output/lending_club_categorical/segment_rules.txt

SEGMENT 4: High Risk
├─ Observations: 48,235 (5.0%)
├─ Defaults: 12,156 (25.2% default rate)
└─ Rules:
   IF interest_rate > 15.0
   AND credit_score <= 680
   AND grade IN ['E', 'F', 'G']
   THEN assign to Segment 4

Risk Level: ⚠️ HIGH RISK (PD = 25.2%)
```

### Excel Summary
Open `segment_summary.xlsx` for a formatted table:

| Segment | Observations | Defaults | Default Rate | Density | Risk Level |
|---------|-------------|----------|--------------|---------|------------|
| 1 | 127,542 | 3,826 | 3.0% | 13.3% | Low |
| 2 | 215,893 | 15,112 | 7.0% | 22.5% | Medium |
| 4 | 48,235 | 12,156 | 25.2% | 5.0% | High |

### HTML Dashboard
Open `dashboard.html` in your browser for:
- Summary statistics
- Segment distribution charts
- Default rate comparisons
- Validation test results

## Available Datasets

The framework includes loaders for 4 public credit datasets:

| Dataset | Size | Download | Best For |
|---------|------|----------|----------|
| **German Credit** | 1K | Auto | Quick testing |
| **Taiwan Credit** | 30K | Manual | IRB validation |
| **Lending Club** | 1.3M | Manual | Production-scale, OOT testing |
| **Home Credit** | 300K | Manual | Complex features |

### Downloading Datasets

```bash
# Use the download script
python scripts/download_datasets.py

# Or manually via Kaggle API
pip install kaggle
kaggle datasets download -d wordsforthewise/lending-club
unzip lending-club.zip -d ./data/
```

See [archive/DATA_SOURCES.md](archive/DATA_SOURCES.md) for detailed instructions.

## Documentation

- **[docs/DATA_SPLITS.md](docs/DATA_SPLITS.md)** - Temporal vs full data split strategies
- **[docs/ORCHESTRATION.md](docs/ORCHESTRATION.md)** - Pipeline architecture and workflow
- **[docs/VISUALIZATION.md](docs/VISUALIZATION.md)** - Output formats and interpretation
- **[docs/VISUALIZATION_AND_EXTRACTION.md](docs/VISUALIZATION_AND_EXTRACTION.md)** - **NEW:** Cluster visualization and segment extraction
- **[archive/](archive/)** - Historical documentation and guides

## Key Features

### Categorical Features Support
Full support for categorical variables alongside numeric features:
- **Membership splits**: "IF grade IN ['E', 'F', 'G']"
- **Forced categorical splits**: Business rules on categorical features
- **Mixed segmentation**: Combine numeric thresholds with categorical membership

### Multiple Output Formats
- **segment_rules.txt**: Human-readable IF...THEN rules
- **baseline_report.json**: Complete validation report with all statistics
- **segment_summary.xlsx**: Excel table for business users
- **dashboard.html**: Interactive visualization with charts
- **config_template.xlsx**: Editable configuration for retraining

### Regulatory Validation
- **Chi-squared tests**: Statistical significance between segments
- **PSI (Population Stability Index)**: Temporal stability < 0.10
- **Binomial confidence intervals**: Default rate precision
- **Minimum defaults**: Basel II/III compliance (≥20 defaults per segment)

## Common Tasks

```bash
# Run temporal split analysis (regulatory approach)
python run_segmentation.py config_examples/lending_club_categorical.yaml

# Run full dataset analysis (maximum power)
python run_segmentation.py config_examples/lending_club_full_data.yaml

# Convert JSON report to Excel
python scripts/convert_to_excel.py output/lending_club_categorical/baseline_report.json

# Extract segment sub-samples (NEW)
python scripts/extract_segments.py data.csv tree_structure.json --segments 1 2 -o low_risk.csv

# Show segment statistics (NEW)
python scripts/extract_segments.py data.csv tree_structure.json --stats

# Download datasets
python scripts/download_datasets.py
```

## Visualization and Extraction (NEW)

The framework now includes powerful tools for visualizing clusters and extracting segment sub-samples:

**Cluster Visualization:**
```python
from scripts.visualize_clusters import ClusterVisualizer

viz = ClusterVisualizer('output/baseline_report.json')
viz.load_data_from_arrays(X_train, y_train, segments_train, feature_names)
viz.reduce_dimensions('pca')
viz.plot_clusters(save_path='clusters.png')
viz.plot_default_rate_heatmap(save_path='heatmap.png')
```

**Segment Extraction:**
```python
from scripts.extract_segments import SegmentExtractor

extractor = SegmentExtractor('output/tree_structure.json')
df_low_risk = extractor.extract_from_csv('data.csv', segment_ids=[1, 2])
extractor.split_by_segments('data.csv', 'output/segments/')
```

See **[docs/VISUALIZATION_AND_EXTRACTION.md](docs/VISUALIZATION_AND_EXTRACTION.md)** for complete documentation.

## License

MIT License
