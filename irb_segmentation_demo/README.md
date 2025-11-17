# IRB Segmentation Framework - Demo Package

Production-ready framework for creating risk segments in credit portfolios using decision trees with regulatory compliance.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Check setup
python setup_demo.py

# 3. Run demo (auto-downloads German Credit dataset)
python run_demo.py demo_german.yaml
```

**Output:** View results in `output/demo_german/`
- `segment_rules.txt` - Human-readable rules
- `segment_summary.xlsx` - Excel summary
- `dashboard.html` - Interactive visualization

## Three Demo Options

### 1. Quick Start (5 seconds)
```bash
python run_demo.py demo_german.yaml
```
- 1K observations, auto-downloads
- Perfect for first-time users

### 2. Medium Scale (30 seconds)
```bash
python run_demo.py demo_taiwan.yaml
```
- 30K observations
- Requires download (see DATASET_DOWNLOAD.md)

### 3. Production Scale (2-3 minutes)
```bash
python run_demo.py demo_lending_club.yaml
```
- 1.3M observations with temporal split
- Demonstrates production deployment
- Requires download (see DATASET_DOWNLOAD.md)

## Documentation

- **[DEMO_INSTRUCTIONS.md](DEMO_INSTRUCTIONS.md)** - Complete step-by-step guide
- **[DATASET_DOWNLOAD.md](DATASET_DOWNLOAD.md)** - How to download larger datasets
- **[OUTPUT_GUIDE.md](OUTPUT_GUIDE.md)** - Understanding your results

## What This Framework Does

- Creates risk segments for credit portfolios
- Enforces Basel II/III regulatory requirements
- Applies business constraints (forced splits, monotonicity)
- Validates segments statistically
- Generates multiple output formats

## Features

- **YAML Configuration**: Define everything in config files
- **Categorical Support**: Mix numeric and categorical features
- **Multiple Outputs**: Text rules, Excel, HTML, JSON
- **Regulatory Validation**: Chi-squared, PSI, binomial tests
- **Temporal Splits**: Train on past, validate on future

## System Requirements

- Python 3.7+
- 4GB RAM minimum (8GB for Lending Club demo)
- ~500MB disk space for datasets

## License

MIT License - Free for commercial and academic use
