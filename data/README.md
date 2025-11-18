# Dataset Directory

This directory contains datasets for IRB segmentation testing.

## Included Datasets

### German Credit Dataset
- **File:** `german_credit.data`
- **Size:** Small (~1,000 observations)
- **Source:** UCI Machine Learning Repository
- **Status:** ✅ Included in repository

## Large Datasets (Download Required)

The following large datasets are **not included** in this repository due to GitHub file size limits.

### Lending Club Loan Data
- **Files:**
  - `accepted_2007_to_2018Q4.csv.gz` (375 MB)
  - `rejected_2007_to_2018Q4.csv.gz` (244 MB)
- **Size:** ~1.3 million loans
- **Source:** [Kaggle - Lending Club](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

**Download Instructions:**
```bash
# Option 1: Kaggle CLI (recommended)
pip install kaggle
kaggle datasets download -d wordsforthewise/lending-club
unzip lending-club.zip -d ./data/

# Option 2: Manual download
# 1. Visit https://www.kaggle.com/datasets/wordsforthewise/lending-club
# 2. Download the dataset
# 3. Extract to ./data/ directory
```

### Taiwan Credit Default
- **Size:** ~30,000 observations
- **Source:** UCI Machine Learning Repository
- **Download:** Will be fetched automatically by data loader

### Home Credit Default Risk
- **Size:** ~300,000 observations
- **Source:** [Kaggle - Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
- **Download:** Manual download required from Kaggle

## Quick Start

For quick testing without downloading large files:

```bash
# Use German Credit dataset (small, included)
python run_segmentation.py config_examples/german_credit.yaml

# Or let the framework download Taiwan Credit automatically
python run_segmentation.py config_examples/taiwan_credit.yaml
```

## File Structure

```
data/
├── README.md                           # This file
├── german_credit.data                  # Small dataset (included)
├── accepted_2007_to_2018Q4.csv.gz     # Large - download separately
└── rejected_2007_to_2018Q4.csv.gz     # Large - download separately
```

## Dataset Loaders

All datasets have corresponding loaders in `data_loaders/`:
- `german_credit.py` - Automatically loads included data
- `lending_club.py` - Requires manual download
- `taiwan_credit.py` - Auto-downloads from UCI
- `home_credit.py` - Requires manual download

See `scripts/download_datasets.py` for automated download helpers.
