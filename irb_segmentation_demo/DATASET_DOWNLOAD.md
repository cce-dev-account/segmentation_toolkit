# Dataset Download Instructions

This guide shows how to download the Taiwan Credit and Lending Club datasets for the demo.

**Note:** German Credit auto-downloads - no manual setup needed!

---

## Quick Reference

| Dataset | Size | Download Method | File Location |
|---------|------|-----------------|---------------|
| German Credit | 1K | Auto-downloads | (automatic) |
| Taiwan Credit | 30K | Kaggle API or Manual | `data/UCI_Credit_Card.csv` |
| Lending Club | 1.3M | Kaggle API or Manual | `data/lending_club_data.csv` |

---

## Option 1: Kaggle API (Recommended)

### Setup Kaggle API

1. **Create Kaggle account** (if you don't have one):
   - Visit https://www.kaggle.com
   - Sign up for free

2. **Get API credentials**:
   - Go to https://www.kaggle.com/settings
   - Scroll to "API" section
   - Click "Create New API Token"
   - This downloads `kaggle.json`

3. **Install Kaggle CLI**:
   ```bash
   pip install kaggle
   ```

4. **Configure credentials**:

   **Linux/Mac:**
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

   **Windows:**
   ```bash
   mkdir %USERPROFILE%\.kaggle
   move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\
   ```

### Download Taiwan Credit Dataset

```bash
# Download from Kaggle
kaggle datasets download -d uciml/default-of-credit-card-clients-dataset

# Unzip to data/ folder
unzip default-of-credit-card-clients-dataset.zip -d ./data/

# Verify file exists
ls data/UCI_Credit_Card.csv
```

**Expected output:** `data/UCI_Credit_Card.csv` (~2.7 MB, 30,000 rows)

### Download Lending Club Dataset

```bash
# Download from Kaggle (WARNING: Large file ~420 MB)
kaggle datasets download -d wordsforthewise/lending-club

# Unzip to data/ folder
unzip lending-club.zip -d ./data/

# Rename file (if needed)
mv data/accepted_2007_to_2018Q4.csv data/lending_club_data.csv

# Verify file exists
ls data/lending_club_data.csv
```

**Expected output:** `data/lending_club_data.csv` (~420 MB, 2.26M rows)

---

## Option 2: Manual Download

If you prefer not to use the Kaggle API, download manually:

### Taiwan Credit Dataset

1. Visit: https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset
2. Click "Download" button (requires Kaggle login)
3. Unzip the downloaded file
4. Copy `UCI_Credit_Card.csv` to `data/UCI_Credit_Card.csv` in this demo folder

### Lending Club Dataset

1. Visit: https://www.kaggle.com/datasets/wordsforthewise/lending-club
2. Click "Download" button (requires Kaggle login)
3. Unzip the downloaded file
4. Find `accepted_2007_to_2018Q4.csv`
5. Copy and rename to `data/lending_club_data.csv` in this demo folder

---

## Verify Downloads

Run the setup script to check which datasets are ready:

```bash
python setup_demo.py
```

Expected output:
```
CHECKING DATASETS
=================

1. German Credit Dataset
   Status: Auto-downloads from UCI (no setup needed)
   Size: 1K rows
   Config: demo_german.yaml

2. Taiwan Credit Dataset
   Status: ✓ Found at data/UCI_Credit_Card.csv
   Size: 30K rows
   Config: demo_taiwan.yaml

3. Lending Club Dataset
   Status: ✓ Found at data/lending_club_data.csv
   Size: 1.3M rows (temporal split: 95K training)
   Config: demo_lending_club.yaml
```

---

## Dataset Details

### German Credit Dataset
- **Size:** 1,000 credit applications
- **Source:** UCI Machine Learning Repository
- **Download:** Automatic (no setup needed)
- **Features:** Age, credit amount, employment status, etc.
- **Target:** Credit risk (good/bad)
- **Best for:** Quick testing, development

### Taiwan Credit Dataset
- **Size:** 30,000 credit card records
- **Source:** UCI via Kaggle
- **Download:** Manual (see above)
- **Features:** Payment history, bill amounts, credit limit
- **Target:** Default next month (yes/no)
- **Best for:** IRB validation, medium-scale testing

### Lending Club Dataset
- **Size:** 2.26M loan records (2007-2018)
- **Preprocessed:** 1.35M after cleaning
- **Source:** Kaggle
- **Download:** Manual (see above)
- **Features:** Interest rate, FICO, DTI, grade, home ownership, purpose
- **Target:** Loan default (yes/no)
- **Time range:** 2007-2018 (covers financial crisis and recovery)
- **Best for:** Production simulation, temporal validation, OOT testing

**Temporal Split:**
- Training: 2007-2012 → 95,902 observations (7%)
- Validation: 2013-2014 → 357,907 observations (27%)
- Out-of-Time: 2015+ → 894,290 observations (66%)

This split demonstrates realistic production deployment where you train on historical data and validate on future periods.

---

## Troubleshooting

### "Kaggle command not found"
```bash
pip install kaggle
```

### "403 Forbidden" error
- Make sure you've accepted the dataset license on Kaggle website
- Check your `kaggle.json` credentials are correctly placed

### "File not found" error
- Check the file is in the correct location: `data/UCI_Credit_Card.csv` or `data/lending_club_data.csv`
- File names are case-sensitive on Linux/Mac

### File size issues
- Taiwan Credit: ~2.7 MB (should download quickly)
- Lending Club: ~420 MB (may take a few minutes)

### Unzip not available (Windows)
Use built-in Windows extraction:
1. Right-click the .zip file
2. Select "Extract All..."
3. Choose the `data/` folder as destination

Or use PowerShell:
```powershell
Expand-Archive -Path .\lending-club.zip -DestinationPath .\data\
```

---

## Alternative Data Sources

If Kaggle is not accessible:

### Taiwan Credit
- UCI Repository: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

### Lending Club
- Official source (historical): https://www.lendingclub.com/info/download-data.action
- Note: May require account and data format might differ

---

## Ready to Run!

Once datasets are downloaded, run:

```bash
# Check setup
python setup_demo.py

# Run medium demo
python run_demo.py demo_taiwan.yaml

# Run production demo
python run_demo.py demo_lending_club.yaml
```

See **DEMO_INSTRUCTIONS.md** for complete usage guide.
