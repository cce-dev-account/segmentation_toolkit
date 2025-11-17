# Lending Club Complete Workflow Guide

## End-to-End: Generate → View → Edit → Re-score → Validate

This guide walks you through the complete workflow using the **full Lending Club dataset (2.26M observations)**.

---

## Prerequisites

```bash
# Ensure Python is installed
python --version  # Should show Python 3.7+

# Install dependencies
pip install numpy pandas scikit-learn scipy openpyxl

# Verify data is available
ls data/lending_club_test.csv  # Should show 2.26M rows
```

---

## Workflow Overview

```
1. Generate Baseline    →    2. Extract Rules    →    3. Edit Excel    →    4. Apply Changes    →    5. Compare Results
   (10-15 min)                  (2-3 min)                (User edits)           (10-15 min)              (View report)
```

---

## Step 1: Generate Baseline Segmentation

### 1.1 Run Initial Model

```bash
python test_lending_club_simple.py
```

**What it does:**
- Loads full 2.26M Lending Club dataset
- Splits into train (70%) / validation (30%)
- Fits IRB segmentation model with parameters:
  - `max_depth=5` (deeper tree for large dataset)
  - `min_samples_leaf=10000` (10K observations per segment)
  - `min_defaults_per_leaf=500` (500 defaults minimum)
  - `min_segment_density=0.05` (5% minimum segment size)

**Expected output:**
```
Loading Lending Club data (ALL observations - ~2.26M rows)...
Loaded 2,260,701 rows
Default rate: 0.1314

Train: 1582490 obs, 207916 defaults (0.1314)
Val: 678211 obs, 89159 defaults (0.1314)

================================================================================
IRB SEGMENTATION - TRAINING SUMMARY
================================================================================

SEGMENT SUMMARY:
  Segment 0: 183,082 obs (11.6%), 3,955 defaults (2.16% PD)
  Segment 1: 111,761 obs (7.1%), 8,919 defaults (7.98% PD)
  Segment 2: 254,388 obs (16.1%), 12,948 defaults (5.09% PD)
  Segment 3: 272,580 obs (17.2%), 30,202 defaults (11.08% PD)
  Segment 4: 126,561 obs (8.0%), 10,568 defaults (8.35% PD)
  Segment 5: 415,442 obs (26.3%), 67,301 defaults (16.20% PD)
  Segment 6: 218,676 obs (13.8%), 54,669 defaults (25.00% PD)

✓ All validation tests passed

[SUCCESS] Lending Club FULL dataset test passed!
```

**Output file created:**
- `lending_club_full_report.json` (full validation report with statistics)

**Time:** ~10-15 minutes

---

## Step 2: Extract Segment Rules

### 2.1 Extract Readable Rules

```bash
python extract_segment_rules.py
```

**What it does:**
- Re-fits model to extract decision tree structure
- Extracts complete decision paths for each segment
- Identifies all feature thresholds used
- Exports in both JSON and human-readable format

**Expected output:**
```
================================================================================
SEGMENT RULE EXTRACTION
================================================================================

Loading Lending Club data to extract segment rules...
Loaded 2,260,701 rows

Fitting model...
Extracting segment rules...
Extracting feature thresholds...

[OK] Rules saved to: segment_rules_detailed.json

================================================================================
SEGMENT RULES SUMMARY
================================================================================

================================================================================
SEGMENT 0: 183,082 obs, 2.16% PD
================================================================================
1. IF int_rate <= 6.86 AND fico_range_high <= 721.50
2. IF int_rate <= 7.86 AND fico_range_high > 721.50 AND annual_inc <= 43336.90
3. IF int_rate <= 7.86 AND fico_range_high > 721.50 AND annual_inc > 43336.90

================================================================================
SEGMENT 1: 111,761 obs, 7.98% PD
================================================================================
1. IF int_rate <= 13.60 AND int_rate > 7.86 AND fico_range_low <= 702.50 AND dti > 14.78
...

================================================================================
FEATURE THRESHOLDS
================================================================================

int_rate:
  - 6.86
  - 7.86
  - 10.58
  - 12.98
  - 13.60
  - 16.97

fico_range_high:
  - 686.50
  - 721.50

annual_inc:
  - 43336.90
  - 58115.50
  - 61077.00
```

**Output files created:**
- `segment_rules_detailed.json` (machine-readable rules)

**Time:** ~2-3 minutes

---

## Step 3: Generate Excel Template for Editing

### 3.1 Create Excel Workbook

```bash
python interfaces/create_excel_template.py
```

**What it does:**
- Reads `lending_club_full_report.json` and `segment_rules_detailed.json`
- Creates multi-worksheet Excel workbook with:
  - **Segment Rules** worksheet (PRIMARY - simplified, easy to edit)
  - **Segment Actions** worksheet (merge similar segments)
  - **Threshold Overview** worksheet (feature-level editing)
  - Individual feature worksheets (int_rate, fico_range_high, etc.)
  - **Model Parameters** worksheet (adjust algorithm parameters)

**Expected output:**
```
================================================================================
CREATING EXCEL WORKBOOK WITH MODIFICATION TEMPLATES
================================================================================

[OK] Created Excel workbook: modification_template.xlsx
  Size: 25,122 bytes
  Worksheets: 12 total
    - Segment Actions
    - Segment Rules (PRIMARY - Start here!)
    - Threshold Overview (with links to 8 feature sheets)
    - Individual sheets: int_rate, annual_inc, fico_range_high, etc.
    - Model Parameters

================================================================================
NEXT STEPS
================================================================================
1. Open modification_template.xlsx in Excel
2. Navigate to "Segment Rules" tab (RECOMMENDED)
3. Edit threshold values in "New_Value" column
4. Save the file
5. Convert to JSON: python interfaces/excel_to_json.py
6. Apply changes: python apply_modifications.py modification.json
```

**Output file created:**
- `modification_template.xlsx` (Excel workbook with 12 worksheets)

**Time:** ~30 seconds

---

## Step 4: Edit Thresholds in Excel

### 4.1 Open Excel File

```bash
# Windows
start modification_template.xlsx

# Mac
open modification_template.xlsx

# Linux
libreoffice modification_template.xlsx
```

### 4.2 Navigate to "Segment Rules" Worksheet

**This is the PRIMARY editing interface** - it shows simplified, segment-level rules.

#### What You'll See:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ SEGMENT DECISION RULES - EDIT THRESHOLDS HERE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ Instructions:                                                               │
│ • Each segment shows its complete decision paths below                      │
│ • Edit threshold values in the 'New_Value' column to change boundaries      │
│ • Leave 'New_Value' blank to keep current threshold                        │
│ • All conditions in a path are combined with AND logic                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ SEGMENT 0: Very Low Risk                                                   │
│ PD: 2.16% | Observations: 183,082 | Defaults: 3,955 | Density: 11.6%     │
├──────────────┬───────────┬────────────────┬─────────────┬─────────┬───────┤
│ Feature      │ Condition │ Current Value  │ New Value   │ Format  │ Reason│
├──────────────┼───────────┼────────────────┼─────────────┼─────────┼───────┤
│ int_rate     │ <=        │ 6.86%          │             │ %       │       │
│ fico_range...│ <=        │ 721.50         │             │ score   │       │
├──────────────┴───────────┴────────────────┴─────────────┴─────────┴───────┤
│                                                                             │
│ SEGMENT 1: Low Risk                                                        │
│ PD: 7.98% | Observations: 111,761 | Defaults: 8,919 | Density: 7.1%      │
├──────────────┬───────────┬────────────────┬─────────────┬─────────┬───────┤
│ int_rate     │ between   │ 7.86% to 13.60%│             │ %       │       │
│ fico_range...│ <=        │ 702.50         │             │ score   │       │
│ dti          │ >         │ 14.78%         │             │ %       │       │
└──────────────┴───────────┴────────────────┴─────────────┴─────────┴───────┘
```

**Key Features:**
- ✅ **Simplified rules** - no redundant conditions (e.g., only shows `int_rate <= 6.86`, not `<= 13.60 AND <= 10.58 AND <= 6.86`)
- ✅ **Range notation** - shows `between 7.86% to 13.60%` instead of separate conditions
- ✅ **Formatted values** - percentages, currency, scores automatically formatted
- ✅ **Color-coded by risk** - segments colored by risk level
- ✅ **Editable fields highlighted** - "New Value" and "Business Reason" columns are blue

### 4.3 Example Edits

#### Example 1: Add Regulatory FICO Threshold

**Business requirement:** Regulatory requirement to split at FICO 650 (prime/subprime cutoff)

**Edit in Segment Rules worksheet:**
1. Find a segment that uses `fico_range_high` or `fico_range_low`
2. In "New_Value" column, enter: `650`
3. In "Business Reason" column, enter: `Regulatory prime/subprime threshold`

#### Example 2: Add Interest Rate Policy Threshold

**Business requirement:** Company policy to separate loans at 15% interest rate

**Edit in Segment Rules worksheet:**
1. Find a segment that uses `int_rate`
2. In "New_Value" column, enter: `15.0`
3. In "Business Reason" column, enter: `Company policy - subprime threshold`

#### Example 3: Add Income-Based Segmentation

**Business requirement:** Separate high-income borrowers (>$75K) for preferential pricing

**Edit in Segment Rules worksheet:**
1. Find a segment that uses `annual_inc`
2. In "New_Value" column, enter: `75000`
3. In "Business Reason" column, enter: `High-income segment for preferential pricing`

### 4.4 Alternative: Merge Similar Segments

Navigate to **"Segment Actions"** worksheet to merge segments with similar default rates.

**Example:** Segments 1 (7.98% PD) and 4 (8.35% PD) are very close - only 0.37pp difference.

```
┌──────────────┬──────────────┬──────────┬──────────┬──────────┬───────────┬────────┬─────────────┬───────────────────────────┐
│ Segment_ID   │ Observations │ Defaults │ PD_Rate_%│ Density_%│ Risk_Level│ Action │ Merge_Into  │ Notes                     │
├──────────────┼──────────────┼──────────┼──────────┼──────────┼───────────┼────────┼─────────────┼───────────────────────────┤
│ 1            │ 111,761      │ 8,919    │ 7.98     │ 7.06     │ Low       │ MERGE  │ 4           │ Similar to Segment 4      │
│ 4            │ 126,561      │ 10,568   │ 8.35     │ 8.00     │ Low       │ KEEP   │             │ Accept merge from Seg 1   │
└──────────────┴──────────────┴──────────┴──────────┴──────────┴───────────┴────────┴─────────────┴───────────────────────────┘
```

**Change:**
1. Set Segment 1 "Action" to `MERGE`
2. Set "Merge_Into" to `4`
3. Add note explaining the merge

### 4.5 Save Excel File

```
File → Save (Ctrl+S / Cmd+S)
```

**Important:** Keep the filename as `modification_template.xlsx` or note the new filename for the next step.

---

## Step 5: Convert Excel to JSON

### 5.1 Convert Excel Edits to JSON Format

```bash
python interfaces/excel_to_json.py modification_template.xlsx
```

**What it does:**
- Reads Excel workbook
- Parses "Segment Actions", "Segment Rules", and "Model Parameters" worksheets
- Generates `modification.json` with all changes

**Expected output:**
```
================================================================================
EXCEL TO JSON CONVERSION
================================================================================

Reading Excel file: modification_template.xlsx
  - Found 12 worksheets

Parsing 'Segment Actions' worksheet...
  - Merge requests: 1 (Segment 1 → 4)

Parsing 'Segment Rules' worksheet...
  - Threshold changes: 3
    • int_rate: 15.0 (Company policy - subprime threshold)
    • fico_range_high: 650 (Regulatory prime/subprime threshold)
    • annual_inc: 75000 (High-income segment for preferential pricing)

Parsing 'Model Parameters' worksheet...
  - No parameter changes

[OK] Created: modification.json

Next step: python apply_modifications.py modification.json
```

**Output file created:**
- `modification.json` (structured modification instructions)

**Time:** ~5 seconds

---

## Step 6: Apply Modifications and Re-score

### 6.1 Apply Changes and Re-train Model

```bash
python apply_modifications.py modification.json
```

**What it does:**
1. Loads Lending Club data (full 2.26M rows)
2. Re-trains model with forced splits at your specified thresholds
3. Applies segment merges
4. Re-validates all IRB requirements
5. Generates new validation report

**Expected output:**
```
================================================================================
APPLYING SEGMENT MODIFICATIONS
================================================================================

Modifications requested:
  Merge segments: [[1, 4]]
  Forced splits: {'int_rate': 15.0, 'fico_range_high': 650, 'annual_inc': 75000}
  Parameter changes: 0

Loading Lending Club data...
Loaded 2,260,701 rows

--------------------------------------------------------------------------------
FITTING MODEL WITH NEW PARAMETERS
--------------------------------------------------------------------------------

Fitting base tree...
Applying forced splits...
  [FORCED SPLIT] Feature 'int_rate' at threshold 15.0
  [FORCED SPLIT] Feature 'fico_range_high' at threshold 650
  [FORCED SPLIT] Feature 'annual_inc' at threshold 75000

Post-processing adjustments...
  [OK] Forced splits applied: 3

Validating segments...
  ✓ Minimum defaults check: PASSED (all segments ≥ 500 defaults)
  ✓ Statistical significance: PASSED
  ✓ Segment density: PASSED (5% ≤ density ≤ 40%)

--------------------------------------------------------------------------------
APPLYING MANUAL MERGES
--------------------------------------------------------------------------------

Merging Segment 4 into Segment 1
  [OK] Merged

--------------------------------------------------------------------------------
RE-VALIDATING AFTER MERGES
--------------------------------------------------------------------------------

SEGMENT SUMMARY:
  Segment 0: 195,234 obs (12.3%), 4,123 defaults (2.11% PD)
  Segment 1: 238,322 obs (15.1%), 19,487 defaults (8.18% PD)  ← MERGED
  Segment 2: 267,451 obs (16.9%), 13,556 defaults (5.07% PD)
  Segment 3: 298,732 obs (18.9%), 31,945 defaults (10.69% PD)
  Segment 4: 401,234 obs (25.4%), 63,234 defaults (15.76% PD)
  Segment 5: 181,517 obs (11.5%), 45,571 defaults (25.11% PD)

✓ All validation tests passed

================================================================================
MODIFICATION COMPLETE
================================================================================

New report saved: modification_result.json

Compare with original:
  Original: lending_club_full_report.json
  Modified: modification_result.json
```

**Output file created:**
- `modification_result.json` (new validation report with changes applied)

**Time:** ~10-15 minutes

---

## Step 7: Compare Results

### 7.1 Compare Original vs Modified Segmentation

```bash
python apply_modifications.py --compare lending_club_full_report.json modification_result.json
```

**Expected output:**
```
================================================================================
COMPARING SEGMENTATION REPORTS
================================================================================

Metric                        Original            Modified            Change
-------------------------------------------------------------------------------------
Number of Segments            7                   6                   -1
PD Range                      22.84%              23.00%              +0.16%
Training Validation           True                True                Same
Validation Set                True                True                Same

SEGMENT CHANGES:
- Segment 1 and 4 merged → New segment has 238,322 obs (8.18% PD)
- Forced splits at int_rate=15.0, fico_range_high=650, annual_inc=75000 applied
- All segments still pass IRB requirements

================================================================================
```

### 7.2 Review New Segment Rules

Extract rules from the modified model:

```bash
python extract_segment_rules.py
```

This will overwrite `segment_rules_detailed.json` with the new rules. Compare the new thresholds to see your forced splits applied.

### 7.3 Generate Updated Excel Template (Optional)

```bash
python interfaces/create_excel_template.py
```

This creates a new `modification_template.xlsx` with the updated segment structure. You can use this to make further refinements.

---

## Complete Workflow Summary

```bash
# Step 1: Generate baseline (10-15 min)
python test_lending_club_simple.py

# Step 2: Extract rules (2-3 min)
python extract_segment_rules.py

# Step 3: Generate Excel template (30 sec)
python interfaces/create_excel_template.py

# Step 4: Edit Excel manually
# Open modification_template.xlsx → Edit "Segment Rules" worksheet → Save

# Step 5: Convert Excel to JSON (5 sec)
python interfaces/excel_to_json.py modification_template.xlsx

# Step 6: Apply modifications (10-15 min)
python apply_modifications.py modification.json

# Step 7: Compare results (5 sec)
python apply_modifications.py --compare lending_club_full_report.json modification_result.json

# Optional: Extract new rules (2-3 min)
python extract_segment_rules.py

# Optional: Generate new Excel with updated rules (30 sec)
python interfaces/create_excel_template.py
```

**Total time:** ~25-35 minutes (plus manual editing time)

---

## Key Files Reference

| File | Description | When Created |
|------|-------------|--------------|
| `lending_club_full_report.json` | Initial validation report | Step 1 |
| `segment_rules_detailed.json` | Segment decision rules | Step 2 |
| `modification_template.xlsx` | Excel editing template | Step 3 |
| `modification.json` | JSON modification instructions | Step 5 |
| `modification_result.json` | New validation report | Step 6 |

---

## Troubleshooting

### Issue: "Python not found"
**Solution:** Install Python from https://python.org or use full path to python.exe

### Issue: "Dataset file not found"
**Solution:** Verify `data/lending_club_test.csv` exists with 2.26M rows

### Issue: "openpyxl not available"
**Solution:** `pip install openpyxl`

### Issue: Excel file won't open
**Solution:** Use LibreOffice Calc or Google Sheets (upload, edit, download as .xlsx)

### Issue: Model runs too slowly
**Solution:** This is normal for 2.26M rows. Consider sampling:
```python
# Edit test_lending_club_simple.py line 42
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y,
    train_size=500000  # Sample 500K for faster testing
)
```

---

## Advanced: Categorical Features

If you want to add categorical splits (e.g., "split high-risk loan purposes"):

**In Excel "Segment Rules" worksheet:**

```
Feature       | Condition | Current Value           | New Value                  | Format      | Reason
loan_purpose  | IN        | education, medical      | education, medical, small_business | categorical | High-risk purposes
```

The system will create a segment specifically for loans with these purposes.

See `THRESHOLD_EDITING_GUIDE.md` for more details.

---

## Next Steps

After completing this workflow, you can:

1. **Export for production:** Use `tree_structure_v2.json` (engine-agnostic format)
2. **Score new data:** Use `irb_segmentation.scorer` module for JSON-based scoring
3. **Iterate:** Repeat workflow with different thresholds to test business scenarios
4. **Visualize:** Use `create_enhanced_dashboard.py` to generate interactive HTML dashboard

---

## Questions?

- **Detailed threshold editing:** See `THRESHOLD_EDITING_GUIDE.md`
- **Architecture details:** See `ARCHITECTURE_ENHANCEMENTS.md`
- **Data sources:** See `DATA_SOURCES.md`
- **Testing:** See `TESTING_GUIDE.md`
