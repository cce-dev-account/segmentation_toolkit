# IRB Segmentation - Visualization & Modification Guide

## Overview

This guide explains how to visualize the Lending Club segmentation results and apply modifications for re-testing.

---

## 📊 Visualization Tools

### 1. Interactive HTML Dashboard (Recommended)

**Best for:** Executive presentations, business reviews, quick exploration

```bash
python create_dashboard.py
```

**Output:** `segmentation_dashboard.html`

**Features:**
- 📈 Visual segment cards with risk levels
- 📊 Default rate bar charts
- ✅ Validation status indicators
- 🎨 Color-coded risk levels
- 📱 Responsive design (works on mobile/tablet)

**Open in browser:** Just double-click `segmentation_dashboard.html`

---

### 2. Text-Based Tree Structure

**Best for:** Understanding decision rules, technical analysis

```python
from visualize_tree import TreeVisualizer

# After fitting a model:
viz = TreeVisualizer(engine)
viz.print_tree_structure(max_depth=3)
```

**Output:**
```
[Node 0] feature_1 <= 50.23
     Samples: 1,582,490 | Defaults: 188,552 (11.92%)
  [Node 1] feature_2 <= 12.45
       Samples: 650,000 | Defaults: 45,000 (6.92%)
    [Node 3] LEAF
         Samples: 183,082 | Defaults: 3,946 (2.16%)
         >>> SEGMENT 0
```

---

### 3. Graphviz Tree Diagram

**Best for:** Academic papers, technical documentation

```python
viz.export_graphviz("tree.dot")
```

**Generate PNG:**
```bash
dot -Tpng tree.dot -o tree.png
```

**Requires:** Graphviz installation (https://graphviz.org/)

---

### 4. Segment Rules Export

**Best for:** Implementation in production systems, code generation

```python
viz.show_segment_rules(output_format="text")  # Human-readable
viz.show_segment_rules(output_format="json")  # Machine-readable
viz.show_segment_rules(output_format="table") # Tabular format
```

**Example Output:**
```
1. IF credit_score <= 6250.0 AND utilization > 0.75 THEN Segment 6
2. IF credit_score <= 6250.0 AND utilization <= 0.75 THEN Segment 5
3. IF credit_score > 6250.0 AND dti <= 0.25 THEN Segment 0
```

---

## 🔧 Modification Workflow

### Current Segmentation Issue

From the Lending Club full dataset results:

- **Problem:** Segments 1 and 4 are very similar
  - Segment 1: 7.98% default rate (111,761 obs)
  - Segment 4: 8.35% default rate (126,561 obs)
  - Difference: Only 0.37 percentage points
  - Chi-squared test: Marginally failed (p=0.00087 vs threshold=0.00048)

- **Options:**
  1. Merge Segments 1 and 4 → 6 total segments
  2. Adjust parameters to create more separation
  3. Add forced split on business-relevant feature

---

## 📝 Step-by-Step Modification Process

### Step 1: Create Modification Template

```bash
python apply_modifications.py --create-sample
```

**Output:** `modify_segments_sample.json`

This creates a pre-configured template that merges Segments 1 and 4.

---

### Step 2: Review and Edit the Template

Open `modify_segments_sample.json`:

```json
{
  "modifications": {
    "merge_segments": {
      "description": "List segment pairs to merge",
      "value": [[1, 4]]  // <-- Merge segments 1 and 4
    },
    "forced_splits": {
      "description": "Add forced split points",
      "value": {}  // <-- Add splits here if needed
    },
    "parameter_changes": {
      "max_depth": 5,
      "min_samples_leaf": 10000,
      "min_defaults_per_leaf": 500,
      "min_segment_density": 0.05,
      "max_segment_density": 0.40
    }
  }
}
```

**Common Modifications:**

#### A. Merge Similar Segments
```json
"merge_segments": {
  "value": [[1, 4], [2, 3]]  // Merge 1→4 and 2→3
}
```

#### B. Add Forced Splits
```json
"forced_splits": {
  "value": {
    "fico_score": 650,  // Always split at FICO 650
    "dti": 0.35         // Always split at DTI 35%
  }
}
```

#### C. Adjust Parameters for More Segments
```json
"parameter_changes": {
  "max_depth": 6,              // Allow deeper tree
  "min_samples_leaf": 5000,    // Smaller minimum (more segments)
  "min_defaults_per_leaf": 250, // Lower threshold
  "min_segment_density": 0.03,  // Allow smaller segments
  "max_segment_density": 0.40
}
```

#### D. Adjust Parameters for Fewer Segments
```json
"parameter_changes": {
  "max_depth": 4,               // Shallower tree
  "min_samples_leaf": 20000,    // Larger minimum (fewer segments)
  "min_defaults_per_leaf": 1000,
  "min_segment_density": 0.10,  // Larger segments
  "max_segment_density": 0.35
}
```

---

### Step 3: Apply Modifications

```bash
python apply_modifications.py modify_segments_sample.json
```

**This will:**
1. Load the full Lending Club dataset (2.26M observations)
2. Re-train the segmentation model with new parameters
3. Apply manual merges (if specified)
4. Run full validation suite
5. Generate new report: `modify_segments_sample_result.json`

**Expected runtime:** ~3-5 minutes

---

### Step 4: Compare Results

```bash
python apply_modifications.py --compare lending_club_full_report.json modify_segments_sample_result.json
```

**Output:**
```
Metric                        Original            Modified            Change
---------------------------------------------------------------------------------
Number of Segments            7                   6                   -1
PD Range                      22.84%              23.10%              +0.26%
Training Validation           False               True                Changed
Validation Set                True                True                Same
```

---

### Step 5: Visualize New Results

```bash
python create_dashboard.py
```

Or specify the new report:

```python
from create_dashboard import create_interactive_dashboard
create_interactive_dashboard(
    report_file="modify_segments_sample_result.json",
    output_file="modified_dashboard.html"
)
```

---

## 🎯 Common Use Cases

### Use Case 1: Merge Similar Segments (Recommended)

**Goal:** Fix chi-squared test failure by merging Segments 1 and 4

**Steps:**
1. Use default template: `python apply_modifications.py --create-sample`
2. Apply directly: `python apply_modifications.py modify_segments_sample.json`
3. Result: 6 segments instead of 7, all validations should pass

**Expected Outcome:**
- Segments reduced from 7 to 6
- Chi-squared test passes
- Combined segment: ~238K observations, ~8.2% default rate
- Statistical power remains excellent

---

### Use Case 2: Add Business Rule Split

**Goal:** Always split at FICO 650 (regulatory or policy requirement)

**Edit template:**
```json
"forced_splits": {
  "value": {
    "fico_score": 650
  }
}
```

**Note:** Feature must exist in dataset. Check feature names:
```python
import pandas as pd
df = pd.read_csv("data/lending_club_test.csv", nrows=1)
print(df.select_dtypes(include=['number']).columns.tolist())
```

---

### Use Case 3: Create Finer Segmentation

**Goal:** More granular segments for pricing

**Edit template:**
```json
"parameter_changes": {
  "max_depth": 6,
  "min_samples_leaf": 5000,
  "min_defaults_per_leaf": 250,
  "min_segment_density": 0.03,
  "max_segment_density": 0.35
}
```

**Expected:** 10-15 segments (depending on data)

---

### Use Case 4: Simplify to High-Level Segments

**Goal:** Fewer segments for simpler risk tiers

**Edit template:**
```json
"parameter_changes": {
  "max_depth": 3,
  "min_samples_leaf": 30000,
  "min_defaults_per_leaf": 1000,
  "min_segment_density": 0.15,
  "max_segment_density": 0.40
}
```

**Expected:** 3-4 segments (Low/Medium/High/Very High risk)

---

## 📋 Validation Checklist

After applying modifications, check:

- [ ] **Chi-Squared Test:** All segment pairs significantly different (p < adjusted alpha)
- [ ] **Minimum Defaults:** Each segment has ≥ 500 defaults (or configured minimum)
- [ ] **Density Bounds:** Each segment between 5-40% of population (or configured)
- [ ] **PSI:** Population Stability Index < 0.1 on validation set
- [ ] **Business Logic:** Segments make business sense (monotonic risk ordering)
- [ ] **Implementation:** Rules can be coded in production systems

---

## 🔍 Interpreting Results

### Dashboard Metrics

**Risk Badges:**
- 🟢 Very Low: PD < 5%
- 🔵 Low: PD 5-10%
- 🟡 Medium: PD 10-15%
- 🟠 High: PD 15-20%
- 🔴 Very High: PD > 20%

**Validation Indicators:**
- ✅ Green: Test passed
- ❌ Red: Test failed
- ⚠️ Yellow: Warning (near threshold)

**Key Metrics:**
- **PD Range:** Wider is better (more risk discrimination)
- **PSI:** Lower is better (< 0.1 excellent, < 0.25 acceptable)
- **Segment Density:** Balanced is better (avoid 1-2 dominant segments)

---

## 💾 Exporting for Production

### 1. Segment Rules (JSON)

```python
import json

with open('lending_club_full_report.json', 'r') as f:
    report = json.load(f)

# Extract statistics
stats = report['segment_statistics']

# Save production config
production_config = {
    'segments': stats,
    'parameters': report['parameters'],
    'validation_date': report['timestamp']
}

with open('production_segments.json', 'w') as f:
    json.dump(production_config, f, indent=2)
```

---

### 2. SQL Implementation

Convert tree rules to SQL CASE statement:

```sql
SELECT
    loan_id,
    CASE
        WHEN credit_score <= 6250 AND utilization > 0.75 THEN 6  -- Very High Risk
        WHEN credit_score <= 6250 AND utilization <= 0.75 THEN 5 -- High Risk
        WHEN credit_score > 6250 AND dti > 0.35 THEN 3           -- Medium Risk
        WHEN credit_score > 6250 AND dti <= 0.35 AND income > 75000 THEN 0  -- Very Low Risk
        ELSE 2  -- Low Risk
    END AS segment_id
FROM loans;
```

---

### 3. Python Scoring Function

```python
def score_loan(credit_score, dti, utilization, income):
    """
    Assign loan to risk segment.

    Returns: segment_id (0-6)
    """
    if credit_score <= 6250:
        if utilization > 0.75:
            return 6  # Very High Risk (25% PD)
        else:
            return 5  # High Risk (16% PD)
    else:
        if dti > 0.35:
            return 3  # Medium Risk (11% PD)
        elif income > 75000:
            return 0  # Very Low Risk (2% PD)
        else:
            return 2  # Low Risk (5% PD)
```

---

## 🐛 Troubleshooting

### Issue: "Dataset file not found"

**Solution:**
```bash
ls data/lending_club_test.csv
```
If missing, re-run: `python test_lending_club_simple.py`

---

### Issue: "Modification failed - segments not found"

**Cause:** Specified segment IDs don't exist after re-training

**Solution:** Check segment IDs in current report first:
```python
import json
with open('lending_club_full_report.json') as f:
    report = json.load(f)
print(list(report['segment_statistics'].keys()))
```

---

### Issue: "All validations still failing"

**Cause:** Parameters too aggressive or data doesn't support fine segmentation

**Solution:** Try conservative parameters:
```json
"parameter_changes": {
  "max_depth": 4,
  "min_samples_leaf": 20000,
  "min_defaults_per_leaf": 1000,
  "min_segment_density": 0.10,
  "max_segment_density": 0.35
}
```

---

## 📚 Additional Resources

- **Framework Documentation:** `README.md`
- **Data Sources:** `DATA_SOURCES.md`
- **Test Results:** `LENDING_CLUB_FULL_RESULTS.md`
- **API Reference:** See docstrings in `irb_segmentation/` modules

---

## 🎓 Best Practices

1. **Start Conservative:** Begin with fewer segments, add granularity as needed
2. **Business First:** Ensure segments align with business logic
3. **Validate Thoroughly:** Always check all statistical tests
4. **Document Changes:** Keep audit trail of all modifications
5. **Monitor Over Time:** Track PSI quarterly to detect population shift
6. **Regulatory Review:** Get approval before production deployment

---

**Need Help?**
- Check dashboard for visual guidance
- Review sample modifications in `modify_segments_sample.json`
- Examine validation results in detail in JSON reports
