# Threshold Editing Guide

## Overview

The enhanced dashboard now shows **segment decision rules** and **feature thresholds**. This guide explains how to view and modify them.

---

## 📊 What You'll See in the Enhanced Dashboard

### 1. Segment Decision Rules

Each segment now displays its complete decision paths:

**Example - Segment 0 (Very Low Risk, 2.16% PD):**
```
Path 1: IF int_rate <= 6.86 AND fico_range_high <= 721.50
Path 2: IF int_rate <= 7.86 AND fico_range_high > 721.50 AND annual_inc <= 43336.90
Path 3: IF int_rate <= 7.86 AND fico_range_high > 721.50 AND annual_inc > 43336.90
```

**Key Insights:**
- Segment 0 (lowest risk) has interest rates ≤ 6.86% - 7.86%
- High FICO scores (>721.5) or low rates
- Multiple paths to qualify

### 2. Feature Thresholds

All thresholds used in the tree:

**int_rate (Interest Rate):**
- 6.86, 7.86, 10.58, 12.98, 13.56, 13.60, 14.01, 16.14, 16.97, 22.37

**fico_range_high (FICO Score):**
- 686.50, 721.50

**annual_inc (Annual Income):**
- $43,336.90, $58,115.50, $61,077, $61,956.50, $70,105

**dti (Debt-to-Income):**
- 14.78

**inq_last_6mths (Recent Inquiries):**
- 0.50, 1.50

---

## 🔧 How to Modify Thresholds

### Step 1: Create Modification Template

```bash
python apply_modifications.py --create-sample
```

This creates `modify_segments_sample.json`

### Step 2: Edit the Template

Open `modify_segments_sample.json` and add forced splits:

```json
{
  "modifications": {
    "forced_splits": {
      "value": {
        "int_rate": 15.0,
        "fico_range_high": 700,
        "annual_inc": 60000,
        "dti": 25.0
      }
    }
  }
}
```

### Step 3: Apply Modifications

```bash
python apply_modifications.py modify_segments_sample.json
```

This will:
- Re-train the model
- Force splits at your specified thresholds
- Generate new segment rules
- Create `modify_segments_sample_result.json`

### Step 4: View Updated Dashboard

```bash
# Extract new rules
python extract_segment_rules.py

# Create new dashboard
python create_enhanced_dashboard.py
```

---

## 💡 Common Threshold Modifications

### Use Case 1: Regulatory Compliance

**Requirement:** Always split at FICO 650 (regulatory threshold)

```json
"forced_splits": {
  "value": {
    "fico_range_high": 650,
    "fico_range_low": 650
  }
}
```

**Result:** All segments will have either FICO < 650 or FICO ≥ 650

---

### Use Case 2: Product Policy

**Requirement:** Different pricing for high vs low interest rate loans

```json
"forced_splits": {
  "value": {
    "int_rate": 12.0
  }
}
```

**Result:** Clear split at 12% interest rate

---

### Use Case 3: Income-Based Segmentation

**Requirement:** Separate high-income borrowers ($75k+)

```json
"forced_splits": {
  "value": {
    "annual_inc": 75000
  }
}
```

**Result:** High-income segment (≥$75k) separated

---

### Use Case 4: Multiple Business Rules

**Requirement:** Apply several policy thresholds at once

```json
"forced_splits": {
  "value": {
    "int_rate": 15.0,
    "fico_range_high": 680,
    "annual_inc": 50000,
    "dti": 30.0,
    "loan_amnt": 15000
  }
}
```

**Result:** All 5 thresholds honored in segmentation

---

## 📋 Understanding Current Thresholds

### Interest Rate (int_rate)

**Current thresholds:** 6.86%, 7.86%, 10.58%, 12.98%, 13.60%, 16.97%

**What they mean:**
- < 6.86%: Super-prime (Segment 0)
- 6.86-10.58%: Prime (Segments 1-2)
- 10.58-13.60%: Near-prime (Segments 3-4)
- 13.60-16.97%: Subprime (Segment 5)
- > 16.97%: Deep subprime (Segment 6)

### FICO Score (fico_range_high)

**Current thresholds:** 686.50, 721.50

**What they mean:**
- > 721.5: Excellent credit
- 686.5-721.5: Good credit
- < 686.5: Fair/Poor credit

### Annual Income (annual_inc)

**Current thresholds:** $43K, $58K, $61K, $70K

**What they mean:**
- < $43K: Low income
- $43K-$61K: Moderate income
- $61K-$70K: Middle income
- > $70K: High income

---

## 🎯 Best Practices

### 1. **Round to Business-Friendly Values**

**Bad:**
```json
"int_rate": 13.5634
```

**Good:**
```json
"int_rate": 13.5  // or 13.0, 14.0
```

### 2. **Use Regulatory/Policy Thresholds**

Align with existing business rules:
- FICO 640, 680, 720 (industry standards)
- DTI 25%, 30%, 35% (common thresholds)
- Income $50K, $75K, $100K (round numbers)

### 3. **Test Impact Before Production**

```bash
# Apply modifications
python apply_modifications.py modify_segments_sample.json

# Compare before/after
python apply_modifications.py --compare \\
    lending_club_full_report.json \\
    modify_segments_sample_result.json
```

### 4. **Document Rationale**

Add comments to your modification file:

```json
{
  "metadata": {
    "reason": "Align with 2025 underwriting policy",
    "approved_by": "Chief Risk Officer",
    "date": "2025-10-01"
  },
  "modifications": {
    "forced_splits": {
      "value": {
        "fico_range_high": 680,
        "int_rate": 15.0
      }
    }
  }
}
```

---

## 📊 Current Segment Characteristics

Based on the dashboard, here's what drives each segment:

### Segment 0 (Very Low Risk - 2.16% PD)
**Key features:** int_rate ≤ 6.86-7.86%, fico_range_high > 721.5
**Profile:** Very low interest rates, excellent credit

### Segment 1 (Low Risk - 7.98% PD)
**Key features:** int_rate 7.86-13.60%, fico_range_low ≤ 702.5, dti > 14.78
**Profile:** Moderate rates, good credit, some DTI risk

### Segment 2 (Low Risk - 5.09% PD)
**Key features:** int_rate 7.86-10.58%, fico_range_low > 702.5, inq_last_6mths ≤ 1.5
**Profile:** Low-moderate rates, good credit, few inquiries

### Segment 3 (Medium Risk - 11.08% PD)
**Key features:** int_rate 10.58-13.56%, fico_range_low ≤ 697.5
**Profile:** Mid rates, moderate credit

### Segment 4 (Low-Medium Risk - 8.35% PD)
**Key features:** int_rate 10.58-12.98%, fico_range_low > 697.5
**Profile:** Mid rates, better credit than Segment 3

### Segment 5 (High Risk - 16.20% PD)
**Key features:** int_rate 12.98-16.97%, annual_inc ≤ $61K
**Profile:** Higher rates, moderate income

### Segment 6 (Very High Risk - 25.00% PD)
**Key features:** int_rate > 16.14-16.97%, inq_last_6mths > 0.5
**Profile:** Highest rates, recent credit shopping

---

## 🔍 Interpreting Rules

### AND Conditions

```
IF int_rate <= 7.86 AND fico_range_high > 721.50
```

**Means:** BOTH conditions must be true
- Interest rate must be ≤ 7.86%
- AND FICO must be > 721.5

### Multiple Paths

If a segment has multiple paths, ANY path qualifies:

```
Segment 0:
  Path 1: int_rate <= 6.86 AND fico_range_high <= 721.50
  Path 2: int_rate <= 7.86 AND annual_inc > 43336.90
```

**Means:** Either path 1 OR path 2 assigns to Segment 0

---

## ⚠️ Important Notes

1. **Re-training Required:** Changing thresholds triggers full model re-train (~3 min)

2. **Validation Re-run:** All statistical tests re-executed automatically

3. **Segment Count May Change:** New thresholds may create more/fewer segments

4. **Feature Availability:** Only use features present in your dataset

5. **Check Feature Names:**
```bash
python -c "import pandas as pd; print(pd.read_csv('data/lending_club_test.csv', nrows=1).columns.tolist())"
```

---

## 📞 Quick Reference

### View Current Rules
```bash
# Open enhanced dashboard
segmentation_dashboard_enhanced.html
```

### Create Modification Template
```bash
python apply_modifications.py --create-sample
```

### Edit Thresholds
```bash
# Edit modify_segments_sample.json
# Add your thresholds to "forced_splits" → "value"
```

### Apply Changes
```bash
python apply_modifications.py modify_segments_sample.json
```

### Update Dashboard
```bash
python extract_segment_rules.py
python create_enhanced_dashboard.py
```

---

## 🎉 What You Can Now Do

✅ **See exactly how each segment is defined**
✅ **View all threshold values used**
✅ **Add business rule thresholds**
✅ **Understand which features drive each segment**
✅ **Export rules for production implementation**
✅ **Compare before/after threshold changes**

---

**Files:**
- `segmentation_dashboard_enhanced.html` - View rules visually
- `segment_rules_detailed.json` - Machine-readable rules
- `modify_segments_sample.json` - Edit thresholds here
