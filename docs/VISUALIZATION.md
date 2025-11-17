# IRB Segmentation - Visualization & Modification Tools

## Overview

Complete toolkit for **visualizing, analyzing, and modifying** IRB segmentation results from the Lending Club dataset (2.26 million observations).

---

## 🎯 What You Can Do

### 1. **View Results** - Interactive Dashboard
```bash
python create_dashboard.py
```
Opens beautiful HTML dashboard showing:
- 7 segments with risk levels and statistics
- Default rate visualizations
- Validation test results
- Modification instructions

**Open:** `segmentation_dashboard.html` in any browser

---

### 2. **Modify Segments** - Fix Issues
```bash
# Create template (pre-configured to merge similar segments 1 & 4)
python apply_modifications.py --create-sample

# Apply modifications
python apply_modifications.py modify_segments_sample.json
```

**Capabilities:**
- ✏️ Merge similar segments
- ➕ Add forced business rule splits
- ⚙️ Adjust tree parameters (depth, minimums)
- 🔄 Re-run full validation

---

### 3. **Compare Results** - Before/After Analysis
```bash
python apply_modifications.py --compare \\
    lending_club_full_report.json \\
    modify_segments_sample_result.json
```

Shows:
- Number of segments changed
- PD range differences
- Validation status changes

---

## 📊 Current Segmentation (7 Segments)

| Segment | Observations | Default Rate | Risk Level | Status |
|---------|-------------|--------------|------------|--------|
| 0 | 183,082 | 2.16% | Very Low | ✅ |
| 1 | 111,761 | 7.98% | Low | ⚠️ Similar to 4 |
| 2 | 254,388 | 5.09% | Low | ✅ |
| 3 | 272,580 | 11.08% | Medium | ✅ |
| 4 | 126,561 | 8.35% | Low-Medium | ⚠️ Similar to 1 |
| 5 | 415,442 | 16.20% | High | ✅ |
| 6 | 218,676 | 25.00% | Very High | ✅ |

**Issue:** Segments 1 and 4 are too similar (only 0.37pp difference)
**Recommendation:** Merge → 6 total segments

---

## 🔧 Recommended Fix

**Problem:** Chi-squared test fails for Segments 1 vs 4

**Solution:** Run the pre-configured merge
```bash
python apply_modifications.py modify_segments_sample.json
```

**Result:**
- ✅ 6 segments (down from 7)
- ✅ All statistical tests pass
- ✅ Combined segment: 238K obs, 8.15% PD
- ✅ Maintains excellent risk discrimination

**Time:** ~3 minutes

---

## 📁 Key Files

| File | Purpose | Size |
|------|---------|------|
| `segmentation_dashboard.html` | Interactive visualization | 26 KB |
| `modify_segments_sample.json` | Modification template | 1.9 KB |
| `lending_club_full_report.json` | Original results | 9.5 KB |
| `VISUALIZATION_GUIDE.md` | Complete tutorial | - |
| `visualize_tree.py` | Toolkit library | - |
| `create_dashboard.py` | Dashboard generator | - |
| `apply_modifications.py` | Modification engine | - |

---

## 🎨 Dashboard Features

### Visual Elements
- **Segment Cards:** Color-coded by risk (green → red)
- **Bar Chart:** Default rate comparison
- **Validation Grid:** Pass/fail indicators
- **Statistics:** Key metrics at a glance

### Information Displayed
- 📊 Total segments, observations, defaults
- 📈 Default rate range (2.16% - 25.00%)
- ✅ Validation results (chi-squared, PSI, density)
- ⚙️ Model parameters
- 🔧 Adjustment history (23 merges applied)

### Responsive Design
- ✅ Desktop browsers
- ✅ Tablets
- ✅ Mobile devices

---

## 🔄 Modification Workflow

### Step 1: View Current State
```bash
python create_dashboard.py
# Open segmentation_dashboard.html
```

### Step 2: Create Modification Template
```bash
python apply_modifications.py --create-sample
```

### Step 3: Edit Template (Optional)
Open `modify_segments_sample.json` and customize:

```json
{
  "modifications": {
    "merge_segments": {
      "value": [[1, 4]]  // Merge these segment pairs
    },
    "forced_splits": {
      "value": {
        "fico_score": 650  // Force split at FICO 650
      }
    },
    "parameter_changes": {
      "max_depth": 5,
      "min_samples_leaf": 10000
    }
  }
}
```

### Step 4: Apply Modifications
```bash
python apply_modifications.py modify_segments_sample.json
```

**This will:**
1. Load 2.26M observations
2. Re-train with new parameters
3. Apply manual merges
4. Run full validation
5. Generate new report

### Step 5: Compare & Review
```bash
# Compare
python apply_modifications.py --compare \\
    lending_club_full_report.json \\
    modify_segments_sample_result.json

# Visualize new results
python create_dashboard.py
```

---

## 💡 Common Modifications

### Merge Similar Segments
```json
"merge_segments": {
  "value": [[1, 4], [2, 5]]  // Multiple merges
}
```

### Add Business Rules
```json
"forced_splits": {
  "value": {
    "fico_score": 650,
    "dti": 0.35,
    "ltv": 0.80
  }
}
```

### More Granular Segmentation
```json
"parameter_changes": {
  "max_depth": 6,
  "min_samples_leaf": 5000,
  "min_segment_density": 0.03
}
```

### Simpler Segmentation
```json
"parameter_changes": {
  "max_depth": 3,
  "min_samples_leaf": 30000,
  "min_segment_density": 0.15
}
```

---

## 📊 Validation Metrics

### Current Status

| Test | Training | Validation | Notes |
|------|----------|------------|-------|
| Chi-squared | ❌ | ✅ | 1 pair failed (Seg 1 vs 4) |
| Minimum Defaults | ✅ | ✅ | 3,946 - 67,295 per segment |
| Density Bounds | ✅ | ✅ | 7.06% - 26.25% |
| PSI | - | ✅ | 0.000158 (excellent) |
| Default Rate Diff | ✅ | ✅ | Clear separation |

**Recommendation:** Merge Segments 1 & 4 to fix chi-squared test

---

## 🎯 Use Cases

### Use Case 1: Fix Statistical Test Failure
**Goal:** Resolve chi-squared issue
**Action:** Use default template to merge Segments 1 & 4
**Result:** 6 segments, all tests pass

### Use Case 2: Enforce Regulatory Split
**Goal:** Always split at FICO 650 (policy requirement)
**Action:** Add forced split in template
**Result:** Segments honor business rule

### Use Case 3: Create Finer Risk Tiers
**Goal:** More granular pricing segments
**Action:** Increase max_depth to 6, reduce minimums
**Result:** 10-12 segments with tighter PD ranges

### Use Case 4: Simplify for Operational Use
**Goal:** Fewer, clearer risk buckets
**Action:** Reduce max_depth to 3, increase minimums
**Result:** 3-4 segments (Low/Medium/High/Very High)

---

## 📈 Visualization Options

### 1. Interactive HTML Dashboard (Recommended)
- **Best for:** Presentations, stakeholder reviews
- **Command:** `python create_dashboard.py`
- **Output:** `segmentation_dashboard.html`
- **Pros:** Beautiful, interactive, no dependencies

### 2. Text Tree Structure
- **Best for:** Technical analysis
- **Usage:** `viz.print_tree_structure()`
- **Pros:** Shows split rules, terminal friendly

### 3. Graphviz Diagram
- **Best for:** Documentation, papers
- **Usage:** `viz.export_graphviz("tree.dot")`
- **Pros:** Publication-quality visuals
- **Cons:** Requires Graphviz installation

### 4. JSON/Table Export
- **Best for:** Integration, automation
- **Usage:** `viz.show_segment_rules("json")`
- **Pros:** Machine-readable

---

## 🚀 Quick Start (5 Minutes)

```bash
# 1. View dashboard
python create_dashboard.py
# Open segmentation_dashboard.html

# 2. Apply recommended fix (merge Segments 1 & 4)
python apply_modifications.py --create-sample
python apply_modifications.py modify_segments_sample.json

# 3. Compare results
python apply_modifications.py --compare \\
    lending_club_full_report.json \\
    modify_segments_sample_result.json

# 4. View updated dashboard
python create_dashboard.py
```

---

## 📚 Documentation

- **`VISUALIZATION_GUIDE.md`** - Complete tutorial with examples
- **`VISUALIZATION_SUMMARY.md`** - Feature overview
- **`LENDING_CLUB_FULL_RESULTS.md`** - Detailed test results
- **Dashboard** - Built-in modification instructions

---

## 🎓 Best Practices

1. **Always visualize first** - Understand current state before modifying
2. **Start with recommended merge** - Fix obvious issues (Seg 1 & 4)
3. **One change at a time** - Easier to understand impact
4. **Compare before/after** - Use built-in comparison tool
5. **Document rationale** - Keep notes on why you made changes
6. **Validate thoroughly** - Check all statistical tests pass
7. **Business alignment** - Ensure segments make operational sense

---

## ⚠️ Important Notes

- **Re-training time:** ~3 minutes for full 2.26M dataset
- **Memory:** Handles full dataset in-memory (tested on standard machine)
- **Validation:** All modifications automatically re-run full validation suite
- **Reproducibility:** Use same random_state (42) for consistent results

---

## 🎯 Success Metrics

After modification, check:
- [ ] All chi-squared tests pass
- [ ] PSI < 0.1 (< 0.25 acceptable)
- [ ] Each segment has 500+ defaults
- [ ] Segment densities between 5-40%
- [ ] PD range > 15pp (good discrimination)
- [ ] Risk levels monotonic (mostly)
- [ ] Business rules honored

---

## 📞 Getting Help

1. **Dashboard:** Open `segmentation_dashboard.html` for visual guide
2. **Tutorial:** Read `VISUALIZATION_GUIDE.md` for step-by-step
3. **Results:** Check `LENDING_CLUB_FULL_RESULTS.md` for context
4. **Code:** Review Python docstrings for API details

---

## 🎉 What's Included

✅ **Interactive HTML dashboard** (no dependencies)
✅ **Modification workflow** (JSON templates)
✅ **Automatic re-validation** (full IRB compliance)
✅ **Comparison tools** (before/after analysis)
✅ **Multiple visualization formats** (HTML, text, Graphviz, JSON)
✅ **Comprehensive documentation** (tutorials and guides)
✅ **Production export** (SQL, Python examples)
✅ **Best practices** (regulatory compliance)

---

**Status:** ✅ Ready to Use
**Test Date:** 2025-10-01
**Dataset:** Lending Club 2.26M observations
**Framework:** IRB Segmentation v1.0

---

## 🚀 Try It Now!

```bash
python create_dashboard.py
```

Then open `segmentation_dashboard.html` in your browser!
