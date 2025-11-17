# IRB Segmentation Framework - Demo Instructions

Welcome! This demo showcases a production-ready IRB (Internal Ratings-Based) PD model segmentation framework for credit risk.

## What This Demo Does

This framework automatically creates risk segments for credit portfolios by:
- Building decision tree-based segments using sklearn
- Enforcing Basel II/III regulatory requirements
- Applying business constraints (forced splits, monotonicity)
- Validating segments (chi-squared, PSI, binomial tests)
- Generating multiple output formats (rules, Excel, HTML, JSON)

---

## Quick Start (5 Minutes)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Setup Script

```bash
python setup_demo.py
```

This checks your environment and recommends which demo to run.

### 3. Run a Demo

**Option A: Quick Start (No Download Needed)**
```bash
python run_demo.py demo_german.yaml
```
- Auto-downloads German Credit dataset (1K rows)
- Runs in ~5 seconds
- Perfect for first-time users

**Option B: Medium Scale (Requires Download)**
```bash
python run_demo.py demo_taiwan.yaml
```
- Uses Taiwan Credit dataset (30K rows)
- Runs in ~30 seconds
- See DATASET_DOWNLOAD.md for download instructions

**Option C: Production Scale (Requires Download)**
```bash
python run_demo.py demo_lending_club.yaml
```
- Uses Lending Club dataset (1.3M rows, temporal split)
- Runs in ~2-3 minutes
- Demonstrates production-ready segmentation with OOT validation

---

## Understanding Your Results

After running a demo, you'll find output files in `output/demo_<dataset>/`:

### 1. segment_rules.txt (Human-Readable Rules)
```
SEGMENT 1: Low Risk
├─ Observations: 127,542 (13.3%)
├─ Defaults: 3,826 (3.0% default rate)
└─ Rules:
   IF credit_score > 680
   AND interest_rate <= 15.0
   THEN assign to Segment 1

Risk Level: 🟢 LOW RISK (PD = 3.0%)
```

### 2. segment_summary.xlsx (Excel Table)
Open in Excel for a formatted summary table with:
- Segment IDs
- Observation counts
- Default rates
- Segment densities
- Risk levels

### 3. dashboard.html (Interactive Visualization)
Open in your browser to see:
- Summary statistics
- Segment distribution bar charts
- Default rate comparisons
- Validation test results

### 4. baseline_report.json (Complete Report)
JSON file with all validation metrics, segment statistics, and configuration.

See **OUTPUT_GUIDE.md** for detailed interpretation.

---

## Three Demo Options Explained

### Option 1: Quick Start (German Credit)

**Dataset:** 1,000 credit applications from German bank
**Download:** Auto-downloads from UCI
**Runtime:** ~5 seconds
**Best for:** First-time users, testing installation

**What it demonstrates:**
- Basic segmentation workflow
- Automatic segment creation
- All output formats
- Validation tests (chi-squared, binomial)

**Command:**
```bash
python run_demo.py demo_german.yaml
```

---

### Option 2: Medium Scale (Taiwan Credit)

**Dataset:** 30,000 credit card records
**Download:** Manual (see DATASET_DOWNLOAD.md)
**Runtime:** ~30 seconds
**Best for:** Understanding IRB requirements, testing with realistic data

**What it demonstrates:**
- Production-scale parameters
- All validation tests (chi-squared, PSI, binomial)
- Segment density controls
- Minimum defaults requirements

**Command:**
```bash
python run_demo.py demo_taiwan.yaml
```

---

### Option 3: Production Scale (Lending Club)

**Dataset:** 1.3M loan records (2007-2018)
**Download:** Manual (see DATASET_DOWNLOAD.md)
**Runtime:** ~2-3 minutes
**Best for:** Production deployment simulation, temporal validation

**What it demonstrates:**
- **Temporal split:** Train on 2007-2012, validate on 2013-2014, test on 2015+
- **Categorical features:** Grade, home ownership, loan purpose
- **Business constraints:** Interest rate, credit score, DTI thresholds
- **Monotone constraints:** Risk ordering (higher FICO = lower risk)
- **Full validation suite:** Chi-squared, PSI, binomial confidence intervals
- **Regulatory compliance:** Basel II/III standards

**Special features:**
- Uses only 95K observations for training (temporal split)
- Tests model stability across time periods (concept drift detection)
- Generates production-ready segments with forced splits
- Creates comprehensive validation report for regulatory review

**Command:**
```bash
python run_demo.py demo_lending_club.yaml
```

**Output highlights:**
- 5-7 risk segments (Low → High risk)
- Human-readable rules combining numeric and categorical conditions
- Excel summary for business users
- HTML dashboard with visual charts
- Full validation report for model validators

---

## Common Questions

### Q: Which demo should I run first?
**A:** Run `demo_german.yaml` first - it auto-downloads and runs in 5 seconds.

### Q: How do I download larger datasets?
**A:** See **DATASET_DOWNLOAD.md** for step-by-step instructions.

### Q: What if validation tests fail?
**A:** The framework will report which tests failed. Common reasons:
- Segments too similar (increase `min_default_rate_diff`)
- Too few defaults per segment (decrease `min_defaults_per_leaf`)
- Temporal instability (PSI > 0.10) - indicates population shift

### Q: Can I use my own data?
**A:** Yes! Create a new config YAML file and point to your dataset. See existing configs for examples.

### Q: What are forced splits?
**A:** Business rules that force the tree to split at specific thresholds. For example:
```yaml
forced_splits:
  credit_score: 680    # Always split at FICO = 680
  ltv: 80.0            # Always split at LTV = 80%
```

### Q: What are monotone constraints?
**A:** Rules ensuring risk increases/decreases in the expected direction:
```yaml
monotone_constraints:
  credit_score: 1      # Higher FICO = lower risk (PD decreases)
  interest_rate: -1    # Higher rate = higher risk (PD increases)
```

---

## Next Steps

1. **Run all three demos** to compare results across dataset sizes
2. **Read OUTPUT_GUIDE.md** to understand validation metrics
3. **Modify configs** to experiment with different parameters
4. **Try your own data** by creating a custom config file

---

## Technical Details

### Framework Architecture
- **Engine:** sklearn DecisionTreeClassifier with custom post-processing
- **Validation:** Chi-squared tests, PSI, binomial confidence intervals
- **Adjustments:** Automatic segment merging/splitting to meet constraints
- **Extensible:** Easy to swap sklearn for LightGBM or custom implementations

### Key Parameters

**Tree Structure:**
- `max_depth`: Maximum tree depth
- `min_samples_leaf`: Minimum observations per segment
- `min_samples_split`: Minimum observations to split

**IRB Requirements:**
- `min_defaults_per_leaf`: Minimum defaults per segment (Basel: ≥20)
- `min_default_rate_diff`: Minimum PD difference between segments
- `significance_level`: Statistical significance threshold (0.01 = 99% confidence)

**Segment Density:**
- `min_segment_density`: Minimum % of population per segment (0.10 = 10%)
- `max_segment_density`: Maximum % of population per segment (0.40 = 40%)

---

## Support

For questions or issues:
1. Check **OUTPUT_GUIDE.md** for result interpretation
2. Review config files in the demo folder
3. Examine output files for detailed error messages

---

## License

MIT License - Free for commercial and academic use.
