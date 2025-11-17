# Lending Club Data Split Strategies

## Two Approaches for Training IRB Segmentation Models

### Option 1: Temporal Split (Recommended for Production)
**Config:** `lending_club_categorical.yaml` (`use_oot: true`)

#### Data Split:
- **Training:** 2007-2012 → 95,902 observations (7%)
- **Validation:** 2013-2014 → 357,907 observations (27%)
- **Out-of-Time (OOT):** 2015+ → 894,290 observations (66%)

#### Advantages ✅:
- **Temporal validation** - Tests model stability across time periods
- **Detects concept drift** - Identifies if credit risk patterns changed
- **Regulatory friendly** - Mimics real-world deployment (train on past, predict future)
- **More realistic** - 2007-2012 excludes financial crisis recovery period
- **Best practice** for credit risk - Standard approach in banking

#### Disadvantages ❌:
- **Less training data** - Only 95k observations (limited statistical power)
- **May miss recent patterns** - Excludes 2013-2018 trends
- **Possible population shift** - Economic conditions changed significantly

#### Best For:
- Production deployments
- Regulatory submissions (Basel II/III)
- When temporal stability matters
- Conservative approach

---

### Option 2: Full Dataset Random Split (Maximum Power)
**Config:** `lending_club_full_data.yaml` (`use_oot: false`)

#### Data Split:
- **Training:** Random 70% → ~944,000 observations
- **Validation:** Random 30% → ~404,000 observations
- **Out-of-Time:** None

#### Advantages ✅:
- **Maximum statistical power** - 10x more training data
- **More stable segments** - Better estimates of PD rates
- **Captures recent trends** - Includes 2007-2018 data
- **Better coverage** - All time periods represented
- **Stronger validation** - Larger validation set (404k vs 358k)

#### Disadvantages ❌:
- **No temporal validation** - Can't assess model stability over time
- **Overfitting risk** - Model may learn time-specific patterns
- **Not regulatory standard** - May not meet Basel guidelines
- **Mixing time periods** - Combines pre/post-crisis data

#### Best For:
- Academic research
- Initial model development
- When maximum accuracy is priority
- Data exploration and feature engineering

---

## Comparison Table

| Metric | Temporal Split | Full Data Split |
|--------|---------------|-----------------|
| **Training Size** | 95,902 (7%) | ~944,000 (70%) |
| **Validation Size** | 357,907 (27%) | ~404,000 (30%) |
| **OOT Test Size** | 894,290 (66%) | None |
| **Statistical Power** | Low | High |
| **Temporal Validation** | ✅ Yes | ❌ No |
| **Regulatory Compliance** | ✅ High | ⚠️ Medium |
| **Segment Stability** | Lower (fewer obs) | Higher (more obs) |
| **Concept Drift Detection** | ✅ Yes | ❌ No |

---

## Recommendation

### Use Temporal Split When:
- Building production models for regulatory approval
- Need to demonstrate model stability over time
- Following Basel II/III guidelines
- Deploying in banking/financial services

### Use Full Data When:
- Developing initial prototypes
- Maximum accuracy is critical
- Working with academic research
- Exploring feature engineering options

### Best Practice: Run Both!
Compare results from both approaches:
1. Temporal split shows if model is stable over time
2. Full data shows maximum achievable performance
3. Large gap suggests concept drift issues

---

## How to Run Both

### Temporal Split (Current):
```bash
python run_segmentation.py config_examples/lending_club_categorical.yaml
```
**Output:** `./output/lending_club_categorical/`

### Full Data:
```bash
python run_segmentation.py config_examples/lending_club_full_data.yaml
```
**Output:** `./output/lending_club_full_data/`

Then compare:
- Segment rules similarity
- Default rate estimates
- Validation metrics (chi-squared, PSI)
- Segment density distributions

---

## Expected Results

### Temporal Split (95k training):
- **5-7 segments** (conservative due to less data)
- **Higher uncertainty** in PD estimates (wider confidence intervals)
- **May fail PSI test** (population shift 2007-2012 → 2013-2014)
- **Training validation passes**, OOT may show drift

### Full Data (944k training):
- **6-10 segments** (more granular segmentation possible)
- **Tighter confidence intervals** (more precise PD estimates)
- **Better statistical significance** (chi-squared tests pass easily)
- **Higher segment densities** (more observations per segment)

---

## Notes on the Data

**Original:** 2,260,701 rows
**After Preprocessing:** 1,348,099 rows (60% retention)

**Preprocessing removes:**
- Missing values in key fields
- Invalid/inconsistent data
- Loans without default status
- Outliers and data quality issues

**Time Range:** 2007-2018 (covers pre-crisis, crisis, and recovery periods)
