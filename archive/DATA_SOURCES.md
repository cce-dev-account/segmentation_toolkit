# Data Sources for IRB Segmentation Testing

This document describes the public credit datasets available for testing the IRB PD segmentation framework.

## Quick Start

```python
# Load a dataset
from data_loaders import load_german_credit

X_train, y_train, X_val, y_val, X_oot, y_oot, feature_names = load_german_credit()

# Use with segmentation engine
from irb_segmentation import IRBSegmentationParams, IRBSegmentationEngine

params = IRBSegmentationParams(max_depth=3, min_defaults_per_leaf=20)
engine = IRBSegmentationEngine(params)
engine.fit(X_train, y_train, X_val, y_val, feature_names=feature_names)
```

---

## 1. German Credit Dataset (UCI) ⭐ RECOMMENDED FOR QUICK TESTING

### Overview
- **Size**: 1,000 loan applications
- **Features**: 20 attributes (credit history, employment, loan amount, purpose, demographics)
- **Target**: Binary credit risk classification (good/bad)
- **Default Rate**: ~30%
- **Time Period**: 1994
- **Country**: Germany

### Source & Download
- **Primary**: UCI Machine Learning Repository
  - URL: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
  - Direct download: https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data
- **Alternative**: Kaggle
  - URL: https://www.kaggle.com/datasets/uciml/german-credit

### Citation
```
Hofmann, H. (1994). Statlog (German Credit Data) [Dataset].
UCI Machine Learning Repository.
https://doi.org/10.24432/C5NC77
```

### License
Creative Commons Attribution 4.0 International (CC BY 4.0)

### Features (Engineered for IRB)
| Feature | Description | Range |
|---------|-------------|-------|
| credit_score | Composite score from checking account, credit history, employment | 0-100 |
| ltv_proxy | Loan-to-value proxy based on credit amount and property | 0-100 |
| dti_proxy | Debt-to-income proxy from installment rate | 0-40 |
| loan_amount | Credit amount | Currency units |
| age | Age in years | 18-75 |
| years_employed | Years at current employment | 0-10+ |
| duration_months | Loan duration | 4-72 |
| purpose_risk | Purpose risk score | 1-4 |
| existing_credits | Number of existing credits | 1-4 |
| num_dependents | Number of dependents | 1-2 |

### Usage
```python
from data_loaders import GermanCreditLoader

loader = GermanCreditLoader(data_dir="./data")
X_train, y_train, X_val, y_val, _, _, feature_names = loader.load()
```

### Notes
- ✓ Automatically downloads from UCI
- ✓ Perfect for unit testing and quick validation
- ✓ Small enough to run locally without GPU
- ✗ No temporal component for OOT testing
- ✗ Limited sample size may require relaxed IRB constraints

---

## 2. Taiwan Credit Card Default Dataset (UCI)

### Overview
- **Size**: 30,000 credit card clients
- **Features**: 24 attributes (credit limit, payment history, bill amounts, demographics)
- **Target**: Binary default indicator (next month)
- **Default Rate**: ~22%
- **Time Period**: April-September 2005
- **Country**: Taiwan

### Source & Download
- **Primary**: Kaggle
  - URL: https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset
  - **Manual download required** (Kaggle account needed)
- **Alternative**: UCI ML Repository
  - URL: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

### Citation
```
Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques
for the predictive accuracy of probability of default of credit card clients.
Expert Systems with Applications, 36(2), 2473-2480.
```

### License
CC BY 4.0

### Features (Engineered for IRB)
| Feature | Description | Range |
|---------|-------------|-------|
| credit_score | Score based on payment history (0-100, higher=better) | 0-100 |
| utilization_rate | Average credit utilization | 0-200% |
| max_utilization | Maximum utilization across months | 0-200% |
| payment_ratio | Payment-to-bill ratio | 0-200% |
| credit_limit | Credit card limit | Currency |
| age | Age in years | 21-79 |
| education_score | Education level | 1-4 |
| avg_payment_delay | Average payment delay (months) | -2 to 8 |
| max_payment_delay | Maximum payment delay | -2 to 8 |
| payment_trend | Recent vs historical payment behavior | -10 to 10 |

### Download Instructions
```bash
# Option 1: Kaggle API (recommended)
pip install kaggle
kaggle datasets download -d uciml/default-of-credit-card-clients-dataset
unzip default-of-credit-card-clients-dataset.zip -d ./data/

# Option 2: Manual download
# 1. Go to https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset
# 2. Click "Download"
# 3. Save to ./data/UCI_Credit_Card.csv
```

### Usage
```python
from data_loaders import TaiwanCreditLoader

loader = TaiwanCreditLoader(data_dir="./data")
X_train, y_train, X_val, y_val, _, _, feature_names = loader.load()
```

### Notes
- ⚠ Requires manual download from Kaggle
- ✓ Good size for testing IRB requirements (>20 defaults per segment)
- ✓ Rich payment history features
- ✗ No temporal split for OOT validation
- ✓ Real credit card data from financial institution

---

## 3. Lending Club Loan Data

### Overview
- **Size**: 2+ million loans (can sample for speed)
- **Features**: 100+ attributes (loan details, grades, income, DTI, employment, credit history)
- **Target**: Loan status (Fully Paid, Charged Off, Default, etc.)
- **Default Rate**: ~15-20%
- **Time Period**: 2007-2018
- **Country**: United States

### Source & Download
- **Primary**: Kaggle
  - URL: https://www.kaggle.com/datasets/wordsforthewise/lending-club
  - **Manual download required** (~2GB compressed)

### Citation
```
Lending Club Loan Data. (2018).
Retrieved from Kaggle: https://www.kaggle.com/datasets/wordsforthewise/lending-club
```

### Features (Engineered for IRB)
| Feature | Description | Range |
|---------|-------------|-------|
| credit_score | FICO credit score (midpoint) | 300-850 |
| dti | Debt-to-income ratio | 0-60% |
| ltv_proxy | Loan-to-income ratio | 0-200% |
| loan_amount | Loan amount | $500-$40,000 |
| interest_rate | Interest rate | 5-30% |
| annual_income | Annual income | $0-$10M+ |
| emp_length_years | Years at current employment | 0-10+ |
| grade_numeric | Lending Club grade (1=G to 7=A) | 1-7 |
| home_ownership_score | Home ownership status | 0-3 |
| term_months | Loan term | 36 or 60 |
| purpose_risk | Loan purpose risk score | 1-5 |
| delinquencies | Delinquencies in last 2 years | 0-30+ |
| inquiries | Credit inquiries in last 6 months | 0-10+ |
| revolving_util | Revolving credit utilization | 0-200% |

### Download Instructions
```bash
# Option 1: Kaggle API
kaggle datasets download -d wordsforthewise/lending-club
unzip lending-club.zip -d ./data/

# Option 2: Manual download
# 1. Go to https://www.kaggle.com/datasets/wordsforthewise/lending-club
# 2. Download "accepted_2007_to_2018Q4.csv" (or latest version)
# 3. Save to ./data/
```

### Usage
```python
from data_loaders import LendingClubLoader

# Use sampling for faster testing
loader = LendingClubLoader(data_dir="./data", sample_size=50000)

# With temporal OOT split (2007-2012 train, 2013-2014 val, 2015+ OOT)
X_train, y_train, X_val, y_val, X_oot, y_oot, feature_names = loader.load(use_oot=True)
```

### Notes
- ⚠ Large file size (~2GB compressed, ~10GB uncompressed)
- ✓ **Temporal splits available** for OOT validation
- ✓ Real P2P lending data with rich feature set
- ✓ Ideal for testing forced splits (loan grades, DTI thresholds)
- ✓ Can test monotonicity constraints (FICO, interest rate)
- ℹ Use `sample_size` parameter for faster testing

---

## 4. Home Credit Default Risk (Kaggle Competition)

### Overview
- **Size**: 307,511 applications (train) with auxiliary tables
- **Features**: 100+ main features, additional features from auxiliary tables
- **Target**: Binary default probability
- **Default Rate**: ~8%
- **Time Period**: Historical applications
- **Country**: Multiple (international)

### Source & Download
- **Primary**: Kaggle Competition
  - URL: https://www.kaggle.com/c/home-credit-default-risk/data
  - **Requires Kaggle account and competition rules acceptance**

### Files Structure
```
home-credit-default-risk/
├── application_train.csv       # Main training data (307,511 rows)
├── application_test.csv        # Test set (48,744 rows, no labels)
├── bureau.csv                  # Credit bureau data
├── bureau_balance.csv          # Monthly bureau balance
├── previous_application.csv    # Previous applications
├── POS_CASH_balance.csv        # Point-of-sale/cash loans
├── credit_card_balance.csv     # Credit card balance
├── installments_payments.csv   # Payment history
└── HomeCredit_columns_description.csv
```

### Features (Engineered for IRB)
| Feature | Description | Range |
|---------|-------------|-------|
| credit_score | Composite from external sources | 0-100 |
| loan_amount | Credit amount | Currency |
| annual_income | Total annual income | Currency |
| ltv_proxy | Loan-to-income ratio | 0-200% |
| dti_proxy | Annuity-to-income ratio | 0-100% |
| age | Age in years | 20-70 |
| years_employed | Years employed | 0-40 |
| family_size | Number of family members | 1-20 |
| owns_car | Car ownership flag | 0/1 |
| owns_property | Property ownership flag | 0/1 |
| region_rating | Region rating | 1-3 |
| ext_source_1/2/3 | External data source scores | 0-1 |

### Download Instructions
```bash
# Kaggle API
kaggle competitions download -c home-credit-default-risk
unzip home-credit-default-risk.zip -d ./data/

# Manual download
# 1. Go to https://www.kaggle.com/c/home-credit-default-risk/data
# 2. Accept competition rules
# 3. Download all CSV files
# 4. Extract to ./data/
```

### Usage
```python
from data_loaders import HomeCreditLoader

# Basic: Use main application data only
loader = HomeCreditLoader(data_dir="./data", use_auxiliary=False)
X_train, y_train, X_val, y_val, X_oot, y_oot, feature_names = loader.load()

# Advanced: Merge auxiliary tables (slower, more features)
loader = HomeCreditLoader(data_dir="./data", use_auxiliary=True)
X_train, y_train, X_val, y_val, X_oot, y_oot, feature_names = loader.load()
```

### Notes
- ⚠ Requires Kaggle account and competition acceptance
- ⚠ Large dataset (300K+ rows)
- ✓ Most realistic for production IRB model testing
- ✓ Complex relational structure with auxiliary tables
- ✓ Test set available (no labels) for OOT validation
- ℹ Set `use_auxiliary=False` for faster loading
- ⚠ Low default rate (~8%) may require more training data per segment

---

## Dataset Comparison

| Dataset | Size | Features | Default Rate | OOT Split | Download | Complexity |
|---------|------|----------|--------------|-----------|----------|------------|
| **German Credit** | 1,000 | 10 | ~30% | No | Auto | ⭐ Easy |
| **Taiwan Credit** | 30,000 | 12 | ~22% | No | Manual | ⭐⭐ Medium |
| **Lending Club** | 2M+ | 14 | ~15-20% | Yes | Manual | ⭐⭐⭐ Hard |
| **Home Credit** | 300K+ | 14+ | ~8% | No | Manual | ⭐⭐⭐⭐ Very Hard |

## Recommended Testing Sequence

### Phase 1: Development & Unit Testing
**Dataset**: German Credit (1K rows)
- Fast iteration
- Validates core functionality
- Tests edge cases
- ✓ Auto-downloads

### Phase 2: IRB Requirements Validation
**Dataset**: Taiwan Credit (30K rows)
- Sufficient defaults per segment
- Tests statistical significance
- Validates density constraints
- Tests binomial confidence

### Phase 3: Production Simulation
**Dataset**: Lending Club (50K-500K sampled)
- Temporal OOT validation
- Tests PSI calculation
- Realistic feature engineering
- Business constraint testing

### Phase 4: Large-Scale Deployment
**Dataset**: Home Credit (300K+ rows)
- Scalability testing
- Complex feature engineering
- Low default rate scenarios
- Production-level validation

---

## Common Issues & Solutions

### Issue: "Dataset file not found"
**Solution**: Follow download instructions for each dataset. Some require manual download from Kaggle.

### Issue: Kaggle API authentication error
**Solution**:
```bash
# Setup Kaggle API credentials
# 1. Go to https://www.kaggle.com/account
# 2. Create API token
# 3. Save kaggle.json to ~/.kaggle/ (Linux/Mac) or C:\Users\<you>\.kaggle\ (Windows)
```

### Issue: Memory error with large datasets
**Solution**:
```python
# Use sampling for Lending Club
loader = LendingClubLoader(sample_size=10000)

# Or process in chunks
# See data loader documentation
```

### Issue: Insufficient defaults per segment
**Solution**: Lower `min_defaults_per_leaf` parameter or use larger dataset

---

## Adding New Datasets

To add a new dataset loader:

1. Create new loader class in `data_loaders/`
2. Inherit from `BaseDataLoader`
3. Implement `load()` method
4. Engineer IRB-relevant features
5. Add tests in `test_with_real_data.py`
6. Document in this file

Example:
```python
from data_loaders.base import BaseDataLoader

class MyDatasetLoader(BaseDataLoader):
    def load(self):
        # 1. Download/read data
        # 2. Preprocess & engineer features
        # 3. Create train/val/OOT splits
        # 4. Return standard tuple
        return X_train, y_train, X_val, y_val, X_oot, y_oot, feature_names
```

---

## License & Citation

When using these datasets, please cite the original sources as indicated in each dataset section. All datasets are provided under their respective licenses for research and educational purposes.
