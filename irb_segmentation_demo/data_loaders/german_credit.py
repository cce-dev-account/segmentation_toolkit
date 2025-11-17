"""
German Credit Dataset Loader

Loads the UCI German Credit dataset with IRB-relevant preprocessing.

Dataset: Statlog (German Credit Data)
Source: UCI Machine Learning Repository
Size: 1,000 observations
Citation: Hofmann, H. (1994). DOI: 10.24432/C5NC77
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
from pathlib import Path
from .base import BaseDataLoader


class GermanCreditLoader(BaseDataLoader):
    """
    Loader for UCI German Credit dataset.

    This dataset classifies people as good or bad credit risks based on
    a set of personal and financial attributes.
    """

    def __init__(self, data_dir: str = "./data", random_state: int = 42):
        super().__init__(data_dir, random_state)
        self.url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
        self.filename = "german_credit.data"

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                            Optional[np.ndarray], Optional[np.ndarray], List[str], Optional[Dict[str, np.ndarray]]]:
        """
        Load and preprocess German Credit dataset.

        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_oot, y_oot, feature_names)
            Note: X_oot and y_oot are None as this dataset has no temporal component
        """
        print("\nLoading German Credit Dataset...")

        # Download if needed
        filepath = self._get_or_download()

        # Load data
        df = self._read_data(filepath)

        # Preprocess
        df = self._preprocess(df)

        # Extract features and target
        feature_names = [col for col in df.columns if col != 'default']
        X = df[feature_names].values
        y = df['default'].values

        # Create train/val split
        X_train, X_val, y_train, y_val = self.create_train_val_split(
            X, y, val_size=0.3, stratify=True
        )

        # No out-of-time data for this dataset
        X_oot = None
        y_oot = None

        # Print summary
        self.print_summary(X_train, y_train, X_val, y_val, X_oot, y_oot, feature_names)

        # Check IRB requirements
        self.check_irb_requirements(y_train, y_val, min_defaults=20)

        # No categorical features for German Credit (all numeric after preprocessing)
        X_categorical = None

        return X_train, y_train, X_val, y_val, X_oot, y_oot, feature_names, X_categorical

    def _get_or_download(self) -> Path:
        """Download dataset if not present."""
        filepath = self.data_dir / self.filename

        if not filepath.exists():
            print(f"Downloading from {self.url}...")
            try:
                import urllib.request
                urllib.request.urlretrieve(self.url, filepath)
                print(f"Downloaded to {filepath}")
            except Exception as e:
                print(f"Error downloading: {e}")
                print("Please manually download from:")
                print("https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)")
                raise
        else:
            print(f"Using existing file: {filepath}")

        return filepath

    def _read_data(self, filepath: Path) -> pd.DataFrame:
        """Read the raw data file."""
        # Define column names based on UCI documentation
        column_names = [
            'checking_account',      # A1: Status of existing checking account
            'duration_months',       # A2: Duration in months
            'credit_history',        # A3: Credit history
            'purpose',               # A4: Purpose
            'credit_amount',         # A5: Credit amount
            'savings_account',       # A6: Savings account/bonds
            'employment_since',      # A7: Present employment since
            'installment_rate',      # A8: Installment rate in percentage of disposable income
            'personal_status',       # A9: Personal status and sex
            'other_debtors',         # A10: Other debtors / guarantors
            'residence_since',       # A11: Present residence since
            'property',              # A12: Property
            'age',                   # A13: Age in years
            'other_installment',     # A14: Other installment plans
            'housing',               # A15: Housing
            'existing_credits',      # A16: Number of existing credits at this bank
            'job',                   # A17: Job
            'num_dependents',        # A18: Number of people being liable to provide maintenance for
            'telephone',             # A19: Telephone
            'foreign_worker',        # A20: foreign worker
            'credit_risk'            # Target: 1 = good, 2 = bad
        ]

        df = pd.read_csv(filepath, sep=' ', names=column_names, header=None)
        return df

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the dataset for IRB segmentation."""
        df = df.copy()

        # Convert target: 1 = good (0), 2 = bad (1)
        df['default'] = (df['credit_risk'] == 2).astype(int)
        df = df.drop('credit_risk', axis=1)

        # Create IRB-relevant features

        # 1. Credit score proxy: combination of checking account, credit history, and employment
        checking_map = {'A11': 4, 'A12': 3, 'A13': 2, 'A14': 1}
        df['checking_score'] = df['checking_account'].map(checking_map).fillna(0)

        credit_history_map = {
            'A30': 1,  # no credits taken/ all credits paid back duly
            'A31': 2,  # all credits at this bank paid back duly
            'A32': 4,  # existing credits paid back duly till now
            'A33': 3,  # delay in paying off in the past
            'A34': 0   # critical account/ other credits existing
        }
        df['credit_history_score'] = df['credit_history'].map(credit_history_map).fillna(2)

        employment_map = {'A71': 0, 'A72': 1, 'A73': 2, 'A74': 3, 'A75': 4}
        df['employment_score'] = df['employment_since'].map(employment_map).fillna(0)

        # Composite credit score (0-100 scale)
        df['credit_score'] = (
            (df['checking_score'] / 4) * 30 +
            (df['credit_history_score'] / 4) * 40 +
            (df['employment_score'] / 4) * 30
        ) * 100

        # 2. LTV proxy: credit amount relative to property value (simplified)
        property_map = {'A121': 4, 'A122': 3, 'A123': 2, 'A124': 1}
        df['property_value'] = df['property'].map(property_map).fillna(2)

        # Normalize credit amount
        max_credit = df['credit_amount'].max()
        normalized_credit = df['credit_amount'] / max_credit

        # LTV proxy (0-100 scale, higher is riskier)
        df['ltv_proxy'] = ((5 - df['property_value']) / 4 + normalized_credit) / 2 * 100

        # 3. DTI proxy: installment rate is already a percentage of disposable income
        df['dti_proxy'] = df['installment_rate'] * 10  # Scale to 0-40 range

        # 4. Loan amount (keep original)
        df['loan_amount'] = df['credit_amount']

        # 5. Age (keep original)
        # df['age'] already exists

        # 6. Years at current job proxy
        employment_years_map = {'A71': 0, 'A72': 1, 'A73': 2.5, 'A74': 5, 'A75': 10}
        df['years_employed'] = df['employment_since'].map(employment_years_map).fillna(0)

        # 7. Purpose as numeric (some purposes are riskier)
        purpose_risk_map = {
            'A40': 1,   # car (new)
            'A41': 2,   # car (used)
            'A42': 3,   # furniture/equipment
            'A43': 2,   # radio/television
            'A44': 4,   # domestic appliances
            'A45': 3,   # repairs
            'A46': 3,   # education
            'A47': 3,   # vacation
            'A48': 4,   # retraining
            'A49': 2,   # business
            'A410': 3   # others
        }
        df['purpose_risk'] = df['purpose'].map(purpose_risk_map).fillna(3)

        # Select final features for modeling
        final_features = [
            'credit_score',
            'ltv_proxy',
            'dti_proxy',
            'loan_amount',
            'age',
            'years_employed',
            'duration_months',
            'purpose_risk',
            'existing_credits',
            'num_dependents',
            'default'
        ]

        return df[final_features]

    def get_feature_descriptions(self) -> dict:
        """Get descriptions of engineered features."""
        return {
            'credit_score': 'Composite score (0-100) from checking account, credit history, and employment',
            'ltv_proxy': 'Loan-to-value proxy (0-100) based on credit amount and property value',
            'dti_proxy': 'Debt-to-income proxy based on installment rate',
            'loan_amount': 'Credit amount in original currency',
            'age': 'Age in years',
            'years_employed': 'Years at current employment',
            'duration_months': 'Loan duration in months',
            'purpose_risk': 'Purpose risk score (1=low, 4=high)',
            'existing_credits': 'Number of existing credits at bank',
            'num_dependents': 'Number of dependents'
        }


# Convenience function
def load_german_credit(data_dir: str = "./data", random_state: int = 42):
    """
    Convenience function to load German Credit dataset.

    Args:
        data_dir: Directory to store/load data
        random_state: Random seed

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_oot, y_oot, feature_names, X_categorical)
    """
    loader = GermanCreditLoader(data_dir, random_state)
    return loader.load()
