"""
Lending Club Loan Data Loader

Loads Lending Club P2P lending dataset with IRB-relevant preprocessing.

Dataset: Lending Club Loan Data
Source: Kaggle
Size: 2+ million loans (2007-2015+)
Features: 100+ including loan amount, grade, income, DTI, employment, etc.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
from pathlib import Path
from .base import BaseDataLoader
import warnings


class LendingClubLoader(BaseDataLoader):
    """
    Loader for Lending Club loan dataset.

    This dataset contains real P2P lending data with rich features
    ideal for IRB segmentation testing.
    """

    def __init__(self, data_dir: str = "./data", random_state: int = 42,
                 sample_size: Optional[int] = None):
        """
        Initialize Lending Club loader.

        Args:
            data_dir: Directory to store/load data
            random_state: Random seed
            sample_size: If provided, randomly sample this many rows for faster processing
        """
        super().__init__(data_dir, random_state)
        self.kaggle_url = "https://www.kaggle.com/datasets/wordsforthewise/lending-club"
        self.filename = "lending_club_data.csv"  # or "accepted_2007_to_2018Q4.csv"
        self.sample_size = sample_size

    def load(self, use_oot: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                            Optional[np.ndarray], Optional[np.ndarray], List[str], Optional[Dict[str, np.ndarray]]]:
        """
        Load and preprocess Lending Club dataset.

        Args:
            use_oot: If True, use temporal split for out-of-time validation

        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_oot, y_oot, feature_names, X_categorical)
        """
        print("\nLoading Lending Club Dataset...")

        # Check if file exists
        filepath = self._get_filepath()

        # Load data
        df = self._read_data(filepath)

        # Preprocess
        df = self._preprocess(df)

        # Create temporal split if requested
        if use_oot and 'issue_year' in df.columns:
            X_train, y_train, X_val, y_val, X_oot, y_oot, feature_names, X_categorical = self._temporal_split(df)
        else:
            # Standard random split
            feature_names = [col for col in df.columns if col != 'default']
            X = df[feature_names].values
            y = df['default'].values

            X_train, X_val, y_train, y_val = self.create_train_val_split(
                X, y, val_size=0.3, stratify=True
            )
            X_oot = None
            y_oot = None
            # No categorical features (all encoded during preprocessing)
            X_categorical = None

        # Print summary
        self.print_summary(X_train, y_train, X_val, y_val, X_oot, y_oot, feature_names)

        # Check IRB requirements
        self.check_irb_requirements(y_train, y_val, min_defaults=20)

        return X_train, y_train, X_val, y_val, X_oot, y_oot, feature_names, X_categorical

    def _get_filepath(self) -> Path:
        """Get filepath, checking multiple possible filenames."""
        possible_files = [
            "lending_club_data.csv",
            "lending_club_test.csv",
            "accepted_2007_to_2018Q4.csv",
            "accepted_2007_to_2018Q4.csv.gz",
            "loan.csv",
            "lending_club_loan_data.csv",
            "accepted_2007_to_2018.csv"
        ]

        for filename in possible_files:
            filepath = self.data_dir / filename
            if filepath.exists():
                print(f"Using file: {filepath}")
                return filepath

        # None found
        print(f"\nLending Club dataset not found in {self.data_dir}")
        print(f"Please download from Kaggle:")
        print(f"  {self.kaggle_url}")
        print(f"Expected filename: {possible_files[0]}")
        print("\nAlternatively, use the Kaggle API:")
        print("  kaggle datasets download -d wordsforthewise/lending-club")
        raise FileNotFoundError(f"Dataset file not found in {self.data_dir}")

    def _read_data(self, filepath: Path) -> pd.DataFrame:
        """Read the CSV file with appropriate handling for large files."""
        print(f"Reading {filepath.name}...")

        # Read with low_memory=False to handle mixed types
        if self.sample_size:
            print(f"Sampling {self.sample_size:,} rows for faster processing...")
            # Read header first
            df_header = pd.read_csv(filepath, nrows=0)
            # Calculate skiprows for random sampling
            total_rows = sum(1 for _ in open(filepath, encoding='utf-8')) - 1  # Exclude header
            skip_rows = sorted(np.random.choice(range(1, total_rows + 1),
                                               size=total_rows - self.sample_size,
                                               replace=False))
            df = pd.read_csv(filepath, skiprows=skip_rows, low_memory=False)
        else:
            df = pd.read_csv(filepath, low_memory=False)

        print(f"Loaded {len(df):,} rows")
        return df

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the dataset for IRB segmentation."""
        df = df.copy()

        print("Preprocessing...")

        # Create binary default indicator
        # loan_status values: 'Fully Paid', 'Charged Off', 'Default', 'Current', etc.
        if 'loan_status' in df.columns:
            default_statuses = ['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off']
            df['default'] = df['loan_status'].isin(default_statuses).astype(int)

            # Filter to completed loans only (exclude current/in grace period)
            completed_statuses = ['Fully Paid', 'Charged Off', 'Default',
                                'Does not meet the credit policy. Status:Charged Off',
                                'Does not meet the credit policy. Status:Fully Paid']
            df = df[df['loan_status'].isin(completed_statuses)]
        else:
            raise ValueError("loan_status column not found")

        # Parse issue date for temporal splitting
        if 'issue_d' in df.columns:
            df['issue_date'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')
            df['issue_year'] = df['issue_date'].dt.year
            df['issue_month'] = df['issue_date'].dt.month

        # 1. Credit Score (FICO)
        if 'fico_range_low' in df.columns and 'fico_range_high' in df.columns:
            df['credit_score'] = (df['fico_range_low'] + df['fico_range_high']) / 2
        elif 'fico_range_low' in df.columns:
            df['credit_score'] = df['fico_range_low'] + 5  # Estimate midpoint
        else:
            df['credit_score'] = 700  # Default

        # 2. DTI (Debt-to-Income)
        if 'dti' in df.columns:
            df['dti'] = df['dti'].fillna(df['dti'].median())
        else:
            df['dti'] = 20

        # 3. Loan Amount
        if 'loan_amnt' in df.columns:
            df['loan_amount'] = df['loan_amnt']
        elif 'funded_amnt' in df.columns:
            df['loan_amount'] = df['funded_amnt']
        else:
            df['loan_amount'] = 10000

        # 4. Interest Rate
        if 'int_rate' in df.columns:
            # Remove % sign and convert to float
            df['interest_rate'] = df['int_rate'].astype(str).str.rstrip('%').astype(float)
        else:
            df['interest_rate'] = 12.0

        # 5. Annual Income
        if 'annual_inc' in df.columns:
            df['annual_income'] = df['annual_inc'].fillna(df['annual_inc'].median())
            # Calculate LTV proxy: loan amount / annual income
            df['ltv_proxy'] = (df['loan_amount'] / df['annual_income'] * 100).clip(0, 200)
        else:
            df['annual_income'] = 50000
            df['ltv_proxy'] = 20

        # 6. Employment Length
        if 'emp_length' in df.columns:
            emp_map = {
                '< 1 year': 0.5,
                '1 year': 1,
                '2 years': 2,
                '3 years': 3,
                '4 years': 4,
                '5 years': 5,
                '6 years': 6,
                '7 years': 7,
                '8 years': 8,
                '9 years': 9,
                '10+ years': 10,
                'n/a': 0
            }
            df['emp_length_years'] = df['emp_length'].map(emp_map).fillna(0)
        else:
            df['emp_length_years'] = 2

        # 7. Loan Grade (A-G)
        if 'grade' in df.columns:
            grade_map = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}
            df['grade_numeric'] = df['grade'].map(grade_map).fillna(4)
        else:
            df['grade_numeric'] = 4

        # 8. Home Ownership
        if 'home_ownership' in df.columns:
            home_map = {'OWN': 3, 'MORTGAGE': 2, 'RENT': 1, 'OTHER': 0, 'NONE': 0, 'ANY': 1}
            df['home_ownership_score'] = df['home_ownership'].map(home_map).fillna(1)
        else:
            df['home_ownership_score'] = 1

        # 9. Loan Term
        if 'term' in df.columns:
            df['term_months'] = df['term'].astype(str).str.extract(r'(\d+)').astype(float).fillna(36)
        else:
            df['term_months'] = 36

        # 10. Loan Purpose
        if 'purpose' in df.columns:
            purpose_risk = {
                'credit_card': 2,
                'debt_consolidation': 2,
                'home_improvement': 2,
                'house': 1,
                'major_purchase': 3,
                'medical': 4,
                'moving': 3,
                'other': 3,
                'renewable_energy': 2,
                'small_business': 5,
                'vacation': 4,
                'wedding': 3,
                'car': 3,
                'educational': 2
            }
            df['purpose_risk'] = df['purpose'].map(purpose_risk).fillna(3)
        else:
            df['purpose_risk'] = 3

        # 11. Delinquencies in last 2 years
        if 'delinq_2yrs' in df.columns:
            df['delinquencies'] = df['delinq_2yrs'].fillna(0)
        else:
            df['delinquencies'] = 0

        # 12. Inquiries in last 6 months
        if 'inq_last_6mths' in df.columns:
            df['inquiries'] = df['inq_last_6mths'].fillna(0)
        else:
            df['inquiries'] = 0

        # 13. Revolving utilization
        if 'revol_util' in df.columns:
            df['revolving_util'] = df['revol_util'].astype(str).str.rstrip('%').astype(float).fillna(50)
        else:
            df['revolving_util'] = 50

        # Select final features
        final_features = [
            'credit_score',
            'dti',
            'ltv_proxy',
            'loan_amount',
            'interest_rate',
            'annual_income',
            'emp_length_years',
            'grade_numeric',
            'home_ownership_score',
            'term_months',
            'purpose_risk',
            'delinquencies',
            'inquiries',
            'revolving_util',
            'default'
        ]

        # Add temporal features if available
        if 'issue_year' in df.columns:
            final_features.insert(0, 'issue_year')
        if 'issue_month' in df.columns:
            final_features.insert(1, 'issue_month')

        # Ensure all features exist
        for feat in final_features:
            if feat not in df.columns:
                warnings.warn(f"Feature {feat} not found, using default value")
                df[feat] = 0

        # Drop rows with missing target
        df = df.dropna(subset=['default'])

        # Drop rows with too many missing features
        feature_cols = [f for f in final_features if f != 'default']
        df = df.dropna(subset=feature_cols, thresh=len(feature_cols) - 2)

        print(f"After preprocessing: {len(df):,} rows")

        return df[final_features]

    def _temporal_split(self, df: pd.DataFrame) -> Tuple:
        """Create temporal train/val/OOT split."""
        if 'issue_year' not in df.columns:
            raise ValueError("issue_year not available for temporal split")

        # Sort by date
        df = df.sort_values('issue_year')

        # Use 2007-2012 for training, 2013-2014 for validation, 2015+ for OOT
        train_df = df[df['issue_year'] <= 2012]
        val_df = df[(df['issue_year'] >= 2013) & (df['issue_year'] <= 2014)]
        oot_df = df[df['issue_year'] >= 2015]

        print(f"\nTemporal split:")
        print(f"  Training: 2007-2012 ({len(train_df):,} loans)")
        print(f"  Validation: 2013-2014 ({len(val_df):,} loans)")
        print(f"  Out-of-Time: 2015+ ({len(oot_df):,} loans)")

        # Extract features (exclude temporal columns from features)
        feature_names = [col for col in df.columns
                        if col not in ['default', 'issue_year', 'issue_month']]

        X_train = train_df[feature_names].values
        y_train = train_df['default'].values

        X_val = val_df[feature_names].values
        y_val = val_df['default'].values

        X_oot = oot_df[feature_names].values if len(oot_df) > 0 else None
        y_oot = oot_df['default'].values if len(oot_df) > 0 else None

        # No categorical features in temporal split (all encoded)
        X_categorical = None

        return X_train, y_train, X_val, y_val, X_oot, y_oot, feature_names, X_categorical

    def get_feature_descriptions(self) -> dict:
        """Get descriptions of features."""
        return {
            'credit_score': 'FICO credit score (300-850)',
            'dti': 'Debt-to-income ratio (%)',
            'ltv_proxy': 'Loan-to-income ratio as LTV proxy (%)',
            'loan_amount': 'Loan amount ($)',
            'interest_rate': 'Interest rate (%)',
            'annual_income': 'Annual income ($)',
            'emp_length_years': 'Years at current employment',
            'grade_numeric': 'Lending Club grade (1=G to 7=A)',
            'home_ownership_score': 'Home ownership score (0-3)',
            'term_months': 'Loan term in months',
            'purpose_risk': 'Loan purpose risk score (1-5)',
            'delinquencies': 'Number of delinquencies in last 2 years',
            'inquiries': 'Number of credit inquiries in last 6 months',
            'revolving_util': 'Revolving credit utilization (%)'
        }


# Convenience function
def load_lending_club(data_dir: str = "./data", random_state: int = 42,
                     sample_size: Optional[int] = None, use_oot: bool = True):
    """
    Convenience function to load Lending Club dataset.

    Args:
        data_dir: Directory to store/load data
        random_state: Random seed
        sample_size: If provided, randomly sample this many rows
        use_oot: Use temporal out-of-time split

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_oot, y_oot, feature_names, X_categorical)
    """
    loader = LendingClubLoader(data_dir, random_state, sample_size)
    return loader.load(use_oot)
