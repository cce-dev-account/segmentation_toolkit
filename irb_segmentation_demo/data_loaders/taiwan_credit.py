"""
Taiwan Credit Card Default Dataset Loader

Loads the UCI Taiwan Credit Card dataset with IRB-relevant preprocessing.

Dataset: Default of Credit Card Clients Dataset
Source: UCI Machine Learning Repository / Kaggle
Size: 30,000 observations
Features: Payment history, bill amounts, payment amounts, demographics
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
from pathlib import Path
from .base import BaseDataLoader


class TaiwanCreditLoader(BaseDataLoader):
    """
    Loader for Taiwan Credit Card Default dataset.

    This dataset contains credit card client data from Taiwan (2005),
    including payment history and default status.
    """

    def __init__(self, data_dir: str = "./data", random_state: int = 42):
        super().__init__(data_dir, random_state)
        # Note: Direct UCI link may not work, user may need to download from Kaggle
        self.kaggle_url = "https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset"
        self.filename = "UCI_Credit_Card.csv"

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                            Optional[np.ndarray], Optional[np.ndarray], List[str], Optional[Dict[str, np.ndarray]]]:
        """
        Load and preprocess Taiwan Credit Card dataset.

        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_oot, y_oot, feature_names, X_categorical)
            Note: X_oot and y_oot are None as this dataset has no temporal component
        """
        print("\nLoading Taiwan Credit Card Dataset...")

        # Check if file exists
        filepath = self.data_dir / self.filename
        if not filepath.exists():
            print(f"\nFile not found: {filepath}")
            print(f"Please download the dataset from Kaggle:")
            print(f"  {self.kaggle_url}")
            print(f"And save it to: {filepath}")
            print("\nAlternatively, use the Kaggle API:")
            print("  kaggle datasets download -d uciml/default-of-credit-card-clients-dataset")
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

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

        # No categorical features (all encoded during preprocessing)
        X_categorical = None

        return X_train, y_train, X_val, y_val, X_oot, y_oot, feature_names, X_categorical

    def _read_data(self, filepath: Path) -> pd.DataFrame:
        """Read the CSV file."""
        # Try different possible formats
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            # Try with different encoding
            df = pd.read_csv(filepath, encoding='latin-1')

        # Handle different possible column name formats
        # Original dataset has 'ID' and 'default.payment.next.month' or similar
        if 'ID' in df.columns:
            df = df.drop('ID', axis=1)

        # Standardize target column name
        target_cols = ['default.payment.next.month', 'default payment next month',
                      'default_payment_next_month', 'Y']
        for col in target_cols:
            if col in df.columns:
                df = df.rename(columns={col: 'default'})
                break

        return df

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the dataset for IRB segmentation."""
        df = df.copy()

        # If 'default' not yet created, look for target variable
        if 'default' not in df.columns:
            # Try to find the target column
            possible_targets = [col for col in df.columns if 'default' in col.lower() or col == 'Y']
            if possible_targets:
                df = df.rename(columns={possible_targets[0]: 'default'})
            else:
                raise ValueError("Cannot find target column")

        # Ensure binary target
        df['default'] = df['default'].astype(int)

        # Create IRB-relevant features

        # 1. Credit Score Proxy: Based on payment history
        # PAY_0 to PAY_6 are payment status (-1 = pay duly, 1 = delay 1 month, etc.)
        pay_cols = [col for col in df.columns if col.startswith('PAY_')]
        if pay_cols:
            # Calculate average payment delay (lower is better)
            df['avg_payment_delay'] = df[pay_cols].mean(axis=1)
            df['max_payment_delay'] = df[pay_cols].max(axis=1)

            # Credit score proxy (0-100, higher is better)
            # Convert payment delay to score
            max_delay = df['avg_payment_delay'].max()
            df['credit_score'] = 100 - ((df['avg_payment_delay'] + 1) / (max_delay + 1)) * 100
            df['credit_score'] = df['credit_score'].clip(0, 100)
        else:
            # Fallback if column names are different
            df['credit_score'] = 70  # Neutral score

        # 2. Utilization Rate (proxy for LTV/DTI)
        # Calculate credit utilization across all months
        bill_cols = [col for col in df.columns if 'BILL_AMT' in col]
        limit_col = [col for col in df.columns if 'LIMIT_BAL' in col or col == 'LIMIT']

        if bill_cols and limit_col:
            limit = df[limit_col[0]]
            # Average bill amount
            df['avg_bill'] = df[bill_cols].mean(axis=1)

            # Utilization rate (0-100+)
            df['utilization_rate'] = (df['avg_bill'] / (limit + 1)) * 100
            df['utilization_rate'] = df['utilization_rate'].clip(0, 200)  # Cap at 200%

            # Max utilization
            df['max_utilization'] = (df[bill_cols].max(axis=1) / (limit + 1)) * 100
            df['max_utilization'] = df['max_utilization'].clip(0, 200)
        else:
            df['utilization_rate'] = 50
            df['max_utilization'] = 50

        # 3. Payment-to-Bill Ratio (DTI proxy)
        pay_amt_cols = [col for col in df.columns if 'PAY_AMT' in col]

        if bill_cols and pay_amt_cols:
            df['avg_payment'] = df[pay_amt_cols].mean(axis=1)
            # Payment ratio (how much they pay vs. owe)
            df['payment_ratio'] = (df['avg_payment'] / (df['avg_bill'] + 1)) * 100
            df['payment_ratio'] = df['payment_ratio'].clip(0, 200)
        else:
            df['payment_ratio'] = 50

        # 4. Credit Limit (loan amount proxy)
        if limit_col:
            df['credit_limit'] = df[limit_col[0]]
        else:
            df['credit_limit'] = df['avg_bill'] * 2  # Estimate

        # 5. Demographics
        # Age
        age_cols = [col for col in df.columns if col in ['AGE', 'age']]
        if age_cols:
            df['age'] = df[age_cols[0]]
        else:
            df['age'] = 35  # Default

        # Education (convert to numeric if needed)
        edu_cols = [col for col in df.columns if 'EDUCATION' in col or col == 'education']
        if edu_cols:
            df['education'] = df[edu_cols[0]]
            # Higher education = higher score (1=grad, 2=university, 3=high school, 4=others)
            df['education_score'] = 5 - df['education'].clip(1, 4)
        else:
            df['education_score'] = 2

        # Marriage status
        marriage_cols = [col for col in df.columns if 'MARRIAGE' in col or col == 'marriage']
        if marriage_cols:
            df['marriage'] = df[marriage_cols[0]]
        else:
            df['marriage'] = 1

        # 6. Recent vs Historical behavior
        if len(pay_cols) >= 3:
            # Recent payment behavior (last 3 months)
            df['recent_payment_delay'] = df[pay_cols[:3]].mean(axis=1)

            # Historical payment behavior (months 4-6)
            if len(pay_cols) >= 6:
                df['historical_payment_delay'] = df[pay_cols[3:6]].mean(axis=1)
                df['payment_trend'] = df['recent_payment_delay'] - df['historical_payment_delay']
            else:
                df['payment_trend'] = 0
        else:
            df['recent_payment_delay'] = df.get('avg_payment_delay', 0)
            df['payment_trend'] = 0

        # Select final features for modeling
        final_features = [
            'credit_score',
            'utilization_rate',
            'max_utilization',
            'payment_ratio',
            'credit_limit',
            'age',
            'education_score',
            'marriage',
            'avg_payment_delay',
            'max_payment_delay',
            'recent_payment_delay',
            'payment_trend',
            'default'
        ]

        # Ensure all features exist
        for feat in final_features:
            if feat not in df.columns:
                df[feat] = 0

        return df[final_features]

    def get_feature_descriptions(self) -> dict:
        """Get descriptions of engineered features."""
        return {
            'credit_score': 'Score (0-100) based on payment history, higher is better',
            'utilization_rate': 'Average credit utilization rate (%)',
            'max_utilization': 'Maximum credit utilization rate across months (%)',
            'payment_ratio': 'Payment-to-bill ratio (%)',
            'credit_limit': 'Credit card limit amount',
            'age': 'Age in years',
            'education_score': 'Education level score (higher = more education)',
            'marriage': 'Marital status (1=married, 2=single, 3=other)',
            'avg_payment_delay': 'Average payment delay in months',
            'max_payment_delay': 'Maximum payment delay in months',
            'recent_payment_delay': 'Recent (3-month) average payment delay',
            'payment_trend': 'Trend in payment behavior (recent - historical)'
        }


# Convenience function
def load_taiwan_credit(data_dir: str = "./data", random_state: int = 42):
    """
    Convenience function to load Taiwan Credit Card dataset.

    Args:
        data_dir: Directory to store/load data
        random_state: Random seed

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_oot, y_oot, feature_names, X_categorical)
    """
    loader = TaiwanCreditLoader(data_dir, random_state)
    return loader.load()
