"""
Home Credit Default Risk Dataset Loader

Loads the Home Credit dataset from Kaggle competition with IRB-relevant preprocessing.

Dataset: Home Credit Default Risk
Source: Kaggle Competition
Size: 300,000+ applications with multiple auxiliary tables
Features: Complex relational structure with 100+ features
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
from pathlib import Path
from .base import BaseDataLoader
import warnings


class HomeCreditLoader(BaseDataLoader):
    """
    Loader for Home Credit Default Risk dataset.

    This dataset contains loan application data with multiple auxiliary
    tables for bureau data, previous applications, etc.
    """

    def __init__(self, data_dir: str = "./data", random_state: int = 42,
                 use_auxiliary: bool = False):
        """
        Initialize Home Credit loader.

        Args:
            data_dir: Directory containing Home Credit CSV files
            random_state: Random seed
            use_auxiliary: If True, merge auxiliary tables (slower but more features)
        """
        super().__init__(data_dir, random_state)
        self.kaggle_url = "https://www.kaggle.com/c/home-credit-default-risk/data"
        self.main_file = "application_train.csv"
        self.test_file = "application_test.csv"
        self.use_auxiliary = use_auxiliary

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                            Optional[np.ndarray], Optional[np.ndarray], List[str], Optional[Dict[str, np.ndarray]]]:
        """
        Load and preprocess Home Credit dataset.

        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_oot, y_oot, feature_names, X_categorical)
            Note: X_oot/y_oot come from test set if available
        """
        print("\nLoading Home Credit Dataset...")

        # Check for main file
        main_path = self.data_dir / self.main_file
        if not main_path.exists():
            print(f"\nHome Credit dataset not found: {main_path}")
            print(f"Please download from Kaggle:")
            print(f"  {self.kaggle_url}")
            print("\nOr use Kaggle API:")
            print("  kaggle competitions download -c home-credit-default-risk")
            raise FileNotFoundError(f"Dataset file not found: {main_path}")

        # Load main application data
        df_train = self._read_main_data(main_path)

        # Load and merge auxiliary tables if requested
        if self.use_auxiliary:
            df_train = self._merge_auxiliary_data(df_train)

        # Preprocess
        df_train = self._preprocess(df_train)

        # Check if test set is available for OOT
        test_path = self.data_dir / self.test_file
        if test_path.exists():
            print(f"Loading test set from {test_path}...")
            df_test = self._read_main_data(test_path)
            if self.use_auxiliary:
                df_test = self._merge_auxiliary_data(df_test)
            df_test = self._preprocess(df_test, is_test=True)
        else:
            df_test = None

        # Extract features and target
        feature_names = [col for col in df_train.columns if col != 'default']
        X = df_train[feature_names].values
        y = df_train['default'].values

        # Split train into train/val
        X_train, X_val, y_train, y_val = self.create_train_val_split(
            X, y, val_size=0.3, stratify=True
        )

        # Use test set as OOT if available
        if df_test is not None:
            X_oot = df_test[[col for col in feature_names if col in df_test.columns]].values
            y_oot = None  # Test set has no labels
            print(f"Test set available: {len(X_oot):,} observations (no labels)")
        else:
            X_oot = None
            y_oot = None

        # Print summary
        self.print_summary(X_train, y_train, X_val, y_val, X_oot, y_oot, feature_names)

        # Check IRB requirements
        self.check_irb_requirements(y_train, y_val, min_defaults=20)

        # No categorical features (all encoded during preprocessing)
        X_categorical = None

        return X_train, y_train, X_val, y_val, X_oot, y_oot, feature_names, X_categorical

    def _read_main_data(self, filepath: Path) -> pd.DataFrame:
        """Read main application CSV."""
        print(f"Reading {filepath.name}...")
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df):,} applications")
        return df

    def _merge_auxiliary_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge auxiliary tables to enrich main application data."""
        print("\nMerging auxiliary data tables...")

        # Bureau data (credit bureau information)
        bureau_path = self.data_dir / "bureau.csv"
        if bureau_path.exists():
            df = self._merge_bureau(df, bureau_path)

        # Previous applications
        prev_path = self.data_dir / "previous_application.csv"
        if prev_path.exists():
            df = self._merge_previous_applications(df, prev_path)

        # Note: Can add more auxiliary tables (installments, credit card balance, etc.)
        # Keeping it simple for initial implementation

        return df

    def _merge_bureau(self, df: pd.DataFrame, bureau_path: Path) -> pd.DataFrame:
        """Aggregate and merge bureau data."""
        print("  Processing bureau data...")
        bureau = pd.read_csv(bureau_path)

        # Aggregate by SK_ID_CURR
        bureau_agg = bureau.groupby('SK_ID_CURR').agg({
            'DAYS_CREDIT': ['min', 'max', 'mean'],
            'CREDIT_DAY_OVERDUE': ['max', 'mean'],
            'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
            'AMT_CREDIT_SUM': ['sum', 'mean', 'max'],
            'AMT_CREDIT_SUM_DEBT': ['sum', 'mean', 'max'],
            'CREDIT_TYPE': 'count'
        }).reset_index()

        # Flatten column names
        bureau_agg.columns = ['_'.join(col).strip('_') for col in bureau_agg.columns.values]
        bureau_agg = bureau_agg.rename(columns={'SK_ID_CURR': 'SK_ID_CURR'})

        # Merge
        df = df.merge(bureau_agg, on='SK_ID_CURR', how='left')
        print(f"    Added {len(bureau_agg.columns) - 1} bureau features")

        return df

    def _merge_previous_applications(self, df: pd.DataFrame, prev_path: Path) -> pd.DataFrame:
        """Aggregate and merge previous application data."""
        print("  Processing previous applications...")
        prev = pd.read_csv(prev_path)

        # Aggregate by SK_ID_CURR
        prev_agg = prev.groupby('SK_ID_CURR').agg({
            'AMT_ANNUITY': ['min', 'max', 'mean'],
            'AMT_APPLICATION': ['min', 'max', 'mean'],
            'AMT_CREDIT': ['min', 'max', 'mean'],
            'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
            'DAYS_DECISION': ['min', 'max', 'mean'],
            'CNT_PAYMENT': ['mean', 'max'],
            'NAME_CONTRACT_STATUS': lambda x: (x == 'Approved').sum()
        }).reset_index()

        # Flatten column names
        prev_agg.columns = ['_'.join(col).strip('_') for col in prev_agg.columns.values]
        prev_agg = prev_agg.rename(columns={'SK_ID_CURR': 'SK_ID_CURR'})

        # Merge
        df = df.merge(prev_agg, on='SK_ID_CURR', how='left')
        print(f"    Added {len(prev_agg.columns) - 1} previous application features")

        return df

    def _preprocess(self, df: pd.DataFrame, is_test: bool = False) -> pd.DataFrame:
        """Preprocess the dataset for IRB segmentation."""
        df = df.copy()

        print("Preprocessing...")

        # Target variable (only in train set)
        if not is_test:
            if 'TARGET' in df.columns:
                df['default'] = df['TARGET'].astype(int)
            else:
                raise ValueError("TARGET column not found in training data")

        # Create IRB-relevant features

        # 1. Credit Score Proxy: Based on external sources
        if 'EXT_SOURCE_1' in df.columns and 'EXT_SOURCE_2' in df.columns and 'EXT_SOURCE_3' in df.columns:
            # External sources are already normalized scores
            df['credit_score'] = (
                df['EXT_SOURCE_1'].fillna(0.5) * 0.3 +
                df['EXT_SOURCE_2'].fillna(0.5) * 0.4 +
                df['EXT_SOURCE_3'].fillna(0.5) * 0.3
            ) * 100
        else:
            df['credit_score'] = 50

        # 2. Loan Amount
        if 'AMT_CREDIT' in df.columns:
            df['loan_amount'] = df['AMT_CREDIT'].fillna(df['AMT_CREDIT'].median())
        else:
            df['loan_amount'] = 0

        # 3. Income
        if 'AMT_INCOME_TOTAL' in df.columns:
            df['annual_income'] = df['AMT_INCOME_TOTAL'].fillna(df['AMT_INCOME_TOTAL'].median())
        else:
            df['annual_income'] = 100000

        # 4. LTV Proxy: Credit / Income
        df['ltv_proxy'] = (df['loan_amount'] / (df['annual_income'] + 1)) * 100
        df['ltv_proxy'] = df['ltv_proxy'].clip(0, 200)

        # 5. DTI Proxy: Annuity / Income
        if 'AMT_ANNUITY' in df.columns:
            df['annuity'] = df['AMT_ANNUITY'].fillna(df['AMT_ANNUITY'].median())
            df['dti_proxy'] = (df['annuity'] * 12 / (df['annual_income'] + 1)) * 100
            df['dti_proxy'] = df['dti_proxy'].clip(0, 100)
        else:
            df['dti_proxy'] = 30

        # 6. Age (in years)
        if 'DAYS_BIRTH' in df.columns:
            df['age'] = (-df['DAYS_BIRTH'] / 365).fillna(40)
        else:
            df['age'] = 40

        # 7. Employment Length (in years)
        if 'DAYS_EMPLOYED' in df.columns:
            # Negative values = currently employed, positive = unemployed
            df['years_employed'] = (-df['DAYS_EMPLOYED'] / 365).clip(0, 40).fillna(0)
        else:
            df['years_employed'] = 2

        # 8. Family Size
        if 'CNT_FAM_MEMBERS' in df.columns:
            df['family_size'] = df['CNT_FAM_MEMBERS'].fillna(2)
        else:
            df['family_size'] = 2

        # 9. Region Rating
        if 'REGION_RATING_CLIENT' in df.columns:
            df['region_rating'] = df['REGION_RATING_CLIENT'].fillna(2)
        else:
            df['region_rating'] = 2

        # 10. Car/Property Ownership Flags
        if 'FLAG_OWN_CAR' in df.columns:
            df['owns_car'] = (df['FLAG_OWN_CAR'] == 'Y').astype(int)
        else:
            df['owns_car'] = 0

        if 'FLAG_OWN_REALTY' in df.columns:
            df['owns_property'] = (df['FLAG_OWN_REALTY'] == 'Y').astype(int)
        else:
            df['owns_property'] = 0

        # 11. Gender
        if 'CODE_GENDER' in df.columns:
            df['gender_f'] = (df['CODE_GENDER'] == 'F').astype(int)
        else:
            df['gender_f'] = 0.5

        # 12. Income Type
        if 'NAME_INCOME_TYPE' in df.columns:
            income_risk = {
                'Working': 1,
                'Commercial associate': 2,
                'Pensioner': 1,
                'State servant': 1,
                'Student': 4,
                'Businessman': 3,
                'Maternity leave': 3,
                'Unemployed': 5
            }
            df['income_type_risk'] = df['NAME_INCOME_TYPE'].map(income_risk).fillna(2)
        else:
            df['income_type_risk'] = 2

        # 13. Number of children
        if 'CNT_CHILDREN' in df.columns:
            df['num_children'] = df['CNT_CHILDREN'].fillna(0)
        else:
            df['num_children'] = 0

        # 14. External source features (if not already used)
        for i in [1, 2, 3]:
            col = f'EXT_SOURCE_{i}'
            if col in df.columns:
                df[f'ext_source_{i}'] = df[col].fillna(0.5)

        # Select final features
        final_features = [
            'credit_score',
            'loan_amount',
            'annual_income',
            'ltv_proxy',
            'dti_proxy',
            'age',
            'years_employed',
            'family_size',
            'num_children',
            'region_rating',
            'owns_car',
            'owns_property',
            'gender_f',
            'income_type_risk'
        ]

        # Add external sources if available
        for i in [1, 2, 3]:
            if f'ext_source_{i}' in df.columns:
                final_features.append(f'ext_source_{i}')

        # Add target if train set
        if not is_test:
            final_features.append('default')

        # Ensure all features exist
        for feat in final_features:
            if feat not in df.columns:
                if feat != 'default' or not is_test:
                    warnings.warn(f"Feature {feat} not found, using default value")
                    df[feat] = 0

        # Drop rows with too many NaNs
        df = df[final_features].dropna(thresh=len(final_features) - 3)

        print(f"After preprocessing: {len(df):,} rows, {len(final_features)} features")

        return df

    def get_feature_descriptions(self) -> dict:
        """Get descriptions of features."""
        return {
            'credit_score': 'Composite credit score from external sources (0-100)',
            'loan_amount': 'Loan credit amount',
            'annual_income': 'Total annual income',
            'ltv_proxy': 'Loan-to-income ratio as LTV proxy (%)',
            'dti_proxy': 'Annuity-to-income ratio as DTI proxy (%)',
            'age': 'Age in years',
            'years_employed': 'Years at current employment',
            'family_size': 'Number of family members',
            'num_children': 'Number of children',
            'region_rating': 'Region rating score',
            'owns_car': 'Owns a car (1=yes, 0=no)',
            'owns_property': 'Owns property/realty (1=yes, 0=no)',
            'gender_f': 'Gender (1=female, 0=male)',
            'income_type_risk': 'Income type risk score (1-5)',
            'ext_source_1': 'External source 1 score',
            'ext_source_2': 'External source 2 score',
            'ext_source_3': 'External source 3 score'
        }


# Convenience function
def load_home_credit(data_dir: str = "./data", random_state: int = 42,
                    use_auxiliary: bool = False):
    """
    Convenience function to load Home Credit dataset.

    Args:
        data_dir: Directory containing Home Credit CSV files
        random_state: Random seed
        use_auxiliary: If True, merge auxiliary tables

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_oot, y_oot, feature_names, X_categorical)
    """
    loader = HomeCreditLoader(data_dir, random_state, use_auxiliary)
    return loader.load()
