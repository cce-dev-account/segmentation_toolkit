"""
Base Data Loader Class

Common utilities and interface for all dataset loaders.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
from abc import ABC, abstractmethod
from pathlib import Path
import warnings


class BaseDataLoader(ABC):
    """
    Abstract base class for dataset loaders.

    All dataset loaders should inherit from this class and implement
    the load() method.
    """

    def __init__(self, data_dir: str = "./data", random_state: int = 42):
        """
        Initialize the data loader.

        Args:
            data_dir: Directory to store/load data
            random_state: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        np.random.seed(random_state)

    @abstractmethod
    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                            Optional[np.ndarray], Optional[np.ndarray], List[str], Optional[Dict[str, np.ndarray]]]:
        """
        Load and preprocess the dataset.

        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_oot, y_oot, feature_names, X_categorical)
            where X_oot and y_oot may be None if no out-of-time data available,
            and X_categorical is a dict mapping categorical feature names to arrays (or None if no categoricals)
        """
        pass

    def print_summary(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     X_oot: Optional[np.ndarray] = None,
                     y_oot: Optional[np.ndarray] = None,
                     feature_names: Optional[List[str]] = None):
        """
        Print summary statistics for the loaded dataset.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            X_oot: Optional out-of-time features
            y_oot: Optional out-of-time targets
            feature_names: Optional feature names
        """
        print("\n" + "=" * 70)
        print("DATASET SUMMARY")
        print("=" * 70)

        # Training set
        print(f"\nTraining Set:")
        print(f"  Observations: {len(X_train):,}")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Defaults: {np.sum(y_train):,} ({np.mean(y_train):.2%})")
        print(f"  Non-defaults: {len(y_train) - np.sum(y_train):,}")

        # Validation set
        print(f"\nValidation Set:")
        print(f"  Observations: {len(X_val):,}")
        print(f"  Defaults: {np.sum(y_val):,} ({np.mean(y_val):.2%})")

        # Out-of-time set
        if X_oot is not None and y_oot is not None:
            print(f"\nOut-of-Time Set:")
            print(f"  Observations: {len(X_oot):,}")
            print(f"  Defaults: {np.sum(y_oot):,} ({np.mean(y_oot):.2%})")

        # Feature information
        if feature_names:
            print(f"\nFeatures ({len(feature_names)}):")
            for i, name in enumerate(feature_names[:10]):
                print(f"  {i+1}. {name}")
            if len(feature_names) > 10:
                print(f"  ... and {len(feature_names) - 10} more")

        # Data quality checks
        print("\nData Quality:")
        print(f"  Missing values (train): {np.sum(np.isnan(X_train))}")
        print(f"  Missing values (val): {np.sum(np.isnan(X_val))}")
        if X_oot is not None:
            print(f"  Missing values (oot): {np.sum(np.isnan(X_oot))}")

        print("=" * 70)

    def create_train_val_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_size: float = 0.3,
        stratify: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and validation sets.

        Args:
            X: Feature matrix
            y: Target vector
            val_size: Proportion for validation set
            stratify: Whether to stratify by target

        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        from sklearn.model_selection import train_test_split

        return train_test_split(
            X, y,
            test_size=val_size,
            random_state=self.random_state,
            stratify=y if stratify else None
        )

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = "median",
        fill_value: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in DataFrame.

        Args:
            df: Input DataFrame
            strategy: "median", "mean", "mode", "drop", or "constant"
            fill_value: Value to use if strategy is "constant"

        Returns:
            DataFrame with missing values handled
        """
        if strategy == "drop":
            return df.dropna()
        elif strategy == "median":
            return df.fillna(df.median(numeric_only=True))
        elif strategy == "mean":
            return df.fillna(df.mean(numeric_only=True))
        elif strategy == "mode":
            return df.fillna(df.mode().iloc[0])
        elif strategy == "constant":
            return df.fillna(fill_value)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def encode_categorical(
        self,
        df: pd.DataFrame,
        categorical_cols: List[str],
        method: str = "onehot"
    ) -> pd.DataFrame:
        """
        Encode categorical variables.

        Args:
            df: Input DataFrame
            categorical_cols: List of categorical column names
            method: "onehot" or "label"

        Returns:
            DataFrame with encoded categorical variables
        """
        if method == "onehot":
            return pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        elif method == "label":
            df_encoded = df.copy()
            from sklearn.preprocessing import LabelEncoder
            for col in categorical_cols:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df[col].astype(str))
            return df_encoded
        else:
            raise ValueError(f"Unknown encoding method: {method}")

    def prepare_categorical_features(
        self,
        df: pd.DataFrame,
        categorical_cols: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Extract categorical features for use with IRB segmentation.

        This method preserves original categorical values (not encoded)
        for use with categorical forced splits.

        Args:
            df: Input DataFrame
            categorical_cols: List of categorical column names

        Returns:
            Dictionary mapping column name to numpy array of categorical values
        """
        categorical_dict = {}
        for col in categorical_cols:
            if col in df.columns:
                categorical_dict[col] = df[col].values
            else:
                warnings.warn(f"Categorical column '{col}' not found in DataFrame")
        return categorical_dict

    def check_irb_requirements(
        self,
        y_train: np.ndarray,
        y_val: np.ndarray,
        min_defaults: int = 20
    ) -> Dict[str, bool]:
        """
        Check if dataset meets basic IRB requirements.

        Args:
            y_train: Training targets
            y_val: Validation targets
            min_defaults: Minimum defaults required

        Returns:
            Dictionary with requirement checks
        """
        checks = {
            "sufficient_train_defaults": np.sum(y_train) >= min_defaults,
            "sufficient_val_defaults": np.sum(y_val) >= min_defaults,
            "sufficient_train_size": len(y_train) >= 500,
            "sufficient_val_size": len(y_val) >= 200,
            "balanced_classes": 0.01 <= np.mean(y_train) <= 0.50,
        }

        all_passed = all(checks.values())

        print("\nIRB Requirements Check:")
        for check, passed in checks.items():
            status = "[PASS]" if passed else "[FAIL]"
            print(f"  {status} {check.replace('_', ' ').title()}")

        if not all_passed:
            warnings.warn("Dataset does not meet all IRB requirements")

        return checks

    def download_file(self, url: str, filename: str) -> Path:
        """
        Download a file from URL if not already present.

        Args:
            url: URL to download from
            filename: Local filename to save

        Returns:
            Path to downloaded file
        """
        filepath = self.data_dir / filename

        if filepath.exists():
            print(f"File already exists: {filepath}")
            return filepath

        print(f"Downloading {filename} from {url}...")
        try:
            import urllib.request
            urllib.request.urlretrieve(url, filepath)
            print(f"Downloaded to: {filepath}")
        except Exception as e:
            print(f"Error downloading file: {e}")
            print(f"Please manually download from {url} and save to {filepath}")
            raise

        return filepath
