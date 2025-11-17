"""
DataFrame Validation and Data Quality Checks

Provides validation for pandas DataFrames:
- Schema validation (column presence, types)
- Target column validation
- Feature type inference and validation
- Comprehensive data quality reporting

All validation includes actionable error messages.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from ..logger import get_logger
from .input_validator import ValidationError

logger = get_logger(__name__)


@dataclass
class DataQualityReport:
    """
    Comprehensive data quality assessment report.

    Attributes:
        n_rows: Total number of rows
        n_cols: Total number of columns
        missing_values: Dict mapping columns to missing value counts
        missing_percentages: Dict mapping columns to missing percentages
        constant_columns: List of columns with only one unique value
        high_cardinality_columns: List of columns with very high cardinality
        duplicated_rows: Number of duplicated rows
        numeric_columns: List of numeric column names
        categorical_columns: List of categorical column names
        outliers: Dict mapping numeric columns to outlier counts
        warnings: List of warning messages
        passed: Whether data passes quality checks
    """

    n_rows: int
    n_cols: int
    missing_values: Dict[str, int] = field(default_factory=dict)
    missing_percentages: Dict[str, float] = field(default_factory=dict)
    constant_columns: List[str] = field(default_factory=list)
    high_cardinality_columns: List[str] = field(default_factory=list)
    duplicated_rows: int = 0
    numeric_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    outliers: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    passed: bool = True

    def __str__(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 70,
            "DATA QUALITY REPORT",
            "=" * 70,
            f"Dataset Shape: {self.n_rows:,} rows x {self.n_cols} columns",
            "",
            "Column Types:",
            f"  Numeric: {len(self.numeric_columns)}",
            f"  Categorical: {len(self.categorical_columns)}",
        ]

        if self.missing_values:
            lines.extend([
                "",
                "Missing Values:",
            ])
            for col, count in sorted(self.missing_values.items(), key=lambda x: -x[1])[:5]:
                pct = self.missing_percentages[col]
                lines.append(f"  {col}: {count:,} ({pct:.1f}%)")
            if len(self.missing_values) > 5:
                lines.append(f"  ... and {len(self.missing_values) - 5} more columns")

        if self.constant_columns:
            lines.extend([
                "",
                f"Constant Columns ({len(self.constant_columns)}):",
                f"  {', '.join(self.constant_columns[:5])}"
            ])
            if len(self.constant_columns) > 5:
                lines.append(f"  ... and {len(self.constant_columns) - 5} more")

        if self.high_cardinality_columns:
            lines.extend([
                "",
                f"High Cardinality Columns ({len(self.high_cardinality_columns)}):",
                f"  {', '.join(self.high_cardinality_columns[:5])}"
            ])

        if self.duplicated_rows > 0:
            lines.extend([
                "",
                f"Duplicated Rows: {self.duplicated_rows:,} ({100*self.duplicated_rows/self.n_rows:.2f}%)"
            ])

        if self.outliers:
            lines.extend([
                "",
                "Outliers (IQR method):",
            ])
            for col, count in list(self.outliers.items())[:5]:
                pct = 100 * count / self.n_rows
                lines.append(f"  {col}: {count:,} ({pct:.1f}%)")

        if self.warnings:
            lines.extend([
                "",
                f"Warnings ({len(self.warnings)}):",
            ])
            for warning in self.warnings[:5]:
                lines.append(f"  - {warning}")
            if len(self.warnings) > 5:
                lines.append(f"  ... and {len(self.warnings) - 5} more warnings")

        lines.extend([
            "",
            f"Status: {'PASSED' if self.passed else 'FAILED'}",
            "=" * 70
        ])

        return "\n".join(lines)


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    min_rows: int = 100,
    max_missing_pct: float = 0.50
) -> DataQualityReport:
    """
    Validate a pandas DataFrame for IRB segmentation.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names (optional)
        min_rows: Minimum number of rows required
        max_missing_pct: Maximum allowed missing percentage per column

    Returns:
        DataQualityReport with comprehensive analysis

    Raises:
        ValidationError: If critical validation fails

    Example:
        >>> df = pd.DataFrame({'age': [25, 30, 35], 'income': [50k, 60k, 70k]})
        >>> report = validate_dataframe(df, min_rows=3)
        >>> print(report)
    """
    # Check if DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValidationError(
            "Input must be a pandas DataFrame",
            field="df",
            expected="pandas.DataFrame",
            actual=str(type(df).__name__),
            fix="Convert to DataFrame: df = pd.DataFrame(data)"
        )

    # Check if empty
    if df.empty:
        raise ValidationError(
            "DataFrame is empty",
            field="df",
            expected="Non-empty DataFrame",
            actual="0 rows",
            fix="Check data loading. Ensure file/query returns data."
        )

    n_rows, n_cols = df.shape

    # Check minimum rows
    if n_rows < min_rows:
        raise ValidationError(
            "DataFrame has too few rows",
            field="df",
            expected=f"At least {min_rows} rows",
            actual=f"{n_rows} rows",
            fix=f"Provide more data or reduce min_rows to {n_rows}"
        )

    # Check required columns
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValidationError(
                "Required columns missing from DataFrame",
                field="df.columns",
                expected=f"Columns: {required_columns}",
                actual=f"Missing: {list(missing_cols)}",
                fix=f"Add missing columns or check column names (case-sensitive)"
            )

    # Initialize report
    report = DataQualityReport(n_rows=n_rows, n_cols=n_cols)

    # Check missing values
    missing_counts = df.isnull().sum()
    for col, count in missing_counts.items():
        if count > 0:
            pct = 100 * count / n_rows
            report.missing_values[col] = int(count)
            report.missing_percentages[col] = float(pct)

            if pct > max_missing_pct * 100:
                report.warnings.append(
                    f"Column '{col}' has {pct:.1f}% missing values (threshold: {max_missing_pct*100:.1f}%)"
                )
                report.passed = False

    # Identify constant columns
    for col in df.columns:
        if df[col].nunique() == 1:
            report.constant_columns.append(col)
            report.warnings.append(f"Column '{col}' has only one unique value")

    # Identify high cardinality columns (potential IDs)
    for col in df.columns:
        n_unique = df[col].nunique()
        if n_unique > 0.9 * n_rows:  # More than 90% unique
            report.high_cardinality_columns.append(col)
            report.warnings.append(
                f"Column '{col}' has very high cardinality ({n_unique:,} unique values). "
                "May be an ID column."
            )

    # Check for duplicated rows
    report.duplicated_rows = int(df.duplicated().sum())
    if report.duplicated_rows > 0:
        dup_pct = 100 * report.duplicated_rows / n_rows
        if dup_pct > 5:  # More than 5% duplicates
            report.warnings.append(
                f"{report.duplicated_rows:,} duplicated rows ({dup_pct:.1f}%)"
            )

    # Classify columns by type
    report.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report.categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Check for outliers in numeric columns (IQR method)
    for col in report.numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        if IQR > 0:  # Avoid division by zero
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR

            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                report.outliers[col] = int(outliers)

    logger.info(
        f"DataFrame validation completed: {n_rows:,} rows, {n_cols} cols, "
        f"{len(report.warnings)} warnings"
    )

    return report


def validate_target_column(
    df: pd.DataFrame,
    target_col: str,
    expected_values: Optional[List] = None
) -> Tuple[pd.Series, int, float]:
    """
    Validate the target column in a DataFrame.

    Args:
        df: DataFrame containing target
        target_col: Name of target column
        expected_values: Expected unique values (e.g., [0, 1] for binary)

    Returns:
        Tuple of (target_series, n_positives, positive_rate)

    Raises:
        ValidationError: If target column is invalid

    Example:
        >>> df = pd.DataFrame({'default': [0, 0, 1, 1, 0]})
        >>> target, n_pos, rate = validate_target_column(df, 'default', [0, 1])
        >>> print(f"Positive rate: {rate:.2%}")
        Positive rate: 40.00%
    """
    # Check if column exists
    if target_col not in df.columns:
        raise ValidationError(
            f"Target column '{target_col}' not found in DataFrame",
            field="target_col",
            expected=f"Column '{target_col}' to exist",
            actual=f"Available columns: {list(df.columns)[:10]}",
            fix=f"Check column name (case-sensitive) or specify correct target_col parameter"
        )

    target = df[target_col]

    # Check for missing values in target
    n_missing = target.isnull().sum()
    if n_missing > 0:
        raise ValidationError(
            f"Target column '{target_col}' contains missing values",
            field=target_col,
            expected="No missing values in target",
            actual=f"{n_missing} missing values",
            fix=f"Remove rows with missing target: df = df[df['{target_col}'].notnull()]"
        )

    # Check unique values
    unique_values = sorted(target.unique())

    if expected_values is not None:
        if not set(unique_values).issubset(set(expected_values)):
            raise ValidationError(
                f"Target column '{target_col}' contains unexpected values",
                field=target_col,
                expected=f"Values in {expected_values}",
                actual=f"Found values: {unique_values}",
                fix=f"Map values to expected range or check data integrity"
            )

    # For binary target, calculate positive rate
    if len(unique_values) == 2:
        # Assume higher value is positive class
        positive_value = max(unique_values)
        n_positives = int((target == positive_value).sum())
        positive_rate = n_positives / len(target)

        logger.info(
            f"Target column '{target_col}' validated: "
            f"{n_positives:,} positives ({positive_rate:.2%})"
        )

        return target, n_positives, positive_rate

    elif len(unique_values) == 1:
        raise ValidationError(
            f"Target column '{target_col}' has only one unique value",
            field=target_col,
            expected="At least 2 classes",
            actual=f"Only value: {unique_values[0]}",
            fix="Check if data filtering removed one class or if target is mislabeled"
        )

    else:
        # Multi-class target
        raise ValidationError(
            f"Target column '{target_col}' has more than 2 unique values",
            field=target_col,
            expected="Binary target (2 classes)",
            actual=f"{len(unique_values)} classes: {unique_values}",
            fix="Use binary target for IRB segmentation or bin multi-class target"
        )


def validate_feature_types(
    df: pd.DataFrame,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None
) -> Tuple[List[str], List[str]]:
    """
    Validate and infer feature types.

    Args:
        df: DataFrame with features
        numeric_features: Explicitly specified numeric features
        categorical_features: Explicitly specified categorical features

    Returns:
        Tuple of (validated_numeric_features, validated_categorical_features)

    Raises:
        ValidationError: If feature types are invalid

    Example:
        >>> df = pd.DataFrame({'age': [25, 30], 'city': ['NYC', 'LA']})
        >>> num_feat, cat_feat = validate_feature_types(df)
        >>> print(num_feat, cat_feat)
        ['age'] ['city']
    """
    all_columns = set(df.columns)

    # If not specified, infer from dtypes
    if numeric_features is None:
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()

    if categorical_features is None:
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Validate specified features exist
    missing_numeric = set(numeric_features) - all_columns
    if missing_numeric:
        raise ValidationError(
            "Specified numeric features not found in DataFrame",
            field="numeric_features",
            expected=f"All features to exist: {numeric_features}",
            actual=f"Missing: {list(missing_numeric)}",
            fix="Check feature names or remove missing features from list"
        )

    missing_categorical = set(categorical_features) - all_columns
    if missing_categorical:
        raise ValidationError(
            "Specified categorical features not found in DataFrame",
            field="categorical_features",
            expected=f"All features to exist: {categorical_features}",
            actual=f"Missing: {list(missing_categorical)}",
            fix="Check feature names or remove missing features from list"
        )

    # Check for overlap
    overlap = set(numeric_features) & set(categorical_features)
    if overlap:
        raise ValidationError(
            "Features specified as both numeric and categorical",
            field="feature_types",
            expected="Disjoint numeric and categorical features",
            actual=f"Overlap: {list(overlap)}",
            fix="Specify each feature as only numeric OR categorical, not both"
        )

    # Validate numeric features actually contain numbers
    for feat in numeric_features:
        if not np.issubdtype(df[feat].dtype, np.number):
            logger.warning(
                f"Feature '{feat}' specified as numeric but has dtype {df[feat].dtype}. "
                "Consider converting or treating as categorical."
            )

    # Validate categorical features
    for feat in categorical_features:
        n_unique = df[feat].nunique()
        if n_unique > 100:
            logger.warning(
                f"Categorical feature '{feat}' has {n_unique} unique values. "
                "Consider grouping rare categories or treating as numeric."
            )

    logger.info(
        f"Feature types validated: {len(numeric_features)} numeric, "
        f"{len(categorical_features)} categorical"
    )

    return numeric_features, categorical_features


def check_data_quality(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    auto_fix: bool = False
) -> Tuple[pd.DataFrame, DataQualityReport]:
    """
    Comprehensive data quality check with optional auto-fixing.

    Args:
        df: DataFrame to check
        target_col: Optional target column name
        auto_fix: Whether to automatically fix issues (remove constant cols, etc.)

    Returns:
        Tuple of (cleaned_df, quality_report)

    Example:
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [1, 1, 1], 'target': [0, 1, 0]})
        >>> cleaned_df, report = check_data_quality(df, target_col='target', auto_fix=True)
        >>> print(report)
    """
    df_clean = df.copy()

    # Run validation
    report = validate_dataframe(df_clean, min_rows=1)

    if auto_fix:
        logger.info("Auto-fixing data quality issues...")

        # Remove constant columns
        if report.constant_columns:
            cols_to_drop = [col for col in report.constant_columns if col != target_col]
            if cols_to_drop:
                logger.info(f"Dropping {len(cols_to_drop)} constant columns: {cols_to_drop}")
                df_clean = df_clean.drop(columns=cols_to_drop)

        # Remove high cardinality columns (likely IDs)
        if report.high_cardinality_columns:
            cols_to_drop = [col for col in report.high_cardinality_columns if col != target_col]
            if cols_to_drop:
                logger.info(
                    f"Dropping {len(cols_to_drop)} high-cardinality columns (likely IDs): "
                    f"{cols_to_drop}"
                )
                df_clean = df_clean.drop(columns=cols_to_drop)

        # Remove duplicated rows
        if report.duplicated_rows > 0:
            logger.info(f"Removing {report.duplicated_rows:,} duplicated rows")
            df_clean = df_clean.drop_duplicates()

        # Re-run validation on cleaned data
        report = validate_dataframe(df_clean, min_rows=1)
        logger.info("Data quality auto-fix completed")

    return df_clean, report
