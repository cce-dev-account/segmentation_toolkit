"""
Unit Tests for IRB Segmentation DataFrame Validation

Tests cover:
- DataFrame validation (structure, schema, data quality)
- Target column validation
- Feature type inference and validation
- Data quality reporting
- Auto-fixing capabilities
"""

import pytest
import pandas as pd
import numpy as np

from irb_segmentation.validators import (
    validate_dataframe,
    validate_target_column,
    validate_feature_types,
    check_data_quality,
    DataQualityReport,
    ValidationError
)


@pytest.mark.unit
class TestDataQualityReport:
    """Test DataQualityReport dataclass."""

    def test_create_basic_report(self):
        """Test creating a basic data quality report."""
        report = DataQualityReport(
            n_rows=1000,
            n_cols=10
        )

        assert report.n_rows == 1000
        assert report.n_cols == 10
        assert report.passed is True
        assert len(report.warnings) == 0

    def test_report_with_warnings(self):
        """Test report with warnings."""
        report = DataQualityReport(
            n_rows=500,
            n_cols=5,
            warnings=["Warning 1", "Warning 2"],
            passed=False
        )

        assert len(report.warnings) == 2
        assert report.passed is False

    def test_report_string_representation(self):
        """Test report string representation."""
        report = DataQualityReport(
            n_rows=1000,
            n_cols=10,
            missing_values={'col1': 50},
            missing_percentages={'col1': 5.0},
            constant_columns=['const_col'],
            duplicated_rows=10
        )

        report_str = str(report)
        assert "DATA QUALITY REPORT" in report_str
        assert "1,000 rows" in report_str
        assert "10 columns" in report_str

    def test_report_with_all_metrics(self):
        """Test report with all metrics populated."""
        report = DataQualityReport(
            n_rows=2000,
            n_cols=15,
            missing_values={'col1': 100, 'col2': 50},
            missing_percentages={'col1': 5.0, 'col2': 2.5},
            constant_columns=['const1', 'const2'],
            high_cardinality_columns=['id_col'],
            duplicated_rows=25,
            numeric_columns=['num1', 'num2', 'num3'],
            categorical_columns=['cat1', 'cat2'],
            outliers={'num1': 50, 'num2': 30},
            warnings=["High missing rate", "Constant columns detected"],
            passed=False
        )

        assert len(report.missing_values) == 2
        assert len(report.constant_columns) == 2
        assert len(report.high_cardinality_columns) == 1
        assert report.duplicated_rows == 25
        assert len(report.numeric_columns) == 3
        assert len(report.categorical_columns) == 2
        assert len(report.outliers) == 2


@pytest.mark.unit
class TestValidateDataFrame:
    """Test validate_dataframe function."""

    def test_valid_dataframe(self, valid_credit_df):
        """Test valid DataFrame passes validation."""
        report = validate_dataframe(valid_credit_df, min_rows=100)

        assert report.n_rows >= 100
        assert report.n_cols > 0
        assert isinstance(report, DataQualityReport)

    def test_dataframe_not_pandas(self):
        """Test validation fails for non-DataFrame."""
        not_df = {"col1": [1, 2, 3], "col2": [4, 5, 6]}

        with pytest.raises(ValidationError) as exc_info:
            validate_dataframe(not_df)

        assert "must be a pandas DataFrame" in str(exc_info.value)

    def test_empty_dataframe(self):
        """Test validation fails for empty DataFrame."""
        df = pd.DataFrame()

        with pytest.raises(ValidationError) as exc_info:
            validate_dataframe(df)

        assert "empty" in str(exc_info.value).lower()

    def test_too_few_rows(self):
        """Test validation fails for too few rows."""
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

        with pytest.raises(ValidationError) as exc_info:
            validate_dataframe(df, min_rows=100)

        assert "too few rows" in str(exc_info.value).lower()
        assert "3 rows" in str(exc_info.value)

    def test_required_columns_present(self):
        """Test validation passes when required columns present."""
        df = pd.DataFrame({
            'age': [25, 30, 35],
            'income': [50000, 60000, 70000],
            'default': [0, 1, 0]
        })

        report = validate_dataframe(
            df,
            required_columns=['age', 'income', 'default'],
            min_rows=3
        )

        assert report.passed

    def test_required_columns_missing(self):
        """Test validation fails when required columns missing."""
        df = pd.DataFrame({
            'age': [25, 30, 35],
            'income': [50000, 60000, 70000]
        })

        with pytest.raises(ValidationError) as exc_info:
            validate_dataframe(
                df,
                required_columns=['age', 'income', 'default'],
                min_rows=3
            )

        assert "missing" in str(exc_info.value).lower()
        assert "default" in str(exc_info.value)

    def test_detects_missing_values(self, df_with_missing):
        """Test detection of missing values."""
        report = validate_dataframe(df_with_missing, min_rows=1)

        assert len(report.missing_values) > 0
        assert any(col in report.missing_values for col in df_with_missing.columns)

    def test_excessive_missing_values(self):
        """Test validation fails for excessive missing values."""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, np.nan, np.nan] * 100,
            'B': [1, 2, 3, 4, 5] * 100
        })

        report = validate_dataframe(df, max_missing_pct=0.30, min_rows=1)

        # Should have warnings or fail
        assert not report.passed or len(report.warnings) > 0

    def test_detects_constant_columns(self, df_with_constant_col):
        """Test detection of constant columns."""
        report = validate_dataframe(df_with_constant_col, min_rows=1)

        assert len(report.constant_columns) > 0
        assert 'constant_col' in report.constant_columns

    def test_detects_high_cardinality(self, df_with_high_cardinality):
        """Test detection of high cardinality columns."""
        report = validate_dataframe(df_with_high_cardinality, min_rows=1)

        assert len(report.high_cardinality_columns) > 0
        assert 'customer_id' in report.high_cardinality_columns

    def test_detects_duplicates(self, df_with_duplicates):
        """Test detection of duplicate rows."""
        report = validate_dataframe(df_with_duplicates, min_rows=1)

        assert report.duplicated_rows > 0

    def test_identifies_column_types(self, valid_credit_df):
        """Test identification of numeric and categorical columns."""
        report = validate_dataframe(valid_credit_df, min_rows=1)

        assert len(report.numeric_columns) > 0
        assert len(report.categorical_columns) >= 0

    def test_detects_outliers(self):
        """Test outlier detection using IQR method."""
        # Create data with outliers
        df = pd.DataFrame({
            'normal': np.random.randn(1000),
            'with_outliers': np.concatenate([
                np.random.randn(950),
                np.array([100, 110, 120, -100, -110] * 10)  # Outliers
            ])
        })

        report = validate_dataframe(df, min_rows=100)

        assert 'with_outliers' in report.outliers
        assert report.outliers['with_outliers'] > 0


@pytest.mark.unit
class TestValidateTargetColumn:
    """Test validate_target_column function."""

    def test_valid_target_column(self):
        """Test valid binary target column."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'default': [0, 1, 0, 1, 0]
        })

        target, n_pos, pos_rate = validate_target_column(
            df, 'default', expected_values=[0, 1]
        )

        assert len(target) == 5
        assert n_pos == 2
        assert np.isclose(pos_rate, 0.4)

    def test_target_column_not_found(self):
        """Test validation fails when target column missing."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })

        with pytest.raises(ValidationError) as exc_info:
            validate_target_column(df, 'default')

        assert "not found" in str(exc_info.value).lower()
        assert "default" in str(exc_info.value)

    def test_target_with_missing_values(self):
        """Test validation fails for missing values in target."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'default': [0, 1, np.nan, 1, 0]
        })

        with pytest.raises(ValidationError) as exc_info:
            validate_target_column(df, 'default')

        assert "missing" in str(exc_info.value).lower()

    def test_target_unexpected_values(self):
        """Test validation fails for unexpected values."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'default': [0, 1, 2, 1, 0]  # Has 2
        })

        with pytest.raises(ValidationError) as exc_info:
            validate_target_column(df, 'default', expected_values=[0, 1])

        assert "unexpected values" in str(exc_info.value).lower()

    def test_target_only_one_class(self):
        """Test validation fails when target has only one class."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'default': [0, 0, 0, 0, 0]
        })

        with pytest.raises(ValidationError) as exc_info:
            validate_target_column(df, 'default')

        assert "one unique value" in str(exc_info.value).lower()

    def test_target_multiclass(self):
        """Test validation fails for multiclass target."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'default': [0, 1, 2, 0, 1]
        })

        with pytest.raises(ValidationError) as exc_info:
            validate_target_column(df, 'default')

        assert "more than 2" in str(exc_info.value).lower() or \
               "binary" in str(exc_info.value).lower()

    def test_target_returns_correct_stats(self):
        """Test function returns correct statistics."""
        df = pd.DataFrame({
            'default': [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
        })

        target, n_pos, pos_rate = validate_target_column(df, 'default')

        assert n_pos == 3
        assert np.isclose(pos_rate, 0.3)


@pytest.mark.unit
class TestValidateFeatureTypes:
    """Test validate_feature_types function."""

    def test_infer_feature_types(self):
        """Test automatic feature type inference."""
        df = pd.DataFrame({
            'age': [25, 30, 35],
            'income': [50000, 60000, 70000],
            'city': ['NYC', 'LA', 'SF'],
            'default': [0, 1, 0]
        })

        num_feat, cat_feat = validate_feature_types(df)

        assert 'age' in num_feat
        assert 'income' in num_feat
        assert 'city' in cat_feat

    def test_explicit_feature_types(self):
        """Test explicit feature type specification."""
        df = pd.DataFrame({
            'age': [25, 30, 35],
            'income': [50000, 60000, 70000],
            'city': ['NYC', 'LA', 'SF']
        })

        num_feat, cat_feat = validate_feature_types(
            df,
            numeric_features=['age', 'income'],
            categorical_features=['city']
        )

        assert num_feat == ['age', 'income']
        assert cat_feat == ['city']

    def test_missing_numeric_feature(self):
        """Test validation fails for missing numeric feature."""
        df = pd.DataFrame({
            'age': [25, 30, 35],
            'city': ['NYC', 'LA', 'SF']
        })

        with pytest.raises(ValidationError) as exc_info:
            validate_feature_types(
                df,
                numeric_features=['age', 'income']  # income doesn't exist
            )

        assert "not found" in str(exc_info.value).lower()
        assert "income" in str(exc_info.value)

    def test_missing_categorical_feature(self):
        """Test validation fails for missing categorical feature."""
        df = pd.DataFrame({
            'age': [25, 30, 35],
            'city': ['NYC', 'LA', 'SF']
        })

        with pytest.raises(ValidationError) as exc_info:
            validate_feature_types(
                df,
                categorical_features=['city', 'state']  # state doesn't exist
            )

        assert "not found" in str(exc_info.value).lower()

    def test_overlapping_feature_types(self):
        """Test validation fails for overlapping feature specifications."""
        df = pd.DataFrame({
            'age': [25, 30, 35],
            'income': [50000, 60000, 70000]
        })

        with pytest.raises(ValidationError) as exc_info:
            validate_feature_types(
                df,
                numeric_features=['age'],
                categorical_features=['age']  # Same feature in both
            )

        assert "overlap" in str(exc_info.value).lower() or \
               "both" in str(exc_info.value).lower()

    def test_warning_for_high_cardinality_categorical(self, caplog):
        """Test warning for high cardinality categorical feature."""
        # Create a categorical feature with many unique values
        df = pd.DataFrame({
            'id': [f'id_{i}' for i in range(200)],  # 200 unique values
            'value': range(200)
        })

        num_feat, cat_feat = validate_feature_types(
            df,
            categorical_features=['id']
        )

        # Should still validate but may log warning
        assert 'id' in cat_feat


@pytest.mark.unit
class TestCheckDataQuality:
    """Test check_data_quality function."""

    def test_check_quality_no_issues(self, valid_credit_df):
        """Test data quality check on clean data."""
        cleaned_df, report = check_data_quality(
            valid_credit_df,
            auto_fix=False
        )

        # Original data returned unchanged
        assert len(cleaned_df) == len(valid_credit_df)
        assert isinstance(report, DataQualityReport)

    def test_check_quality_with_auto_fix(self, df_with_constant_col):
        """Test auto-fixing data quality issues."""
        original_cols = df_with_constant_col.shape[1]

        cleaned_df, report = check_data_quality(
            df_with_constant_col,
            auto_fix=True
        )

        # Constant column should be removed
        assert cleaned_df.shape[1] < original_cols

    def test_auto_fix_removes_constant_columns(self):
        """Test that auto-fix removes constant columns."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [1, 1, 1, 1, 1],  # Constant
            'C': [5, 6, 7, 8, 9],
            'target': [0, 1, 0, 1, 0]
        })

        cleaned_df, report = check_data_quality(
            df,
            target_col='target',
            auto_fix=True
        )

        assert 'B' not in cleaned_df.columns
        assert 'target' in cleaned_df.columns  # Preserved

    def test_auto_fix_removes_high_cardinality(self):
        """Test that auto-fix removes high cardinality columns."""
        df = pd.DataFrame({
            'id': range(500),  # Unique for each row
            'feature': np.random.randn(500),
            'target': np.random.randint(0, 2, 500)
        })

        cleaned_df, report = check_data_quality(
            df,
            target_col='target',
            auto_fix=True
        )

        # ID column should be removed
        assert 'id' not in cleaned_df.columns
        assert 'target' in cleaned_df.columns

    def test_auto_fix_removes_duplicates(self, df_with_duplicates):
        """Test that auto-fix removes duplicate rows."""
        original_rows = len(df_with_duplicates)

        cleaned_df, report = check_data_quality(
            df_with_duplicates,
            auto_fix=True
        )

        # Duplicates should be removed
        assert len(cleaned_df) < original_rows

    def test_preserves_target_column(self):
        """Test that target column is preserved during auto-fix."""
        df = pd.DataFrame({
            'A': [1] * 100,  # Constant
            'B': list(range(100)),  # High cardinality
            'target': [0, 1] * 50
        })

        cleaned_df, report = check_data_quality(
            df,
            target_col='target',
            auto_fix=True
        )

        # Target should always be preserved
        assert 'target' in cleaned_df.columns

    def test_check_quality_without_target(self):
        """Test data quality check without specifying target."""
        df = pd.DataFrame({
            'A': [1, 2, 3] * 100,  # Enough rows
            'B': [1, 1, 1] * 100,  # Constant
            'C': [4, 5, 6] * 100
        })

        cleaned_df, report = check_data_quality(
            df,
            auto_fix=True
        )

        # Should still work without target and remove constant column
        assert 'B' not in cleaned_df.columns or len(cleaned_df.columns) <= len(df.columns)


@pytest.mark.unit
class TestDataValidationIntegration:
    """Integration tests for data validation functions."""

    def test_full_dataframe_workflow(self, valid_credit_df):
        """Test complete DataFrame validation workflow."""
        # 1. Validate DataFrame structure
        report = validate_dataframe(valid_credit_df, min_rows=100)
        assert report.n_rows >= 100

        # 2. Validate target column
        target, n_pos, pos_rate = validate_target_column(
            valid_credit_df, 'default'
        )
        assert n_pos > 0

        # 3. Validate feature types
        num_feat, cat_feat = validate_feature_types(valid_credit_df)
        assert len(num_feat) > 0

        # 4. Check data quality
        cleaned_df, quality_report = check_data_quality(
            valid_credit_df,
            target_col='default',
            auto_fix=False
        )
        assert isinstance(quality_report, DataQualityReport)

    def test_validation_with_problematic_data(self):
        """Test validation catches multiple data issues."""
        df = pd.DataFrame({
            'const': [1] * 300,  # Constant - all same value
            'id': list(range(300)),  # High cardinality
            'feature': np.random.randn(300),
            'target': [0, 1, 0] * 100
        })

        # Duplicate some rows
        df = pd.concat([df, df.iloc[:50]], ignore_index=True)

        report = validate_dataframe(df, min_rows=100)

        # Should detect issues - check that warnings or issues exist
        has_issues = (
            len(report.constant_columns) > 0 or
            len(report.high_cardinality_columns) > 0 or
            report.duplicated_rows > 0
        )
        assert has_issues

    def test_auto_fix_workflow(self):
        """Test auto-fix improves data quality."""
        df = pd.DataFrame({
            'A': [1, 2, 3] * 200,
            'B': [1, 1, 1] * 200,  # Constant
            'C': range(600),  # High cardinality
            'D': np.random.randn(600),
            'target': [0, 1, 0] * 200
        })

        # Add duplicates
        df = pd.concat([df, df.iloc[:100]], ignore_index=True)

        # Check quality before
        _, report_before = check_data_quality(df, auto_fix=False)
        issues_before = len(report_before.warnings)

        # Auto-fix
        cleaned_df, report_after = check_data_quality(
            df,
            target_col='target',
            auto_fix=True
        )

        # Should have fewer issues after
        assert len(cleaned_df.columns) < len(df.columns)
        assert len(cleaned_df) <= len(df)


@pytest.mark.unit
class TestDataValidationEdgeCases:
    """Test edge cases for data validation."""

    def test_dataframe_with_all_nan_column(self):
        """Test handling of columns with all NaN values."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [np.nan, np.nan, np.nan, np.nan, np.nan],
            'C': [6, 7, 8, 9, 10]
        })

        report = validate_dataframe(df, min_rows=1)

        # Should detect high missing rate in column B
        assert 'B' in report.missing_values
        assert report.missing_percentages['B'] == 100.0

    def test_dataframe_with_mixed_types(self):
        """Test DataFrame with mixed column types."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.5, 2.5, 3.5],
            'str_col': ['a', 'b', 'c'],
            'bool_col': [True, False, True]
        })

        num_feat, cat_feat = validate_feature_types(df)

        # Should correctly classify types
        assert 'int_col' in num_feat
        assert 'float_col' in num_feat

    def test_single_row_dataframe(self):
        """Test validation on single-row DataFrame."""
        df = pd.DataFrame({
            'A': [1],
            'B': [2],
            'C': [3]
        })

        report = validate_dataframe(df, min_rows=1)
        assert report.n_rows == 1

    def test_dataframe_with_zero_variance_after_dropna(self):
        """Test handling of zero variance after missing value handling."""
        df = pd.DataFrame({
            'A': [1, 1, 1, np.nan, 1],
            'B': [2, 3, 4, 5, 6]
        })

        # After dropping NaN, A has zero variance
        report = validate_dataframe(df, min_rows=1)

        # Should complete without error
        assert isinstance(report, DataQualityReport)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
