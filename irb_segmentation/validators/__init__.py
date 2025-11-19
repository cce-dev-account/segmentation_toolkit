"""
Input Validation Module for IRB Segmentation

This module provides comprehensive validation for:
- NumPy arrays (shape, dtype, NaN/inf handling)
- pandas DataFrames (schema, data quality)
- IRB parameters and constraints
- Data preprocessing and transformation
- Binary classification models (PD)
- Regression models (LGD, EAD)

All validators provide actionable error messages with specific guidance
on how to fix validation issues.
"""

from .input_validator import (
    validate_array,
    validate_binary_target,
    validate_feature_names,
    validate_train_val_compatibility,
    ValidationError
)

from .data_validator import (
    validate_dataframe,
    validate_target_column,
    validate_feature_types,
    check_data_quality,
    DataQualityReport
)

from .performance_metrics import (
    gini_coefficient,
    ks_statistic,
    brier_score,
    accuracy_ratio,
    calculate_all_metrics,
    PerformanceMetrics
)

from .calibration import (
    hosmer_lemeshow_test,
    traffic_light_test,
    central_tendency_test,
    run_all_calibration_tests,
    CalibrationResults
)

from .backtesting import (
    binomial_backtest,
    chi_squared_backtest,
    traffic_light_backtest,
    run_backtest,
    BacktestResults
)

from .monotonicity import (
    check_strict_monotonicity,
    calculate_rank_correlation,
    check_monotonic_trend,
    run_monotonicity_validation,
    MonotonicityResults
)

from .regression_metrics import (
    r_squared,
    adjusted_r_squared,
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
    calculate_all_regression_metrics,
    RegressionMetrics
)

from .residual_analysis import (
    analyze_residuals,
    breusch_pagan_test,
    white_test,
    run_all_residual_diagnostics,
    ResidualAnalysisResults,
    HeteroscedasticityTestResults
)

from .quantile_regression import (
    quantile_loss,
    fit_quantile_regression,
    estimate_lgd_floor,
    compare_quantile_models,
    validate_downturn_calibration,
    QuantileRegressionResults
)

# Import SegmentValidator from parent validators.py module
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

try:
    from validators import SegmentValidator
except ImportError:
    # If running from a different context, try direct import
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "validators_legacy",
        Path(__file__).parent.parent / "validators.py"
    )
    if spec and spec.loader:
        validators_legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(validators_legacy)
        SegmentValidator = validators_legacy.SegmentValidator

__all__ = [
    # Input validation
    'validate_array',
    'validate_binary_target',
    'validate_feature_names',
    'validate_train_val_compatibility',
    'ValidationError',

    # Data validation
    'validate_dataframe',
    'validate_target_column',
    'validate_feature_types',
    'check_data_quality',
    'DataQualityReport',

    # Performance metrics
    'gini_coefficient',
    'ks_statistic',
    'brier_score',
    'accuracy_ratio',
    'calculate_all_metrics',
    'PerformanceMetrics',

    # Calibration testing
    'hosmer_lemeshow_test',
    'traffic_light_test',
    'central_tendency_test',
    'run_all_calibration_tests',
    'CalibrationResults',

    # Back-testing
    'binomial_backtest',
    'chi_squared_backtest',
    'traffic_light_backtest',
    'run_backtest',
    'BacktestResults',

    # Monotonicity validation
    'check_strict_monotonicity',
    'calculate_rank_correlation',
    'check_monotonic_trend',
    'run_monotonicity_validation',
    'MonotonicityResults',

    # Regression metrics (LGD/EAD)
    'r_squared',
    'adjusted_r_squared',
    'mean_absolute_error',
    'root_mean_squared_error',
    'mean_absolute_percentage_error',
    'calculate_all_regression_metrics',
    'RegressionMetrics',

    # Residual analysis and diagnostics
    'analyze_residuals',
    'breusch_pagan_test',
    'white_test',
    'run_all_residual_diagnostics',
    'ResidualAnalysisResults',
    'HeteroscedasticityTestResults',

    # Quantile regression (downturn LGD)
    'quantile_loss',
    'fit_quantile_regression',
    'estimate_lgd_floor',
    'compare_quantile_models',
    'validate_downturn_calibration',
    'QuantileRegressionResults',

    # Segment validation (from parent validators.py)
    'SegmentValidator',
]

__version__ = '1.1.0'
