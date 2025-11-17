"""
Shared Test Fixtures and Configuration for IRB Segmentation Test Suite

This module provides reusable fixtures for:
- Sample data (arrays, DataFrames, targets)
- Engine configurations
- Mock data generators
- Common test utilities

Usage:
    pytest automatically discovers fixtures from conftest.py
    Any test can use these fixtures by including them as function parameters
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from typing import Tuple, List, Dict

from irb_segmentation.params import IRBSegmentationParams
from irb_segmentation.config import (
    SegmentationConfig,
    DataConfig,
    OutputConfig,
    LoggingConfig
)
from irb_segmentation.engine import IRBSegmentationEngine


# ==============================================================================
# PYTEST CONFIGURATION
# ==============================================================================

def pytest_configure(config):
    """Register custom markers for test organization."""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual functions/classes"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for workflows"
    )
    config.addinivalue_line(
        "markers", "edge_case: Tests for edge cases and boundary conditions"
    )
    config.addinivalue_line(
        "markers", "property: Property-based tests with hypothesis"
    )
    config.addinivalue_line(
        "markers", "slow: Slow-running tests (>1 second)"
    )


# ==============================================================================
# RANDOM SEED FIXTURES
# ==============================================================================

@pytest.fixture(scope="session")
def random_seed():
    """Global random seed for reproducible tests."""
    return 42


@pytest.fixture(autouse=True)
def set_random_seed(random_seed):
    """Automatically set random seed before each test."""
    np.random.seed(random_seed)


# ==============================================================================
# SAMPLE DATA FIXTURES - ARRAYS
# ==============================================================================

@pytest.fixture
def valid_X_small():
    """Small valid feature array (200 samples, 3 features)."""
    return np.random.randn(200, 3)


@pytest.fixture
def valid_X_medium():
    """Medium valid feature array (1000 samples, 10 features)."""
    return np.random.randn(1000, 10)


@pytest.fixture
def valid_X_large():
    """Large valid feature array (5000 samples, 20 features)."""
    return np.random.randn(5000, 20)


@pytest.fixture
def valid_y_small():
    """Small valid binary target (200 samples, 15% default rate)."""
    y = np.zeros(200, dtype=int)
    y[:30] = 1  # 15% default rate
    return y


@pytest.fixture
def valid_y_medium():
    """Medium valid binary target (1000 samples, 10% default rate)."""
    y = np.zeros(1000, dtype=int)
    y[:100] = 1  # 10% default rate
    return y


@pytest.fixture
def valid_y_large():
    """Large valid binary target (5000 samples, 8% default rate)."""
    y = np.zeros(5000, dtype=int)
    y[:400] = 1  # 8% default rate
    return y


@pytest.fixture
def imbalanced_y():
    """Highly imbalanced target (1000 samples, 2% default rate)."""
    y = np.zeros(1000, dtype=int)
    y[:20] = 1  # 2% default rate
    return y


@pytest.fixture
def balanced_y():
    """Balanced binary target (1000 samples, 50% default rate)."""
    y = np.zeros(1000, dtype=int)
    y[:500] = 1  # 50% default rate
    return y


# ==============================================================================
# SAMPLE DATA FIXTURES - DATAFRAMES
# ==============================================================================

@pytest.fixture
def valid_credit_df():
    """Valid credit risk DataFrame with realistic features."""
    n_samples = 1000

    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.lognormal(10.5, 0.8, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'debt_to_income': np.random.uniform(0, 1.5, n_samples),
        'employment_years': np.random.randint(0, 40, n_samples),
        'num_credit_lines': np.random.randint(0, 20, n_samples),
        'utilization_rate': np.random.uniform(0, 1, n_samples),
        'default': np.random.binomial(1, 0.1, n_samples)
    })

    return df


@pytest.fixture
def df_with_missing():
    """DataFrame with missing values in various columns."""
    n_samples = 500

    df = pd.DataFrame({
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.lognormal(10.5, 0.8, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'default': np.random.binomial(1, 0.1, n_samples)
    })

    # Add missing values
    df.loc[df.sample(frac=0.1).index, 'income'] = np.nan
    df.loc[df.sample(frac=0.05).index, 'credit_score'] = np.nan

    return df


@pytest.fixture
def df_with_constant_col():
    """DataFrame with a constant column."""
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, 500),
        'constant_col': [1] * 500,  # Constant column
        'income': np.random.lognormal(10.5, 0.8, 500),
        'default': np.random.binomial(1, 0.1, 500)
    })

    return df


@pytest.fixture
def df_with_high_cardinality():
    """DataFrame with high cardinality ID-like column."""
    n_samples = 500

    df = pd.DataFrame({
        'customer_id': range(n_samples),  # Unique for each row
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.lognormal(10.5, 0.8, n_samples),
        'default': np.random.binomial(1, 0.1, n_samples)
    })

    return df


@pytest.fixture
def df_with_duplicates():
    """DataFrame with duplicate rows."""
    df = pd.DataFrame({
        'age': [25, 30, 25, 40, 30],
        'income': [50000, 60000, 50000, 70000, 60000],
        'default': [0, 1, 0, 0, 1]
    })

    # Duplicate rows 10 times
    df = pd.concat([df] * 10, ignore_index=True)

    return df


# ==============================================================================
# CONFIGURATION FIXTURES
# ==============================================================================

@pytest.fixture
def default_params():
    """Default IRB parameters for testing."""
    return IRBSegmentationParams(
        min_samples_leaf=50,
        min_defaults_per_leaf=10
    )


@pytest.fixture
def strict_params():
    """Strict IRB parameters with tighter constraints."""
    return IRBSegmentationParams(
        min_samples_leaf=100,
        min_defaults_per_leaf=20,
        min_segment_density=0.08
    )


@pytest.fixture
def relaxed_params():
    """Relaxed IRB parameters for edge case testing."""
    return IRBSegmentationParams(
        min_samples_leaf=30,
        min_defaults_per_leaf=5,
        min_segment_density=0.03
    )


@pytest.fixture
def default_segmentation_config(tmp_path):
    """Default segmentation configuration for testing."""
    return SegmentationConfig(
        data=DataConfig(
            source=str(tmp_path / "test_data.csv"),
            data_type='csv',
            target_column='default'
        ),
        irb_params=IRBSegmentationParams(
            max_depth=5,
            min_samples_split=100,
            min_samples_leaf=50,
            min_defaults_per_leaf=10,
            random_state=42
        ),
        output=OutputConfig(
            output_dir=str(tmp_path / "output")
        ),
        logging=LoggingConfig(
            level='INFO',
            log_file=None,
            log_format=None
        )
    )


@pytest.fixture
def logging_config_with_file(tmp_path):
    """Logging configuration with file output."""
    log_file = tmp_path / "test.log"
    return LoggingConfig(
        level='DEBUG',
        log_file=str(log_file),
        log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


# ==============================================================================
# ENGINE FIXTURES
# ==============================================================================

@pytest.fixture
def default_engine(default_params):
    """Segmentation engine with default parameters."""
    return IRBSegmentationEngine(params=default_params, verbose=False)


@pytest.fixture
def strict_engine(strict_params):
    """Segmentation engine with strict parameters."""
    return IRBSegmentationEngine(params=strict_params, verbose=False)


@pytest.fixture
def fitted_engine(default_engine, valid_X_medium, valid_y_medium):
    """Pre-fitted segmentation engine."""
    default_engine.fit(valid_X_medium, valid_y_medium)
    return default_engine


# ==============================================================================
# TRAIN/VAL SPLIT FIXTURES
# ==============================================================================

@pytest.fixture
def train_val_split_small(valid_X_small, valid_y_small):
    """Small train/val split (70/30)."""
    split_idx = int(len(valid_X_small) * 0.7)
    return (
        valid_X_small[:split_idx],
        valid_y_small[:split_idx],
        valid_X_small[split_idx:],
        valid_y_small[split_idx:]
    )


@pytest.fixture
def train_val_split_medium(valid_X_medium, valid_y_medium):
    """Medium train/val split (80/20)."""
    split_idx = int(len(valid_X_medium) * 0.8)
    return (
        valid_X_medium[:split_idx],
        valid_y_medium[:split_idx],
        valid_X_medium[split_idx:],
        valid_y_medium[split_idx:]
    )


@pytest.fixture
def train_val_split_large(valid_X_large, valid_y_large):
    """Large train/val split (80/20)."""
    split_idx = int(len(valid_X_large) * 0.8)
    return (
        valid_X_large[:split_idx],
        valid_y_large[:split_idx],
        valid_X_large[split_idx:],
        valid_y_large[split_idx:]
    )


# ==============================================================================
# EDGE CASE FIXTURES
# ==============================================================================

@pytest.fixture
def all_defaults_y():
    """Edge case: All observations are defaults."""
    return np.ones(200, dtype=int)


@pytest.fixture
def no_defaults_y():
    """Edge case: No defaults (all zeros)."""
    return np.zeros(200, dtype=int)


@pytest.fixture
def single_default_y():
    """Edge case: Only one default."""
    y = np.zeros(200, dtype=int)
    y[0] = 1
    return y


@pytest.fixture
def tiny_X():
    """Edge case: Minimal dataset (50 samples, 2 features)."""
    return np.random.randn(50, 2)


@pytest.fixture
def tiny_y():
    """Edge case: Minimal target (50 samples, 5 defaults)."""
    y = np.zeros(50, dtype=int)
    y[:5] = 1
    return y


@pytest.fixture
def X_with_nans():
    """Edge case: Feature array with NaN values."""
    X = np.random.randn(500, 5)
    X[np.random.choice(500, 50, replace=False),
      np.random.choice(5, 50, replace=True)] = np.nan
    return X


@pytest.fixture
def X_with_infs():
    """Edge case: Feature array with infinity values."""
    X = np.random.randn(500, 5)
    X[np.random.choice(500, 10, replace=False),
      np.random.choice(5, 10, replace=True)] = np.inf
    return X


@pytest.fixture
def single_feature_X():
    """Edge case: Only one feature."""
    return np.random.randn(1000, 1)


# ==============================================================================
# TEMPORARY DIRECTORY FIXTURES
# ==============================================================================

@pytest.fixture
def temp_output_dir():
    """Temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp(prefix="irb_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_data_file(tmp_path, valid_credit_df):
    """Temporary CSV file with sample data."""
    file_path = tmp_path / "test_data.csv"
    valid_credit_df.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def temp_config_file(tmp_path):
    """Temporary YAML config file."""
    config_path = tmp_path / "test_config.yaml"
    config_content = """
data:
  source: "test_data.csv"
  target_column: "default"
  feature_columns: null

model:
  max_depth: 5
  min_samples_split: 100
  min_samples_leaf: 50
  random_state: 42

optimization:
  max_iterations: 10
  min_defaults_per_leaf: 10
  max_segments: 10
  min_segments: 3
  monotonicity_penalty: 1.0

logging:
  level: "INFO"
  log_file: null
"""
    config_path.write_text(config_content)
    return config_path


# ==============================================================================
# MOCK DATA GENERATORS
# ==============================================================================

@pytest.fixture
def generate_credit_data():
    """Factory fixture to generate synthetic credit data."""
    def _generate(
        n_samples: int = 1000,
        n_features: int = 10,
        default_rate: float = 0.1,
        add_noise: bool = True,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic credit risk data.

        Args:
            n_samples: Number of samples
            n_features: Number of features
            default_rate: Target default rate
            add_noise: Whether to add random noise
            random_state: Random seed

        Returns:
            Tuple of (X, y) arrays
        """
        np.random.seed(random_state)

        # Generate features
        X = np.random.randn(n_samples, n_features)

        # Generate target with desired default rate
        n_defaults = int(n_samples * default_rate)
        y = np.zeros(n_samples, dtype=int)
        y[:n_defaults] = 1

        # Shuffle
        shuffle_idx = np.random.permutation(n_samples)
        X = X[shuffle_idx]
        y = y[shuffle_idx]

        if add_noise:
            X += np.random.randn(n_samples, n_features) * 0.1

        return X, y

    return _generate


@pytest.fixture
def generate_dataframe():
    """Factory fixture to generate DataFrames with specified properties."""
    def _generate(
        n_samples: int = 500,
        n_numeric: int = 5,
        n_categorical: int = 3,
        default_rate: float = 0.1,
        missing_rate: float = 0.0,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Generate DataFrame with mixed feature types.

        Args:
            n_samples: Number of samples
            n_numeric: Number of numeric features
            n_categorical: Number of categorical features
            default_rate: Target default rate
            missing_rate: Proportion of missing values
            random_state: Random seed

        Returns:
            DataFrame with features and target
        """
        np.random.seed(random_state)

        data = {}

        # Numeric features
        for i in range(n_numeric):
            data[f'num_feat_{i}'] = np.random.randn(n_samples)

        # Categorical features
        for i in range(n_categorical):
            categories = [f'cat_{j}' for j in range(5)]
            data[f'cat_feat_{i}'] = np.random.choice(categories, n_samples)

        # Target
        n_defaults = int(n_samples * default_rate)
        target = np.zeros(n_samples, dtype=int)
        target[:n_defaults] = 1
        np.random.shuffle(target)
        data['default'] = target

        df = pd.DataFrame(data)

        # Add missing values
        if missing_rate > 0:
            n_missing = int(n_samples * missing_rate)
            for col in df.columns:
                if col != 'default':
                    missing_idx = np.random.choice(n_samples, n_missing, replace=False)
                    df.loc[missing_idx, col] = np.nan

        return df

    return _generate


# ==============================================================================
# ASSERTION HELPERS
# ==============================================================================

@pytest.fixture
def assert_valid_segments():
    """Helper to assert segment properties."""
    def _assert(segments: np.ndarray, y: np.ndarray, params: IRBSegmentationParams):
        """
        Assert that segments meet IRB constraints.

        Args:
            segments: Segment assignments
            y: Binary target
            params: IRB parameters
        """
        unique_segments = np.unique(segments)
        n_segments = len(unique_segments)

        # Check each segment
        for seg_id in unique_segments:
            seg_mask = segments == seg_id
            seg_size = seg_mask.sum()
            seg_defaults = y[seg_mask].sum()

            # Check minimum size
            assert seg_size >= params.min_samples_leaf, \
                f"Segment {seg_id} has {seg_size} samples (min: {params.min_samples_leaf})"

            # Check minimum defaults
            assert seg_defaults >= params.min_defaults_per_leaf, \
                f"Segment {seg_id} has {seg_defaults} defaults (min: {params.min_defaults_per_leaf})"

    return _assert


@pytest.fixture
def assert_monotonic_segments():
    """Helper to assert monotonicity of segment default rates."""
    def _assert(segments: np.ndarray, y: np.ndarray):
        """
        Assert that segment default rates are monotonic.

        Args:
            segments: Segment assignments
            y: Binary target
        """
        unique_segments = sorted(np.unique(segments))

        default_rates = []
        for seg_id in unique_segments:
            seg_mask = segments == seg_id
            dr = y[seg_mask].mean()
            default_rates.append(dr)

        # Check if monotonically increasing or decreasing
        is_increasing = all(default_rates[i] <= default_rates[i+1]
                          for i in range(len(default_rates)-1))
        is_decreasing = all(default_rates[i] >= default_rates[i+1]
                          for i in range(len(default_rates)-1))

        assert is_increasing or is_decreasing, \
            f"Default rates not monotonic: {default_rates}"

    return _assert


# ==============================================================================
# COMPARISON UTILITIES
# ==============================================================================

@pytest.fixture
def compare_arrays():
    """Helper to compare numpy arrays with tolerance."""
    def _compare(arr1: np.ndarray, arr2: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8):
        """
        Compare two arrays with specified tolerance.

        Args:
            arr1: First array
            arr2: Second array
            rtol: Relative tolerance
            atol: Absolute tolerance
        """
        assert arr1.shape == arr2.shape, \
            f"Shape mismatch: {arr1.shape} vs {arr2.shape}"

        assert np.allclose(arr1, arr2, rtol=rtol, atol=atol), \
            f"Arrays not close. Max diff: {np.abs(arr1 - arr2).max()}"

    return _compare


# ==============================================================================
# MODULE-LEVEL SETUP/TEARDOWN
# ==============================================================================

@pytest.fixture(scope="session", autouse=True)
def test_session_setup():
    """Setup run once at the beginning of the test session."""
    print("\n" + "="*80)
    print("STARTING IRB SEGMENTATION TEST SUITE")
    print("="*80)
    yield
    print("\n" + "="*80)
    print("TEST SESSION COMPLETED")
    print("="*80)
