"""
Example Usage of IRB Segmentation Framework

This script demonstrates how to use the IRB PD model segmentation framework
with synthetic credit data.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from irb_segmentation import IRBSegmentationParams, IRBSegmentationEngine


def generate_synthetic_credit_data(n_samples: int = 10000) -> tuple:
    """
    Generate synthetic credit portfolio data.

    Returns:
        Tuple of (X, y, feature_names, X_oot, y_oot) where oot = out-of-time validation
    """
    np.random.seed(42)

    print(f"Generating {n_samples} synthetic credit observations...")

    # Generate features
    credit_score = np.random.normal(700, 100, n_samples)
    credit_score = np.clip(credit_score, 300, 850)

    ltv = np.random.uniform(40, 120, n_samples)
    dti = np.random.uniform(5, 60, n_samples)
    loan_amount = np.random.lognormal(12, 0.5, n_samples)
    years_at_current_job = np.random.exponential(5, n_samples)

    X = np.column_stack([credit_score, ltv, dti, loan_amount, years_at_current_job])
    feature_names = ['credit_score', 'ltv', 'dti', 'loan_amount', 'years_at_current_job']

    # Generate realistic default probabilities
    # Lower credit score -> higher default
    # Higher LTV -> higher default
    # Higher DTI -> higher default
    # More job stability -> lower default
    logit = (
        -5.0
        + 0.015 * (650 - credit_score)
        + 0.03 * (ltv - 80)
        + 0.02 * (dti - 30)
        - 0.1 * np.log1p(years_at_current_job)
    )

    default_prob = 1 / (1 + np.exp(-logit))
    y = np.random.binomial(1, default_prob)

    print(f"  Overall default rate: {np.mean(y):.2%}")
    print(f"  Number of defaults: {np.sum(y)}")

    # Generate out-of-time data with slight distribution shift
    print("\nGenerating out-of-time validation data...")
    n_oot = n_samples // 5

    credit_score_oot = np.random.normal(690, 105, n_oot)  # Slightly worse
    credit_score_oot = np.clip(credit_score_oot, 300, 850)

    ltv_oot = np.random.uniform(45, 125, n_oot)  # Slightly higher
    dti_oot = np.random.uniform(8, 62, n_oot)
    loan_amount_oot = np.random.lognormal(12.1, 0.5, n_oot)
    years_at_current_job_oot = np.random.exponential(4.5, n_oot)

    X_oot = np.column_stack([
        credit_score_oot, ltv_oot, dti_oot, loan_amount_oot, years_at_current_job_oot
    ])

    logit_oot = (
        -5.0
        + 0.015 * (650 - credit_score_oot)
        + 0.03 * (ltv_oot - 80)
        + 0.02 * (dti_oot - 30)
        - 0.1 * np.log1p(years_at_current_job_oot)
    )

    default_prob_oot = 1 / (1 + np.exp(-logit_oot))
    y_oot = np.random.binomial(1, default_prob_oot)

    print(f"  OOT default rate: {np.mean(y_oot):.2%}")

    return X, y, feature_names, X_oot, y_oot


def example_basic_usage():
    """Example 1: Basic usage with default parameters"""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 80)

    # Generate data
    X, y, feature_names, X_oot, y_oot = generate_synthetic_credit_data(n_samples=5000)

    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Define parameters
    params = IRBSegmentationParams(
        max_depth=3,
        min_samples_leaf=300,
        min_defaults_per_leaf=20,
        min_segment_density=0.10,
        max_segment_density=0.50,
        validation_tests=['chi_squared', 'binomial']
    )

    print("\nParameters:")
    print(params.get_summary())

    # Create and fit engine
    engine = IRBSegmentationEngine(params)
    engine.fit(X_train, y_train, X_val, y_val, feature_names=feature_names)

    # Get production segments
    segments = engine.get_production_segments()
    print(f"\nProduction segments created: {len(np.unique(segments))}")

    # Generate validation report
    report = engine.get_validation_report()
    print("\nValidation Status:")
    print(f"  All validations passed: {report['validation_results']['train']['all_passed']}")


def example_with_business_constraints():
    """Example 2: Using business constraints"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Business Constraints")
    print("=" * 80)

    # Generate data
    X, y, feature_names, X_oot, y_oot = generate_synthetic_credit_data(n_samples=5000)

    # Define parameters with business rules
    params = IRBSegmentationParams(
        max_depth=4,
        min_samples_leaf=250,
        min_defaults_per_leaf=25,
        min_segment_density=0.08,
        max_segment_density=0.45,
        # Business constraints
        forced_splits={'ltv': 80.0},  # Regulatory threshold
        monotone_constraints={'credit_score': 1},  # Higher score = lower risk
        validation_tests=['chi_squared', 'psi', 'binomial']
    )

    print("\nBusiness Constraints Applied:")
    print(f"  Forced split at LTV=80%")
    print(f"  Monotone constraint: credit_score increases with lower risk")

    # Create and fit engine
    engine = IRBSegmentationEngine(params)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    engine.fit(X_train, y_train, X_val, y_val, feature_names=feature_names)

    # Check adjustments
    print("\nAdjustments Applied:")
    print(f"  Merges: {len(engine.adjustment_log_['merges'])}")
    print(f"  Splits: {len(engine.adjustment_log_['splits'])}")
    print(f"  Forced splits: {len(engine.adjustment_log_['forced_splits'])}")
    print(f"  Monotonicity violations: {len(engine.adjustment_log_['monotonicity_violations'])}")

    # Extract segment rules
    rules = engine.get_segment_rules()
    print("\nSegment Rules (first 5):")
    for i, rule in enumerate(rules[:5], 1):
        print(f"  {i}. {rule}")


def example_oot_validation():
    """Example 3: Out-of-time validation"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Out-of-Time Validation")
    print("=" * 80)

    # Generate data
    X, y, feature_names, X_oot, y_oot = generate_synthetic_credit_data(n_samples=8000)

    # Define parameters
    params = IRBSegmentationParams(
        max_depth=3,
        min_samples_leaf=400,
        min_defaults_per_leaf=30,
        validation_tests=['chi_squared', 'psi']
    )

    # Fit on in-time data
    engine = IRBSegmentationEngine(params)
    engine.fit(X, y, feature_names=feature_names)

    # Predict on out-of-time data
    segments_oot = engine.predict(X_oot)

    # Calculate PSI
    from irb_segmentation.validators import SegmentValidator
    psi_result = SegmentValidator.calculate_psi(
        engine.segments_train_, segments_oot, threshold=0.1
    )

    print("\nOut-of-Time Validation:")
    print(f"  PSI: {psi_result['psi']:.4f}")
    print(f"  Stability: {psi_result['stability']}")
    print(f"  Passed: {psi_result['passed']}")

    print("\nPSI by Segment:")
    for seg, psi in psi_result['psi_per_segment'].items():
        print(f"  Segment {seg}: {psi:.4f}")


def example_export_and_analysis():
    """Example 4: Export results and detailed analysis"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Export and Analysis")
    print("=" * 80)

    # Generate data
    X, y, feature_names, X_oot, y_oot = generate_synthetic_credit_data(n_samples=5000)

    # Fit model
    params = IRBSegmentationParams(
        max_depth=3,
        min_samples_leaf=300,
        min_defaults_per_leaf=20
    )

    engine = IRBSegmentationEngine(params)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    engine.fit(X_train, y_train, X_val, y_val, feature_names=feature_names)

    # Export report
    report_path = "segmentation_report.json"
    engine.export_report(report_path)

    # Create detailed DataFrame
    segments = engine.get_production_segments()
    df = pd.DataFrame(X_train, columns=feature_names)
    df['segment'] = segments
    df['default'] = y_train

    # Analyze segments
    print("\nDetailed Segment Analysis:")
    segment_analysis = df.groupby('segment').agg({
        'credit_score': ['mean', 'min', 'max'],
        'ltv': ['mean', 'min', 'max'],
        'default': ['count', 'sum', 'mean']
    })

    print(segment_analysis)

    print(f"\nReport exported to: {report_path}")


def main():
    """Run all examples"""
    print("IRB PD Model Segmentation Framework - Examples")
    print("=" * 80)

    try:
        example_basic_usage()
        example_with_business_constraints()
        example_oot_validation()
        example_export_and_analysis()

        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\nError running examples: {e}")
        raise


if __name__ == "__main__":
    main()
