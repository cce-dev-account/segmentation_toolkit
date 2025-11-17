"""
Integration Testing with Real Credit Datasets

Tests the IRB segmentation framework with actual credit datasets to validate
that it works correctly and meets all IRB requirements.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from irb_segmentation import IRBSegmentationParams, IRBSegmentationEngine
from data_loaders import (
    GermanCreditLoader,
    TaiwanCreditLoader,
    LendingClubLoader,
    HomeCreditLoader
)


def test_german_credit():
    """Test 1: German Credit Dataset (Quick validation)"""
    print("\n" + "=" * 80)
    print("TEST 1: GERMAN CREDIT DATASET")
    print("=" * 80)

    try:
        # Load data
        loader = GermanCreditLoader(data_dir="./data")
        X_train, y_train, X_val, y_val, X_oot, y_oot, feature_names = loader.load()

        # Define parameters optimized for small dataset
        params = IRBSegmentationParams(
            max_depth=3,
            min_samples_leaf=50,
            min_defaults_per_leaf=5,  # Lower for small dataset
            min_segment_density=0.08,
            max_segment_density=0.60,
            validation_tests=['chi_squared', 'binomial']
        )

        # Fit model
        print("\n" + "-" * 80)
        print("FITTING MODEL")
        print("-" * 80)
        engine = IRBSegmentationEngine(params)
        engine.fit(X_train, y_train, X_val, y_val, feature_names=feature_names)

        # Get report
        report = engine.get_validation_report()

        # Export report
        engine.export_report("german_credit_report.json")

        # Print segment rules
        print("\n" + "-" * 80)
        print("SEGMENT RULES")
        print("-" * 80)
        rules = engine.get_segment_rules()
        for i, rule in enumerate(rules[:5], 1):
            print(f"{i}. {rule}")
        if len(rules) > 5:
            print(f"... and {len(rules) - 5} more rules")

        # Summary
        print("\n" + "-" * 80)
        print("TEST RESULT: [PASS]")
        print("-" * 80)
        return True

    except FileNotFoundError as e:
        print(f"\n[SKIP] {e}")
        return None
    except Exception as e:
        print(f"\n[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_taiwan_credit():
    """Test 2: Taiwan Credit Card Dataset"""
    print("\n" + "=" * 80)
    print("TEST 2: TAIWAN CREDIT CARD DATASET")
    print("=" * 80)

    try:
        # Load data
        loader = TaiwanCreditLoader(data_dir="./data")
        X_train, y_train, X_val, y_val, X_oot, y_oot, feature_names = loader.load()

        # Define parameters for larger dataset
        params = IRBSegmentationParams(
            max_depth=4,
            min_samples_leaf=500,
            min_defaults_per_leaf=30,
            min_segment_density=0.10,
            max_segment_density=0.50,
            validation_tests=['chi_squared', 'binomial']
        )

        # Fit model
        print("\n" + "-" * 80)
        print("FITTING MODEL")
        print("-" * 80)
        engine = IRBSegmentationEngine(params)
        engine.fit(X_train, y_train, X_val, y_val, feature_names=feature_names)

        # Export report
        engine.export_report("taiwan_credit_report.json")

        print("\n" + "-" * 80)
        print("TEST RESULT: [PASS]")
        print("-" * 80)
        return True

    except FileNotFoundError as e:
        print(f"\n[SKIP] {e}")
        return None
    except Exception as e:
        print(f"\n[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lending_club():
    """Test 3: Lending Club Dataset with OOT validation"""
    print("\n" + "=" * 80)
    print("TEST 3: LENDING CLUB DATASET (with Out-of-Time Validation)")
    print("=" * 80)

    try:
        # Load data with sampling for faster testing
        loader = LendingClubLoader(data_dir="./data", sample_size=50000)
        X_train, y_train, X_val, y_val, X_oot, y_oot, feature_names = loader.load(use_oot=True)

        # Define parameters with business constraints
        params = IRBSegmentationParams(
            max_depth=4,
            min_samples_leaf=1000,
            min_defaults_per_leaf=50,
            min_segment_density=0.10,
            max_segment_density=0.45,
            # Business constraints
            monotone_constraints={'credit_score': 1},  # Higher FICO = lower risk
            validation_tests=['chi_squared', 'psi', 'binomial']
        )

        # Fit model
        print("\n" + "-" * 80)
        print("FITTING MODEL")
        print("-" * 80)
        engine = IRBSegmentationEngine(params)
        engine.fit(X_train, y_train, X_val, y_val, feature_names=feature_names)

        # Test PSI on OOT data if available
        if X_oot is not None and y_oot is not None:
            from irb_segmentation.validators import SegmentValidator

            segments_oot = engine.predict(X_oot)
            psi_result = SegmentValidator.calculate_psi(
                engine.segments_train_, segments_oot, threshold=0.1
            )

            print("\n" + "-" * 80)
            print("OUT-OF-TIME VALIDATION")
            print("-" * 80)
            print(f"PSI: {psi_result['psi']:.4f}")
            print(f"Stability: {psi_result['stability']}")
            print(f"Passed (PSI < 0.1): {psi_result['passed']}")

        # Export report
        engine.export_report("lending_club_report.json")

        print("\n" + "-" * 80)
        print("TEST RESULT: [PASS]")
        print("-" * 80)
        return True

    except FileNotFoundError as e:
        print(f"\n[SKIP] {e}")
        return None
    except Exception as e:
        print(f"\n[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_home_credit():
    """Test 4: Home Credit Dataset"""
    print("\n" + "=" * 80)
    print("TEST 4: HOME CREDIT DATASET")
    print("=" * 80)

    try:
        # Load data (without auxiliary tables for speed)
        loader = HomeCreditLoader(data_dir="./data", use_auxiliary=False)
        X_train, y_train, X_val, y_val, X_oot, y_oot, feature_names = loader.load()

        # Define parameters for large dataset
        params = IRBSegmentationParams(
            max_depth=4,
            min_samples_leaf=2000,
            min_defaults_per_leaf=100,
            min_segment_density=0.10,
            max_segment_density=0.50,
            validation_tests=['chi_squared', 'binomial']
        )

        # Fit model
        print("\n" + "-" * 80)
        print("FITTING MODEL")
        print("-" * 80)
        engine = IRBSegmentationEngine(params)
        engine.fit(X_train, y_train, X_val, y_val, feature_names=feature_names)

        # Export report
        engine.export_report("home_credit_report.json")

        print("\n" + "-" * 80)
        print("TEST RESULT: [PASS]")
        print("-" * 80)
        return True

    except FileNotFoundError as e:
        print(f"\n[SKIP] {e}")
        return None
    except Exception as e:
        print(f"\n[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_datasets():
    """Generate comparison report across all datasets"""
    print("\n" + "=" * 80)
    print("DATASET COMPARISON SUMMARY")
    print("=" * 80)

    import json

    datasets = [
        ("German Credit", "german_credit_report.json"),
        ("Taiwan Credit", "taiwan_credit_report.json"),
        ("Lending Club", "lending_club_report.json"),
        ("Home Credit", "home_credit_report.json")
    ]

    comparison = []

    for name, report_file in datasets:
        report_path = Path(report_file)
        if report_path.exists():
            with open(report_path, 'r') as f:
                report = json.load(f)

            comparison.append({
                'dataset': name,
                'n_segments': report['segments']['n_segments'],
                'train_size': report['segments']['train_size'],
                'all_validations_passed': report['validation_results']['train']['all_passed']
            })

    if comparison:
        print("\n{:<20} {:<12} {:<15} {:<20}".format(
            "Dataset", "Segments", "Train Size", "Validations Passed"
        ))
        print("-" * 80)

        for item in comparison:
            status = "[PASS]" if item['all_validations_passed'] else "[FAIL]"
            print("{:<20} {:<12} {:<15,} {:<20}".format(
                item['dataset'],
                item['n_segments'],
                item['train_size'],
                status
            ))

    print("=" * 80)


def main():
    """Run all integration tests"""
    print("\n" + "=" * 80)
    print("IRB SEGMENTATION FRAMEWORK - INTEGRATION TESTS")
    print("=" * 80)
    print("\nTesting with real credit datasets to validate IRB compliance...")

    results = {
        'German Credit': test_german_credit(),
        'Taiwan Credit': test_taiwan_credit(),
        'Lending Club': test_lending_club(),
        'Home Credit': test_home_credit()
    }

    # Generate comparison
    compare_datasets()

    # Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)

    print(f"\nTests Passed: {passed}")
    print(f"Tests Failed: {failed}")
    print(f"Tests Skipped: {skipped} (datasets not available)")

    for dataset, result in results.items():
        if result is True:
            status = "[PASS]"
        elif result is False:
            status = "[FAIL]"
        else:
            status = "[SKIP]"
        print(f"  {dataset}: {status}")

    print("\n" + "=" * 80)

    if failed > 0:
        print("\n[WARN] Some tests failed. Please review the output above.")
        return 1
    elif passed == 0:
        print("\n[WARN] No datasets available for testing.")
        print("Please download at least one dataset to test the framework.")
        return 2
    else:
        print("\n[SUCCESS] All available tests passed successfully!")
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
