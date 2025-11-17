"""
Apply Segment Modifications and Re-validate

Allows users to:
1. Merge similar segments
2. Add forced split points
3. Adjust parameters
4. Re-run validation

Usage:
    python apply_modifications.py modify_segments.json
"""

import numpy as np
import pandas as pd
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from irb_segmentation import IRBSegmentationParams, IRBSegmentationEngine
from irb_segmentation.logger import get_logger
from sklearn.model_selection import train_test_split

# Module-level logger
logger = get_logger(__name__)


def load_lending_club_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load Lending Club data (same as test script).

    Returns:
        X_train, y_train, X_val, y_val, feature_names
    """
    logger.info("\nLoading Lending Club data...")

    df = pd.read_csv("data/lending_club_test.csv", low_memory=False)
    logger.info(f"Loaded {len(df):,} rows")

    # Create default target
    df['default'] = df['loan_status'].apply(
        lambda x: 1 if 'Charged Off' in str(x) or 'Default' in str(x) else 0
    )

    # Select numerical features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'default'][:14]

    X = df[numeric_cols].fillna(0).values
    y = df['default'].values

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    logger.info(f"Train: {len(X_train):,} obs, {y_train.sum():,} defaults ({y_train.mean():.4f})")
    logger.info(f"Val: {len(X_val):,} obs, {y_val.sum():,} defaults ({y_val.mean():.4f})")

    return X_train, y_train, X_val, y_val, numeric_cols


def apply_modifications(modification_file: str):
    """
    Apply modifications from JSON file and re-validate.

    Args:
        modification_file: Path to modification JSON
    """
    logger.info("=" * 80)
    logger.info("APPLYING SEGMENT MODIFICATIONS")
    logger.info("=" * 80)

    # Load modification file
    if not Path(modification_file).exists():
        logger.info(f"\nError: {modification_file} not found")
        return

    with open(modification_file, 'r') as f:
        config = json.load(f)

    mods = config['modifications']

    logger.info("\nModifications requested:")
    logger.info(f"  Merge segments: {mods['merge_segments']['value']}")
    logger.info(f"  Forced splits: {mods['forced_splits']['value']}")
    logger.info(f"  Parameter changes: {len([k for k, v in mods['parameter_changes'].items() if k != 'description'])}")

    # Load data
    X_train, y_train, X_val, y_val, feature_names = load_lending_club_data()

    # Create parameters from modification file
    param_changes = mods['parameter_changes']
    params = IRBSegmentationParams(
        max_depth=param_changes.get('max_depth', 5),
        min_samples_split=param_changes.get('min_samples_leaf', 10000) * 2,
        min_samples_leaf=param_changes.get('min_samples_leaf', 10000),
        min_defaults_per_leaf=param_changes.get('min_defaults_per_leaf', 500),
        min_segment_density=param_changes.get('min_segment_density', 0.05),
        max_segment_density=param_changes.get('max_segment_density', 0.40),
        forced_splits=mods['forced_splits']['value']
    )

    logger.info("-" * 80)
    logger.info("FITTING MODEL WITH NEW PARAMETERS")
    logger.info("-" * 80)

    # Fit model
    engine = IRBSegmentationEngine(params)
    engine.fit(X_train, y_train, X_val, y_val, feature_names=feature_names)

    # Apply manual merges if specified
    merge_pairs = mods['merge_segments']['value']
    if merge_pairs:
        logger.info("-" * 80)
        logger.info("APPLYING MANUAL MERGES")
        logger.info("-" * 80)

        segments = engine.segments_train_.copy()

        for seg1, seg2 in merge_pairs:
            logger.info(f"\nMerging Segment {seg2} into Segment {seg1}")

            # Check if segments exist
            if seg1 not in segments and seg2 not in segments:
                logger.info(f"  [SKIP] Segments {seg1} and {seg2} not found")
                continue

            # Merge seg2 into seg1
            segments[segments == seg2] = seg1
            logger.info(f"  [OK] Merged")

        # Re-label segments sequentially
        unique_segs = sorted(np.unique(segments))
        segment_map = {old: new for new, old in enumerate(unique_segs)}
        segments = np.array([segment_map[s] for s in segments])

        engine.segments_train_ = segments

        # Re-validate
        logger.info("-" * 80)
        logger.info("RE-VALIDATING AFTER MERGES")
        logger.info("-" * 80)

        from irb_segmentation.validators import SegmentValidator

        engine.validation_results_['train'] = SegmentValidator.run_all_validations(
            segments, y_train, params
        )

        # Update validation set segments
        engine.segments_val_ = engine.predict(X_val)
        engine.validation_results_['validation'] = SegmentValidator.run_all_validations(
            engine.segments_val_, y_val, params, reference_segments=segments
        )

        # Print updated summary
        engine._print_fit_summary()

    # Export new report
    output_file = modification_file.replace('.json', '_result.json')
    engine.export_report(output_file)

    logger.info("=" * 80)
    logger.info("MODIFICATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nNew report saved: {output_file}")
    logger.info("\nCompare with original:")
    logger.info(f"  Original: lending_club_full_report.json")
    logger.info(f"  Modified: {output_file}")


def create_sample_modification():
    """Create sample modification file for Lending Club."""
    logger.info("=" * 80)
    logger.info("CREATING SAMPLE MODIFICATION FILE")
    logger.info("=" * 80)

    template = {
        "metadata": {
            "instructions": "Modify segments by specifying merges or forced splits, then run: python apply_modifications.py",
            "dataset": "Lending Club Full (2.26M observations)",
            "original_segments": 7,
            "note": "This template merges Segments 1 and 4 (similar default rates)"
        },
        "modifications": {
            "merge_segments": {
                "description": "List segment pairs to merge. Format: [[seg1, seg2], [seg3, seg4]]",
                "note": "Merging segments 1 (7.98% PD) and 4 (8.35% PD) - only 0.37pp difference",
                "value": [[1, 4]]
            },
            "forced_splits": {
                "description": "Add forced split points. Format: {feature_name: threshold}",
                "note": "Example: Split at FICO 650 or DTI 0.35 if those features exist",
                "value": {}
            },
            "parameter_changes": {
                "description": "Modify IRB parameters and re-fit (changes here trigger full re-training)",
                "max_depth": 5,
                "min_samples_leaf": 10000,
                "min_defaults_per_leaf": 500,
                "min_segment_density": 0.05,
                "max_segment_density": 0.40
            }
        },
        "current_segments": {
            "0": {"default_rate": 0.0216, "n_observations": 183082, "risk": "Very Low"},
            "1": {"default_rate": 0.0798, "n_observations": 111761, "risk": "Low"},
            "2": {"default_rate": 0.0509, "n_observations": 254388, "risk": "Low"},
            "3": {"default_rate": 0.1108, "n_observations": 272580, "risk": "Medium"},
            "4": {"default_rate": 0.0835, "n_observations": 126561, "risk": "Low-Medium"},
            "5": {"default_rate": 0.1620, "n_observations": 415442, "risk": "High"},
            "6": {"default_rate": 0.2500, "n_observations": 218676, "risk": "Very High"}
        }
    }

    output_file = "modify_segments_sample.json"
    with open(output_file, 'w') as f:
        json.dump(template, f, indent=2)

    logger.info(f"\nSample modification file created: {output_file}")
    logger.info("\nThis sample will:")
    logger.info("  1. Merge Segments 1 and 4 (similar PD rates)")
    logger.info("  2. Keep all other parameters the same")
    logger.info("\nTo apply:")
    logger.info(f"  python apply_modifications.py {output_file}")
    logger.info("\nTo customize:")
    logger.info(f"  1. Edit {output_file}")
    logger.info(f"  2. Modify merge_segments, forced_splits, or parameter_changes")
    logger.info(f"  3. Run: python apply_modifications.py {output_file}")


def compare_reports(original: str, modified: str):
    """
    Compare two segmentation reports.

    Args:
        original: Path to original report
        modified: Path to modified report
    """
    logger.info("=" * 80)
    logger.info("COMPARING SEGMENTATION REPORTS")
    logger.info("=" * 80)

    with open(original, 'r') as f:
        orig = json.load(f)

    with open(modified, 'r') as f:
        mod = json.load(f)

    orig_stats = orig['segment_statistics']
    mod_stats = mod['segment_statistics']

    logger.info(f"\n{'Metric':<30}{'Original':<20}{'Modified':<20}{'Change':<15}")
    logger.info("-" * 85)

    # Number of segments
    n_orig = len(orig_stats)
    n_mod = len(mod_stats)
    logger.info(f"{'Number of Segments':<30}{n_orig:<20}{n_mod:<20}{n_mod - n_orig:+d}")

    # PD range
    orig_pds = [s['default_rate'] for s in orig_stats.values()]
    mod_pds = [s['default_rate'] for s in mod_stats.values()]

    orig_range = max(orig_pds) - min(orig_pds)
    mod_range = max(mod_pds) - min(mod_pds)
    logger.info(f"{'PD Range':<30}{orig_range:<20.2%}{mod_range:<20.2%}{mod_range - orig_range:+.2%}")

    # Validation results
    orig_pass = orig['validation_results']['train']['all_passed']
    mod_pass = mod['validation_results']['train']['all_passed']
    logger.info(f"{'Training Validation':<30}{str(orig_pass):<20}{str(mod_pass):<20}{'Same' if orig_pass == mod_pass else 'Changed'}")

    orig_val_pass = orig['validation_results']['validation']['all_passed']
    mod_val_pass = mod['validation_results']['validation']['all_passed']
    logger.info(f"{'Validation Set':<30}{str(orig_val_pass):<20}{str(mod_val_pass):<20}{'Same' if orig_val_pass == mod_val_pass else 'Changed'}")

    logger.info("=" * 80)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        logger.info("=" * 80)
        logger.info("SEGMENT MODIFICATION TOOL")
        logger.info("=" * 80)
        logger.info("\nUsage:")
        logger.info("  python apply_modifications.py <modification_file.json>")
        logger.info("\nCommands:")
        logger.info("  python apply_modifications.py --create-sample")
        logger.info("    Creates a sample modification file")
        logger.info("\n  python apply_modifications.py --compare original.json modified.json")
        logger.info("    Compares two report files")
        logger.info("\nExample workflow:")
        logger.info("  1. python apply_modifications.py --create-sample")
        logger.info("  2. Edit modify_segments_sample.json")
        logger.info("  3. python apply_modifications.py modify_segments_sample.json")
        logger.info("  4. python apply_modifications.py --compare lending_club_full_report.json modify_segments_sample_result.json")
        return

    if sys.argv[1] == '--create-sample':
        create_sample_modification()

    elif sys.argv[1] == '--compare':
        if len(sys.argv) < 4:
            logger.info("Error: --compare requires two report files")
            logger.info("Usage: python apply_modifications.py --compare original.json modified.json")
            return
        compare_reports(sys.argv[2], sys.argv[3])

    else:
        apply_modifications(sys.argv[1])


if __name__ == "__main__":
    main()
