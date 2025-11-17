"""
IRB Segmentation Demo Setup Script

Interactive script to:
- Check dependencies
- Guide dataset downloads
- Validate data files
- Recommend which demo to run
"""

import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    print("=" * 80)
    print("CHECKING DEPENDENCIES")
    print("=" * 80)
    print()

    required = ['numpy', 'pandas', 'sklearn', 'scipy', 'yaml', 'openpyxl']
    missing = []

    for package in required:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            elif package == 'yaml':
                __import__('yaml')
            else:
                __import__(package)
            print(f"✓ {package:<15} installed")
        except ImportError:
            print(f"✗ {package:<15} MISSING")
            missing.append(package)

    print()

    if missing:
        print("Missing packages detected!")
        print("Install with: pip install -r requirements.txt")
        print()
        return False
    else:
        print("All dependencies installed! ✓")
        print()
        return True


def check_datasets():
    """Check which datasets are available"""
    print("=" * 80)
    print("CHECKING DATASETS")
    print("=" * 80)
    print()

    data_dir = Path("data")
    datasets = {
        'german': {'file': None, 'status': 'Auto-downloads', 'size': '1K rows'},
        'taiwan': {'file': data_dir / 'UCI_Credit_Card.csv', 'status': 'Not found', 'size': '30K rows'},
        'lending_club': {'file': data_dir / 'lending_club_data.csv', 'status': 'Not found', 'size': '1.3M rows'}
    }

    # Check German Credit (auto-downloads)
    print("1. German Credit Dataset")
    print(f"   Status: Auto-downloads from UCI (no setup needed)")
    print(f"   Size: 1K rows")
    print(f"   Config: demo_german.yaml")
    print()

    # Check Taiwan Credit
    print("2. Taiwan Credit Dataset")
    if datasets['taiwan']['file'].exists():
        print(f"   Status: ✓ Found at {datasets['taiwan']['file']}")
        datasets['taiwan']['status'] = 'Ready'
    else:
        print(f"   Status: ✗ Not found")
        print(f"   Download: See DATASET_DOWNLOAD.md")
    print(f"   Size: 30K rows")
    print(f"   Config: demo_taiwan.yaml")
    print()

    # Check Lending Club
    print("3. Lending Club Dataset")
    if datasets['lending_club']['file'].exists():
        print(f"   Status: ✓ Found at {datasets['lending_club']['file']}")
        datasets['lending_club']['status'] = 'Ready'
    else:
        print(f"   Status: ✗ Not found")
        print(f"   Download: See DATASET_DOWNLOAD.md")
    print(f"   Size: 1.3M rows (temporal split: 95K training)")
    print(f"   Config: demo_lending_club.yaml")
    print()

    return datasets


def recommend_demo(datasets):
    """Recommend which demo to run"""
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print()

    if datasets['lending_club']['status'] == 'Ready':
        print("Recommended: Start with PRODUCTION SCALE demo")
        print("  python run_demo.py demo_lending_club.yaml")
        print()
        print("This will:")
        print("  - Use temporal split (2007-2012 training, 2013-2014 validation, 2015+ OOT)")
        print("  - Generate production-ready segments")
        print("  - Create all output formats (rules, Excel, HTML, JSON)")
        print("  - Run in ~2-3 minutes")
    elif datasets['taiwan']['status'] == 'Ready':
        print("Recommended: Start with MEDIUM SCALE demo")
        print("  python run_demo.py demo_taiwan.yaml")
        print()
        print("This will:")
        print("  - Use 30K credit card records")
        print("  - Generate IRB-compliant segments")
        print("  - Create all output formats")
        print("  - Run in ~30 seconds")
    else:
        print("Recommended: Start with QUICK START demo")
        print("  python run_demo.py demo_german.yaml")
        print()
        print("This will:")
        print("  - Auto-download German Credit dataset (1K rows)")
        print("  - Generate segments in ~5 seconds")
        print("  - Create all output formats")
        print("  - No manual download needed!")

    print()
    print("For larger datasets, see DATASET_DOWNLOAD.md for download instructions.")
    print()


def main():
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "IRB SEGMENTATION DEMO SETUP" + " " * 31 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    # Check dependencies
    deps_ok = check_dependencies()

    if not deps_ok:
        print("Please install dependencies first, then run this script again.")
        sys.exit(1)

    # Check datasets
    datasets = check_datasets()

    # Recommend demo
    recommend_demo(datasets)

    print("=" * 80)
    print("Setup complete! Run the recommended command above to start.")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
