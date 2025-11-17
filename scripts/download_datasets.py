"""
Dataset Download Helper

Downloads datasets from direct URLs where possible.
For Kaggle datasets, provides instructions for manual download.
"""

import urllib.request
import os
from pathlib import Path

def download_taiwan_credit():
    """
    Taiwan Credit Card dataset requires Kaggle account.
    Provide download instructions.
    """
    print("\n" + "=" * 80)
    print("TAIWAN CREDIT CARD DATASET")
    print("=" * 80)
    print("\nThis dataset requires a Kaggle account to download.")
    print("\nOptions:")
    print("\n1. Manual Download:")
    print("   a. Visit: https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset")
    print("   b. Click 'Download' (requires Kaggle login)")
    print("   c. Extract and save as: ./data/UCI_Credit_Card.csv")

    print("\n2. Kaggle API (if you have credentials):")
    print("   a. Get API token from: https://www.kaggle.com/account")
    print("   b. Save to: ~/.kaggle/kaggle.json (Linux/Mac) or C:\\Users\\<you>\\.kaggle\\kaggle.json (Windows)")
    print("   c. Run: kaggle datasets download -d uciml/default-of-credit-card-clients-dataset")

    print("\n3. Alternative UCI Source:")
    print("   The dataset is also available at:")
    print("   https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients")
    print("   Download and save as: ./data/UCI_Credit_Card.csv")


def download_lending_club():
    """
    Lending Club dataset - very large, requires Kaggle.
    """
    print("\n" + "=" * 80)
    print("LENDING CLUB DATASET")
    print("=" * 80)
    print("\nThis is a large dataset (~2GB compressed, ~10GB uncompressed).")
    print("\nDownload Options:")
    print("\n1. Kaggle (Recommended):")
    print("   https://www.kaggle.com/datasets/wordsforthewise/lending-club")
    print("   Download: accepted_2007_to_2018Q4.csv")
    print("   Save to: ./data/")

    print("\n2. Official Lending Club:")
    print("   Note: Lending Club shut down in 2020, official source no longer available")

    print("\nFor testing purposes, you can:")
    print("   - Use sampling (framework supports sample_size parameter)")
    print("   - Test with German Credit (already works) or Taiwan Credit")


def download_home_credit():
    """
    Home Credit dataset - Kaggle competition.
    """
    print("\n" + "=" * 80)
    print("HOME CREDIT DATASET")
    print("=" * 80)
    print("\nThis is a Kaggle competition dataset.")
    print("\nDownload:")
    print("   1. Visit: https://www.kaggle.com/c/home-credit-default-risk/data")
    print("   2. Accept competition rules")
    print("   3. Download application_train.csv and application_test.csv")
    print("   4. Save to: ./data/")

    print("\nOptional (for advanced features):")
    print("   - bureau.csv")
    print("   - previous_application.csv")
    print("   - Other auxiliary tables")


def main():
    """Main function to show download instructions."""
    print("\n" + "=" * 80)
    print("DATASET DOWNLOAD HELPER")
    print("=" * 80)
    print("\nThe IRB Segmentation Framework is tested with 4 public datasets.")
    print("Currently, only German Credit auto-downloads.")
    print("\nFor the other datasets, follow the instructions below:\n")

    # Create data directory
    Path("./data").mkdir(exist_ok=True)
    print("[OK] Created ./data/ directory")

    # Show instructions for each dataset
    download_taiwan_credit()
    download_lending_club()
    download_home_credit()

    print("\n" + "=" * 80)
    print("CURRENT STATUS")
    print("=" * 80)

    data_dir = Path("./data")
    datasets = {
        "German Credit": "german_credit.data",
        "Taiwan Credit": "UCI_Credit_Card.csv",
        "Lending Club": "accepted_2007_to_2018Q4.csv",
        "Home Credit": "application_train.csv"
    }

    print()
    for name, filename in datasets.items():
        filepath = data_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  [YES] {name:<20} Found ({size_mb:.1f} MB)")
        else:
            print(f"  [NO]  {name:<20} Not found")

    print("\n" + "=" * 80)
    print("\nAfter downloading datasets, run:")
    print("  python test_with_real_data.py")
    print("\nTo test with only German Credit (already working), run:")
    print("  python test_with_real_data.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
