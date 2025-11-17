"""
Simple Lending Club test without sampling
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from irb_segmentation import IRBSegmentationParams, IRBSegmentationEngine

print("Loading Lending Club data (ALL observations - ~2.26M rows)...")
print("This may take a few minutes...")

# Read all data
df = pd.read_csv("data/lending_club_test.csv", low_memory=False)
print(f"Loaded {len(df):,} rows")

# Simple preprocessing - just get the columns we need
target_col = 'loan_status'
if target_col not in df.columns:
    print(f"Available columns: {list(df.columns)[:20]}")
    sys.exit(1)

# Create default binary (Charged Off / Default = 1, Fully Paid = 0)
df['default'] = df[target_col].apply(
    lambda x: 1 if 'Charged Off' in str(x) or 'Default' in str(x) else 0
)

print(f"Default rate: {df['default'].mean():.4f}")

# Select numerical features
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != 'default'][:14]  # Take first 14

X = df[numeric_cols].fillna(0).values
y = df['default'].values

# Split
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTrain: {len(X_train)} obs, {y_train.sum()} defaults ({y_train.mean():.4f})")
print(f"Val: {len(X_val)} obs, {y_val.sum()} defaults ({y_val.mean():.4f})")

# Fit model with parameters for large dataset
params = IRBSegmentationParams(
    max_depth=5,  # Allow deeper tree for more granular segments
    min_samples_split=20000,  # Much larger minimum for 2M+ dataset
    min_samples_leaf=10000,   # Scale up proportionally
    min_defaults_per_leaf=500,  # Higher minimum defaults
    min_segment_density=0.05,   # Allow smaller segments (5% of 2M = 100K)
    max_segment_density=0.40    # Prevent any single segment dominating
)

engine = IRBSegmentationEngine(params)
engine.fit(X_train, y_train, X_val, y_val, feature_names=numeric_cols)

engine.export_report("lending_club_full_report.json")
print("\n[SUCCESS] Lending Club FULL dataset test passed!")
print(f"Processed {len(df):,} total observations")
