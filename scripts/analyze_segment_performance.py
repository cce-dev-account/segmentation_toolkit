"""
Segment Performance Analysis and Visualization

Creates comprehensive visualizations showing:
1. Calibration plots by segment (predicted vs actual)
2. ROC curves and performance metrics by segment
3. Feature importance analysis per segment
4. Performance comparison dashboard

Usage:
    python scripts/analyze_segment_performance.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score, brier_score_loss
)
from scipy import stats


class SegmentPerformanceAnalyzer:
    """
    Analyze and visualize segment performance.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, segments: np.ndarray,
                 feature_names: List[str]):
        """
        Initialize analyzer.

        Args:
            X: Feature matrix
            y: Binary outcomes (0/1)
            segments: Segment assignments
            feature_names: List of feature names
        """
        self.X = X
        self.y = y
        self.segments = segments
        self.feature_names = feature_names
        self.unique_segments = np.unique(segments)

        # Fit models for each segment
        self.segment_models = {}
        self.segment_predictions = {}
        self._fit_segment_models()

    def _fit_segment_models(self):
        """Fit simple logistic regression for each segment."""
        print("\nFitting segment-specific models...")

        for seg in self.unique_segments:
            mask = self.segments == seg
            X_seg = self.X[mask]
            y_seg = self.y[mask]

            if len(np.unique(y_seg)) < 2 or len(y_seg) < 50:
                print(f"  Segment {seg}: Skipped (insufficient data or no variation)")
                continue

            # Fit logistic regression
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_seg, y_seg)

            # Store model and predictions
            self.segment_models[seg] = model
            self.segment_predictions[seg] = model.predict_proba(X_seg)[:, 1]

            print(f"  Segment {seg}: Model fitted ({len(y_seg):,} observations)")

    def plot_calibration_curves(self, n_bins: int = 10,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Create calibration plot showing predicted vs actual by segment.

        Args:
            n_bins: Number of bins for predicted probabilities
            save_path: Optional path to save figure

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        colors = plt.cm.Set1(np.linspace(0, 1, len(self.segment_models)))

        for idx, (seg, model) in enumerate(self.segment_models.items()):
            mask = self.segments == seg
            y_true = self.y[mask]
            y_pred = self.segment_predictions[seg]

            # Bin predictions
            bins = np.linspace(0, 1, n_bins + 1)
            bin_indices = np.digitize(y_pred, bins) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)

            # Calculate mean predicted and actual per bin
            pred_means = []
            actual_means = []
            bin_centers = []

            for i in range(n_bins):
                bin_mask = bin_indices == i
                if np.sum(bin_mask) > 0:
                    pred_means.append(y_pred[bin_mask].mean())
                    actual_means.append(y_true[bin_mask].mean())
                    bin_centers.append((bins[i] + bins[i+1]) / 2)

            # Plot calibration curve
            ax.plot(pred_means, actual_means, 'o-',
                   color=colors[idx], linewidth=2, markersize=8,
                   label=f'Segment {seg} (n={len(y_true):,})', alpha=0.8)

            # Add confidence intervals
            for pred, actual in zip(pred_means, actual_means):
                bin_mask = (y_pred >= pred - 0.05) & (y_pred < pred + 0.05)
                n = np.sum(bin_mask)
                if n > 0:
                    ci = 1.96 * np.sqrt(actual * (1 - actual) / n)
                    ax.plot([pred, pred], [actual - ci, actual + ci],
                           color=colors[idx], alpha=0.3, linewidth=1)

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration', alpha=0.5)

        # Formatting
        ax.set_xlabel('Predicted Default Probability', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual Default Rate', fontsize=12, fontweight='bold')
        ax.set_title('Calibration Plot by Segment\n(Predicted vs Actual Default Rates)',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', frameon=True, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

        # Add text box with interpretation
        text = "Interpretation:\n"
        text += "• Points on diagonal = well-calibrated\n"
        text += "• Separated lines = distinct risk profiles\n"
        text += "• Higher lines = higher-risk segments"
        ax.text(0.02, 0.98, text, transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
               fontsize=9, family='monospace')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nCalibration plot saved: {save_path}")

        return fig

    def plot_roc_curves(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot ROC curves for each segment.

        Args:
            save_path: Optional path to save figure

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.cm.Set1(np.linspace(0, 1, len(self.segment_models)))
        auc_scores = {}

        for idx, (seg, model) in enumerate(self.segment_models.items()):
            mask = self.segments == seg
            y_true = self.y[mask]
            y_pred = self.segment_predictions[seg]

            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred)
            auc_scores[seg] = auc

            # Plot
            ax.plot(fpr, tpr, color=colors[idx], linewidth=2,
                   label=f'Segment {seg} (AUC={auc:.3f})', alpha=0.8)

        # Diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random', alpha=0.5)

        # Formatting
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curves by Segment\n(Discriminative Power within Segments)',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', frameon=True, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

        # Add AUC interpretation
        mean_auc = np.mean(list(auc_scores.values()))
        text = f"Mean AUC: {mean_auc:.3f}\n\n"
        text += "Interpretation:\n"
        text += "• AUC > 0.7 = good separation\n"
        text += "• Higher = better prediction\n"
        text += "• Different AUCs = segments\n  have different predictability"
        ax.text(0.98, 0.02, text, transform=ax.transAxes,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
               fontsize=9, family='monospace')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved: {save_path}")

        return fig

    def plot_feature_importance_heatmap(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create heatmap of feature coefficients by segment.

        Args:
            save_path: Optional path to save figure

        Returns:
            Figure object
        """
        # Extract coefficients
        coef_data = []
        for seg in sorted(self.segment_models.keys()):
            model = self.segment_models[seg]
            coef_data.append(model.coef_[0])

        coef_df = pd.DataFrame(
            coef_data,
            columns=self.feature_names,
            index=[f'Segment {seg}' for seg in sorted(self.segment_models.keys())]
        ).T

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(coef_df, annot=True, fmt='.3f', cmap='RdYlGn_r',
                   center=0, cbar_kws={'label': 'Logistic Regression Coefficient'},
                   linewidths=0.5, ax=ax)

        ax.set_title('Feature Importance by Segment\n(Logistic Regression Coefficients)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Segment', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=12, fontweight='bold')

        # Rotate labels
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
        plt.setp(ax.get_yticklabels(), rotation=0)

        # Add interpretation
        text = "Interpretation:\n"
        text += "• Red = increases default risk\n"
        text += "• Green = decreases default risk\n"
        text += "• Different patterns = segments\n  have different risk drivers"
        fig.text(0.02, 0.02, text,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
                fontsize=9, family='monospace')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance heatmap saved: {save_path}")

        return fig

    def plot_performance_comparison(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create multi-panel performance comparison dashboard.

        Args:
            save_path: Optional path to save figure

        Returns:
            Figure object
        """
        n_segments = len(self.segment_models)
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Panel 1: Confusion matrices
        for idx, (seg, model) in enumerate(self.segment_models.items()):
            if idx >= 6:  # Max 6 segments to fit
                break

            row = idx // 3
            col = idx % 3
            ax = fig.add_subplot(gs[row, col])

            mask = self.segments == seg
            y_true = self.y[mask]
            y_pred = (self.segment_predictions[seg] > 0.5).astype(int)

            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       cbar=False, square=True)

            # Calculate metrics
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            ax.set_title(f'Segment {seg}\nPrec={precision:.2f}, Rec={recall:.2f}, F1={f1:.2f}',
                        fontsize=10, fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_xticklabels(['Non-Default', 'Default'])
            ax.set_yticklabels(['Non-Default', 'Default'])

        fig.suptitle('Segment Performance Comparison Dashboard',
                    fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance comparison saved: {save_path}")

        return fig

    def plot_lift_curves(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot lift curves showing improvement over baseline.

        Args:
            save_path: Optional path to save figure

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        colors = plt.cm.Set1(np.linspace(0, 1, len(self.segment_models)))
        overall_default_rate = self.y.mean()

        for idx, (seg, model) in enumerate(self.segment_models.items()):
            mask = self.segments == seg
            y_true = self.y[mask]
            y_pred = self.segment_predictions[seg]

            # Sort by predicted probability
            sorted_indices = np.argsort(y_pred)[::-1]
            y_true_sorted = y_true[sorted_indices]

            # Calculate cumulative default rate
            cumulative_defaults = np.cumsum(y_true_sorted)
            cumulative_total = np.arange(1, len(y_true_sorted) + 1)
            cumulative_default_rate = cumulative_defaults / cumulative_total

            # Calculate lift
            lift = cumulative_default_rate / overall_default_rate

            # Sample every N points for plotting
            step = max(1, len(lift) // 100)
            percentiles = (cumulative_total / len(y_true_sorted) * 100)[::step]
            lift_sampled = lift[::step]

            # Plot
            ax.plot(percentiles, lift_sampled, color=colors[idx], linewidth=2,
                   label=f'Segment {seg}', alpha=0.8)

        # Baseline
        ax.axhline(y=1, color='k', linestyle='--', linewidth=2,
                  label='Baseline (No Model)', alpha=0.5)

        # Formatting
        ax.set_xlabel('Percentage of Population (Ranked by Risk)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Lift (vs Overall Default Rate)', fontsize=12, fontweight='bold')
        ax.set_title('Lift Curves by Segment\n(Improvement Over Baseline)',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', frameon=True, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)

        # Add interpretation
        text = "Interpretation:\n"
        text += "• Lift > 1 = better than random\n"
        text += "• Higher lift = better targeting\n"
        text += "• Area under curve = overall value"
        ax.text(0.02, 0.98, text, transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
               fontsize=9, family='monospace')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Lift curves saved: {save_path}")

        return fig

    def generate_all_visualizations(self, output_prefix: str):
        """
        Generate all performance visualizations.

        Args:
            output_prefix: Prefix for output files
        """
        print("\n" + "=" * 70)
        print("GENERATING SEGMENT PERFORMANCE VISUALIZATIONS")
        print("=" * 70)

        # 1. Calibration plot
        print("\n1. Creating calibration plot...")
        self.plot_calibration_curves(save_path=f"{output_prefix}_calibration.png")
        plt.close()

        # 2. ROC curves
        print("\n2. Creating ROC curves...")
        self.plot_roc_curves(save_path=f"{output_prefix}_roc_curves.png")
        plt.close()

        # 3. Feature importance
        print("\n3. Creating feature importance heatmap...")
        self.plot_feature_importance_heatmap(save_path=f"{output_prefix}_feature_importance.png")
        plt.close()

        # 4. Performance comparison
        print("\n4. Creating performance comparison dashboard...")
        self.plot_performance_comparison(save_path=f"{output_prefix}_performance_comparison.png")
        plt.close()

        # 5. Lift curves
        print("\n5. Creating lift curves...")
        self.plot_lift_curves(save_path=f"{output_prefix}_lift_curves.png")
        plt.close()

        print("\n" + "=" * 70)
        print("ALL VISUALIZATIONS GENERATED")
        print("=" * 70)
        print(f"\nFiles saved with prefix: {output_prefix}")
        print("\nGenerated files:")
        print(f"  {output_prefix}_calibration.png")
        print(f"  {output_prefix}_roc_curves.png")
        print(f"  {output_prefix}_feature_importance.png")
        print(f"  {output_prefix}_performance_comparison.png")
        print(f"  {output_prefix}_lift_curves.png")


def main():
    """Command-line usage example."""
    print("\n" + "=" * 70)
    print("SEGMENT PERFORMANCE ANALYZER")
    print("=" * 70)
    print("\nThis tool analyzes segment performance and creates visualizations.")
    print("\nUsage in Python:")
    print("""
    from scripts.analyze_segment_performance import SegmentPerformanceAnalyzer

    # After loading data and segments
    analyzer = SegmentPerformanceAnalyzer(X, y, segments, feature_names)
    analyzer.generate_all_visualizations('output/performance')
    """)


if __name__ == '__main__':
    main()
