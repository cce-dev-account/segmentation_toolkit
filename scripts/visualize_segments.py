"""
Segment Visualization Module

Generates comprehensive visualizations for segment characterization including:
- Feature distributions by segment
- Box plots comparing segments
- Radar charts
- 2D scatter plots
- Default rate drivers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Tuple
from pathlib import Path


class SegmentVisualizer:
    """
    Create comprehensive visualizations for segment analysis.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target array (n_samples,)
        segments: Segment assignments (n_samples,)
        feature_names: List of feature names
        output_dir: Directory to save visualizations
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        segments: np.ndarray,
        feature_names: List[str],
        output_dir: str = "output"
    ):
        self.X = X
        self.y = y
        self.segments = segments
        self.feature_names = feature_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.unique_segments = sorted(np.unique(segments))
        self.n_segments = len(self.unique_segments)

        # Define color palette for consistency
        self.colors = plt.cm.Set2(np.linspace(0, 1, self.n_segments))

    def plot_feature_distributions(
        self,
        features: Optional[List[str]] = None,
        n_features: int = 6,
        output_file: str = "segment_viz_distributions.png"
    ) -> str:
        """
        Plot overlaid distributions of key features by segment.

        Args:
            features: List of feature names to plot. If None, selects top n_features
            n_features: Number of features to plot if features is None
            output_file: Output filename

        Returns:
            Path to saved file
        """
        print(f"\nGenerating feature distribution plots...")

        # Select features to plot
        if features is None:
            # Select features with highest variance or most discriminative
            feature_indices = self._select_top_features(n_features)
            features = [self.feature_names[i] for i in feature_indices]
        else:
            feature_indices = [self.feature_names.index(f) for f in features]

        n_features_plot = len(features)
        n_cols = 3
        n_rows = (n_features_plot + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for idx, (feat_idx, feat_name) in enumerate(zip(feature_indices, features)):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            # Plot distribution for each segment
            for seg_idx, seg in enumerate(self.unique_segments):
                mask = self.segments == seg
                data = self.X[mask, feat_idx]

                # Remove outliers for better visualization
                q1, q99 = np.percentile(data, [1, 99])
                data_clean = data[(data >= q1) & (data <= q99)]

                ax.hist(
                    data_clean,
                    bins=30,
                    alpha=0.5,
                    label=f'Seg {seg}',
                    color=self.colors[seg_idx],
                    density=True
                )

            ax.set_xlabel(feat_name)
            ax.set_ylabel('Density')
            ax.set_title(f'{feat_name} Distribution')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Hide empty subplots
        for idx in range(n_features_plot, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

        plt.suptitle('Feature Distributions by Segment', fontsize=16, y=1.00)
        plt.tight_layout()

        output_path = self.output_dir / output_file
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {output_path}")
        return str(output_path)

    def plot_boxplots_by_segment(
        self,
        features: Optional[List[str]] = None,
        n_features: int = 6,
        output_file: str = "segment_viz_boxplots.png"
    ) -> str:
        """
        Create side-by-side box plots for key features across segments.

        Args:
            features: List of feature names to plot
            n_features: Number of features if features is None
            output_file: Output filename

        Returns:
            Path to saved file
        """
        print(f"\nGenerating box plots...")

        # Select features
        if features is None:
            feature_indices = self._select_top_features(n_features)
            features = [self.feature_names[i] for i in feature_indices]
        else:
            feature_indices = [self.feature_names.index(f) for f in features]

        n_features_plot = len(features)
        n_cols = 3
        n_rows = (n_features_plot + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for idx, (feat_idx, feat_name) in enumerate(zip(feature_indices, features)):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            # Prepare data for box plot
            data_by_segment = []
            labels = []

            for seg in self.unique_segments:
                mask = self.segments == seg
                data = self.X[mask, feat_idx]
                data_by_segment.append(data)
                labels.append(f'Seg {seg}')

            bp = ax.boxplot(
                data_by_segment,
                labels=labels,
                patch_artist=True,
                showfliers=False  # Hide outliers for clarity
            )

            # Color boxes
            for patch, color in zip(bp['boxes'], self.colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_ylabel(feat_name)
            ax.set_title(f'{feat_name} by Segment')
            ax.grid(True, alpha=0.3, axis='y')

        # Hide empty subplots
        for idx in range(n_features_plot, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

        plt.suptitle('Feature Comparison Across Segments', fontsize=16, y=1.00)
        plt.tight_layout()

        output_path = self.output_dir / output_file
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {output_path}")
        return str(output_path)

    def plot_radar_chart(
        self,
        features: Optional[List[str]] = None,
        n_features: int = 8,
        output_file: str = "segment_viz_radar.png"
    ) -> str:
        """
        Create radar/spider chart comparing segment profiles.

        Args:
            features: Features to include in radar chart
            n_features: Number of features if features is None
            output_file: Output filename

        Returns:
            Path to saved file
        """
        print(f"\nGenerating radar chart...")

        # Select features
        if features is None:
            feature_indices = self._select_top_features(n_features)
            features = [self.feature_names[i] for i in feature_indices]
        else:
            feature_indices = [self.feature_names.index(f) for f in features]

        n_features_plot = len(features)

        # Calculate normalized median values for each segment
        segment_profiles = []
        for seg in self.unique_segments:
            mask = self.segments == seg
            profile = []
            for feat_idx in feature_indices:
                median_val = np.median(self.X[mask, feat_idx])
                # Normalize to 0-1 based on global min/max
                global_min = np.min(self.X[:, feat_idx])
                global_max = np.max(self.X[:, feat_idx])
                normalized = (median_val - global_min) / (global_max - global_min + 1e-10)
                profile.append(normalized)
            segment_profiles.append(profile)

        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, n_features_plot, endpoint=False).tolist()
        segment_profiles = [profile + profile[:1] for profile in segment_profiles]  # Complete the circle
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        for seg_idx, (seg, profile) in enumerate(zip(self.unique_segments, segment_profiles)):
            ax.plot(
                angles,
                profile,
                'o-',
                linewidth=2,
                label=f'Segment {seg}',
                color=self.colors[seg_idx]
            )
            ax.fill(angles, profile, alpha=0.15, color=self.colors[seg_idx])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features, size=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['25%', '50%', '75%', '100%'])
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        plt.title('Segment Profile Comparison\n(Normalized Median Values)', size=14, pad=20)

        output_path = self.output_dir / output_file
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {output_path}")
        return str(output_path)

    def plot_2d_scatter(
        self,
        feature_pairs: Optional[List[Tuple[str, str]]] = None,
        output_file: str = "segment_viz_scatter_2d.png"
    ) -> str:
        """
        Create 2D scatter plots for feature pairs, colored by segment.

        Args:
            feature_pairs: List of (feature1, feature2) tuples. If None, selects interesting pairs
            output_file: Output filename

        Returns:
            Path to saved file
        """
        print(f"\nGenerating 2D scatter plots...")

        # Select interesting feature pairs if not provided
        if feature_pairs is None:
            feature_pairs = self._select_feature_pairs(n_pairs=4)

        n_pairs = len(feature_pairs)
        n_cols = 2
        n_rows = (n_pairs + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for idx, (feat1, feat2) in enumerate(feature_pairs):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            feat1_idx = self.feature_names.index(feat1)
            feat2_idx = self.feature_names.index(feat2)

            # Plot each segment
            for seg_idx, seg in enumerate(self.unique_segments):
                mask = self.segments == seg
                defaults = self.y[mask]

                # Separate defaulters and non-defaulters
                default_mask = defaults == 1
                non_default_mask = defaults == 0

                # Plot non-defaulters (smaller, more transparent)
                ax.scatter(
                    self.X[mask][non_default_mask, feat1_idx],
                    self.X[mask][non_default_mask, feat2_idx],
                    c=[self.colors[seg_idx]],
                    alpha=0.3,
                    s=20,
                    edgecolors='none'
                )

                # Plot defaulters (larger, more visible)
                ax.scatter(
                    self.X[mask][default_mask, feat1_idx],
                    self.X[mask][default_mask, feat2_idx],
                    c=[self.colors[seg_idx]],
                    alpha=0.7,
                    s=40,
                    edgecolors='black',
                    linewidths=0.5,
                    label=f'Seg {seg}' if seg_idx < len(self.unique_segments) else None
                )

            ax.set_xlabel(feat1)
            ax.set_ylabel(feat2)
            ax.set_title(f'{feat1} vs {feat2}')
            ax.grid(True, alpha=0.3)

            if idx == 0:  # Only add legend to first plot
                ax.legend(fontsize=8, loc='best')

        # Hide empty subplots
        for idx in range(n_pairs, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

        plt.suptitle('Feature Space Visualization by Segment\n(Larger points = defaults)', fontsize=14, y=1.00)
        plt.tight_layout()

        output_path = self.output_dir / output_file
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {output_path}")
        return str(output_path)

    def plot_default_rate_drivers(
        self,
        output_file: str = "segment_viz_default_drivers.png"
    ) -> str:
        """
        Plot feature importance/correlation with default rate per segment.

        Args:
            output_file: Output filename

        Returns:
            Path to saved file
        """
        print(f"\nGenerating default rate driver analysis...")

        # Calculate correlation with default for each feature in each segment
        correlations = np.zeros((self.n_segments, len(self.feature_names)))

        for seg_idx, seg in enumerate(self.unique_segments):
            mask = self.segments == seg
            X_seg = self.X[mask]
            y_seg = self.y[mask]

            for feat_idx in range(len(self.feature_names)):
                # Pearson correlation
                if len(np.unique(X_seg[:, feat_idx])) > 1:  # Check for variation
                    corr = np.corrcoef(X_seg[:, feat_idx], y_seg)[0, 1]
                    correlations[seg_idx, feat_idx] = corr if not np.isnan(corr) else 0

        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 8))

        im = ax.imshow(correlations.T, cmap='RdBu_r', aspect='auto', vmin=-0.3, vmax=0.3)

        # Set ticks
        ax.set_xticks(np.arange(self.n_segments))
        ax.set_yticks(np.arange(len(self.feature_names)))
        ax.set_xticklabels([f'Segment {seg}' for seg in self.unique_segments])
        ax.set_yticklabels(self.feature_names)

        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation with Default', rotation=270, labelpad=20)

        # Add correlation values as text
        for i in range(len(self.feature_names)):
            for j in range(self.n_segments):
                text = ax.text(
                    j, i, f'{correlations[j, i]:.2f}',
                    ha="center", va="center",
                    color="black" if abs(correlations[j, i]) < 0.15 else "white",
                    fontsize=8
                )

        ax.set_title('Feature Correlation with Default Rate by Segment\n(Red = increases default risk, Blue = decreases risk)',
                     fontsize=12, pad=20)

        plt.tight_layout()

        output_path = self.output_dir / output_file
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {output_path}")
        return str(output_path)

    def _select_top_features(self, n_features: int = 6) -> List[int]:
        """
        Select most discriminative features across segments.

        Returns:
            List of feature indices
        """
        # Calculate variance of medians across segments (features with different behavior)
        median_variance = []

        for feat_idx in range(len(self.feature_names)):
            segment_medians = []
            for seg in self.unique_segments:
                mask = self.segments == seg
                median_val = np.median(self.X[mask, feat_idx])
                segment_medians.append(median_val)

            # Variance of medians (normalized by global std)
            global_std = np.std(self.X[:, feat_idx])
            if global_std > 0:
                variance = np.var(segment_medians) / global_std
            else:
                variance = 0

            median_variance.append(variance)

        # Return indices of top n features
        top_indices = np.argsort(median_variance)[-n_features:][::-1]
        return top_indices.tolist()

    def _select_feature_pairs(self, n_pairs: int = 4) -> List[Tuple[str, str]]:
        """
        Select interesting feature pairs for scatter plots.

        Returns:
            List of (feature1, feature2) tuples
        """
        # Common important feature pairs for credit risk
        common_pairs = [
            ('credit_score', 'interest_rate'),
            ('annual_income', 'loan_amount'),
            ('dti', 'interest_rate'),
            ('credit_score', 'annual_income')
        ]

        # Filter to pairs that exist in our features
        valid_pairs = []
        for feat1, feat2 in common_pairs:
            if feat1 in self.feature_names and feat2 in self.feature_names:
                valid_pairs.append((feat1, feat2))

        # If we need more pairs or none of the common ones exist, use top features
        if len(valid_pairs) < n_pairs:
            top_features = self._select_top_features(min(6, len(self.feature_names)))
            for i in range(len(top_features) - 1):
                for j in range(i + 1, len(top_features)):
                    feat1 = self.feature_names[top_features[i]]
                    feat2 = self.feature_names[top_features[j]]
                    if (feat1, feat2) not in valid_pairs:
                        valid_pairs.append((feat1, feat2))
                    if len(valid_pairs) >= n_pairs:
                        break
                if len(valid_pairs) >= n_pairs:
                    break

        return valid_pairs[:n_pairs]

    def generate_all_visualizations(self, prefix: str = "segment_viz") -> Dict[str, str]:
        """
        Generate all visualization types.

        Args:
            prefix: Prefix for output filenames

        Returns:
            Dictionary mapping visualization type to file path
        """
        print("\n" + "=" * 70)
        print("GENERATING SEGMENT VISUALIZATIONS")
        print("=" * 70)

        outputs = {}

        # 1. Feature distributions
        outputs['distributions'] = self.plot_feature_distributions(
            output_file=f"{prefix}_distributions.png"
        )

        # 2. Box plots
        outputs['boxplots'] = self.plot_boxplots_by_segment(
            output_file=f"{prefix}_boxplots.png"
        )

        # 3. Radar chart
        outputs['radar'] = self.plot_radar_chart(
            output_file=f"{prefix}_radar.png"
        )

        # 4. 2D scatter plots
        outputs['scatter'] = self.plot_2d_scatter(
            output_file=f"{prefix}_scatter_2d.png"
        )

        # 5. Default rate drivers
        outputs['drivers'] = self.plot_default_rate_drivers(
            output_file=f"{prefix}_default_drivers.png"
        )

        print("\n" + "=" * 70)
        print("VISUALIZATION GENERATION COMPLETE")
        print("=" * 70)
        print(f"\nGenerated {len(outputs)} visualization files:")
        for viz_type, path in outputs.items():
            print(f"  {viz_type}: {path}")

        return outputs
