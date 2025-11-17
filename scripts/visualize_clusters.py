"""
Cluster Visualization for IRB Segmentation

Creates scatter plots and visualizations of segmentation clusters using
dimensionality reduction techniques (PCA, t-SNE, UMAP).

Usage:
    python scripts/visualize_clusters.py <baseline_report.json> [options]

    # Using fitted engine directly
    python scripts/visualize_clusters.py output/lending_club_categorical/baseline_report.json --method pca

    # Interactive mode
    python scripts/visualize_clusters.py output/lending_club_categorical/baseline_report.json --interactive
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
import argparse


class ClusterVisualizer:
    """
    Visualize segmentation clusters in 2D using dimensionality reduction.
    """

    def __init__(self, baseline_report_path: str, data_path: Optional[str] = None):
        """
        Initialize visualizer with segmentation results.

        Args:
            baseline_report_path: Path to baseline_report.json
            data_path: Optional path to original data CSV (if not using built-in loaders)
        """
        self.report_path = Path(baseline_report_path)
        self.data_path = data_path

        # Load report
        with open(self.report_path, 'r') as f:
            self.report = json.load(f)

        # Extract metadata
        self.output_dir = self.report_path.parent

        # Data storage
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.segments_train = None
        self.segments_val = None
        self.feature_names = None

        # Reduced data
        self.X_reduced = None
        self.reduction_method = None

    def load_data_from_engine(self):
        """
        Load data by re-running the segmentation.

        This reconstructs the engine to access the original data and segment assignments.
        """
        # Try to find tree structure to determine data source
        tree_path = self.output_dir / "tree_structure.json"

        if not tree_path.exists():
            raise FileNotFoundError(
                f"tree_structure.json not found in {self.output_dir}. "
                f"Please ensure all outputs were generated."
            )

        # For now, require manual data loading
        # In a full implementation, we could reconstruct from config
        raise NotImplementedError(
            "Automatic data loading not yet implemented. "
            "Please use load_data_from_arrays() or load_data_from_csv()"
        )

    def load_data_from_arrays(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        segments_train: np.ndarray,
        feature_names: List[str],
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        segments_val: Optional[np.ndarray] = None
    ):
        """
        Load data from numpy arrays (for use after running pipeline).

        Args:
            X_train: Training features
            y_train: Training labels
            segments_train: Training segment assignments
            feature_names: List of feature names
            X_val: Optional validation features
            y_val: Optional validation labels
            segments_val: Optional validation segment assignments
        """
        self.X_train = X_train
        self.y_train = y_train
        self.segments_train = segments_train
        self.feature_names = feature_names

        if X_val is not None:
            self.X_val = X_val
            self.y_val = y_val
            self.segments_val = segments_val

        print(f"Loaded data: {len(X_train):,} training samples, {len(feature_names)} features")

    def load_data_from_csv(self, csv_path: str, target_col: str = 'default'):
        """
        Load data from CSV file (requires tree_structure.json to assign segments).

        Args:
            csv_path: Path to CSV file with features
            target_col: Name of target column
        """
        # Load CSV
        df = pd.read_csv(csv_path)

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in CSV")

        self.y_train = df[target_col].values
        X_df = df.drop(columns=[target_col])

        # Select numeric columns
        numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
        self.X_train = X_df[numeric_cols].fillna(0).values
        self.feature_names = numeric_cols

        # Load tree structure to assign segments
        tree_path = self.output_dir / "tree_structure.json"
        if not tree_path.exists():
            raise FileNotFoundError("tree_structure.json required to assign segments from CSV")

        # Assign segments using tree
        self.segments_train = self._assign_segments_from_tree(self.X_train, tree_path)

        print(f"Loaded data from CSV: {len(self.X_train):,} samples, {len(self.feature_names)} features")

    def _assign_segments_from_tree(self, X: np.ndarray, tree_path: Path) -> np.ndarray:
        """
        Assign segments to observations using tree structure JSON.

        Args:
            X: Feature matrix
            tree_path: Path to tree_structure.json

        Returns:
            Array of segment assignments
        """
        with open(tree_path, 'r') as f:
            tree_data = json.load(f)

        nodes = {node['id']: node for node in tree_data['nodes']}
        leaf_to_segment = {int(k): v for k, v in tree_data['segment_mapping']['leaf_to_segment'].items()}

        # Create feature name to index mapping
        tree_features = tree_data['feature_metadata']['names']

        segments = np.zeros(len(X), dtype=int)

        # Traverse tree for each observation
        for i in range(len(X)):
            node_id = 0  # Start at root

            # Traverse until leaf
            while nodes[node_id]['type'] != 'leaf':
                node = nodes[node_id]
                feature_idx = node['feature_index']
                threshold = node['threshold']

                # Navigate tree
                if X[i, feature_idx] <= threshold:
                    node_id = node['left_child']
                else:
                    node_id = node['right_child']

            # Assign segment from leaf mapping
            segments[i] = leaf_to_segment.get(node_id, 0)

        return segments

    def reduce_dimensions(self, method: str = 'pca', n_components: int = 2, **kwargs):
        """
        Reduce feature dimensions to 2D for visualization.

        Args:
            method: Reduction method ('pca', 'tsne', 'umap')
            n_components: Number of dimensions (default: 2)
            **kwargs: Additional arguments for the reduction method
        """
        if self.X_train is None:
            raise RuntimeError("Data not loaded. Call load_data_from_* first.")

        print(f"\nReducing dimensions using {method.upper()}...")

        if method.lower() == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=n_components, **kwargs)
            self.X_reduced = reducer.fit_transform(self.X_train)

            # Print explained variance
            print(f"  Explained variance: {reducer.explained_variance_ratio_.sum():.2%}")
            print(f"  PC1: {reducer.explained_variance_ratio_[0]:.2%}")
            print(f"  PC2: {reducer.explained_variance_ratio_[1]:.2%}")

        elif method.lower() == 'tsne':
            from sklearn.manifold import TSNE
            perplexity = kwargs.pop('perplexity', 30)
            reducer = TSNE(n_components=n_components, perplexity=perplexity, **kwargs)
            self.X_reduced = reducer.fit_transform(self.X_train)
            print(f"  t-SNE completed with perplexity={perplexity}")

        elif method.lower() == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_components=n_components, **kwargs)
                self.X_reduced = reducer.fit_transform(self.X_train)
                print(f"  UMAP completed")
            except ImportError:
                raise ImportError("UMAP requires 'umap-learn' package. Install with: pip install umap-learn")
        else:
            raise ValueError(f"Unknown method: {method}. Choose 'pca', 'tsne', or 'umap'")

        self.reduction_method = method.upper()

    def plot_clusters(
        self,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
        show_centroids: bool = True,
        show_stats: bool = True
    ):
        """
        Create scatter plot of clusters colored by segment.

        Args:
            title: Plot title
            figsize: Figure size
            save_path: Optional path to save figure
            show_centroids: Whether to show segment centroids
            show_stats: Whether to show segment statistics
        """
        if self.X_reduced is None:
            raise RuntimeError("Must call reduce_dimensions() first")

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Get unique segments and stats
        unique_segments = np.unique(self.segments_train)
        segment_stats = self.report['segment_statistics']

        # Create color map based on default rate
        colors = self._get_risk_colors(segment_stats)

        # Plot each segment
        for seg_id in unique_segments:
            mask = self.segments_train == seg_id
            stats = segment_stats[str(seg_id)]

            # Scatter plot
            ax.scatter(
                self.X_reduced[mask, 0],
                self.X_reduced[mask, 1],
                c=[colors[seg_id]],
                label=f"Seg {seg_id} (PD={stats['default_rate']:.1%})",
                alpha=0.6,
                s=30,
                edgecolors='white',
                linewidth=0.5
            )

            # Show centroids
            if show_centroids:
                centroid = self.X_reduced[mask].mean(axis=0)
                ax.scatter(
                    centroid[0],
                    centroid[1],
                    c=[colors[seg_id]],
                    s=200,
                    marker='*',
                    edgecolors='black',
                    linewidth=2,
                    zorder=10
                )

        # Labels
        if title is None:
            title = f"Segmentation Clusters ({self.reduction_method})"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(f'{self.reduction_method} Component 1', fontsize=12)
        ax.set_ylabel(f'{self.reduction_method} Component 2', fontsize=12)

        # Legend
        ax.legend(
            title='Segments',
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            frameon=True,
            fontsize=9
        )

        # Add statistics text box
        if show_stats:
            stats_text = self._format_stats_text()
            ax.text(
                0.02, 0.98,
                stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=9,
                family='monospace'
            )

        plt.tight_layout()

        # Save
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved: {save_path}")

        return fig, ax

    def plot_default_rate_heatmap(
        self,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
        resolution: int = 100
    ):
        """
        Create heatmap showing default rate across 2D space.

        Args:
            figsize: Figure size
            save_path: Optional path to save figure
            resolution: Grid resolution for heatmap
        """
        if self.X_reduced is None:
            raise RuntimeError("Must call reduce_dimensions() first")

        fig, ax = plt.subplots(figsize=figsize)

        # Create grid
        x_min, x_max = self.X_reduced[:, 0].min(), self.X_reduced[:, 0].max()
        y_min, y_max = self.X_reduced[:, 1].min(), self.X_reduced[:, 1].max()

        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= 0.1 * x_range
        x_max += 0.1 * x_range
        y_min -= 0.1 * y_range
        y_max += 0.1 * y_range

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )

        # Calculate default rate for each grid cell using KNN
        from scipy.spatial.distance import cdist
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # Find nearest neighbors
        k = 50  # Number of neighbors
        distances = cdist(grid_points, self.X_reduced)
        nearest_indices = np.argpartition(distances, k, axis=1)[:, :k]

        # Calculate default rate
        default_rates = np.zeros(len(grid_points))
        for i, indices in enumerate(nearest_indices):
            default_rates[i] = self.y_train[indices].mean()

        default_rates = default_rates.reshape(xx.shape)

        # Plot heatmap
        contour = ax.contourf(
            xx, yy, default_rates,
            levels=20,
            cmap='RdYlGn_r',
            alpha=0.7
        )

        # Overlay scatter
        scatter = ax.scatter(
            self.X_reduced[:, 0],
            self.X_reduced[:, 1],
            c=self.y_train,
            cmap='RdYlGn_r',
            s=10,
            alpha=0.3,
            edgecolors='none'
        )

        # Colorbar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Default Rate', fontsize=12)

        # Labels
        ax.set_title(f'Default Rate Heatmap ({self.reduction_method})', fontsize=14, fontweight='bold')
        ax.set_xlabel(f'{self.reduction_method} Component 1', fontsize=12)
        ax.set_ylabel(f'{self.reduction_method} Component 2', fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nHeatmap saved: {save_path}")

        return fig, ax

    def plot_feature_distributions(
        self,
        features: Optional[List[str]] = None,
        n_features: int = 6,
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[str] = None
    ):
        """
        Plot distributions of top features across segments.

        Args:
            features: Specific features to plot (if None, use top by variance)
            n_features: Number of features to plot
            figsize: Figure size
            save_path: Optional path to save figure
        """
        if self.X_train is None:
            raise RuntimeError("Data not loaded")

        # Select features
        if features is None:
            # Use features with highest variance
            variances = np.var(self.X_train, axis=0)
            top_indices = np.argsort(variances)[-n_features:][::-1]
            features = [self.feature_names[i] for i in top_indices]
        else:
            features = features[:n_features]

        # Create subplot grid
        n_cols = 3
        n_rows = (len(features) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

        # Get colors
        segment_stats = self.report['segment_statistics']
        colors = self._get_risk_colors(segment_stats)
        unique_segments = np.unique(self.segments_train)

        # Plot each feature
        for idx, feature in enumerate(features):
            ax = axes[idx]
            feature_idx = self.feature_names.index(feature)

            # Plot distribution for each segment
            for seg_id in unique_segments:
                mask = self.segments_train == seg_id
                data = self.X_train[mask, feature_idx]

                ax.hist(
                    data,
                    bins=30,
                    alpha=0.5,
                    label=f'Seg {seg_id}',
                    color=colors[seg_id],
                    edgecolor='black',
                    linewidth=0.5
                )

            ax.set_xlabel(feature, fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'{feature} Distribution', fontsize=11, fontweight='bold')

            if idx == 0:
                ax.legend(fontsize=8)

        # Remove empty subplots
        for idx in range(len(features), len(axes)):
            fig.delaxes(axes[idx])

        plt.suptitle('Feature Distributions by Segment', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nFeature distributions saved: {save_path}")

        return fig, axes

    def _get_risk_colors(self, segment_stats: Dict) -> Dict[int, str]:
        """
        Assign colors to segments based on default rate.

        Args:
            segment_stats: Segment statistics dict

        Returns:
            Dictionary mapping segment ID to color
        """
        colors = {}
        for seg_id, stats in segment_stats.items():
            dr = stats['default_rate']
            seg_id_int = int(seg_id)

            if dr < 0.05:
                colors[seg_id_int] = '#2ecc71'  # Green
            elif dr < 0.10:
                colors[seg_id_int] = '#3498db'  # Blue
            elif dr < 0.15:
                colors[seg_id_int] = '#f39c12'  # Orange
            elif dr < 0.20:
                colors[seg_id_int] = '#e74c3c'  # Red
            else:
                colors[seg_id_int] = '#c0392b'  # Dark red

        return colors

    def _format_stats_text(self) -> str:
        """Format summary statistics as text."""
        stats = self.report['segment_statistics']
        n_segments = len(stats)
        total_obs = sum(s['n_observations'] for s in stats.values())
        total_defaults = sum(s['n_defaults'] for s in stats.values())
        overall_dr = total_defaults / total_obs

        text = f"Overall Statistics:\n"
        text += f"  Segments: {n_segments}\n"
        text += f"  Observations: {total_obs:,}\n"
        text += f"  Defaults: {total_defaults:,}\n"
        text += f"  Default Rate: {overall_dr:.2%}"

        return text

    def create_all_visualizations(self, output_prefix: Optional[str] = None):
        """
        Create all visualization types and save to files.

        Args:
            output_prefix: Prefix for output files (default: uses output_dir)
        """
        if output_prefix is None:
            output_prefix = str(self.output_dir / "viz")

        print("\n" + "=" * 70)
        print("GENERATING ALL VISUALIZATIONS")
        print("=" * 70)

        # 1. PCA scatter
        print("\n1. Creating PCA scatter plot...")
        self.reduce_dimensions('pca')
        self.plot_clusters(save_path=f"{output_prefix}_pca_scatter.png")
        plt.close()

        # 2. PCA heatmap
        print("\n2. Creating PCA heatmap...")
        self.plot_default_rate_heatmap(save_path=f"{output_prefix}_pca_heatmap.png")
        plt.close()

        # 3. t-SNE scatter (if dataset not too large)
        if len(self.X_train) < 50000:
            print("\n3. Creating t-SNE scatter plot...")
            self.reduce_dimensions('tsne', perplexity=30)
            self.plot_clusters(save_path=f"{output_prefix}_tsne_scatter.png")
            plt.close()
        else:
            print("\n3. Skipping t-SNE (dataset too large)")

        # 4. Feature distributions
        print("\n4. Creating feature distributions...")
        self.plot_feature_distributions(save_path=f"{output_prefix}_feature_distributions.png")
        plt.close()

        print("\n" + "=" * 70)
        print(f"All visualizations saved with prefix: {output_prefix}")
        print("=" * 70)


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description='Visualize segmentation clusters')
    parser.add_argument('report_path', help='Path to baseline_report.json')
    parser.add_argument('--method', choices=['pca', 'tsne', 'umap'], default='pca',
                        help='Dimensionality reduction method')
    parser.add_argument('--output', help='Output file prefix for visualizations')
    parser.add_argument('--interactive', action='store_true',
                        help='Show interactive plots (don\'t auto-save)')
    parser.add_argument('--all', action='store_true',
                        help='Generate all visualization types')

    args = parser.parse_args()

    # Create visualizer
    viz = ClusterVisualizer(args.report_path)

    print("\n" + "=" * 70)
    print("CLUSTER VISUALIZER")
    print("=" * 70)
    print(f"\nReport: {args.report_path}")
    print("\nNote: This visualizer requires data to be loaded separately.")
    print("      Use load_data_from_arrays() or load_data_from_csv() in your script.")
    print("\nExample usage in Python:")
    print("""
    from scripts.visualize_clusters import ClusterVisualizer

    # After running pipeline
    viz = ClusterVisualizer('output/baseline_report.json')
    viz.load_data_from_arrays(X_train, y_train, segments_train, feature_names)
    viz.reduce_dimensions('pca')
    viz.plot_clusters(save_path='clusters.png')
    """)


if __name__ == '__main__':
    main()
