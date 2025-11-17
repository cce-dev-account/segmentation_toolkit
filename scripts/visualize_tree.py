"""
Tree Visualization and Interactive Modification Tool

Provides multiple visualization options for IRB segmentation trees:
1. Text-based tree structure
2. Graphviz decision tree diagram
3. Interactive segment rules editor
4. Segment statistics dashboard
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent))

from irb_segmentation import IRBSegmentationEngine, IRBSegmentationParams


class TreeVisualizer:
    """
    Visualize and interact with IRB segmentation trees.
    """

    def __init__(self, engine: IRBSegmentationEngine):
        """
        Initialize visualizer with fitted engine.

        Args:
            engine: Fitted IRBSegmentationEngine
        """
        self.engine = engine
        self.tree = engine.tree_model.tree_
        self.feature_names = engine.feature_names_

    def print_tree_structure(self, max_depth: Optional[int] = None):
        """
        Print ASCII tree structure with statistics.

        Args:
            max_depth: Maximum depth to display (None = all)
        """
        from sklearn.tree import _tree

        tree = self.tree
        feature_names = self.feature_names

        # Get segment assignments
        train_leaves = self.engine.tree_model.apply(self.engine.X_train_)
        leaf_to_segment = {}
        for leaf, segment in zip(train_leaves, self.engine.segments_train_):
            leaf_to_segment[leaf] = segment

        # Calculate statistics per node
        node_samples = {}
        node_defaults = {}
        node_segments = {}

        for i, leaf in enumerate(train_leaves):
            node_id = leaf
            if node_id not in node_samples:
                node_samples[node_id] = 0
                node_defaults[node_id] = 0
            node_samples[node_id] += 1
            node_defaults[node_id] += self.engine.y_train_[i]
            node_segments[node_id] = leaf_to_segment.get(node_id, -1)

        print("\n" + "=" * 80)
        print("DECISION TREE STRUCTURE")
        print("=" * 80)
        print("\nLegend:")
        print("  [Node ID] Feature <= Threshold")
        print("  Samples: N | Defaults: M (X.XX%) | Segment: S")
        print("=" * 80)

        def recurse(node, depth=0, parent_rule=""):
            if max_depth is not None and depth > max_depth:
                return

            indent = "  " * depth

            # Get node info
            n_samples = node_samples.get(node, tree.n_node_samples[node])
            n_defaults = node_defaults.get(node, 0)
            default_rate = n_defaults / n_samples if n_samples > 0 else 0

            if tree.feature[node] != _tree.TREE_UNDEFINED:
                # Internal node
                feature = feature_names[tree.feature[node]]
                threshold = tree.threshold[node]

                print(f"{indent}[{node}] {feature} <= {threshold:.2f}")
                print(f"{indent}     Samples: {n_samples:,} | Defaults: {n_defaults:,} ({default_rate:.2%})")

                # Left child
                recurse(tree.children_left[node], depth + 1, f"{feature} <= {threshold:.2f}")

                # Right child
                recurse(tree.children_right[node], depth + 1, f"{feature} > {threshold:.2f}")

            else:
                # Leaf node
                segment = node_segments.get(node, -1)
                print(f"{indent}[{node}] LEAF")
                print(f"{indent}     Samples: {n_samples:,} | Defaults: {n_defaults:,} ({default_rate:.2%})")
                print(f"{indent}     >>> SEGMENT {segment}")

        recurse(0)
        print("=" * 80)

    def export_graphviz(self, output_file: str = "tree.dot", include_stats: bool = True):
        """
        Export tree to Graphviz DOT format for visualization.

        Args:
            output_file: Output .dot file path
            include_stats: Whether to include detailed statistics
        """
        from sklearn.tree import export_graphviz

        # Get segment info for coloring
        train_leaves = self.engine.tree_model.apply(self.engine.X_train_)
        leaf_to_segment = {}
        for leaf, segment in zip(train_leaves, self.engine.segments_train_):
            leaf_to_segment[leaf] = segment

        # Export with styling
        export_graphviz(
            self.engine.tree_model,
            out_file=output_file,
            feature_names=self.feature_names,
            filled=True,
            rounded=True,
            special_characters=True,
            proportion=True,
            precision=2
        )

        print(f"\nGraphviz file exported: {output_file}")
        print("To generate image, run:")
        print(f"  dot -Tpng {output_file} -o tree.png")
        print(f"  dot -Tsvg {output_file} -o tree.svg")
        print(f"  dot -Tpdf {output_file} -o tree.pdf")
        print("\nNote: Requires Graphviz installation (https://graphviz.org/)")

    def show_segment_rules(self, output_format: str = "text"):
        """
        Display segment rules in various formats.

        Args:
            output_format: 'text', 'table', or 'json'
        """
        rules = self.engine.get_segment_rules()
        stats = self.engine._get_segment_statistics()

        if output_format == "text":
            print("\n" + "=" * 80)
            print("SEGMENT RULES")
            print("=" * 80)

            for i, rule in enumerate(rules, 1):
                print(f"\n{i}. {rule}")

        elif output_format == "table":
            print("\n" + "=" * 80)
            print("SEGMENT SUMMARY TABLE")
            print("=" * 80)

            # Create table
            rows = []
            for seg_id in sorted(stats.keys()):
                s = stats[seg_id]
                rows.append({
                    'Segment': seg_id,
                    'Observations': f"{s['n_observations']:,}",
                    'Defaults': f"{s['n_defaults']:,}",
                    'Default_Rate': f"{s['default_rate']:.2%}",
                    'Density': f"{s['density']:.2%}"
                })

            df = pd.DataFrame(rows)
            print(df.to_string(index=False))

        elif output_format == "json":
            output = {
                "rules": rules,
                "statistics": stats
            }
            print(json.dumps(output, indent=2))

    def generate_interactive_html(self, output_file: str = "tree_interactive.html"):
        """
        Generate interactive HTML visualization with D3.js tree.

        Args:
            output_file: Output HTML file path
        """
        # Get tree data
        stats = self.engine._get_segment_statistics()
        rules = self.engine.get_segment_rules()

        # Create HTML with embedded D3.js visualization
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>IRB Segmentation Tree - Interactive Viewer</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}

        #header {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        h1 {{
            margin: 0;
            color: #333;
        }}

        #stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}

        .stat-card {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .stat-card h3 {{
            margin: 0 0 10px 0;
            color: #666;
            font-size: 14px;
        }}

        .stat-card .value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}

        #tree-container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow-x: auto;
        }}

        .segment-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}

        .segment-table th, .segment-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}

        .segment-table th {{
            background: #f0f0f0;
            font-weight: bold;
        }}

        .segment-table tr:hover {{
            background: #f9f9f9;
        }}

        .risk-very-low {{ color: #2ecc71; font-weight: bold; }}
        .risk-low {{ color: #27ae60; font-weight: bold; }}
        .risk-medium {{ color: #f39c12; font-weight: bold; }}
        .risk-high {{ color: #e74c3c; font-weight: bold; }}
        .risk-very-high {{ color: #c0392b; font-weight: bold; }}
    </style>
</head>
<body>
    <div id="header">
        <h1>IRB Segmentation Tree - Lending Club Dataset</h1>
        <p>Interactive visualization of decision tree segments with risk statistics</p>
    </div>

    <div id="stats">
        <div class="stat-card">
            <h3>Total Segments</h3>
            <div class="value">{len(stats)}</div>
        </div>
        <div class="stat-card">
            <h3>Total Observations</h3>
            <div class="value">{sum(s['n_observations'] for s in stats.values()):,}</div>
        </div>
        <div class="stat-card">
            <h3>Total Defaults</h3>
            <div class="value">{sum(s['n_defaults'] for s in stats.values()):,}</div>
        </div>
        <div class="stat-card">
            <h3>Overall Default Rate</h3>
            <div class="value">{sum(s['n_defaults'] for s in stats.values()) / sum(s['n_observations'] for s in stats.values()):.2%}</div>
        </div>
    </div>

    <div id="tree-container">
        <h2>Segment Statistics</h2>
        <table class="segment-table">
            <thead>
                <tr>
                    <th>Segment</th>
                    <th>Observations</th>
                    <th>Defaults</th>
                    <th>Default Rate</th>
                    <th>Density</th>
                    <th>Risk Level</th>
                </tr>
            </thead>
            <tbody>
"""

        # Add segment rows
        for seg_id in sorted(stats.keys()):
            s = stats[seg_id]
            dr = s['default_rate']

            # Assign risk level
            if dr < 0.05:
                risk_class = "risk-very-low"
                risk_level = "Very Low"
            elif dr < 0.10:
                risk_class = "risk-low"
                risk_level = "Low"
            elif dr < 0.15:
                risk_class = "risk-medium"
                risk_level = "Medium"
            elif dr < 0.20:
                risk_class = "risk-high"
                risk_level = "High"
            else:
                risk_class = "risk-very-high"
                risk_level = "Very High"

            html_content += f"""
                <tr>
                    <td><strong>Segment {seg_id}</strong></td>
                    <td>{s['n_observations']:,}</td>
                    <td>{s['n_defaults']:,}</td>
                    <td>{s['default_rate']:.2%}</td>
                    <td>{s['density']:.2%}</td>
                    <td class="{risk_class}">{risk_level}</td>
                </tr>
"""

        html_content += """
            </tbody>
        </table>

        <h2 style="margin-top: 40px;">Segment Rules</h2>
        <div style="background: #f9f9f9; padding: 15px; border-radius: 4px; margin-top: 10px;">
"""

        # Add rules
        for i, rule in enumerate(rules[:10], 1):  # Show first 10 rules
            html_content += f"<p><strong>{i}.</strong> {rule}</p>\n"

        if len(rules) > 10:
            html_content += f"<p><em>... and {len(rules) - 10} more rules</em></p>\n"

        html_content += """
        </div>
    </div>

    <div style="margin-top: 20px; padding: 15px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <h3>Instructions</h3>
        <p>This visualization shows the IRB segmentation results. Each segment represents a group of loans with similar risk characteristics.</p>
        <ul>
            <li><strong>Default Rate:</strong> Percentage of loans that defaulted in each segment</li>
            <li><strong>Density:</strong> Percentage of total portfolio in each segment</li>
            <li><strong>Risk Level:</strong> Classification based on default rate thresholds</li>
        </ul>
        <p>To modify the segmentation, adjust parameters in the Python script and re-run validation.</p>
    </div>
</body>
</html>
"""

        with open(output_file, 'w') as f:
            f.write(html_content)

        print(f"\nInteractive HTML visualization created: {output_file}")
        print("Open in browser to view interactive dashboard")

    def export_modification_template(self, output_file: str = "modify_segments.json"):
        """
        Export template for manual segment modifications.

        Args:
            output_file: Output JSON file path
        """
        stats = self.engine._get_segment_statistics()

        template = {
            "metadata": {
                "instructions": "Modify segments by specifying merges or forced splits",
                "timestamp": self.engine.validation_results_.get("timestamp", ""),
                "original_segments": len(stats)
            },
            "modifications": {
                "merge_segments": {
                    "description": "List segment pairs to merge. Format: [[seg1, seg2], [seg3, seg4]]",
                    "example": [[1, 4]],
                    "value": []
                },
                "forced_splits": {
                    "description": "Add forced split points. Format: {feature_name: threshold}",
                    "example": {"fico_score": 700, "dti": 0.30},
                    "value": {}
                },
                "parameter_changes": {
                    "description": "Modify IRB parameters and re-fit",
                    "max_depth": self.engine.params.max_depth,
                    "min_samples_leaf": self.engine.params.min_samples_leaf,
                    "min_defaults_per_leaf": self.engine.params.min_defaults_per_leaf,
                    "min_segment_density": self.engine.params.min_segment_density,
                    "max_segment_density": self.engine.params.max_segment_density
                }
            },
            "current_segments": stats
        }

        with open(output_file, 'w') as f:
            json.dump(template, f, indent=2)

        print(f"\nModification template exported: {output_file}")
        print("\nTo apply modifications:")
        print("1. Edit the JSON file with your desired changes")
        print("2. Run: python apply_modifications.py modify_segments.json")


def main():
    """Demo visualization with Lending Club results."""
    print("\n" + "=" * 80)
    print("IRB SEGMENTATION TREE VISUALIZER")
    print("=" * 80)

    # Check if we have a fitted model
    report_file = "lending_club_full_report.json"
    if not Path(report_file).exists():
        print(f"\nError: {report_file} not found")
        print("Please run test_lending_club_simple.py first")
        return

    # We need to reload the model - for now, show what we can from the report
    with open(report_file, 'r') as f:
        report = json.load(f)

    print("\n" + "=" * 80)
    print("CURRENT SEGMENTATION SUMMARY")
    print("=" * 80)

    stats = report['segment_statistics']

    # Create summary table
    print(f"\n{'Segment':<10}{'Observations':<15}{'Defaults':<12}{'PD Rate':<12}{'Density':<10}{'Risk Level':<15}")
    print("-" * 80)

    for seg_id in sorted([int(k) for k in stats.keys()]):
        s = stats[str(seg_id)]
        dr = s['default_rate']

        if dr < 0.05:
            risk = "Very Low"
        elif dr < 0.10:
            risk = "Low"
        elif dr < 0.15:
            risk = "Medium"
        elif dr < 0.20:
            risk = "High"
        else:
            risk = "Very High"

        print(f"{seg_id:<10}{s['n_observations']:<15,}{s['n_defaults']:<12,}{s['default_rate']:<12.2%}{s['density']:<10.2%}{risk:<15}")

    print("\n" + "=" * 80)
    print("VISUALIZATION OPTIONS")
    print("=" * 80)
    print("\n1. Tree Structure: Shows decision tree with split points")
    print("2. Graphviz Export: Generates .dot file for visual tree diagram")
    print("3. Interactive HTML: Creates web-based dashboard")
    print("4. Modification Template: JSON file for segment modifications")
    print("\nNote: Full visualization requires reloading the fitted model")
    print("      (feature in development)")


if __name__ == "__main__":
    main()
