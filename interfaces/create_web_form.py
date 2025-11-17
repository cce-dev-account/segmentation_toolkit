"""
Interactive Web Form for Segment Modifications

Creates an HTML form with drag-and-drop segment merging,
threshold sliders, and real-time preview.
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def classify_risk(pd_rate):
    """Classify risk level based on PD rate."""
    pd_pct = pd_rate * 100
    if pd_pct < 5:
        return "Very Low"
    elif pd_pct < 8:
        return "Low"
    elif pd_pct < 12:
        return "Low-Medium"
    elif pd_pct < 20:
        return "High"
    else:
        return "Very High"


def create_web_form(
    report_file: str = "lending_club_full_report.json",
    rules_file: str = "segment_rules_detailed.json",
    output_file: str = "modification_form.html"
):
    """Create interactive web form for segment modifications."""

    print("\n" + "=" * 80)
    print("CREATING INTERACTIVE WEB FORM")
    print("=" * 80)

    # Load data
    report_path = Path(__file__).parent.parent / report_file
    rules_path = Path(__file__).parent.parent / rules_file
    output_path = Path(__file__).parent / output_file

    if not report_path.exists():
        print(f"\nError: Report file not found: {report_path}")
        return False

    if not rules_path.exists():
        print(f"\nError: Rules file not found: {rules_path}")
        return False

    with open(report_path, 'r') as f:
        report = json.load(f)

    with open(rules_path, 'r') as f:
        rules_data = json.load(f)

    # Extract segment statistics from validation results
    segments = {}
    pd_stats = report.get('validation_results', {}).get('train', {}).get('validations', {}).get('binomial', {}).get('confidence_intervals', {})
    density_stats = report.get('validation_results', {}).get('train', {}).get('validations', {}).get('density', {}).get('densities', {})

    for key in list(pd_stats.keys()):
        # Skip non-segment keys
        if not key.isdigit():
            continue

        pd_data = pd_stats[key]
        segments[key] = {
            'default_rate': pd_data['default_rate'],
            'n_observations': pd_data['n_observations'],
            'n_defaults': pd_data['n_defaults'],
            'density': density_stats.get(key, 0),
            'risk_rating': classify_risk(pd_data['default_rate'])
        }

    params = report.get('parameters', {})
    feature_thresholds = rules_data.get('feature_thresholds', {})
    segment_rules = rules_data.get('segment_rules', {})

    print(f"\nLoaded {len(segments)} segments")
    print(f"Found {len(feature_thresholds)} features with thresholds")

    # Generate HTML
    html = generate_html(segments, params, feature_thresholds, segment_rules)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print("\n" + "=" * 80)
    print("WEB FORM CREATED")
    print("=" * 80)
    print(f"\nOutput: {output_path}")
    print(f"Size: {output_path.stat().st_size:,} bytes")
    print("\nTo use:")
    print(f"  1. Open {output_path.name} in browser")
    print("  2. Drag segments to merge them")
    print("  3. Adjust threshold sliders")
    print("  4. Modify parameters")
    print("  5. Click 'Generate JSON' to download modification.json")

    return True


def generate_html(segments, params, feature_thresholds, segment_rules):
    """Generate complete HTML with JavaScript."""

    # Convert data to JavaScript
    segments_js = json.dumps(segments, indent=2)
    params_js = json.dumps(params, indent=2)
    thresholds_js = json.dumps(feature_thresholds, indent=2)
    rules_js = json.dumps(segment_rules, indent=2)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IRB Segmentation Editor</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}

        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}

        header h1 {{
            font-size: 28px;
            margin-bottom: 10px;
        }}

        header p {{
            opacity: 0.9;
            font-size: 14px;
        }}

        .tabs {{
            display: flex;
            background: #f5f5f5;
            border-bottom: 2px solid #ddd;
        }}

        .tab {{
            flex: 1;
            padding: 15px;
            text-align: center;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s;
            border: none;
            background: transparent;
            font-size: 14px;
        }}

        .tab:hover {{
            background: #e0e0e0;
        }}

        .tab.active {{
            background: white;
            color: #667eea;
            border-bottom: 3px solid #667eea;
        }}

        .tab-content {{
            display: none;
            padding: 30px;
        }}

        .tab-content.active {{
            display: block;
        }}

        .segment-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .segment-card {{
            border: 2px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            cursor: move;
            transition: all 0.3s;
            background: white;
        }}

        .segment-card:hover {{
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transform: translateY(-2px);
        }}

        .segment-card.dragging {{
            opacity: 0.5;
        }}

        .segment-card.drag-over {{
            border-color: #667eea;
            border-style: dashed;
            background: #f0f4ff;
        }}

        .segment-card.merged {{
            opacity: 0.6;
            background: #f5f5f5;
        }}

        .segment-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}

        .segment-id {{
            font-size: 24px;
            font-weight: bold;
        }}

        .segment-risk {{
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }}

        .risk-very-low {{ background: #d4edda; color: #155724; }}
        .risk-low {{ background: #d1ecf1; color: #0c5460; }}
        .risk-medium {{ background: #fff3cd; color: #856404; }}
        .risk-high {{ background: #f8d7da; color: #721c24; }}
        .risk-very-high {{ background: #f5c6cb; color: #721c24; }}

        .segment-stats {{
            font-size: 13px;
            color: #666;
            line-height: 1.8;
        }}

        .pd-rate {{
            font-size: 20px;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }}

        .merge-indicator {{
            margin-top: 10px;
            padding: 8px;
            background: #fff3cd;
            border-radius: 4px;
            font-size: 12px;
            color: #856404;
        }}

        .threshold-section {{
            margin-bottom: 30px;
        }}

        .threshold-item {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 15px;
        }}

        .threshold-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}

        .threshold-name {{
            font-weight: 600;
            font-size: 16px;
            color: #333;
        }}

        .threshold-value {{
            font-size: 18px;
            font-weight: bold;
            color: #667eea;
        }}

        .threshold-slider {{
            width: 100%;
            margin-bottom: 10px;
        }}

        .threshold-current {{
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }}

        .param-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
        }}

        .param-item {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
        }}

        .param-label {{
            font-weight: 600;
            margin-bottom: 10px;
            color: #333;
        }}

        .param-input {{
            width: 100%;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 6px;
            font-size: 16px;
        }}

        .param-input:focus {{
            outline: none;
            border-color: #667eea;
        }}

        .param-hint {{
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }}

        .action-bar {{
            position: sticky;
            bottom: 0;
            background: white;
            padding: 20px 30px;
            border-top: 2px solid #ddd;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 -4px 12px rgba(0,0,0,0.1);
        }}

        .btn {{
            padding: 12px 24px;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }}

        .btn-primary {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}

        .btn-primary:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }}

        .btn-secondary {{
            background: #f5f5f5;
            color: #333;
        }}

        .btn-secondary:hover {{
            background: #e0e0e0;
        }}

        .summary {{
            font-size: 14px;
            color: #666;
        }}

        .summary strong {{
            color: #667eea;
        }}

        .instructions {{
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 4px;
        }}

        .instructions h3 {{
            color: #1976d2;
            margin-bottom: 10px;
            font-size: 16px;
        }}

        .instructions ul {{
            margin-left: 20px;
            color: #555;
            font-size: 14px;
        }}

        .instructions li {{
            margin: 5px 0;
        }}

        .add-threshold-btn {{
            display: inline-block;
            margin-top: 10px;
            padding: 8px 16px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
        }}

        .add-threshold-btn:hover {{
            background: #5568d3;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>IRB Segmentation Editor</h1>
            <p>Interactive interface for modifying credit risk segments</p>
        </header>

        <div class="tabs">
            <button class="tab active" onclick="switchTab(0)">Segment Merging</button>
            <button class="tab" onclick="switchTab(1)">Threshold Editor</button>
            <button class="tab" onclick="switchTab(2)">Parameters</button>
            <button class="tab" onclick="switchTab(3)">Preview</button>
        </div>

        <div class="tab-content active" id="tab-segments">
            <div class="instructions">
                <h3>How to Merge Segments</h3>
                <ul>
                    <li>Drag a segment card and drop it onto another segment to merge them</li>
                    <li>The dragged segment will be merged INTO the target segment</li>
                    <li>Merged segments will be grayed out</li>
                    <li>To undo, click "Reset All Changes"</li>
                </ul>
            </div>
            <div class="segment-grid" id="segment-grid"></div>
        </div>

        <div class="tab-content" id="tab-thresholds">
            <div class="instructions">
                <h3>Business Rule Thresholds</h3>
                <ul>
                    <li>Add forced splits by entering feature name and threshold value</li>
                    <li>Current thresholds from the model are shown for reference</li>
                    <li>Use exact feature names from the dropdown</li>
                </ul>
            </div>
            <div id="threshold-list"></div>
            <button class="add-threshold-btn" onclick="addThreshold()">+ Add New Threshold</button>
        </div>

        <div class="tab-content" id="tab-parameters">
            <div class="instructions">
                <h3>Model Parameters</h3>
                <ul>
                    <li>Increase max_depth for more granular segments</li>
                    <li>Decrease min_samples_leaf for smaller segments</li>
                    <li>Adjust density constraints to control segment balance</li>
                </ul>
            </div>
            <div class="param-grid" id="param-grid"></div>
        </div>

        <div class="tab-content" id="tab-preview">
            <div class="instructions">
                <h3>Modification Summary</h3>
                <ul>
                    <li>Review all changes before generating JSON</li>
                    <li>Download the modification.json file</li>
                    <li>Run: python apply_modifications.py modification.json</li>
                </ul>
            </div>
            <pre id="json-preview" style="background: #f5f5f5; padding: 20px; border-radius: 8px; overflow-x: auto; font-size: 12px;"></pre>
        </div>

        <div class="action-bar">
            <div class="summary">
                <strong id="merge-count">0</strong> merges •
                <strong id="threshold-count">0</strong> thresholds •
                <strong id="param-count">0</strong> parameter changes
            </div>
            <div>
                <button class="btn btn-secondary" onclick="resetChanges()">Reset All Changes</button>
                <button class="btn btn-primary" onclick="generateJSON()">Generate JSON</button>
            </div>
        </div>
    </div>

    <script>
        // Data
        const segments = {segments_js};
        const parameters = {params_js};
        const featureThresholds = {thresholds_js};
        const segmentRules = {rules_js};

        // State
        let merges = [];
        let forcedSplits = {{}};
        let paramChanges = {{}};
        let customThresholds = [];

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {{
            renderSegments();
            renderThresholds();
            renderParameters();
            updateSummary();
        }});

        function switchTab(index) {{
            const tabs = document.querySelectorAll('.tab');
            const contents = document.querySelectorAll('.tab-content');

            tabs.forEach((tab, i) => {{
                tab.classList.toggle('active', i === index);
            }});

            contents.forEach((content, i) => {{
                content.classList.toggle('active', i === index);
            }});

            if (index === 3) {{
                updatePreview();
            }}
        }}

        function renderSegments() {{
            const grid = document.getElementById('segment-grid');
            grid.innerHTML = '';

            Object.entries(segments).forEach(([id, seg]) => {{
                const isMerged = merges.some(m => m[0] == id);
                const mergeTarget = merges.find(m => m[0] == id);

                const card = document.createElement('div');
                card.className = 'segment-card' + (isMerged ? ' merged' : '');
                card.draggable = !isMerged;
                card.dataset.segmentId = id;

                const pdRate = seg.default_rate * 100;
                const riskClass = pdRate < 5 ? 'very-low' :
                                 pdRate < 8 ? 'low' :
                                 pdRate < 12 ? 'medium' :
                                 pdRate < 20 ? 'high' : 'very-high';

                card.innerHTML = `
                    <div class="segment-header">
                        <div class="segment-id">Segment ${{id}}</div>
                        <div class="segment-risk risk-${{riskClass}}">${{seg.risk_rating}}</div>
                    </div>
                    <div class="pd-rate">${{pdRate.toFixed(2)}}% PD</div>
                    <div class="segment-stats">
                        <div>Observations: ${{seg.n_observations.toLocaleString()}}</div>
                        <div>Defaults: ${{seg.n_defaults.toLocaleString()}}</div>
                        <div>Density: ${{(seg.density * 100).toFixed(2)}}%</div>
                    </div>
                    ${{isMerged ? `<div class="merge-indicator">Will merge into Segment ${{mergeTarget[1]}}</div>` : ''}}
                `;

                card.addEventListener('dragstart', handleDragStart);
                card.addEventListener('dragover', handleDragOver);
                card.addEventListener('drop', handleDrop);
                card.addEventListener('dragleave', handleDragLeave);

                grid.appendChild(card);
            }});
        }}

        function handleDragStart(e) {{
            e.target.classList.add('dragging');
            e.dataTransfer.effectAllowed = 'move';
            e.dataTransfer.setData('text/html', e.target.dataset.segmentId);
        }}

        function handleDragOver(e) {{
            if (e.preventDefault) {{
                e.preventDefault();
            }}
            e.dataTransfer.dropEffect = 'move';
            const card = e.currentTarget;
            if (!card.classList.contains('dragging')) {{
                card.classList.add('drag-over');
            }}
            return false;
        }}

        function handleDragLeave(e) {{
            e.currentTarget.classList.remove('drag-over');
        }}

        function handleDrop(e) {{
            if (e.stopPropagation) {{
                e.stopPropagation();
            }}

            e.currentTarget.classList.remove('drag-over');
            const draggedId = e.dataTransfer.getData('text/html');
            const targetId = e.currentTarget.dataset.segmentId;

            document.querySelector('.dragging')?.classList.remove('dragging');

            if (draggedId !== targetId) {{
                // Remove any existing merge for this segment
                merges = merges.filter(m => m[0] != draggedId);
                // Add new merge
                merges.push([parseInt(draggedId), parseInt(targetId)]);
                renderSegments();
                updateSummary();
            }}

            return false;
        }}

        function renderThresholds() {{
            const list = document.getElementById('threshold-list');
            list.innerHTML = '';

            Object.entries(featureThresholds).forEach(([feature, thresholds]) => {{
                const item = document.createElement('div');
                item.className = 'threshold-item';
                item.innerHTML = `
                    <div class="threshold-header">
                        <div class="threshold-name">${{feature}}</div>
                    </div>
                    <div class="threshold-current">
                        Current thresholds: ${{Array.isArray(thresholds) ? thresholds.join(', ') : thresholds}}
                    </div>
                    <div style="margin-top: 10px;">
                        <input type="number" step="0.01" placeholder="Enter new threshold"
                               class="param-input" id="threshold-${{feature}}"
                               onchange="updateForcedSplit('${{feature}}', this.value)">
                    </div>
                `;
                list.appendChild(item);
            }});
        }}

        function addThreshold() {{
            const feature = prompt('Enter feature name:');
            if (feature && !featureThresholds[feature]) {{
                const value = prompt('Enter threshold value:');
                if (value) {{
                    updateForcedSplit(feature, value);
                    renderThresholds();
                }}
            }}
        }}

        function updateForcedSplit(feature, value) {{
            if (value && value.trim() !== '') {{
                forcedSplits[feature] = parseFloat(value);
            }} else {{
                delete forcedSplits[feature];
            }}
            updateSummary();
        }}

        function renderParameters() {{
            const grid = document.getElementById('param-grid');
            grid.innerHTML = '';

            const paramInfo = {{
                max_depth: {{ range: '3-7', desc: 'Maximum tree depth' }},
                min_samples_leaf: {{ range: '1000-50000', desc: 'Minimum observations per segment' }},
                min_defaults_per_leaf: {{ range: '50-2000', desc: 'Minimum defaults per segment' }},
                min_segment_density: {{ range: '0.01-0.20', desc: 'Minimum segment size (% of population)' }},
                max_segment_density: {{ range: '0.30-0.60', desc: 'Maximum segment size (% of population)' }}
            }};

            Object.entries(parameters).forEach(([param, value]) => {{
                const info = paramInfo[param] || {{ range: '', desc: param }};
                const item = document.createElement('div');
                item.className = 'param-item';
                item.innerHTML = `
                    <div class="param-label">${{param}}</div>
                    <input type="number" step="any" value="${{value}}"
                           class="param-input" id="param-${{param}}"
                           onchange="updateParameter('${{param}}', this.value)">
                    <div class="param-hint">${{info.desc}}</div>
                    <div class="param-hint">Valid range: ${{info.range}}</div>
                `;
                grid.appendChild(item);
            }});
        }}

        function updateParameter(param, value) {{
            const numValue = value.includes('.') ? parseFloat(value) : parseInt(value);
            if (numValue !== parameters[param]) {{
                paramChanges[param] = numValue;
            }} else {{
                delete paramChanges[param];
            }}
            updateSummary();
        }}

        function updateSummary() {{
            document.getElementById('merge-count').textContent = merges.length;
            document.getElementById('threshold-count').textContent = Object.keys(forcedSplits).length;
            document.getElementById('param-count').textContent = Object.keys(paramChanges).length;
        }}

        function updatePreview() {{
            const modification = {{
                metadata: {{
                    instructions: "Generated from interactive web form",
                    source: "modification_form.html",
                    modification_notes: generateNotes()
                }},
                modifications: {{
                    merge_segments: {{
                        description: "Segment pairs to merge",
                        value: merges
                    }},
                    forced_splits: {{
                        description: "Forced split thresholds",
                        value: forcedSplits
                    }},
                    parameter_changes: paramChanges
                }}
            }};

            document.getElementById('json-preview').textContent = JSON.stringify(modification, null, 2);
        }}

        function generateNotes() {{
            const notes = [];

            merges.forEach(([from, to]) => {{
                const fromPD = (segments[from].default_rate * 100).toFixed(2);
                const toPD = (segments[to].default_rate * 100).toFixed(2);
                notes.push(`Segment ${{from}} -> ${{to}}: Merge ${{fromPD}}% PD into ${{toPD}}% PD`);
            }});

            Object.entries(forcedSplits).forEach(([feature, threshold]) => {{
                notes.push(`${{feature}} @ ${{threshold}}: Business rule threshold`);
            }});

            Object.entries(paramChanges).forEach(([param, value]) => {{
                notes.push(`${{param}}: ${{parameters[param]}} -> ${{value}}`);
            }});

            return notes;
        }}

        function resetChanges() {{
            if (confirm('Reset all changes?')) {{
                merges = [];
                forcedSplits = {{}};
                paramChanges = {{}};
                renderSegments();
                renderThresholds();
                renderParameters();
                updateSummary();
            }}
        }}

        function generateJSON() {{
            updatePreview();

            const modification = {{
                metadata: {{
                    instructions: "Generated from interactive web form",
                    source: "modification_form.html",
                    modification_notes: generateNotes()
                }},
                modifications: {{
                    merge_segments: {{
                        description: "Segment pairs to merge",
                        value: merges
                    }},
                    forced_splits: {{
                        description: "Forced split thresholds",
                        value: forcedSplits
                    }},
                    parameter_changes: paramChanges
                }}
            }};

            const blob = new Blob([JSON.stringify(modification, null, 2)], {{ type: 'application/json' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'modification.json';
            a.click();
            URL.revokeObjectURL(url);

            alert('modification.json downloaded!\\n\\nNext step:\\npython apply_modifications.py modification.json');
        }}
    </script>
</body>
</html>
"""

    return html


def main():
    """Main entry point."""
    success = create_web_form()
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
