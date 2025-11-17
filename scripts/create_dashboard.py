"""
Create Interactive HTML Dashboard from Report

Generates a user-friendly web interface for viewing and modifying segments.
"""

import json
from pathlib import Path


def create_interactive_dashboard(report_file: str = "lending_club_full_report.json",
                                  output_file: str = "segmentation_dashboard.html"):
    """
    Create interactive HTML dashboard from JSON report.

    Args:
        report_file: Path to segmentation report JSON
        output_file: Path to output HTML file
    """
    print(f"\nCreating interactive dashboard from {report_file}...")

    with open(report_file, 'r') as f:
        report = json.load(f)

    stats = report['segment_statistics']
    params = report['parameters']
    val_results = report['validation_results']
    adjustments = report['adjustments']

    # Calculate aggregate stats
    total_obs = sum(s['n_observations'] for s in stats.values())
    total_defaults = sum(s['n_defaults'] for s in stats.values())
    overall_dr = total_defaults / total_obs

    # Create HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IRB Segmentation Dashboard - Lending Club</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        .header {{
            background: white;
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        .header h1 {{
            color: #333;
            margin-bottom: 10px;
            font-size: 32px;
        }}

        .header p {{
            color: #666;
            font-size: 16px;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}

        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        .stat-card h3 {{
            color: #666;
            font-size: 14px;
            font-weight: 500;
            margin-bottom: 8px;
            text-transform: uppercase;
        }}

        .stat-card .value {{
            color: #333;
            font-size: 28px;
            font-weight: bold;
        }}

        .stat-card .subvalue {{
            color: #999;
            font-size: 14px;
            margin-top: 4px;
        }}

        .card {{
            background: white;
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        .card h2 {{
            color: #333;
            margin-bottom: 20px;
            font-size: 24px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}

        .segments-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}

        .segment-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 8px;
            padding: 20px;
            border-left: 4px solid #667eea;
            transition: transform 0.2s, box-shadow 0.2s;
            cursor: pointer;
        }}

        .segment-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }}

        .segment-card.risk-very-low {{ border-left-color: #10b981; }}
        .segment-card.risk-low {{ border-left-color: #3b82f6; }}
        .segment-card.risk-medium {{ border-left-color: #f59e0b; }}
        .segment-card.risk-high {{ border-left-color: #ef4444; }}
        .segment-card.risk-very-high {{ border-left-color: #7f1d1d; }}

        .segment-card h3 {{
            font-size: 18px;
            margin-bottom: 15px;
            color: #333;
        }}

        .segment-metric {{
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            font-size: 14px;
        }}

        .segment-metric .label {{
            color: #666;
        }}

        .segment-metric .value {{
            font-weight: bold;
            color: #333;
        }}

        .risk-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
            margin-top: 10px;
        }}

        .risk-badge.very-low {{ background: #d1fae5; color: #065f46; }}
        .risk-badge.low {{ background: #dbeafe; color: #1e40af; }}
        .risk-badge.medium {{ background: #fef3c7; color: #92400e; }}
        .risk-badge.high {{ background: #fee2e2; color: #991b1b; }}
        .risk-badge.very-high {{ background: #fecaca; color: #7f1d1d; }}

        .validation-section {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }}

        .validation-item {{
            padding: 15px;
            background: #f9fafb;
            border-radius: 8px;
            border-left: 3px solid #d1d5db;
        }}

        .validation-item.passed {{ border-left-color: #10b981; }}
        .validation-item.failed {{ border-left-color: #ef4444; }}

        .validation-item h4 {{
            color: #333;
            margin-bottom: 5px;
            font-size: 14px;
        }}

        .validation-item .status {{
            font-weight: bold;
            font-size: 12px;
        }}

        .validation-item.passed .status {{ color: #10b981; }}
        .validation-item.failed .status {{ color: #ef4444; }}

        .modification-section {{
            background: #f0f9ff;
            border: 2px dashed #3b82f6;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
        }}

        .modification-section h3 {{
            color: #1e40af;
            margin-bottom: 15px;
        }}

        .button {{
            display: inline-block;
            padding: 12px 24px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 8px;
            text-decoration: none;
            font-weight: bold;
            transition: transform 0.2s, box-shadow 0.2s;
            border: none;
            cursor: pointer;
            margin-right: 10px;
            margin-top: 10px;
        }}

        .button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(102,126,234,0.4);
        }}

        .code-block {{
            background: #1f2937;
            color: #f9fafb;
            padding: 15px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            overflow-x: auto;
            margin-top: 10px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}

        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e5e7eb;
        }}

        th {{
            background: #f9fafb;
            font-weight: 600;
            color: #374151;
        }}

        tr:hover {{
            background: #f9fafb;
        }}

        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: bold;
        }}

        .badge.success {{ background: #d1fae5; color: #065f46; }}
        .badge.warning {{ background: #fef3c7; color: #92400e; }}
        .badge.error {{ background: #fee2e2; color: #991b1b; }}

        .chart-container {{
            margin-top: 20px;
            padding: 20px;
            background: #f9fafb;
            border-radius: 8px;
        }}

        .bar-chart {{
            display: flex;
            align-items: flex-end;
            height: 200px;
            gap: 10px;
        }}

        .bar {{
            flex: 1;
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
            border-radius: 4px 4px 0 0;
            position: relative;
            transition: opacity 0.2s;
        }}

        .bar:hover {{
            opacity: 0.8;
        }}

        .bar-label {{
            position: absolute;
            bottom: -25px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 11px;
            color: #666;
            white-space: nowrap;
        }}

        .bar-value {{
            position: absolute;
            top: -25px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 12px;
            font-weight: bold;
            color: #333;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 IRB Segmentation Dashboard</h1>
            <p>Lending Club Dataset - 2.26 Million Observations</p>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Segments</h3>
                <div class="value">{len(stats)}</div>
                <div class="subvalue">Validated segments</div>
            </div>
            <div class="stat-card">
                <h3>Total Observations</h3>
                <div class="value">{total_obs:,}</div>
                <div class="subvalue">Training set</div>
            </div>
            <div class="stat-card">
                <h3>Total Defaults</h3>
                <div class="value">{total_defaults:,}</div>
                <div class="subvalue">{overall_dr:.2%} overall rate</div>
            </div>
            <div class="stat-card">
                <h3>PD Range</h3>
                <div class="value">{max(s['default_rate'] for s in stats.values()) - min(s['default_rate'] for s in stats.values()):.1%}</div>
                <div class="subvalue">{min(s['default_rate'] for s in stats.values()):.1%} - {max(s['default_rate'] for s in stats.values()):.1%}</div>
            </div>
        </div>

        <div class="card">
            <h2>📊 Segment Overview</h2>
            <div class="segments-grid">
"""

    # Add segment cards
    for seg_id in sorted([int(k) for k in stats.keys()]):
        s = stats[str(seg_id)]
        dr = s['default_rate']

        # Determine risk level
        if dr < 0.05:
            risk_class = "risk-very-low"
            risk_badge = "very-low"
            risk_label = "Very Low Risk"
        elif dr < 0.10:
            risk_class = "risk-low"
            risk_badge = "low"
            risk_label = "Low Risk"
        elif dr < 0.15:
            risk_class = "risk-medium"
            risk_badge = "medium"
            risk_label = "Medium Risk"
        elif dr < 0.20:
            risk_class = "risk-high"
            risk_badge = "high"
            risk_label = "High Risk"
        else:
            risk_class = "risk-very-high"
            risk_badge = "very-high"
            risk_label = "Very High Risk"

        html += f"""
                <div class="segment-card {risk_class}">
                    <h3>Segment {seg_id}</h3>
                    <div class="segment-metric">
                        <span class="label">Observations:</span>
                        <span class="value">{s['n_observations']:,}</span>
                    </div>
                    <div class="segment-metric">
                        <span class="label">Defaults:</span>
                        <span class="value">{s['n_defaults']:,}</span>
                    </div>
                    <div class="segment-metric">
                        <span class="label">Default Rate:</span>
                        <span class="value">{s['default_rate']:.2%}</span>
                    </div>
                    <div class="segment-metric">
                        <span class="label">Portfolio %:</span>
                        <span class="value">{s['density']:.1%}</span>
                    </div>
                    <span class="risk-badge {risk_badge}">{risk_label}</span>
                </div>
"""

    html += """
            </div>
        </div>

        <div class="card">
            <h2>📈 Default Rate Distribution</h2>
            <div class="chart-container">
                <div class="bar-chart">
"""

    # Add bars for default rates
    max_dr = max(s['default_rate'] for s in stats.values())
    for seg_id in sorted([int(k) for k in stats.keys()]):
        s = stats[str(seg_id)]
        height_pct = (s['default_rate'] / max_dr) * 100
        html += f"""
                    <div class="bar" style="height: {height_pct}%;">
                        <div class="bar-value">{s['default_rate']:.1%}</div>
                        <div class="bar-label">Seg {seg_id}</div>
                    </div>
"""

    html += """
                </div>
            </div>
        </div>

        <div class="card">
            <h2>✅ Validation Results</h2>
"""

    # Training validation
    train_val = val_results['train']['validations']
    html += """
            <h3 style="margin-bottom: 15px; color: #666;">Training Set Validation</h3>
            <div class="validation-section">
"""

    for test_name, result in train_val.items():
        passed = result.get('passed', True)
        status_class = "passed" if passed else "failed"
        status_text = "[PASS]" if passed else "[FAIL]"

        html += f"""
                <div class="validation-item {status_class}">
                    <h4>{test_name.replace('_', ' ').title()}</h4>
                    <div class="status">{status_text}</div>
                </div>
"""

    html += """
            </div>
"""

    # Validation set
    val_val = val_results['validation']['validations']
    html += """
            <h3 style="margin: 30px 0 15px 0; color: #666;">Validation Set</h3>
            <div class="validation-section">
"""

    for test_name, result in val_val.items():
        if test_name == 'psi':
            passed = result.get('passed', True)
            psi_value = result.get('psi', 0)
            status_class = "passed" if passed else "failed"
            html += f"""
                <div class="validation-item {status_class}">
                    <h4>PSI (Population Stability)</h4>
                    <div class="status">PSI = {psi_value:.6f} ({result.get('stability', 'N/A')})</div>
                </div>
"""
        else:
            passed = result.get('passed', True)
            status_class = "passed" if passed else "failed"
            status_text = "[PASS]" if passed else "[FAIL]"
            html += f"""
                <div class="validation-item {status_class}">
                    <h4>{test_name.replace('_', ' ').title()}</h4>
                    <div class="status">{status_text}</div>
                </div>
"""

    html += """
            </div>
        </div>

        <div class="card">
            <h2>⚙️ Model Parameters</h2>
            <table>
                <thead>
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
"""

    param_descriptions = {
        'max_depth': 'Maximum tree depth',
        'min_samples_leaf': 'Minimum samples per leaf',
        'min_defaults_per_leaf': 'Minimum defaults per segment',
        'min_segment_density': 'Minimum segment size (%)',
        'max_segment_density': 'Maximum segment size (%)'
    }

    for param, value in params.items():
        desc = param_descriptions.get(param, '')
        if isinstance(value, float):
            value_str = f"{value:.2%}" if param.endswith('density') else f"{value:.4f}"
        else:
            value_str = str(value)

        html += f"""
                    <tr>
                        <td><strong>{param}</strong></td>
                        <td>{value_str}</td>
                        <td>{desc}</td>
                    </tr>
"""

    html += f"""
                </tbody>
            </table>
        </div>

        <div class="card">
            <h2>🔧 Adjustments Applied</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                <div style="padding: 15px; background: #f9fafb; border-radius: 8px;">
                    <h4 style="color: #666; margin-bottom: 5px;">Merges</h4>
                    <div style="font-size: 24px; font-weight: bold;">{len(adjustments['merges'])}</div>
                </div>
                <div style="padding: 15px; background: #f9fafb; border-radius: 8px;">
                    <h4 style="color: #666; margin-bottom: 5px;">Splits</h4>
                    <div style="font-size: 24px; font-weight: bold;">{len(adjustments['splits'])}</div>
                </div>
                <div style="padding: 15px; background: #f9fafb; border-radius: 8px;">
                    <h4 style="color: #666; margin-bottom: 5px;">Forced Splits</h4>
                    <div style="font-size: 24px; font-weight: bold;">{len(adjustments['forced_splits'])}</div>
                </div>
                <div style="padding: 15px; background: #f9fafb; border-radius: 8px;">
                    <h4 style="color: #666; margin-bottom: 5px;">Monotonicity Violations</h4>
                    <div style="font-size: 24px; font-weight: bold;">{len(adjustments['monotonicity_violations'])}</div>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="modification-section">
                <h3>💡 Modify Segmentation</h3>
                <p style="margin-bottom: 15px; color: #1e40af;">Want to adjust the segments? Follow these steps:</p>

                <h4 style="margin: 15px 0 10px 0; color: #1e40af;">Step 1: Create Modification Template</h4>
                <div class="code-block">python apply_modifications.py --create-sample</div>

                <h4 style="margin: 15px 0 10px 0; color: #1e40af;">Step 2: Edit the JSON File</h4>
                <p style="margin-bottom: 10px; color: #374151;">Modify <code>modify_segments_sample.json</code> to:</p>
                <ul style="margin-left: 20px; color: #374151;">
                    <li>Merge similar segments (e.g., Segments 1 and 4)</li>
                    <li>Add forced split points on specific features</li>
                    <li>Adjust IRB parameters (depth, minimums, etc.)</li>
                </ul>

                <h4 style="margin: 15px 0 10px 0; color: #1e40af;">Step 3: Apply Modifications</h4>
                <div class="code-block">python apply_modifications.py modify_segments_sample.json</div>

                <h4 style="margin: 15px 0 10px 0; color: #1e40af;">Step 4: Compare Results</h4>
                <div class="code-block">python apply_modifications.py --compare lending_club_full_report.json modify_segments_sample_result.json</div>

                <div style="margin-top: 20px;">
                    <span class="badge warning">⚠️ Note</span>
                    <span style="color: #374151; margin-left: 10px;">Modifications will re-run the full segmentation and validation process (~3 minutes)</span>
                </div>
            </div>
        </div>

        <div style="text-align: center; padding: 20px; color: white;">
            <p>Generated by IRB Segmentation Framework | Report: {report_file}</p>
        </div>
    </div>
</body>
</html>
"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"[OK] Dashboard created: {output_file}")
    print(f"\nOpen {output_file} in your browser to view the interactive dashboard")


if __name__ == "__main__":
    create_interactive_dashboard()
