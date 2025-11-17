"""
Create Enhanced Interactive Dashboard with Segment Rules and Threshold Editor
"""

import json
from pathlib import Path


def create_enhanced_dashboard(
    report_file: str = "lending_club_full_report.json",
    rules_file: str = "segment_rules_detailed.json",
    output_file: str = "segmentation_dashboard_enhanced.html"
):
    """Create enhanced dashboard with segment rules and threshold editor."""

    print(f"\nCreating enhanced dashboard...")

    # Load report
    with open(report_file, 'r') as f:
        report = json.load(f)

    # Load rules
    if Path(rules_file).exists():
        with open(rules_file, 'r') as f:
            rules_data = json.load(f)
    else:
        print(f"Warning: {rules_file} not found. Run extract_segment_rules.py first.")
        rules_data = {"segment_rules": {}, "feature_thresholds": {}}

    stats = report['segment_statistics']
    params = report['parameters']
    val_results = report['validation_results']
    segment_rules = rules_data.get('segment_rules', {})
    feature_thresholds = rules_data.get('feature_thresholds', {})

    # Calculate aggregate stats
    total_obs = sum(s['n_observations'] for s in stats.values())
    total_defaults = sum(s['n_defaults'] for s in stats.values())
    overall_dr = total_defaults / total_obs

    # Start HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IRB Segmentation Dashboard - Enhanced with Rules</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{ max-width: 1600px; margin: 0 auto; }}

        .header {{
            background: white;
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        .header h1 {{ color: #333; margin-bottom: 10px; font-size: 32px; }}
        .header p {{ color: #666; font-size: 16px; }}

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

        .stat-card .value {{ color: #333; font-size: 28px; font-weight: bold; }}
        .stat-card .subvalue {{ color: #999; font-size: 14px; margin-top: 4px; }}

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

        .segment-card-detailed {{
            background: white;
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }}

        .segment-card-detailed.risk-very-low {{ border-left-color: #10b981; }}
        .segment-card-detailed.risk-low {{ border-left-color: #3b82f6; }}
        .segment-card-detailed.risk-medium {{ border-left-color: #f59e0b; }}
        .segment-card-detailed.risk-high {{ border-left-color: #ef4444; }}
        .segment-card-detailed.risk-very-high {{ border-left-color: #7f1d1d; }}

        .segment-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }}

        .segment-title {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}

        .risk-badge {{
            display: inline-block;
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: bold;
        }}

        .risk-badge.very-low {{ background: #d1fae5; color: #065f46; }}
        .risk-badge.low {{ background: #dbeafe; color: #1e40af; }}
        .risk-badge.medium {{ background: #fef3c7; color: #92400e; }}
        .risk-badge.high {{ background: #fee2e2; color: #991b1b; }}
        .risk-badge.very-high {{ background: #fecaca; color: #7f1d1d; }}

        .segment-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}

        .stat-item {{
            background: #f9fafb;
            padding: 12px;
            border-radius: 8px;
        }}

        .stat-item .label {{ color: #666; font-size: 12px; margin-bottom: 4px; }}
        .stat-item .value {{ color: #333; font-size: 20px; font-weight: bold; }}

        .segment-rules {{
            background: #f9fafb;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
        }}

        .segment-rules h4 {{
            color: #333;
            margin-bottom: 15px;
            font-size: 16px;
        }}

        .rule-item {{
            background: white;
            padding: 12px 15px;
            margin-bottom: 10px;
            border-radius: 6px;
            border-left: 3px solid #667eea;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            color: #374151;
            line-height: 1.6;
        }}

        .rule-item .condition {{
            color: #dc2626;
            font-weight: bold;
        }}

        .threshold-editor {{
            background: #f0f9ff;
            border: 2px solid #3b82f6;
            border-radius: 12px;
            padding: 25px;
            margin-top: 20px;
        }}

        .threshold-editor h3 {{
            color: #1e40af;
            margin-bottom: 20px;
            font-size: 20px;
        }}

        .threshold-list {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
        }}

        .threshold-group {{
            background: white;
            border-radius: 8px;
            padding: 15px;
        }}

        .threshold-group h4 {{
            color: #333;
            margin-bottom: 12px;
            font-size: 14px;
            font-weight: bold;
        }}

        .threshold-value {{
            background: #f9fafb;
            padding: 8px 12px;
            margin-bottom: 8px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            color: #374151;
            border-left: 3px solid #3b82f6;
        }}

        .edit-button {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            font-weight: bold;
            cursor: pointer;
            margin-top: 15px;
            transition: transform 0.2s;
        }}

        .edit-button:hover {{ transform: translateY(-2px); }}

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

        .info-box {{
            background: #fef3c7;
            border-left: 4px solid #f59e0b;
            padding: 15px;
            border-radius: 6px;
            margin-top: 15px;
        }}

        .info-box p {{ color: #92400e; font-size: 14px; line-height: 1.6; }}

        .collapsible {{
            cursor: pointer;
            user-select: none;
        }}

        .collapsible:hover {{ opacity: 0.8; }}

        .content {{
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
        }}

        .content.active {{ max-height: 2000px; }}

        .toggle-icon {{
            float: right;
            transition: transform 0.3s;
        }}

        .toggle-icon.active {{ transform: rotate(180deg); }}

        .interface-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}

        .interface-card {{
            background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
            border: 2px solid #e5e7eb;
            border-radius: 12px;
            padding: 24px;
            transition: all 0.3s ease;
        }}

        .interface-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
            border-color: #667eea;
        }}

        .interface-icon {{
            font-size: 48px;
            margin-bottom: 12px;
        }}

        .interface-card h4 {{
            color: #333;
            margin: 12px 0;
            font-size: 18px;
        }}

        .interface-card p {{
            color: #6b7280;
            font-size: 14px;
            margin-bottom: 16px;
            line-height: 1.5;
        }}

        .interface-steps {{
            background: white;
            border-radius: 8px;
            padding: 16px;
            margin: 16px 0;
        }}

        .step {{
            color: #374151;
            font-size: 13px;
            margin: 8px 0;
            padding-left: 8px;
        }}

        .step code {{
            background: #f3f4f6;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 12px;
            color: #667eea;
        }}

        .interface-badge {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            display: inline-block;
            margin-top: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 IRB Segmentation Dashboard</h1>
            <p>Lending Club Dataset - 2.26 Million Observations | Enhanced with Segment Rules & Threshold Editor</p>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Segments</h3>
                <div class="value">{len(stats)}</div>
                <div class="subvalue">With decision rules</div>
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
            <h2>📊 Detailed Segment Analysis with Decision Rules</h2>
"""

    # Add detailed segment cards
    for seg_id in sorted([int(k) for k in stats.keys()]):
        s = stats[str(seg_id)]
        dr = s['default_rate']

        # Risk level
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

        # Get rules for this segment
        seg_rules = segment_rules.get(str(seg_id), {}).get('rules', [])
        seg_desc = segment_rules.get(str(seg_id), {}).get('description', '')

        html += f"""
            <div class="segment-card-detailed {risk_class}">
                <div class="segment-header">
                    <div class="segment-title">Segment {seg_id}</div>
                    <span class="risk-badge {risk_badge}">{risk_label}</span>
                </div>

                <div class="segment-stats">
                    <div class="stat-item">
                        <div class="label">Observations</div>
                        <div class="value">{s['n_observations']:,}</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Defaults</div>
                        <div class="value">{s['n_defaults']:,}</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Default Rate</div>
                        <div class="value">{s['default_rate']:.2%}</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Portfolio %</div>
                        <div class="value">{s['density']:.1%}</div>
                    </div>
                </div>

                {f'<p style="color: #666; margin-top: 15px; font-size: 14px;">{seg_desc}</p>' if seg_desc else ''}
"""

        if seg_rules:
            # Show first 3 rules, make rest collapsible
            html += f"""
                <div class="segment-rules">
                    <h4>Decision Rules ({len(seg_rules)} path{'s' if len(seg_rules) > 1 else ''})</h4>
"""

            for i, rule in enumerate(seg_rules[:3], 1):
                html += f"""
                    <div class="rule-item">
                        <strong>Path {i}:</strong> IF {rule.replace(' AND ', ' <span class="condition">AND</span> ')}
                    </div>
"""

            if len(seg_rules) > 3:
                html += f"""
                    <div class="collapsible" onclick="toggleRules('seg{seg_id}')" id="toggle-seg{seg_id}">
                        <span>Show {len(seg_rules) - 3} more paths</span>
                        <span class="toggle-icon" id="icon-seg{seg_id}">▼</span>
                    </div>
                    <div class="content" id="content-seg{seg_id}">
"""
                for i, rule in enumerate(seg_rules[3:], 4):
                    html += f"""
                        <div class="rule-item">
                            <strong>Path {i}:</strong> IF {rule.replace(' AND ', ' <span class="condition">AND</span> ')}
                        </div>
"""
                html += """
                    </div>
"""

            html += """
                </div>
"""

        html += """
            </div>
"""

    html += """
        </div>

        <div class="card">
            <div class="modification-interfaces">
                <h3>✏️ Modification Interfaces</h3>
                <p style="color: #374151; margin-bottom: 20px;">Choose your preferred method to modify the segmentation:</p>

                <div class="interface-grid">
                    <div class="interface-card">
                        <div class="interface-icon">📊</div>
                        <h4>Excel/CSV Spreadsheets</h4>
                        <p>Edit segments in familiar spreadsheet format with fill-in-the-blank worksheets</p>
                        <div class="interface-steps">
                            <div class="step">1. Generate templates: <code>python interfaces/create_excel_template.py</code></div>
                            <div class="step">2. Edit CSV files in Excel/LibreOffice</div>
                            <div class="step">3. Convert back: <code>python interfaces/excel_to_json.py</code></div>
                            <div class="step">4. Apply: <code>python apply_modifications.py modification.json</code></div>
                        </div>
                        <div class="interface-badge">✅ Recommended for Business Users</div>
                    </div>

                    <div class="interface-card">
                        <div class="interface-icon">🎨</div>
                        <h4>Interactive Web Form</h4>
                        <p>Drag-and-drop visual interface with real-time preview and threshold sliders</p>
                        <div class="interface-steps">
                            <div class="step">1. Generate form: <code>python interfaces/create_web_form.py</code></div>
                            <div class="step">2. Open interfaces/modification_form.html in browser</div>
                            <div class="step">3. Drag segments to merge, adjust sliders</div>
                            <div class="step">4. Click "Generate JSON" to download</div>
                            <div class="step">5. Apply: <code>python apply_modifications.py modification.json</code></div>
                        </div>
                        <div class="interface-badge">✅ Most User-Friendly</div>
                    </div>

                    <div class="interface-card">
                        <div class="interface-icon">📝</div>
                        <h4>YAML Text Format</h4>
                        <p>Human-readable text format with inline comments, perfect for version control</p>
                        <div class="interface-steps">
                            <div class="step">1. Create template: <code>python interfaces/yaml_converter.py --create-template</code></div>
                            <div class="step">2. Edit modification_template.yaml in text editor</div>
                            <div class="step">3. Convert: <code>python interfaces/yaml_converter.py --to-json modification_template.yaml</code></div>
                            <div class="step">4. Apply: <code>python apply_modifications.py modification.json</code></div>
                        </div>
                        <div class="interface-badge">✅ Best for Version Control</div>
                    </div>

                    <div class="interface-card">
                        <div class="interface-icon">⚙️</div>
                        <h4>Direct JSON Editing</h4>
                        <p>For advanced users who prefer working directly with JSON format</p>
                        <div class="interface-steps">
                            <div class="step">1. Edit modification.json directly</div>
                            <div class="step">2. Follow structure in example below</div>
                            <div class="step">3. Apply: <code>python apply_modifications.py modification.json</code></div>
                        </div>
                        <div class="interface-badge">⚡ Advanced Users</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="threshold-editor">
                <h3>🎯 Feature Thresholds Used in Segmentation</h3>
                <p style="color: #374151; margin-bottom: 20px;">These are the actual threshold values used to split the data. Use one of the interfaces above to modify them.</p>

                <div class="threshold-list">
"""

    # Add feature thresholds
    for feature, thresholds in sorted(feature_thresholds.items()):
        html += f"""
                    <div class="threshold-group">
                        <h4>{feature}</h4>
"""
        for threshold in thresholds:
            html += f"""
                        <div class="threshold-value">{threshold:.2f}</div>
"""
        html += """
                    </div>
"""

    html += f"""
                </div>

                <div class="info-box">
                    <p><strong>💡 How to Modify Thresholds:</strong></p>
                    <p>1. Create a modification template with: <code style="background:#fff; padding:2px 6px; border-radius:3px;">python apply_modifications.py --create-sample</code></p>
                    <p>2. Edit <code style="background:#fff; padding:2px 6px; border-radius:3px;">modify_segments_sample.json</code> to add forced splits at your desired thresholds</p>
                    <p>3. Apply changes with: <code style="background:#fff; padding:2px 6px; border-radius:3px;">python apply_modifications.py modify_segments_sample.json</code></p>
                </div>

                <h4 style="margin-top: 25px; color: #1e40af;">Example: Force split at specific thresholds</h4>
                <div class="code-block">{{
  "forced_splits": {{
    "value": {{
      "int_rate": 15.0,          // Always split at 15% interest rate
      "fico_range_high": 700,    // Always split at FICO 700
      "annual_inc": 60000,       // Always split at $60k income
      "dti": 25.0                // Always split at 25% DTI
    }}
  }}
}}</div>

                <button class="edit-button" onclick="alert('To edit thresholds:\\n1. Run: python apply_modifications.py --create-sample\\n2. Edit modify_segments_sample.json\\n3. Run: python apply_modifications.py modify_segments_sample.json')">
                    📝 Create Modification Template
                </button>
            </div>
        </div>

        <div style="text-align: center; padding: 20px; color: white;">
            <p>IRB Segmentation Framework | Enhanced Dashboard with Rules & Threshold Editor</p>
            <p style="font-size: 12px; margin-top: 5px;">Report: {report_file} | Rules: {rules_file}</p>
        </div>
    </div>

    <script>
        function toggleRules(segId) {{
            var content = document.getElementById('content-' + segId);
            var icon = document.getElementById('icon-' + segId);

            if (content.classList.contains('active')) {{
                content.classList.remove('active');
                icon.classList.remove('active');
            }} else {{
                content.classList.add('active');
                icon.classList.add('active');
            }}
        }}
    </script>
</body>
</html>
"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"[OK] Enhanced dashboard created: {output_file}")
    print(f"\nOpen {output_file} in your browser to view:")
    print("  - Detailed segment decision rules")
    print("  - Feature thresholds")
    print("  - Interactive threshold editor")


if __name__ == "__main__":
    create_enhanced_dashboard()
