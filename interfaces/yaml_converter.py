"""
YAML Format Support for Segment Modifications

Provides a more readable alternative to JSON with inline comments
and better version control compatibility.
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def json_to_yaml(json_file: str = "modification.json", yaml_file: str = "modification.yaml"):
    """Convert modification JSON to YAML format."""

    print("\n" + "=" * 80)
    print("CONVERTING JSON TO YAML")
    print("=" * 80)

    json_path = Path(__file__).parent.parent / json_file
    yaml_path = Path(__file__).parent.parent / yaml_file

    if not json_path.exists():
        print(f"\nError: JSON file not found: {json_path}")
        return False

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Generate YAML manually (simple format, no dependencies)
    yaml_content = generate_yaml(data)

    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"\nConverted: {json_path.name} -> {yaml_path.name}")
    print(f"Size: {yaml_path.stat().st_size:,} bytes")
    print("\n" + "=" * 80)
    print("YAML file created successfully")
    print("=" * 80)
    print(f"\nOutput: {yaml_path}")
    print("\nTo convert back to JSON:")
    print(f"  python interfaces/yaml_converter.py --to-json {yaml_file}")

    return True


def yaml_to_json(yaml_file: str = "modification.yaml", json_file: str = "modification.json"):
    """Convert modification YAML back to JSON format."""

    print("\n" + "=" * 80)
    print("CONVERTING YAML TO JSON")
    print("=" * 80)

    yaml_path = Path(__file__).parent.parent / yaml_file
    json_path = Path(__file__).parent.parent / json_file

    if not yaml_path.exists():
        print(f"\nError: YAML file not found: {yaml_path}")
        return False

    with open(yaml_path, 'r', encoding='utf-8') as f:
        yaml_content = f.read()

    # Parse YAML (simple parser for our specific format)
    data = parse_yaml(yaml_content)

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nConverted: {yaml_path.name} -> {json_path.name}")
    print(f"Size: {json_path.stat().st_size:,} bytes")
    print("\n" + "=" * 80)
    print("JSON file created successfully")
    print("=" * 80)
    print(f"\nOutput: {json_path}")
    print("\nTo apply modifications:")
    print(f"  python apply_modifications.py {json_file}")

    return True


def generate_yaml(data):
    """Generate YAML content from modification data."""

    yaml = """# IRB Segmentation Modification File
# YAML Format - Human-readable alternative to JSON
#
# This file defines changes to the segmentation model:
# - merge_segments: Combine similar segments
# - forced_splits: Business rule thresholds
# - parameter_changes: Adjust model parameters
#
# Syntax Notes:
# - Lines starting with # are comments
# - Use 2 spaces for indentation (not tabs)
# - Segment IDs and feature names must match exactly
#

"""

    # Metadata
    metadata = data.get('metadata', {})
    yaml += "metadata:\n"
    yaml += f"  instructions: \"{metadata.get('instructions', '')}\"\n"

    source_files = metadata.get('source_files', [])
    if source_files:
        yaml += "  source_files:\n"
        for sf in source_files:
            yaml += f"    - {sf}\n"

    notes = metadata.get('modification_notes', [])
    if notes:
        yaml += "  modification_notes:\n"
        for note in notes:
            yaml += f"    - \"{note}\"\n"
    else:
        yaml += "  modification_notes: []\n"

    yaml += "\n"

    # Modifications
    mods = data.get('modifications', {})
    yaml += "modifications:\n\n"

    # Merge segments
    yaml += "  # Segment Merging\n"
    yaml += "  # Format: [source_segment, target_segment]\n"
    yaml += "  # Example: [1, 4] means merge segment 1 into segment 4\n"
    yaml += "  merge_segments:\n"
    yaml += "    description: \"Segment pairs to merge\"\n"

    merges = mods.get('merge_segments', {}).get('value', [])
    if merges:
        yaml += "    value:\n"
        for merge in merges:
            yaml += f"      - [{merge[0]}, {merge[1]}]  # Merge segment {merge[0]} into {merge[1]}\n"
    else:
        yaml += "    value: []\n"

    yaml += "\n"

    # Forced splits
    yaml += "  # Business Rule Thresholds\n"
    yaml += "  # Format: feature_name: threshold_value\n"
    yaml += "  # These thresholds will be enforced regardless of model optimization\n"
    yaml += "  # Use for regulatory requirements or business policy cutoffs\n"
    yaml += "  forced_splits:\n"
    yaml += "    description: \"Forced split thresholds\"\n"

    splits = mods.get('forced_splits', {}).get('value', {})
    if splits:
        yaml += "    value:\n"
        for feature, threshold in splits.items():
            yaml += f"      {feature}: {threshold}\n"
    else:
        yaml += "    value: {}\n"

    yaml += "\n"

    # Parameter changes
    yaml += "  # Model Parameter Adjustments\n"
    yaml += "  # Only specify parameters you want to change\n"
    yaml += "  # Available parameters:\n"
    yaml += "  #   max_depth: 3-7 (higher = more segments)\n"
    yaml += "  #   min_samples_leaf: 1000-50000 (lower = more segments)\n"
    yaml += "  #   min_defaults_per_leaf: 50-2000 (higher = fewer segments)\n"
    yaml += "  #   min_segment_density: 0.01-0.20 (lower = allow smaller segments)\n"
    yaml += "  #   max_segment_density: 0.30-0.60 (lower = force balanced segments)\n"
    yaml += "  parameter_changes:\n"

    params = mods.get('parameter_changes', {})
    if params:
        for param, value in params.items():
            yaml += f"    {param}: {value}\n"
    else:
        yaml += "    {}  # No parameter changes\n"

    return yaml


def parse_yaml(yaml_content):
    """Parse YAML content into Python dict (simple parser for our format)."""

    lines = yaml_content.split('\n')
    data = {
        'metadata': {},
        'modifications': {}
    }

    current_section = None
    current_subsection = None
    current_list = None

    for line in lines:
        # Skip comments and empty lines
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue

        # Detect section level by indentation
        indent = len(line) - len(line.lstrip())

        # Top level sections
        if indent == 0:
            if stripped.startswith('metadata:'):
                current_section = 'metadata'
            elif stripped.startswith('modifications:'):
                current_section = 'modifications'

        # Level 1 (under metadata or modifications)
        elif indent == 2:
            if current_section == 'metadata':
                if stripped.startswith('instructions:'):
                    data['metadata']['instructions'] = extract_value(stripped)
                elif stripped.startswith('source_files:'):
                    data['metadata']['source_files'] = []
                    current_list = 'source_files'
                elif stripped.startswith('modification_notes:'):
                    data['metadata']['modification_notes'] = []
                    current_list = 'modification_notes'

            elif current_section == 'modifications':
                if stripped.startswith('merge_segments:'):
                    current_subsection = 'merge_segments'
                    data['modifications']['merge_segments'] = {}
                elif stripped.startswith('forced_splits:'):
                    current_subsection = 'forced_splits'
                    data['modifications']['forced_splits'] = {}
                elif stripped.startswith('parameter_changes:'):
                    current_subsection = 'parameter_changes'
                    data['modifications']['parameter_changes'] = {}

        # Level 2 (under subsections)
        elif indent == 4:
            if current_section == 'metadata' and current_list:
                # List item
                if stripped.startswith('- '):
                    value = stripped[2:].strip('"')
                    data['metadata'][current_list].append(value)

            elif current_section == 'modifications' and current_subsection:
                if stripped.startswith('description:'):
                    data['modifications'][current_subsection]['description'] = extract_value(stripped)
                elif stripped.startswith('value:'):
                    value_str = stripped[6:].strip()
                    if value_str == '[]':
                        data['modifications'][current_subsection]['value'] = []
                    elif value_str == '{}':
                        data['modifications'][current_subsection]['value'] = {}
                    # else will be populated by deeper levels
                    elif not data['modifications'][current_subsection].get('value'):
                        if current_subsection == 'merge_segments':
                            data['modifications'][current_subsection]['value'] = []
                        elif current_subsection == 'forced_splits':
                            data['modifications'][current_subsection]['value'] = {}
                else:
                    # Parameter changes at this level
                    if current_subsection == 'parameter_changes':
                        key, value = parse_key_value(stripped)
                        if key and value is not None:
                            data['modifications']['parameter_changes'][key] = value

        # Level 3 (list items, dict entries)
        elif indent == 6:
            if current_subsection == 'merge_segments':
                # Parse list item: - [1, 4]
                if stripped.startswith('- ['):
                    merge_str = stripped[2:].split('#')[0].strip()  # Remove comments
                    merge_list = eval(merge_str)  # Safe since we control format
                    if 'value' not in data['modifications']['merge_segments']:
                        data['modifications']['merge_segments']['value'] = []
                    data['modifications']['merge_segments']['value'].append(merge_list)

            elif current_subsection == 'forced_splits':
                # Parse dict entry: feature: threshold
                key, value = parse_key_value(stripped)
                if key and value is not None:
                    if 'value' not in data['modifications']['forced_splits']:
                        data['modifications']['forced_splits']['value'] = {}
                    data['modifications']['forced_splits']['value'][key] = value

    return data


def extract_value(line):
    """Extract value from 'key: value' line."""
    parts = line.split(':', 1)
    if len(parts) == 2:
        return parts[1].strip().strip('"')
    return ""


def parse_key_value(line):
    """Parse 'key: value' and convert to appropriate type."""
    parts = line.split(':', 1)
    if len(parts) != 2:
        return None, None

    key = parts[0].strip()
    value_str = parts[1].strip()

    # Try to parse as number
    try:
        if '.' in value_str:
            return key, float(value_str)
        else:
            return key, int(value_str)
    except ValueError:
        # Return as string
        return key, value_str.strip('"')


def create_template_yaml(output_file: str = "modification_template.yaml"):
    """Create a template YAML file with examples and comments."""

    yaml = """# IRB Segmentation Modification Template
# YAML Format - Human-readable alternative to JSON
#
# Fill in this template to make changes to your segmentation model
# Then convert to JSON: python interfaces/yaml_converter.py --to-json
#

metadata:
  instructions: "Manual modifications via YAML template"
  source_files:
    - modification_template.yaml
  modification_notes:
    - "Example modification - replace with your changes"

modifications:

  # =============================================================================
  # SEGMENT MERGING
  # =============================================================================
  # Merge similar segments to simplify the model
  # Format: [source_segment_id, target_segment_id]
  # The source segment will be merged INTO the target segment
  #
  # Example: Merge segment 1 into segment 4 (similar PD rates)
  #
  merge_segments:
    description: "Segment pairs to merge"
    value:
      - [1, 4]  # Merge segment 1 (7.98% PD) into segment 4 (8.35% PD)
      # Add more merges below:
      # - [source, target]

  # =============================================================================
  # BUSINESS RULE THRESHOLDS (Forced Splits)
  # =============================================================================
  # Enforce specific thresholds regardless of model optimization
  # Use for regulatory requirements or business policy cutoffs
  #
  # Available features (from model):
  #   - int_rate: Interest rate (%)
  #   - fico_range_high: FICO score (high range)
  #   - fico_range_low: FICO score (low range)
  #   - annual_inc: Annual income ($)
  #   - dti: Debt-to-income ratio (%)
  #   - loan_amnt: Loan amount ($)
  #   - installment: Monthly installment ($)
  #   - inq_last_6mths: Credit inquiries (last 6 months)
  #
  # Examples:
  #   fico_range_high: 650  # Prime/subprime regulatory cutoff
  #   int_rate: 15.0        # Subprime threshold
  #   annual_inc: 60000     # High-income segment
  #
  forced_splits:
    description: "Forced split thresholds"
    value:
      int_rate: 15.0           # Company policy: subprime cutoff
      fico_range_high: 650     # Regulatory: prime/subprime distinction
      annual_inc: 60000        # Business rule: high-income preferential pricing
      # Add more thresholds below:
      # feature_name: threshold_value

  # =============================================================================
  # MODEL PARAMETER ADJUSTMENTS
  # =============================================================================
  # Fine-tune the segmentation algorithm
  # Only specify parameters you want to change from current values
  #
  # Available parameters:
  #
  #   max_depth: 3-7
  #     - Controls maximum tree depth
  #     - Higher = more segments, more granular
  #     - Current: 5
  #
  #   min_samples_leaf: 1000-50000
  #     - Minimum observations per segment
  #     - Lower = more segments, smaller sizes
  #     - Current: 10000
  #
  #   min_defaults_per_leaf: 50-2000
  #     - Minimum defaults required per segment
  #     - Higher = fewer segments, better statistical power
  #     - Current: 500
  #
  #   min_segment_density: 0.01-0.20
  #     - Minimum segment size as % of population
  #     - Lower = allow smaller segments
  #     - Current: 0.05 (5%)
  #
  #   max_segment_density: 0.30-0.60
  #     - Maximum segment size as % of population
  #     - Lower = force more balanced segments
  #     - Current: 0.40 (40%)
  #
  parameter_changes:
    # Uncomment and modify as needed:
    # max_depth: 6
    # min_samples_leaf: 8000
    # min_defaults_per_leaf: 600
    # min_segment_density: 0.04
    # max_segment_density: 0.35

# =============================================================================
# WORKFLOW
# =============================================================================
# 1. Edit this file with your changes
# 2. Save the file
# 3. Convert to JSON: python interfaces/yaml_converter.py --to-json modification_template.yaml
# 4. Apply changes: python apply_modifications.py modification.json
# 5. Review new segmentation in dashboard
#
# Tips:
# - Start with segment merges (consolidate similar risk profiles)
# - Add forced splits for regulatory/business requirements
# - Adjust parameters last (for fine-tuning)
# - Document your reasons in modification_notes above
# - Version control friendly: commit this file to track changes over time
#
"""

    output_path = Path(__file__).parent.parent / output_file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(yaml)

    print("\n" + "=" * 80)
    print("YAML TEMPLATE CREATED")
    print("=" * 80)
    print(f"\nOutput: {output_path}")
    print(f"Size: {output_path.stat().st_size:,} bytes")
    print("\nTo use:")
    print(f"  1. Edit {output_file} with your modifications")
    print("  2. Convert to JSON: python interfaces/yaml_converter.py --to-json")
    print("  3. Apply: python apply_modifications.py modification.json")

    return True


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='YAML <-> JSON converter for modifications')
    parser.add_argument('--to-json', metavar='YAML_FILE', help='Convert YAML to JSON')
    parser.add_argument('--to-yaml', metavar='JSON_FILE', help='Convert JSON to YAML')
    parser.add_argument('--create-template', action='store_true', help='Create YAML template')

    args = parser.parse_args()

    if args.to_json:
        success = yaml_to_json(args.to_json)
    elif args.to_yaml:
        success = json_to_yaml(args.to_yaml)
    elif args.create_template:
        success = create_template_yaml()
    else:
        # Default: convert existing modification.json to YAML
        success = json_to_yaml()

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
