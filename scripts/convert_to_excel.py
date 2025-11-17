"""
Convert YAML configuration files to Excel format
"""
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from irb_segmentation import SegmentationConfig

def main():
    # Convert each YAML config to Excel
    yaml_files = [
        'config_examples/german_credit.yaml',
        'config_examples/lending_club.yaml',
        'config_examples/taiwan_credit.yaml'
    ]

    for yaml_file in yaml_files:
        try:
            # Load YAML
            config = SegmentationConfig.from_yaml(yaml_file)

            # Export to Excel
            excel_file = yaml_file.replace('.yaml', '.xlsx')
            config.to_excel(excel_file, template='standard')

            print(f'✓ Converted: {yaml_file} -> {excel_file}')
        except Exception as e:
            print(f'✗ Error converting {yaml_file}: {e}')

    print('\n✓ All Excel configs created successfully!')

if __name__ == '__main__':
    main()
