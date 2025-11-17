SEGMENTATION MODIFICATION TEMPLATES
====================================

You have 3 CSV files to edit:

1. modification_template_segment_actions.csv
   - Decide which segments to merge
   - Mark segments as KEEP or MERGE
   - Specify merge targets

2. modification_template_thresholds.csv
   - Add forced business rule splits
   - Specify feature thresholds (e.g., FICO 650)
   - Document regulatory requirements

3. modification_template_parameters.csv
   - Adjust model parameters
   - Change depth, minimums, density constraints
   - Document reasons for changes

WORKFLOW:
---------
1. Open files in Excel/LibreOffice/Google Sheets
2. Make your modifications
3. Save the files (keep CSV format)
4. Run: python interfaces/excel_to_json.py
5. This generates modification.json
6. Apply with: python apply_modifications.py modification.json

TIPS:
-----
- Edit one file at a time
- Start with segment_actions (merge similar segments)
- Then add thresholds (business rules)
- Finally adjust parameters (if needed)
- Always document your reasons in Notes column

HELP:
-----
- See MODELER_GUIDE.md for detailed instructions
- Check INTERFACE_COMPARISON.md for which interface to use
- View examples in interfaces/templates/examples/
