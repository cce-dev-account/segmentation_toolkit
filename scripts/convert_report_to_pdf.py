"""
PDF Conversion Helper

Converts markdown segment reports to PDF format using pandoc.
Checks for pandoc installation and provides helpful error messages.
"""

import sys
import subprocess
import shutil
from pathlib import Path
from typing import Optional


def check_pandoc_installed() -> bool:
    """
    Check if pandoc is installed and accessible.

    Returns:
        True if pandoc is installed, False otherwise
    """
    return shutil.which('pandoc') is not None


def check_latex_installed() -> bool:
    """
    Check if a LaTeX engine is installed (xelatex preferred).

    Returns:
        True if LaTeX is installed, False otherwise
    """
    return shutil.which('xelatex') is not None or shutil.which('pdflatex') is not None


def convert_markdown_to_pdf(
    markdown_path: str,
    output_path: Optional[str] = None,
    pdf_engine: str = 'xelatex',
    verbose: bool = True
) -> bool:
    """
    Convert markdown file to PDF using pandoc.

    Args:
        markdown_path: Path to markdown file
        output_path: Path for output PDF (default: same name with .pdf extension)
        pdf_engine: LaTeX engine to use ('xelatex', 'pdflatex', 'lualatex')
        verbose: Print progress messages

    Returns:
        True if successful, False otherwise
    """
    markdown_path = Path(markdown_path)

    if not markdown_path.exists():
        print(f"Error: Markdown file not found: {markdown_path}")
        return False

    # Default output path
    if output_path is None:
        output_path = markdown_path.with_suffix('.pdf')
    else:
        output_path = Path(output_path)

    if verbose:
        print(f"\nConverting markdown to PDF...")
        print(f"  Input:  {markdown_path}")
        print(f"  Output: {output_path}")

    # Check pandoc installation
    if not check_pandoc_installed():
        print("\nError: pandoc is not installed!")
        print("\nTo install pandoc:")
        print("  - Windows: Download from https://pandoc.org/installing.html")
        print("  - macOS: brew install pandoc")
        print("  - Linux: sudo apt-get install pandoc (Ubuntu/Debian)")
        print("           sudo yum install pandoc (Fedora/RHEL)")
        return False

    # Check LaTeX installation
    if not check_latex_installed():
        print("\nWarning: LaTeX engine not found!")
        print("PDF conversion requires a LaTeX installation.")
        print("\nTo install LaTeX:")
        print("  - Windows: Install MiKTeX (https://miktex.org/)")
        print("  - macOS: Install MacTeX (brew install --cask mactex)")
        print("  - Linux: sudo apt-get install texlive-xetex (Ubuntu/Debian)")
        return False

    # Build pandoc command
    cmd = [
        'pandoc',
        str(markdown_path),
        '-o', str(output_path),
        '--pdf-engine=' + pdf_engine,
        '--toc',  # Table of contents
        '--toc-depth=2',  # 2 levels in TOC
        '-V', 'geometry:margin=1in',  # 1 inch margins
        '-V', 'linkcolor:blue',  # Blue links
        '-V', 'fontsize=11pt'  # 11pt font
    ]

    try:
        if verbose:
            print("\nRunning pandoc...")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        if verbose:
            print(f"\nSuccess! PDF created: {output_path}")
            print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

        return True

    except subprocess.CalledProcessError as e:
        print(f"\nError during PDF conversion:")
        print(f"  Return code: {e.returncode}")
        if e.stdout:
            print(f"  Output: {e.stdout}")
        if e.stderr:
            print(f"  Error: {e.stderr}")
        return False
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return False


def main():
    """Command-line interface for PDF conversion."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert markdown segment report to PDF',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert segment report to PDF
  python convert_report_to_pdf.py output/lending_club_categorical/segment_report.md

  # Specify output path
  python convert_report_to_pdf.py segment_report.md -o my_report.pdf

  # Use different LaTeX engine
  python convert_report_to_pdf.py segment_report.md --engine pdflatex

  # Quiet mode
  python convert_report_to_pdf.py segment_report.md --quiet
        """
    )

    parser.add_argument(
        'markdown_file',
        type=str,
        help='Path to markdown file to convert'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output PDF path (default: same name as input with .pdf extension)'
    )

    parser.add_argument(
        '--engine',
        type=str,
        default='xelatex',
        choices=['xelatex', 'pdflatex', 'lualatex'],
        help='LaTeX engine to use (default: xelatex)'
    )

    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )

    args = parser.parse_args()

    # Convert
    success = convert_markdown_to_pdf(
        markdown_path=args.markdown_file,
        output_path=args.output,
        pdf_engine=args.engine,
        verbose=not args.quiet
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
