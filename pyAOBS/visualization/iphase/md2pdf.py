#!/usr/bin/env python3
"""
Convert Markdown (with LaTeX math) to PDF via pandoc + xelatex.

Display math: use Pandoc-style ``$$ ... $$`` blocks. Avoid ``\\[ ... \\]`` inside
indented list items (pandoc can mis-parse it and break LaTeX).

Usage:
  python md2pdf.py input.md [output.pdf]

Examples:
  python md2pdf.py TIME_DIFF_FORMULAS.md
  python md2pdf.py TIME_DIFF_FORMULAS.md TIME_DIFF_FORMULAS.pdf
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    if shutil.which("pandoc") is None:
        print("Error: pandoc not found.")
        print("Install (Ubuntu/Debian): sudo apt install pandoc")
        return 1

    if shutil.which("xelatex") is None:
        print("Error: xelatex not found.")
        print("Install (Ubuntu/Debian): sudo apt install texlive-xetex texlive-lang-chinese")
        return 1

    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("TIME_DIFF_FORMULAS.md")
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path.with_suffix(".pdf")

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        return 1

    cmd = [
        "pandoc",
        "-f",
        "markdown+tex_math_dollars",
        str(input_path),
        "-o",
        str(output_path),
        "--pdf-engine=xelatex",
        "-V",
        "mainfont=Noto Serif CJK SC",
        "-V",
        "sansfont=Noto Sans CJK SC",
        "-V",
        "geometry:margin=1in",
    ]

    print(f"Converting: {input_path} -> {output_path}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Conversion failed with exit code: {exc.returncode}")
        return exc.returncode

    print(f"Done: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
