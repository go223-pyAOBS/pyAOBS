#!/usr/bin/env bash
set -euo pipefail

# Convert Markdown (with LaTeX math) to PDF using pandoc + xelatex.
# Display math: use $$ ... $$ (Pandoc). Avoid \[ ... \] under list-indent.
# Usage:
#   ./md2pdf.sh input.md [output.pdf]
# Examples:
#   ./md2pdf.sh TIME_DIFF_FORMULAS.md
#   ./md2pdf.sh TIME_DIFF_FORMULAS.md TIME_DIFF_FORMULAS.pdf

if ! command -v pandoc >/dev/null 2>&1; then
  echo "Error: pandoc not found. Install with: sudo apt install pandoc"
  exit 1
fi

if ! command -v xelatex >/dev/null 2>&1; then
  echo "Error: xelatex not found. Install with: sudo apt install texlive-xetex texlive-lang-chinese"
  exit 1
fi

INPUT="${1:-TIME_DIFF_FORMULAS.md}"
OUTPUT="${2:-${INPUT%.md}.pdf}"

if [[ ! -f "$INPUT" ]]; then
  echo "Error: input file not found: $INPUT"
  exit 1
fi

echo "Converting: $INPUT -> $OUTPUT"
pandoc -f markdown+tex_math_dollars "$INPUT" \
  -o "$OUTPUT" \
  --pdf-engine=xelatex \
  -V mainfont="Noto Serif CJK SC" \
  -V sansfont="Noto Sans CJK SC" \
  -V geometry:margin=1in

echo "Done: $OUTPUT"
