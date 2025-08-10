#!/usr/bin/env bash
set -euo pipefail

# Convert a single PNG/JPG to SVG via ImageMagick + Potrace (grayscale outline tracing)
# Usage:
#   bash utils/vectorize/png_to_svg.sh <input.(png|jpg|jpeg|tif|tiff|bmp)> <output.svg>

INPNG="${1:-}"
OUTSVG="${2:-}"

if [[ -z "${INPNG}" || -z "${OUTSVG}" ]]; then
  echo "Usage: bash utils/vectorize/png_to_svg.sh <input.png> <output.svg>"
  exit 1
fi

mkdir -p "$(dirname "${OUTSVG}")"
TMPPGM="${OUTSVG%.svg}.pgm"

# Choose ImageMagick cmd: magick (IM7) or convert (IM6)
IMCMD=""
if command -v magick >/dev/null 2>&1; then IMCMD="magick"; fi
if [[ -z "${IMCMD}" && "$(command -v convert || true)" != "" ]]; then IMCMD="convert"; fi
if [[ -z "${IMCMD}" ]]; then echo "ImageMagick not found."; exit 1; fi

if ! command -v potrace >/dev/null 2>&1; then
  echo "Potrace not found."
  exit 1
fi

# Preprocess to clean, high-contrast 1-bit-ish PGM
"${IMCMD}" "${INPNG}" \
  -colorspace Gray \
  -auto-level \
  -blur 0x0.4 \
  -threshold 60% \
  -despeckle -despeckle \
  "${TMPPGM}"

# Trace to SVG
potrace "${TMPPGM}" -s -t 2 -a 1.0 -O 0.2 -o "${OUTSVG}" -q

echo "Wrote: ${OUTSVG}"