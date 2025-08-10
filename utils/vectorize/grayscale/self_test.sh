#!/usr/bin/env bash
set -euo pipefail

# Self-test: creates a simple PNG doodle, runs png_to_svg.sh, checks outputs exist

DIR="utils/vectorize"
IN="${DIR}/_test_input"
OUT="${DIR}/_test_out"
mkdir -p "${IN}" "${OUT}"

# Simple black squiggle on white using ImageMagick
IMCMD=""
if command -v magick >/dev/null 2>&1; then IMCMD="magick"; fi
if [[ -z "${IMCMD}" && "$(command -v convert || true)" != "" ]]; then IMCMD="convert"; fi
if [[ -z "${IMCMD}" ]]; then echo "ImageMagick not found. Install first."; exit 1; fi
if ! command -v potrace >/dev/null 2>&1; then echo "Potrace not found. Install first."; exit 1; fi

# Generate a sample doodle
"${IMCMD}" -size 256x256 xc:white \
  -fill none -stroke black -strokewidth 6 \
  -draw "path 'M 20,200 C 40,20 220,20 236,200'" \
  -draw "circle 128,128 128,90" \
  "${IN}/doodle.png"

echo "Generated ${IN}/doodle.png"

bash "${DIR}/png_to_svg.sh" "${IN}/doodle.png" "${OUT}/doodle.svg"

if [[ -s "${OUT}/doodle.svg" ]]; then
  echo "Self-test OK: ${OUT}/doodle.svg exists and is non-empty"
else
  echo "Self-test FAILED: output SVG missing or empty"; exit 1
fi
