#!/usr/bin/env bash
set -euo pipefail

# Color-preserving PNG/JPG -> SVG using ImageMagick + Potrace layering
# Strategy:
# 1) Quantize to a limited palette (default 8 colors, tunable via $PALETTE)
# 2) For each unique color, build a binary mask and trace it with Potrace, assigning that color as fill
# 3) Merge all traced paths into a single SVG with proper viewBox and transform
#
# Usage:
#   bash utils/vectorize/png_to_svg_color.sh <input.png> <output.svg> [palette_size]
# Env overrides (optional):
#   PALETTE=8 TURDSIZE=2 ALPHAMAX=1.0 OPTTOL=0.2 BLUR=0x0.4

INIMG="${1:-}"
OUTSVG="${2:-}"
PALETTE="${3:-${PALETTE:-8}}"

if [[ -z "${INIMG}" || -z "${OUTSVG}" ]]; then
  echo "Usage: bash utils/vectorize/png_to_svg_color.sh <input.png> <output.svg> [palette_size]"
  exit 1
fi

TURDSIZE=${TURDSIZE:-2}
ALPHAMAX=${ALPHAMAX:-1.0}
OPTTOL=${OPTTOL:-0.2}
BLUR=${BLUR:-0x0.4}

# Choose ImageMagick cmd
IMCMD=""
if command -v magick >/dev/null 2>&1; then IMCMD="magick"; fi
if [[ -z "${IMCMD}" && "$(command -v convert || true)" != "" ]]; then IMCMD="convert"; fi
if [[ -z "${IMCMD}" ]]; then echo "ImageMagick not found."; exit 1; fi
if ! command -v potrace >/dev/null 2>&1; then echo "Potrace not found."; exit 1; fi

mkdir -p "$(dirname "${OUTSVG}")"
WORKDIR="$(dirname "${OUTSVG}")/._colortrace_$RANDOM$RANDOM"
mkdir -p "${WORKDIR}"
trap 'rm -rf "${WORKDIR}"' EXIT

TMPQ="${WORKDIR}/quant.png"
WH=$("${IMCMD}" identify -format "%w %h" "${INIMG}")
W=$(printf "%s" "${WH}" | awk '{print $1}')
H=$(printf "%s" "${WH}" | awk '{print $2}')

# 1) Quantize: disable dithering to get clean flat regions
"${IMCMD}" "${INIMG}" -alpha off -blur "${BLUR}" -dither None -colors "${PALETTE}" +dither -define png:color-type=2 "${TMPQ}"

# 2) Collect unique colors (hex #RRGGBB)
COLORS_FILE="${WORKDIR}/colors.txt"
"${IMCMD}" "${TMPQ}" -unique-colors -depth 8 txt:- | sed -n 's/.*#\([0-9A-Fa-f]\{6\}\).*/#\1/p' | sort -u > "${COLORS_FILE}"
if [[ ! -s "${COLORS_FILE}" ]]; then
  echo "No colors found after quantization."; exit 1
fi

# 3) For each color, create mask and trace with fill color
SVGPATHS_FILE="${WORKDIR}/paths.svgfrag"
: > "${SVGPATHS_FILE}"

idx=0
while IFS= read -r COLOR; do
  ((idx++))
  mask="${WORKDIR}/mask_${idx}.pgm"
  # Build binary mask: target color -> black, others -> white
  "${IMCMD}" "${TMPQ}" -alpha off \
    -fill white -colorize 100 \
    -fill black -opaque "${COLOR}" \
    -threshold 50% \
    "${mask}"

  # Trace mask as filled paths with the original COLOR
  svgfile="${WORKDIR}/layer_${idx}.svg"
  potrace "${mask}" -s -t "${TURDSIZE}" -a "${ALPHAMAX}" -O "${OPTTOL}" -C "${COLOR}" -o "${svgfile}" -q

  # Extract <path .../> elements and append
  # This grabs all self-closing path tags regardless of spacing
  grep -o '<path[^>]*/>' "${svgfile}" >> "${SVGPATHS_FILE}" || true

done < "${COLORS_FILE}"

# 4) Assemble final SVG
cat > "${OUTSVG}" <<EOF
<svg xmlns="http://www.w3.org/2000/svg" version="1.1" viewBox="0 0 ${W} ${H}">
  <g transform="translate(0,${H}) scale(1,-1)">
$(cat "${SVGPATHS_FILE}")
  </g>
</svg>
EOF

echo "Wrote (color): ${OUTSVG}"