#!/usr/bin/env bash
set -euo pipefail

# Batch convert a folder of PNG/JPG/JPEG/TIF/TIFF/BMP to SVG
# Usage:
#   bash utils/vectorize/batch_convert.sh <input_dir> <output_dir>

INDIR="${1:-./input}"
OUTDIR="${2:-./out_svg}"
TMPDIR="${OUTDIR}/_tmp_pgm"

mkdir -p "${OUTDIR}" "${TMPDIR}"

# Choose ImageMagick cmd
IMCMD=""
if command -v magick >/dev/null 2>&1; then IMCMD="magick"; fi
if [[ -z "${IMCMD}" && "$(command -v convert || true)" != "" ]]; then IMCMD="convert"; fi
if [[ -z "${IMCMD}" ]]; then echo "ImageMagick not found."; exit 1; fi
if ! command -v potrace >/dev/null 2>&1; then echo "Potrace not found."; exit 1; fi

shopt -s nullglob
count=0
files=("${INDIR}"/*.{png,PNG,jpg,JPG,jpeg,JPEG,tif,TIF,tiff,TIFF,bmp,BMP})
if [[ ${#files[@]} -eq 0 ]]; then
  echo "No raster files found in ${INDIR}"; exit 1
fi

for f in "${files[@]}"; do
  name="$(basename "${f}")"; base="${name%.*}"
  echo ">> ${name}"
  "${IMCMD}" "${f}" -colorspace Gray -auto-level -blur 0x0.4 -threshold 60% -despeckle -despeckle "${TMPDIR}/${base}.pgm"
  potrace "${TMPDIR}/${base}.pgm" -s -t 2 -a 1.0 -O 0.2 -o "${OUTDIR}/${base}.svg" -q
  ((count++))
done

echo "All done → ${OUTDIR} (${count} files)"