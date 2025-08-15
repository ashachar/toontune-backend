#!/bin/bash

# Enhanced FFmpeg chromakey script with edge cleanup
# Usage: ./ffmpeg_chromakey_enhanced.sh input.mp4 output.webm

INPUT="$1"
OUTPUT="$2"

if [ -z "$INPUT" ] || [ -z "$OUTPUT" ]; then
    echo "Usage: $0 input.mp4 output.webm"
    exit 1
fi

# Advanced chromakey with edge processing
ffmpeg -i "$INPUT" \
  -filter_complex "\
    [0:v]chromakey=green:0.12:0.06[ck]; \
    [ck]despill=type=green:mix=0.5:expand=0[dsp]; \
    [dsp]erosion=coordinates=3x3:threshold0=200[eroded]; \
    [eroded]format=yuva420p[out]" \
  -map "[out]" \
  -c:v libvpx-vp9 \
  -pix_fmt yuva420p \
  -b:v 0 \
  -crf 25 \
  "$OUTPUT" -y

echo "✅ Processed: $OUTPUT"