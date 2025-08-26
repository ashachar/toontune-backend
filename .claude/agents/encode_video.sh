#!/bin/bash
#
# Quick wrapper for the video encoder agent
# Usage: ./encode_video.sh <input_video> [output_video]
#

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if input is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <input_video> [output_video]"
    echo "       $0 --batch <directory> [pattern]"
    exit 1
fi

# Run the video encoder
python3 "$SCRIPT_DIR/video_encoder.py" "$@"