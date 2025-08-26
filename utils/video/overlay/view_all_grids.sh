#!/bin/bash

# Script to view all generated grid frames
echo "Opening all grid frames for review..."
echo "================================="
echo ""

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
GRID_DIR="$SCRIPT_DIR/grid_overlays"

# List all grid images with their corresponding prompts
echo "Grid frames generated for text overlays:"
echo ""

for grid_file in "$GRID_DIR"/grid_*.png; do
    if [ -f "$grid_file" ]; then
        filename=$(basename "$grid_file")
        # Extract info from filename
        scene=$(echo "$filename" | sed 's/grid_scene\([0-9]*\)_overlay.*/Scene \1/')
        word=$(echo "$filename" | sed 's/.*overlay[0-9]*_\(.*\)\.png/\1/')
        
        echo "- $scene: '$word' - $filename"
        
        # Show corresponding prompt file
        prompt_file="${grid_file%.png}.txt"
        prompt_file=$(echo "$prompt_file" | sed 's/grid_/prompt_/')
        
        if [ -f "$prompt_file" ]; then
            echo "  Prompt file: $(basename "$prompt_file")"
        fi
        echo ""
    fi
done

echo "================================="
echo "Total grid frames: $(ls -1 "$GRID_DIR"/grid_*.png 2>/dev/null | wc -l)"
echo ""
echo "To view all grids in Preview (macOS):"
echo "open $GRID_DIR/grid_*.png"
echo ""
echo "To view a specific grid:"
echo "open $GRID_DIR/grid_scene1_overlay1_Let\'s.png"