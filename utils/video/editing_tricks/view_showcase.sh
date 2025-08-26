#!/bin/bash

echo "============================================"
echo "    VIDEO EDITING TRICKS SHOWCASE"
echo "============================================"
echo ""
echo "Available showcase videos:"
echo "1. Grid Layout (3x4) - All effects simultaneously"
echo "2. Sequential - Each effect one after another"
echo ""

# Check if videos exist
if [ -f "test_output/showcase_grid_h264.mp4" ]; then
    echo "✅ Grid showcase: test_output/showcase_grid_h264.mp4"
    echo "   Duration: 3 seconds"
    echo "   Size: $(ls -lh test_output/showcase_grid_h264.mp4 | awk '{print $5}')"
else
    echo "❌ Grid showcase not found"
fi

if [ -f "test_output/showcase_sequential_h264.mp4" ]; then
    echo "✅ Sequential showcase: test_output/showcase_sequential_h264.mp4"
    echo "   Duration: $(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 test_output/showcase_sequential_h264.mp4 | cut -d. -f1) seconds"
    echo "   Size: $(ls -lh test_output/showcase_sequential_h264.mp4 | awk '{print $5}')"
else
    echo "❌ Sequential showcase not found"
fi

echo ""
echo "Individual effect videos in test_output/:"
ls -1 test_output/*.mp4 | grep -v showcase | while read file; do
    name=$(basename "$file" .mp4)
    echo "  - $name"
done

echo ""
echo "To view in QuickTime Player (macOS):"
echo "  open test_output/showcase_grid_h264.mp4"
echo "  open test_output/showcase_sequential_h264.mp4"
echo ""
echo "To view in VLC or other player:"
echo "  vlc test_output/showcase_grid_h264.mp4"
echo ""