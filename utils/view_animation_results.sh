#!/bin/bash

# Script to view animation results

echo "============================================================"
echo "🎬 Animation Pipeline Results Viewer"
echo "============================================================"
echo ""
echo "Choose what to view:"
echo "1) Labeled frame (segmentation visualization)"
echo "2) Segmented video with annotations"
echo "3) Sailor emergence animation"
echo "4) Open all in separate windows"
echo "5) Exit"
echo ""
read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo "Opening labeled frame..."
        open output/animate_object_in_video/segmentation/labeled_frame.png
        ;;
    2)
        echo "Playing segmented video..."
        ffplay -autoexit output/animate_object_in_video/segmentation/final_video_h264.mp4
        ;;
    3)
        echo "Playing sailor emergence animation..."
        ffplay -autoexit -loop 0 output/animate_object_in_video/emergence/emergence_animation.mp4
        ;;
    4)
        echo "Opening all results..."
        open output/animate_object_in_video/segmentation/labeled_frame.png
        ffplay output/animate_object_in_video/segmentation/final_video_h264.mp4 &
        ffplay -loop 0 output/animate_object_in_video/emergence/emergence_animation.mp4 &
        ;;
    5)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Please run the script again."
        ;;
esac