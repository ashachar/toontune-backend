#!/bin/bash

# Process all do_re_mi videos with the video description generator in dry-run mode

echo "=========================================="
echo "Processing all do_re_mi videos (DRY RUN)"
echo "=========================================="
echo ""

# Array of all do_re_mi video files
videos=(
    "utils/editing_tricks/do_re_mi_effects/02_selective_color.mp4"
    "utils/editing_tricks/do_re_mi_effects/03_text_behind.mp4"
    "utils/editing_tricks/do_re_mi_effects/04_motion_text.mp4"
    "utils/editing_tricks/do_re_mi_effects/05_animated_subtitles.mp4"
    "utils/editing_tricks/do_re_mi_effects/06_floating.mp4"
    "utils/editing_tricks/do_re_mi_effects/07_smooth_zoom.mp4"
    "utils/editing_tricks/do_re_mi_effects/08_3d_photo.mp4"
    "utils/editing_tricks/do_re_mi_effects/09_rotation.mp4"
    "utils/editing_tricks/do_re_mi_effects/10_highlight_focus.mp4"
    "utils/editing_tricks/do_re_mi_effects/11_progress_bar.mp4"
    "utils/editing_tricks/do_re_mi_effects/12_video_in_text.mp4"
    "utils/editing_tricks/do_re_mi_effects/do_re_mi_showcase_master.mp4"
    "utils/editing_tricks/do_re_mi_quick_test/00_original.mp4"
    "utils/editing_tricks/do_re_mi_quick_test/02_floating.mp4"
    "utils/editing_tricks/do_re_mi_quick_test/03_zoom.mp4"
    "utils/editing_tricks/do_re_mi_quick_test/04_progress.mp4"
    "utils/editing_tricks/do_re_mi_quick_test/05_motion_text.mp4"
    "utils/editing_tricks/do_re_mi_quick_test/short_clip.mp4"
    "utils/editing_tricks/do_re_mi_quick_test/do_re_mi_showcase_final.mp4"
    "output/downsample_test/do_re_mi_small_full.mp4"
    "output/downsample_test/do_re_mi_tiny_full.mp4"
)

# Counter for processed videos
count=0
total=${#videos[@]}

# Process each video
for video in "${videos[@]}"; do
    count=$((count + 1))
    
    echo "[$count/$total] Processing: $video"
    echo "----------------------------------------"
    
    # Run the video description generator in dry-run mode
    python utils/video_description_generator.py "$video" --dry-run
    
    echo ""
done

echo "=========================================="
echo "✅ BATCH PROCESSING COMPLETE"
echo "Processed $count videos in dry-run mode"
echo "=========================================="

# Show summary of generated files
echo ""
echo "📁 Generated prompt files:"
find . -name "*_prompt_*_dryrun.txt" -type f -newermt "5 minutes ago" 2>/dev/null | sort

echo ""
echo "📊 prompts.yaml updated with $count video entries"