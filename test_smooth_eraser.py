#!/usr/bin/env python3
"""Test smooth eraser wipe directly"""

from utils.animations.smooth_eraser_wipe import create_masked_eraser_wipe

# Test with the files that exist
character_video = "outputs/temp_eraser_source.mp4"
original_video = "uploads/assets/runway_experiment/runway_demo_input.mp4"
eraser_image = "uploads/assets/images/eraser.png"
output_video = "outputs/test_smooth_eraser.mp4"

print("Testing smooth eraser wipe...")
result = create_masked_eraser_wipe(
    character_video=character_video,
    original_video=original_video,
    eraser_image=eraser_image,
    output_video=output_video,
    wipe_start=0,
    wipe_duration=0.6
)

if result:
    print(f"Success! Output: {output_video}")
else:
    print("Failed to create eraser wipe")