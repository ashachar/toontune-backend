#!/usr/bin/env python3
"""Test with debug logging to find stale mask bug."""

import os
os.environ['DEBUG_3D_TEXT'] = '1'

from utils.animations.apply_3d_text_animation import apply_animation_to_video

# Use the test video with moving person
result = apply_animation_to_video(
    video_path="test_person_h264.mp4",
    text="Hello",
    font_size=80,
    position=(640, 360),
    motion_duration=0.5,  # Short motion
    dissolve_duration=1.5,  # Longer dissolve to see movement
    output_path="outputs/mask_debug_test.mp4",
    final_opacity=0.7,
    supersample=2,  # Low for speed
    debug=True
)

print(f"\n✅ Created: {result}")
print("\nCheck the debug output above for [MASK_DEBUG] and [OCCLUSION_DEBUG] lines.")