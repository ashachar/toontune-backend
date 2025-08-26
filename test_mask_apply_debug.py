#!/usr/bin/env python3
"""Test with detailed mask application debug."""

import os
os.environ['DEBUG_3D_TEXT'] = '1'

from utils.animations.apply_3d_text_animation import apply_animation_to_video

# Run test with just H to track clearly
result = apply_animation_to_video(
    video_path="test_person_h264.mp4",
    text="H",
    font_size=100,
    position=(500, 360),
    motion_duration=0.3,  # Very short motion
    dissolve_duration=1.0,  # Focus on dissolve
    output_path="outputs/h_mask_debug.mp4",
    final_opacity=0.7,
    supersample=2,
    debug=True
)

print(f"\n✅ Created: {result}")
print("\nCheck [APPLY_DEBUG] lines above to see mask edge position each frame.")