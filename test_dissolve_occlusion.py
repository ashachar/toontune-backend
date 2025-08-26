#!/usr/bin/env python3
"""Test dissolve with debug output to prove occlusion is/isn't working."""

import sys
import os
sys.path.append('.')

from utils.animations.apply_3d_text_animation import apply_3d_text_to_video

# Enable debug mode
os.environ['DEBUG_3D_TEXT'] = '1'

# Run with debug to see occlusion proof
result = apply_3d_text_to_video(
    video_path="cartoons/man_medium.mp4",
    text="Hello World",
    font_size=40,
    position=(640, 300),
    motion_frames=19,
    dissolve_frames=50,
    debug=True
)

print(f"\n✅ Video created: {result}")
print("\nNow check the console output above for [OCCLUSION_PROOF] lines.")