#!/usr/bin/env python3
"""Test the improved depth smoothness and 80% reduction"""

import cv2
import numpy as np
from utils.animations.text_3d_behind_segment import Text3DBehindSegment

# Quick test
W, H, fps = 1166, 534, 60
bg = np.full((H, W, 3), 245, dtype=np.uint8)

print("Testing improved depth (80% smaller, smoother)...")
anim = Text3DBehindSegment(
    duration=3.0,
    fps=fps,
    resolution=(W, H),
    text="HELLO WORLD",
    segment_mask=None,
    font_size=140,
    text_color=(255, 220, 0),
    depth_color=(200, 170, 0),
    depth_layers=10,  # Will be doubled internally for smoothness
    depth_offset=3,    # Will be reduced by 80% internally
    start_scale=1.5,   # Test at medium scale
    end_scale=1.5,
    phase1_duration=3.0,
    phase2_duration=0.0,
    phase3_duration=0.0,
    center_position=(W//2, H//2),
    shadow_offset=6,
    outline_width=2,
    perspective_angle=25,
    supersample_factor=4,  # Higher quality
    debug=False,
    perspective_during_shrink=False,
)

# Generate one frame
print("Generating test frame...")
frame = anim.generate_frame(0, bg)

if frame.shape[2] == 4:
    frame = frame[:, :, :3]

cv2.imwrite("smooth_depth_test.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
print("✓ Saved: smooth_depth_test.png")
print("\nImprovements:")
print("  • Depth reduced by 80% (more subtle)")
print("  • 2x more depth layers for smoother gradient")
print("  • 4x supersampling for all letters")
print("  • Shadow also reduced to match")