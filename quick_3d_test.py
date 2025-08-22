#!/usr/bin/env python3
"""Quick test of 3D text fixes - generates just a few frames"""

import os
import cv2
import numpy as np
from utils.animations.text_3d_behind_segment import Text3DBehindSegment

# Simple test without mask extraction
W, H, fps = 1166, 534, 60

# Create animation
print("Creating 3D text animation with fixes...")
anim = Text3DBehindSegment(
    duration=3.0,
    fps=fps, 
    resolution=(W, H),
    text="HELLO WORLD",
    segment_mask=None,  # No mask for speed
    font_size=140,
    text_color=(255, 220, 0),
    depth_color=(200, 170, 0),
    depth_layers=10,
    depth_offset=3,
    start_scale=2.2,
    end_scale=0.9,
    phase1_duration=1.2,
    phase2_duration=0.6, 
    phase3_duration=1.2,
    center_position=(W//2, H//2),
    shadow_offset=6,
    outline_width=2,
    perspective_angle=25,
    supersample_factor=3,
    debug=True,
    perspective_during_shrink=False,
)

# Test just a few key frames
test_frames = [0, 30, 60, 90, 120, 150, 179]
bg = np.full((H, W, 3), 245, dtype=np.uint8)  # Light gray

print("\nGenerating test frames...")
for i in test_frames:
    print(f"\nFrame {i}:")
    frame = anim.generate_frame(i, bg)
    
    # Save this frame
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    cv2.imwrite(f"quick_3d_frame_{i:03d}.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

print("\n✅ Test complete! Check quick_3d_frame_*.png files")