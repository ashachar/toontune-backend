#!/usr/bin/env python3
"""Quick verification that the depth fix is applied"""

import cv2
import numpy as np
from utils.animations.text_3d_behind_segment import Text3DBehindSegment

# Test with just a few frames
W, H, fps = 1166, 534, 60
bg = np.full((H, W, 3), 100, dtype=np.uint8)  # Dark gray

print("Generating 3 test frames with current code...")

anim = Text3DBehindSegment(
    duration=3.0,
    fps=fps,
    resolution=(W, H),
    text="HELLO WORLD",
    segment_mask=None,
    font_size=140,
    text_color=(255, 220, 0),
    depth_color=(200, 170, 0),
    depth_layers=10,
    depth_offset=3,
    start_scale=1.8,
    end_scale=0.9,
    phase1_duration=1.2,
    phase2_duration=0.6,
    phase3_duration=1.2,
    center_position=(W//2, H//2),
    shadow_offset=6,
    outline_width=2,
    perspective_angle=25,
    supersample_factor=4,
    debug=False,
    perspective_during_shrink=False,
)

# Test frames at different scales
test_frames = [0, 36, 72]  # Start, 0.6s, 1.2s

for i in test_frames:
    frame = anim.generate_frame(i, bg)
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    cv2.imwrite(f"verify_frame_{i:03d}.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f"  ✓ verify_frame_{i:03d}.png")

print("\n✅ Check these frames - they should have:")
print("  • Smooth depth (no pixelation)")
print("  • Subtle depth (80% smaller)")
print("  • All letters smooth like 'O'")