#!/usr/bin/env python3
"""Quick test of fade timing and slant fixes"""

import cv2
import numpy as np
from utils.animations.text_3d_behind_segment import Text3DBehindSegment

W, H, fps = 1166, 534, 60
bg = np.full((H, W, 3), 100, dtype=np.uint8)

print("Quick test of fixes...")

# Simple mask - bottom half is foreground
mask = np.zeros((H, W), dtype=np.uint8)
mask[H//2:, :] = 255  # Bottom half is foreground

anim = Text3DBehindSegment(
    duration=2.0,
    fps=fps,
    resolution=(W, H),
    text="HELLO WORLD",
    segment_mask=mask,
    font_size=140,
    text_color=(255, 220, 0),
    depth_color=(200, 170, 0),
    depth_layers=10,
    depth_offset=3,
    start_scale=2.0,
    end_scale=1.0,
    phase1_duration=1.2,  # 72 frames
    phase2_duration=0.4,
    phase3_duration=0.4,
    center_position=(W//2, H//2),
    shadow_offset=6,
    outline_width=2,
    perspective_angle=30,  # Should be ignored
    supersample_factor=2,  # Lower for speed
    debug=False,
    perspective_during_shrink=False,
)

# Test key frames
test_frames = [
    (0, "start"),
    (36, "midpoint_shrink"),  # 50% of phase1
    (72, "end_shrink"),       # End of phase1
    (119, "final")            # Last frame
]

print("\nGenerating key frames...")
for frame_num, name in test_frames:
    frame = anim.generate_frame(frame_num, bg)
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    
    filename = f"quick_fix_{name}.png"
    cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f"  Frame {frame_num:3d} ({name}): {filename}")

print("\n✅ Check the images:")
print("  • quick_fix_midpoint_shrink.png - should START fading here")
print("  • quick_fix_final.png - should be STRAIGHT (no slant)")
print("\nThe fade should happen at the midpoint, not after!")
print("The text should never become slanted!")