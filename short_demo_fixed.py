#!/usr/bin/env python3
"""Create a SHORT demo video showing the fixes work"""

import cv2
import numpy as np
from utils.animations.text_3d_behind_segment import Text3DBehindSegment

# Simple test - 30 frames (0.5 seconds)
W, H, fps = 1166, 534, 60
frames = []

# Create gradient background
for i in range(30):
    bg = np.full((H, W, 3), 100 + i*2, dtype=np.uint8)
    frames.append(bg)

print("Creating SHORT demo with smooth depth...")

anim = Text3DBehindSegment(
    duration=0.5,
    fps=fps,
    resolution=(W, H),
    text="HELLO WORLD",
    segment_mask=None,  # No mask for speed
    font_size=140,
    text_color=(255, 220, 0),
    depth_color=(200, 170, 0),
    depth_layers=10,
    depth_offset=3,
    start_scale=1.8,
    end_scale=1.2,
    phase1_duration=0.5,
    phase2_duration=0.0,
    phase3_duration=0.0,
    center_position=(W//2, H//2),
    shadow_offset=6,
    outline_width=2,
    perspective_angle=25,
    supersample_factor=4,
    debug=False,
    perspective_during_shrink=False,
)

output_frames = []
for i in range(30):
    print(f"Frame {i+1}/30...", end='\r')
    frame = anim.generate_frame(i, frames[i])
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    output_frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

print("\nSaving short demo...")
output_path = "SHORT_smooth_depth_demo.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
for f in output_frames:
    out.write(f)
out.release()

# Save first and last frame
cv2.imwrite("demo_first_frame.png", output_frames[0])
cv2.imwrite("demo_last_frame.png", output_frames[-1])

print(f"\n✅ Done!")
print(f"Video: {output_path} (0.5 seconds)")
print("First frame: demo_first_frame.png")
print("Last frame: demo_last_frame.png")
print("\nThis SHORT video shows:")
print("  • Smooth depth (no pixelation)")
print("  • Subtle 3D effect (80% reduced)")
print("  • All letters smooth like 'O'")