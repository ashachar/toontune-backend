#!/usr/bin/env python3
"""Final demo with both fixes: proper fade timing and no slant"""

import os
import cv2
import numpy as np
from utils.animations.text_3d_behind_segment import Text3DBehindSegment
from utils.segmentation.segment_extractor import extract_foreground_mask

video_path = "test_element_3sec.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

ret, frame = cap.read()
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
print("Extracting mask...")
mask = extract_foreground_mask(frame_rgb)

# Load 60 frames (1 second)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frames = []
for i in range(60):
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

print("Creating animation with BOTH fixes...")

anim = Text3DBehindSegment(
    duration=1.0,
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
    phase1_duration=0.8,  # Most of the time is shrinking
    phase2_duration=0.1,
    phase3_duration=0.1,
    center_position=(W//2, H//2),
    shadow_offset=6,
    outline_width=2,
    perspective_angle=30,  # Ignored due to fix
    supersample_factor=3,
    debug=False,
)

print("Generating frames...")
output_frames = []

for i in range(60):
    if i % 10 == 0:
        print(f"  Frame {i}/60...")
    frame = anim.generate_frame(i, frames[i])
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    output_frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

print("Saving...")
output_path = "BOTH_FIXES_demo.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
for f in output_frames:
    out.write(f)
out.release()

# H.264
import subprocess
cmd = ['ffmpeg', '-y', '-i', output_path, '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', output_path.replace('.mp4', '_h264.mp4')]
subprocess.run(cmd, capture_output=True)

print(f"\n✅ Done!")
print(f"Video: BOTH_FIXES_demo_h264.mp4")
print("\nFixes applied:")
print("  1. ✓ Fade starts at 50% of shrink (when passing behind)")
print("  2. ✓ No slanting (perspective disabled)")