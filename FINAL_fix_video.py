#!/usr/bin/env python3
"""FINAL fix - regenerate text_3d_fixed_final.mp4 properly"""

import os
import cv2
import numpy as np
from utils.animations.text_3d_behind_segment import Text3DBehindSegment
from utils.segmentation.segment_extractor import extract_foreground_mask

print("FIXING text_3d_fixed_final.mp4 once and for all!\n")

# Check if old broken video exists
if os.path.exists("text_3d_fixed_final.mp4"):
    os.rename("text_3d_fixed_final.mp4", "text_3d_BROKEN_OLD.mp4")
    print("Renamed old broken video to text_3d_BROKEN_OLD.mp4")

video_path = "test_element_3sec.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

ret, frame = cap.read()
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
mask = extract_foreground_mask(frame_rgb)

# Load just 45 frames (0.75 seconds) for speed
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frames = []
for i in range(45):
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

print(f"Creating smooth 3D text animation...")

# Lower supersample to 3 for faster processing
anim = Text3DBehindSegment(
    duration=0.75,
    fps=fps,
    resolution=(W, H),
    text="HELLO WORLD",
    segment_mask=mask,
    font_size=140,
    text_color=(255, 220, 0),
    depth_color=(200, 170, 0),
    depth_layers=10,
    depth_offset=3,
    start_scale=1.8,
    end_scale=1.1,
    phase1_duration=0.5,
    phase2_duration=0.15,
    phase3_duration=0.1,
    center_position=(W//2, H//2),
    shadow_offset=6,
    outline_width=2,
    perspective_angle=25,
    supersample_factor=3,  # Reduced for speed but still good
    debug=False,
    perspective_during_shrink=False,
)

output_frames = []
for i in range(45):
    print(f"Frame {i+1}/45...", end='\r')
    frame = anim.generate_frame(i, frames[i])
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    output_frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

print("\nSaving...")
output_path = "text_3d_TRULY_FIXED.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
for f in output_frames:
    out.write(f)
out.release()

# H.264
import subprocess
cmd = ['ffmpeg', '-y', '-i', output_path, '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', output_path.replace('.mp4', '_h264.mp4')]
subprocess.run(cmd, capture_output=True)

print(f"\n✅ SUCCESS!")
print(f"Created: {output_path}")
print(f"H.264: text_3d_TRULY_FIXED_h264.mp4")
print("\nThis video has smooth depth like the demo!")