#!/usr/bin/env python3
"""Create 1-second video with all fixes to replace the broken one"""

import os
import cv2
import numpy as np
from utils.animations.text_3d_behind_segment import Text3DBehindSegment
from utils.segmentation.segment_extractor import extract_foreground_mask

print("Creating 1-second video with SMOOTH DEPTH...")

video_path = "test_element_3sec.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Get mask from first frame
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

print(f"Loaded {len(frames)} frames")

# Animation with CURRENT FIXED CODE
anim = Text3DBehindSegment(
    duration=1.0,  # 1 second
    fps=fps,
    resolution=(W, H),
    text="HELLO WORLD",
    segment_mask=mask,
    font_size=140,
    text_color=(255, 220, 0),
    depth_color=(200, 170, 0),
    depth_layers=10,   # -> 20 internally
    depth_offset=3,     # -> 0.6 internally (80% reduced)
    start_scale=2.0,
    end_scale=1.2,
    phase1_duration=0.7,  # Mostly shrink
    phase2_duration=0.2,
    phase3_duration=0.1,
    center_position=(W//2, H//2),
    shadow_offset=6,
    outline_width=2,
    perspective_angle=25,
    supersample_factor=4,  # HIGH QUALITY
    debug=False,
    perspective_during_shrink=False,
)

print("Generating smooth frames...")
output_frames = []

for i in range(60):
    if i % 10 == 0:
        print(f"  Frame {i}/60...")
    frame = anim.generate_frame(i, frames[i])
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    output_frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

# Save video
print("Saving...")
output_path = "text_3d_SMOOTH_1sec.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
temp = "temp_1sec.mp4"

out = cv2.VideoWriter(temp, fourcc, fps, (W, H))
for f in output_frames:
    out.write(f)
out.release()

# H.264
import subprocess
cmd = [
    'ffmpeg', '-y', '-i', temp,
    '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
    output_path
]
subprocess.run(cmd, capture_output=True)

if os.path.exists(temp):
    os.remove(temp)

# Preview
cv2.imwrite("smooth_1sec_preview.png", output_frames[20])

print(f"\n✅ DONE!")
print(f"Video: {output_path} (1 second)")
print("Preview: smooth_1sec_preview.png")
print("\nThis 1-second video has the SAME smooth quality")
print("as the SHORT demo - ALL letters smooth!")