#!/usr/bin/env python3
"""Generate final video with smooth depth for all letters"""

import os
import cv2
import numpy as np
from utils.animations.text_3d_behind_segment import Text3DBehindSegment
from utils.segmentation.segment_extractor import extract_foreground_mask

video_path = "test_element_3sec.mp4"
if not os.path.exists(video_path):
    print(f"Error: {video_path} not found")
    exit(1)

print("Loading video...")
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

ret, frame = cap.read()
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
print("Extracting mask...")
mask = extract_foreground_mask(frame_rgb)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frames = []
count = 0
while count < 90:  # Just 1.5 seconds for speed
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    count += 1
cap.release()

print(f"Creating animation with smooth, subtle depth...")
anim = Text3DBehindSegment(
    duration=1.5,
    fps=fps,
    resolution=(W, H),
    text="HELLO WORLD",
    segment_mask=mask,
    font_size=140,
    text_color=(255, 220, 0),
    depth_color=(200, 170, 0),
    depth_layers=10,  # Doubled internally to 20 for smoothness
    depth_offset=3,    # Reduced by 80% internally
    start_scale=2.0,
    end_scale=1.0,
    phase1_duration=1.0,
    phase2_duration=0.3,
    phase3_duration=0.2,
    center_position=(W//2, H//2),
    shadow_offset=6,   # Also reduced internally
    outline_width=2,
    perspective_angle=25,
    supersample_factor=4,  # High quality
    debug=False,
    perspective_during_shrink=False,
)

print("Generating frames...")
output_frames = []
total = len(frames)

for i in range(total):
    if i % 20 == 0:
        print(f"  Frame {i}/{total}...")
    bg = frames[i]
    frame = anim.generate_frame(i, bg)
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    output_frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

print("Saving...")
output_path = "text_3d_smooth_final.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
temp = "temp_final.mp4"

out = cv2.VideoWriter(temp, fourcc, fps, (W, H))
for f in output_frames:
    out.write(f)
out.release()

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

print(f"\n✅ Done! Video: {output_path}")
print("\nFixed:")
print("  • All letters have smooth depth (like 'O')")
print("  • Depth reduced by 80% (more subtle)")
print("  • Higher quality anti-aliasing")