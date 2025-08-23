#!/usr/bin/env python3
"""Test the fixed depth solution"""

import cv2
import numpy as np
from utils.animations.text_3d_behind_segment_fixed_depth import Text3DBehindSegment

print("TESTING FIXED DEPTH")
print("-" * 40)
print("Depth now stays constant after going behind!")

video_path = "test_element_3sec.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frames = []
for i in range(45):
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

anim = Text3DBehindSegment(
    duration=0.75,
    fps=fps,
    resolution=(W, H),
    text="HELLO WORLD",
    segment_mask=None,
    font_size=140,
    text_color=(255, 220, 0),
    depth_color=(200, 170, 0),
    depth_layers=8,
    depth_offset=3,
    start_scale=2.0,
    end_scale=1.0,
    shrink_duration=0.6,
    wiggle_duration=0.15,
    center_position=(W//2, H//2),
    shadow_offset=6,
    outline_width=2,
    perspective_angle=0,
    supersample_factor=2,
    debug=True,  # Enable to see depth_scale values
)

output_frames = []
for i in range(len(frames)):
    if i % 10 == 0:
        print(f"Frame {i}/{len(frames)}...")
    frame = anim.generate_frame(i, frames[i])
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    output_frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

print("\nSaving...")
out = cv2.VideoWriter("depth_fixed.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
for f in output_frames:
    out.write(f)
out.release()

import subprocess
subprocess.run([
    'ffmpeg', '-y', '-i', 'depth_fixed.mp4',
    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
    'DEPTH_FIXED_h264.mp4'
], capture_output=True)

import os
os.remove('depth_fixed.mp4')

print("\n✅ Done! Video: DEPTH_FIXED_h264.mp4")
print("The depth now stays constant after text goes behind!")