#!/usr/bin/env python3
"""Quick final demo with per-layer masking fix"""

import os
import cv2
import numpy as np
from utils.animations.text_3d_behind_segment_fixed import Text3DBehindSegment

print("QUICK DEMO - Per-layer masking fix")
print("-" * 40)

video_path = "test_element_3sec.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Just 30 frames focusing on the critical moment
cap.set(cv2.CAP_PROP_POS_FRAMES, 35)
frames = []
for i in range(30):
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

anim = Text3DBehindSegment(
    duration=0.5,
    fps=fps,
    resolution=(W, H),
    text="HELLO WORLD",
    segment_mask=None,
    font_size=140,
    text_color=(255, 220, 0),
    depth_color=(200, 170, 0),
    depth_layers=8,  # Fewer layers for speed
    depth_offset=3,
    start_scale=1.6,
    end_scale=1.0,
    phase1_duration=0.2,
    phase2_duration=0.25,
    phase3_duration=0.05,
    center_position=(W//2, H//2),
    shadow_offset=6,
    outline_width=2,
    perspective_angle=0,
    supersample_factor=2,  # Lower for speed
    debug=False,
)

output_frames = []
for i in range(len(frames)):
    print(f"Frame {i+1}/{len(frames)}", end='\r')
    frame = anim.generate_frame(i, frames[i])
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    output_frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

print("\nSaving...")
out = cv2.VideoWriter("quick_demo.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
for f in output_frames:
    out.write(f)
out.release()

# H.264 conversion
import subprocess
subprocess.run([
    'ffmpeg', '-y', '-i', 'quick_demo.mp4',
    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
    'QUICK_DEMO_FIXED_h264.mp4'
], capture_output=True)

os.remove('quick_demo.mp4')

# Save critical frame
cv2.imwrite("critical_W_frame.png", output_frames[13])

print("\n✅ Done!")
print("Video: QUICK_DEMO_FIXED_h264.mp4")
print("Frame: critical_W_frame.png")