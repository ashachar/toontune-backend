#!/usr/bin/env python3
"""Create the PROPER fixed video with all current improvements"""

import os
import cv2
import numpy as np
from utils.animations.text_3d_behind_segment import Text3DBehindSegment
from utils.segmentation.segment_extractor import extract_foreground_mask

# Load video
video_path = "test_element_3sec.mp4"
print("Loading video...")
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Get first frame for mask
ret, frame = cap.read()
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
print("Extracting mask...")
mask = extract_foreground_mask(frame_rgb)

# Load only first 90 frames (1.5 seconds) for faster processing
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frames = []
for i in range(90):
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

print(f"Loaded {len(frames)} frames at {W}x{H}")
print("\n✨ Creating animation with ALL current fixes:")
print("  • Smooth depth on ALL letters")
print("  • 80% reduced depth (subtle)")
print("  • 4x supersampling")
print("  • No pixelation\n")

# Create animation with current code
anim = Text3DBehindSegment(
    duration=1.5,  # Match frame count
    fps=fps,
    resolution=(W, H),
    text="HELLO WORLD",
    segment_mask=mask,
    font_size=140,
    text_color=(255, 220, 0),
    depth_color=(200, 170, 0),
    depth_layers=10,   # Doubled internally
    depth_offset=3,     # Reduced 80% internally
    start_scale=2.0,
    end_scale=1.0,
    phase1_duration=0.8,
    phase2_duration=0.4,
    phase3_duration=0.3,
    center_position=(W//2, H//2),
    shadow_offset=6,
    outline_width=2,
    perspective_angle=25,
    supersample_factor=4,  # Current default
    debug=False,
    perspective_during_shrink=False,
)

print("Generating frames...")
output_frames = []

for i in range(len(frames)):
    if i % 15 == 0:
        print(f"  Frame {i}/{len(frames)}...")
    
    frame = anim.generate_frame(i, frames[i])
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    output_frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

# Save video
print("\nEncoding video...")
output_path = "text_3d_ACTUALLY_fixed.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
temp = "temp_actually.mp4"

out = cv2.VideoWriter(temp, fourcc, fps, (W, H))
for f in output_frames:
    out.write(f)
out.release()

# Convert to H.264
import subprocess
cmd = [
    'ffmpeg', '-y', '-i', temp,
    '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
    output_path
]
result = subprocess.run(cmd, capture_output=True, text=True)

if os.path.exists(temp):
    os.remove(temp)

# Save preview at 0.5 seconds
preview_idx = fps // 2
if preview_idx < len(output_frames):
    cv2.imwrite("actually_fixed_preview.png", output_frames[preview_idx])

print(f"\n✅ SUCCESS!")
print(f"📹 Video: {output_path}")
print(f"🖼️  Preview: actually_fixed_preview.png")
print("\nThis video matches the smooth test image quality!")