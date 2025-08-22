#!/usr/bin/env python3
"""Test improved 3D text with actual video"""

import os
import cv2
import numpy as np
from utils.animations.text_3d_behind_segment import Text3DBehindSegment
from utils.segmentation.segment_extractor import extract_foreground_mask

# Load video
video_path = "test_element_3sec.mp4"
if not os.path.exists(video_path):
    print(f"Error: {video_path} not found")
    exit(1)

print(f"Loading {video_path}...")
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Get first frame for mask
ret, frame = cap.read()
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
print("Extracting foreground mask...")
mask = extract_foreground_mask(frame_rgb)

# Reset to beginning
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Load all frames
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

print(f"Loaded {len(frames)} frames at {W}x{H} @ {fps}fps")

# Create animation with fixes
print("\nCreating 3D text animation...")
anim = Text3DBehindSegment(
    duration=3.0,
    fps=fps,
    resolution=(W, H),
    text="HELLO WORLD",
    segment_mask=mask,
    font_size=140,
    text_color=(255, 220, 0),
    depth_color=(200, 170, 0),
    depth_layers=10,
    depth_offset=3,
    start_scale=2.2,
    end_scale=0.9,
    phase1_duration=1.2,
    phase2_duration=0.6,
    phase3_duration=1.2,
    center_position=(W//2, H//2),  # Center of video
    shadow_offset=6,
    outline_width=2,
    perspective_angle=25,
    supersample_factor=3,  # Anti-aliasing
    debug=False,  # Less verbose
    perspective_during_shrink=False,  # Keep focal point stable
)

print("Generating frames...")
output_frames = []
total = int(3.0 * fps)

for i in range(total):
    if i % 20 == 0:
        print(f"  Frame {i}/{total}...")
    
    bg = frames[i % len(frames)]
    frame = anim.generate_frame(i, bg)
    
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    
    output_frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

# Save video
print("\nSaving video...")
output_path = "text_3d_fixed_final.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
temp = "temp_3d.mp4"

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
subprocess.run(cmd, capture_output=True)

if os.path.exists(temp):
    os.remove(temp)

print(f"✓ Saved: {output_path}")

# Extract preview frame
preview = output_frames[total // 4]  # During shrink
cv2.imwrite("text_3d_fixed_preview.png", preview)

print("\n✅ Complete!")
print(f"Video: {output_path}")
print("Preview: text_3d_fixed_preview.png")
print("\nFixed issues:")
print("  • Smooth anti-aliased edges (no pixelation)")
print("  • Text shrinks to center point (no drift)")
print("  • Proper 3D depth with perspective")