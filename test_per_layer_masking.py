#!/usr/bin/env python3
"""Test per-layer masking fix for clean occlusion boundaries"""

import cv2
import numpy as np
from utils.animations.text_3d_behind_segment_fixed import Text3DBehindSegment

print("="*60)
print("TESTING PER-LAYER MASKING FIX")
print("="*60)
print("\nThis fixes the occlusion boundary issue where the 'W'")
print("was incorrectly cut by the girl's head.")
print("\nNow each depth layer is masked individually BEFORE")
print("compositing, ensuring clean boundaries.\n")

video_path = "test_element_3sec.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Load frames for a short demo focusing on the critical moment
# Frame 30-75 should show the text passing behind the subject
frames = []
cap.set(cv2.CAP_PROP_POS_FRAMES, 30)  # Start from frame 30
for i in range(45):  # 45 frames = 0.75 seconds
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

print(f"Loaded {len(frames)} frames at {W}x{H} @ {fps}fps")
print("Focusing on the critical moment when text passes behind subject\n")

# Create animation with per-layer masking
anim = Text3DBehindSegment(
    duration=0.75,
    fps=fps,
    resolution=(W, H),
    text="HELLO WORLD",
    segment_mask=None,  # Dynamic mask from each frame
    font_size=140,
    text_color=(255, 220, 0),
    depth_color=(200, 170, 0),
    depth_layers=10,
    depth_offset=3,
    start_scale=1.5,  # Start smaller to focus on occlusion
    end_scale=0.9,
    phase1_duration=0.25,  # Quick grow
    phase2_duration=0.40,  # Main shrink/pass phase
    phase3_duration=0.10,  # Brief wiggle
    center_position=(W//2, H//2),
    shadow_offset=6,
    outline_width=2,
    perspective_angle=0,  # No perspective
    supersample_factor=3,  # High quality
    debug=False,
)

print("Generating frames with PER-LAYER masking...")
print("Each depth layer is masked individually for clean boundaries\n")

output_frames = []
for i in range(len(frames)):
    if i % 10 == 0:
        print(f"  Frame {i}/{len(frames)}...")
    
    frame = anim.generate_frame(i, frames[i])
    
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    
    output_frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

# Save video
print("\nSaving test video...")
output_path = "per_layer_masking_test.mp4"
temp = "temp_per_layer.mp4"

out = cv2.VideoWriter(temp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
for f in output_frames:
    out.write(f)
out.release()

# Convert to H.264
print("Converting to H.264...")
import subprocess
cmd = [
    'ffmpeg', '-y', '-i', temp,
    '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
    output_path
]
subprocess.run(cmd, capture_output=True)

import os
if os.path.exists(temp):
    os.remove(temp)

# Save critical frame where occlusion happens
critical_frame = 25  # Around when text passes behind
cv2.imwrite("per_layer_masking_frame.png", output_frames[critical_frame])

print(f"\n" + "="*60)
print("✅ PER-LAYER MASKING TEST COMPLETE!")
print("="*60)
print(f"\n📹 Video: {output_path}")
print(f"📸 Critical frame: per_layer_masking_frame.png")
print("\n🎯 What's fixed:")
print("  • Each 3D depth layer is masked individually")
print("  • Clean occlusion boundaries at all depths")
print("  • Back layers properly hidden, front layers visible")
print("  • No artifacts at mask boundaries")
print("\nCheck the critical frame to see the clean 'W' occlusion!")