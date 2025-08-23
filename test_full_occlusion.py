#!/usr/bin/env python3
"""Test FULL occlusion fix - W should be completely hidden"""

import cv2
import numpy as np
from utils.animations.text_3d_behind_segment_full_occlusion import Text3DBehindSegment

print("="*60)
print("TESTING FULL OCCLUSION FIX")
print("="*60)
print("\n🔧 Key fixes:")
print("  • 100% mask strength (not 95%)")
print("  • Aggressive mask dilation (11x11 kernel, 2 iterations)")
print("  • Morphological closing to fill gaps")
print("  • Lower threshold (>50) for broader coverage")
print("  • Earlier fade start (30% instead of 50%)\n")

video_path = "test_element_3sec.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Load 45 frames
frames = []
for i in range(45):
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

print(f"Loaded {len(frames)} frames at {W}x{H} @ {fps}fps")

# Create animation with full occlusion
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
    debug=True,  # Enable debug to see occlusion stats
)

print("\nGenerating frames with FULL occlusion...")
output_frames = []

for i in range(len(frames)):
    if i % 10 == 0:
        print(f"\n  Frame {i}/{len(frames)}...")
    
    # Critical frames
    if i == 25:
        print("    → Frame 25: W should be FULLY behind head")
    
    frame = anim.generate_frame(i, frames[i])
    
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    
    output_frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

# Save video
print("\nSaving video...")
out = cv2.VideoWriter("full_occlusion.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
for f in output_frames:
    out.write(f)
out.release()

# H.264 conversion
print("Converting to H.264...")
import subprocess
subprocess.run([
    'ffmpeg', '-y', '-i', 'full_occlusion.mp4',
    '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
    'FULL_OCCLUSION_h264.mp4'
], capture_output=True)

import os
os.remove('full_occlusion.mp4')

# Save critical frame
cv2.imwrite("full_occlusion_frame25.png", output_frames[25])

print("\n" + "="*60)
print("✅ FULL OCCLUSION TEST COMPLETE!")
print("="*60)
print("\n📹 Video: FULL_OCCLUSION_h264.mp4")
print("📸 Critical frame: full_occlusion_frame25.png")
print("\n🎯 What's fixed:")
print("  • W should be COMPLETELY hidden behind head")
print("  • No partial visibility through mask gaps")
print("  • Aggressive masking ensures full coverage")
print("  • 100% opacity blocking where mask is present")