#!/usr/bin/env python3
"""Create the FINAL video with minimal masking - tight with letters"""

import cv2
import numpy as np
from utils.animations.text_3d_behind_segment_final import Text3DBehindSegment

print("="*60)
print("CREATING FINAL VIDEO")
print("="*60)
print("\n✅ Final approach:")
print("  • Minimal mask processing (3x3 kernel, 1 iteration)")
print("  • Mask stays tight with letters")
print("  • Accept rembg limitations for seated figures")
print("  • No excessive dilation\n")

video_path = "test_element_3sec.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Load 90 frames (1.5 seconds)
frames = []
for i in range(90):
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

print(f"Loaded {len(frames)} frames at {W}x{H} @ {fps}fps")

# Create final animation
anim = Text3DBehindSegment(
    duration=1.5,
    fps=fps,
    resolution=(W, H),
    text="HELLO WORLD",
    segment_mask=None,
    font_size=140,
    text_color=(255, 220, 0),
    depth_color=(200, 170, 0),
    depth_layers=10,
    depth_offset=3,
    start_scale=2.2,
    end_scale=0.9,
    shrink_duration=1.2,
    wiggle_duration=0.3,
    center_position=(W//2, H//2),
    shadow_offset=6,
    outline_width=2,
    perspective_angle=0,
    supersample_factor=3,
    debug=False,
)

print("\nGenerating frames...")
output_frames = []

for i in range(len(frames)):
    if i % 15 == 0:
        print(f"  Frame {i}/{len(frames)}...")
    
    frame = anim.generate_frame(i, frames[i])
    
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    
    output_frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

# Save video
print("\nSaving video...")
out = cv2.VideoWriter("final_temp.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
for f in output_frames:
    out.write(f)
out.release()

# H.264 conversion
print("Converting to H.264...")
import subprocess
subprocess.run([
    'ffmpeg', '-y', '-i', 'final_temp.mp4',
    '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
    'FINAL_VIDEO_h264.mp4'
], capture_output=True)

import os
os.remove('final_temp.mp4')

print("\n" + "="*60)
print("✅ FINAL VIDEO COMPLETE!")
print("="*60)
print("\n📹 Video: FINAL_VIDEO_h264.mp4")
print("\n🎯 Final characteristics:")
print("  • Text moves backwards only (no forward growth)")
print("  • Smooth 3D depth effect (80% reduced)")
print("  • Proper fade timing (40-60% transition)")
print("  • Minimal masking - stays tight with letters")
print("  • No excessive dilation or artifacts")
print("\n⚠️ Known limitation:")
print("  • O may show through seated girl (rembg limitation)")
print("  • This is accepted rather than over-dilating")