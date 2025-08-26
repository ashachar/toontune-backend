#!/usr/bin/env python3
"""
Extract 2:20-2:24 segment from ai_math1.mp4 and apply Hello World animation.
"""

import os
import sys
import numpy as np
from PIL import Image

print("="*80)
print("🎬 EXTRACTING AND ANIMATING 2:20-2:24 SEGMENT")
print("="*80)

# Configuration
INPUT_VIDEO = "uploads/assets/videos/ai_math1.mp4"
SEGMENT_VIDEO = "outputs/ai_math1_segment_2m20s.mp4"
OUTPUT_VIDEO = "outputs/hello_world_2m20s_final.mp4"

# Step 1: Extract segment using FFmpeg
print("\n📹 Step 1: Extracting 2:20-2:24 segment...")
extract_cmd = (
    f'ffmpeg -i {INPUT_VIDEO} -ss 140 -t 4 '
    f'-c:v copy -c:a copy {SEGMENT_VIDEO} -y'
)
result = os.system(extract_cmd + ' 2>/dev/null')

if result != 0 or not os.path.exists(SEGMENT_VIDEO):
    print("❌ Failed to extract segment")
    sys.exit(1)

# Check segment
import cv2
cap = cv2.VideoCapture(SEGMENT_VIDEO)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

print(f"✅ Extracted {frame_count} frames at {fps} fps ({width}x{height})")

# Step 2: Apply animation using refactored module
print("\n🎨 Step 2: Applying Hello World animation...")

# Use the existing refactored animation
from utils.animations.letter_3d_dissolve import Letter3DDissolve

# Create animation (turn off debug)
animation = Letter3DDissolve(
    text="Hello World",
    initial_position=(width//2, height//2 - 50),  # Center, slightly above
    is_behind=True,
    font_size=72,
    resolution=(width, height),
    fps=fps,
    duration=4.0,  # Total duration
    stable_duration=0.8,  # Motion phase
    dissolve_duration=2.5,
    supersample_factor=8,
    debug=False  # Disable debug output
)

# Open videos
cap = cv2.VideoCapture(SEGMENT_VIDEO)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

print(f"  Processing {frame_count} frames...")

for frame_idx in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply animation
    frame_with_text = animation.generate_frame(frame_idx, frame)
    
    # Ensure it's a numpy array
    if isinstance(frame_with_text, Image.Image):
        frame_with_text = np.array(frame_with_text)
    
    # Convert RGB to BGR if needed
    if len(frame_with_text.shape) == 3 and frame_with_text.shape[2] == 3:
        # Assume it's RGB, convert to BGR
        frame_with_text = cv2.cvtColor(frame_with_text, cv2.COLOR_RGB2BGR)
    
    # Add timestamp
    time_s = frame_idx / fps
    cv2.putText(frame_with_text, f"2:{20+int(time_s):02d}",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
               (255, 255, 255), 2)
    
    out.write(frame_with_text)
    
    if frame_idx % 25 == 0:
        print(f"    Frame {frame_idx}/{frame_count}")

cap.release()
out.release()

print(f"\n✅ Animation saved to: {OUTPUT_VIDEO}")

# Step 3: Convert to H.264
print("\n🎬 Step 3: Converting to H.264...")
h264_output = OUTPUT_VIDEO.replace('.mp4', '_h264.mp4')
convert_cmd = (
    f'ffmpeg -i {OUTPUT_VIDEO} -c:v libx264 -preset fast -crf 23 '
    f'-pix_fmt yuv420p -movflags +faststart {h264_output} -y'
)
os.system(convert_cmd + ' 2>/dev/null')

if os.path.exists(h264_output):
    size_mb = os.path.getsize(h264_output) / (1024 * 1024)
    print(f"✅ H.264 version: {h264_output} ({size_mb:.2f} MB)")
    
    # Open the video
    print(f"\n🎥 Opening video...")
    os.system(f"open {h264_output}")
else:
    print("⚠️ H.264 conversion may have failed, opening original...")
    os.system(f"open {OUTPUT_VIDEO}")

print("\n" + "="*80)
print("✅ COMPLETE: Hello World animation on 2:20-2:24 segment")
print("="*80)