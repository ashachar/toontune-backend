#!/usr/bin/env python3
"""
Create a sample burn animation that actually completes.
"""

import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Use the simpler burn effect that works
from utils.animations.letter_3d_burn.burn import Letter3DBurn

print("Creating burn animation sample...")

# Lower resolution for faster rendering
width, height = 640, 360
fps = 24
duration = 2.5

# Create burn animation
burn = Letter3DBurn(
    duration=duration,
    fps=fps,
    resolution=(width, height),
    text="BURN",
    font_size=100,
    text_color=(255, 200, 0),  # Yellow/orange
    burn_color=(255, 50, 0),    # Red/orange burn
    burn_duration=0.6,
    burn_stagger=0.2,
    supersample_factor=2,
    debug=False
)

# Dark background
background = np.ones((height, width, 3), dtype=np.uint8) * 40

# Generate frames
output_path = "outputs/burn_sample.mp4"
os.makedirs("outputs", exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

total_frames = int(duration * fps)
print(f"Rendering {total_frames} frames...")

for frame_num in range(total_frames):
    frame = burn.generate_frame(frame_num, background.copy())
    writer.write(frame)
    
    if frame_num % 10 == 0:
        print(f"  Frame {frame_num}/{total_frames}")

writer.release()

# Convert to H.264
h264_output = "outputs/burn_sample_h264.mp4"
os.system(f"ffmpeg -i {output_path} -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -movflags +faststart -y {h264_output} 2>/dev/null")

print(f"\n✅ Burn animation saved to: {h264_output}")