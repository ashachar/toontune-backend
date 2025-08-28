#!/usr/bin/env python3
"""
Quick test of photorealistic burn - lower resolution for faster rendering.
"""

import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.animations.letter_3d_burn.photorealistic_burn import PhotorealisticLetterBurn

# Quick test with lower resolution
width, height = 1280, 720
fps = 24
duration = 3.0

print("Creating photorealistic burn animation...")
print(f"Resolution: {width}x{height}, Duration: {duration}s")

# Create burn animation
burn = PhotorealisticLetterBurn(
    duration=duration,
    fps=fps,
    resolution=(width, height),
    text="FIRE",
    font_size=200,
    text_color=(200, 200, 200),
    flame_height=150,
    flame_intensity=1.0,
    burn_stagger=0.3,
    supersample_factor=1,  # Lower for speed
    debug=False
)

# Dark background
background = np.ones((height, width, 3), dtype=np.uint8) * 30

# Generate frames
output_path = "outputs/burn_quick_test.mp4"
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
h264_output = "outputs/burn_quick_test_h264.mp4"
os.system(f"ffmpeg -i {output_path} -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -y {h264_output} 2>/dev/null")

print(f"\n✅ Photorealistic burn saved to: {h264_output}")
print("\nFeatures rendered:")
print("  • Realistic flames with turbulence")  
print("  • Volumetric smoke")
print("  • Heat propagation and charring")
print("  • Glowing embers")
print("  • Progressive burning")