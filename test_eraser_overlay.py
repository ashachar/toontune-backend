#!/usr/bin/env python3
"""Simple test to overlay eraser image on video"""

import subprocess
import os

# Paths
eraser_path = "uploads/assets/images/eraser.png"
input_video = "uploads/assets/runway_experiment/runway_scaled_cropped.mp4"
output_video = "uploads/assets/runway_experiment/test_eraser_overlay.mp4"

# Check if eraser exists
if not os.path.exists(eraser_path):
    print(f"Eraser not found at: {eraser_path}")
    print("Looking for eraser...")
    # Try to find it
    possible_paths = [
        "uploads/assets/images/eraser.png",
        "uploads/assets/eraser.png",
        "assets/images/eraser.png",
        "eraser.png"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            eraser_path = path
            print(f"Found eraser at: {eraser_path}")
            break
    else:
        print("Could not find eraser.png anywhere!")
        exit(1)

print(f"Using eraser from: {eraser_path}")
print(f"Input video: {input_video}")

# Simple FFmpeg command to overlay eraser on video
# Position it at center of screen
cmd = [
    'ffmpeg', '-y',
    '-i', input_video,
    '-i', eraser_path,
    '-filter_complex', '[1:v]scale=200:-1[eraser];[0:v][eraser]overlay=(W-w)/2:(H-h)/2',
    '-c:v', 'libx264',
    '-preset', 'fast',
    '-crf', '18',
    output_video
]

print("Running command...")
print(" ".join(cmd))

result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode == 0:
    print(f"Success! Output saved to: {output_video}")
else:
    print(f"Error: {result.stderr}")