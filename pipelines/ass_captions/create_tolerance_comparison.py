#!/usr/bin/env python3
"""
Create a video showing how different tolerance values affect the binary mask.
"""

import cv2
import numpy as np
import subprocess

# Input
green_screen_video = "../../uploads/assets/videos/ai_math1b/raw_video_6sec_mask_lossless.mp4"
output_path = "../../outputs/tolerance_comparison.mp4"

# Green screen color
GREEN_SCREEN_BGR = np.array([154, 254, 119], dtype=np.uint8)

# Test different tolerance values
TOLERANCES = [1, 3, 5, 10]

# Open video
cap = cv2.VideoCapture(green_screen_video)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Create output video with 2x2 grid
out_width = width * 2
out_height = height * 2
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

print(f"Creating tolerance comparison video...")
print(f"Tolerances: {TOLERANCES}")

# Process each frame
for frame_idx in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        break
    
    # Create 2x2 grid
    grid = np.zeros((out_height, out_width, 3), dtype=np.uint8)
    
    for i, tolerance in enumerate(TOLERANCES):
        # Calculate position in grid
        row = i // 2
        col = i % 2
        y_start = row * height
        x_start = col * width
        
        # Calculate binary mask with this tolerance
        diff = np.abs(frame.astype(np.int16) - GREEN_SCREEN_BGR.astype(np.int16))
        is_green_screen = np.all(diff <= tolerance, axis=2)
        binary_mask = (~is_green_screen).astype(np.uint8) * 255
        
        # Apply morphological operations
        kernel = np.ones((3,3), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        # Convert to 3-channel for display
        mask_3ch = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        
        # Add text label
        label = f"Tolerance: {tolerance}"
        cv2.putText(mask_3ch, label, (30, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (128, 128, 128), 3)
        
        # Calculate and show statistics
        fg_percent = 100 * np.sum(binary_mask == 255) / binary_mask.size
        stats = f"FG: {fg_percent:.1f}%"
        cv2.putText(mask_3ch, stats, (30, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (128, 128, 128), 2)
        
        # Place in grid
        grid[y_start:y_start+height, x_start:x_start+width] = mask_3ch
    
    # Write frame
    out.write(grid)
    
    if (frame_idx + 1) % 30 == 0:
        print(f"Processed {frame_idx + 1}/{total_frames} frames")

# Clean up
cap.release()
out.release()

print(f"\n✅ Tolerance comparison saved: {output_path}")

# Convert to H.264
output_h264 = output_path.replace('.mp4', '_h264.mp4')
print("Converting to H.264...")
cmd = [
    "ffmpeg", "-y",
    "-i", output_path,
    "-c:v", "libx264",
    "-preset", "fast",
    "-crf", "23",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
    output_h264
]
subprocess.run(cmd, check=True, capture_output=True)
print(f"H.264 version: {output_h264}")

# Clean up temp
import os
os.remove(output_path)

print(f"\n📊 Tolerance Comparison Video: {output_h264}")
print("Shows 2x2 grid with different tolerance values:")
print("- Top-left: Tolerance = 1 (very strict)")
print("- Top-right: Tolerance = 3")
print("- Bottom-left: Tolerance = 5 (recommended)")
print("- Bottom-right: Tolerance = 10 (very loose)")