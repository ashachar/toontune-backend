#!/usr/bin/env python3
"""
Create binary mask video showing foreground (white) vs background (black)
using tolerance-based green screen detection.
"""

import cv2
import numpy as np
from pathlib import Path

# Input: green screen video from Replicate
green_screen_video = "../../uploads/assets/videos/ai_math1b/raw_video_6sec_mask_lossless.mp4"
output_path = "../../outputs/binary_mask_demo.mp4"

# Green screen color and tolerance
GREEN_SCREEN_BGR = np.array([154, 254, 119], dtype=np.uint8)
TOLERANCE = 5  # Adjust as needed

# Open video
cap = cv2.VideoCapture(green_screen_video)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Processing {total_frames} frames...")
print(f"Green screen color: BGR {GREEN_SCREEN_BGR}")
print(f"Tolerance: ±{TOLERANCE}")

# Create output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

# Process each frame
frame_count = 0
for frame_idx in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        break
    
    # Calculate difference from green screen color
    # This handles the color variation from Replicate
    diff = np.abs(frame.astype(np.int16) - GREEN_SCREEN_BGR.astype(np.int16))
    
    # Check if within tolerance for all channels
    is_green_screen = np.all(diff <= TOLERANCE, axis=2)
    
    # Create binary mask: 
    # 0 (black) = green screen (background)
    # 255 (white) = NOT green screen (foreground/person)
    binary_mask = (~is_green_screen).astype(np.uint8) * 255
    
    # Optional: Apply morphological operations to clean up edges
    kernel = np.ones((3,3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)  # Remove noise
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes
    
    # Write frame
    out.write(binary_mask)
    
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Processed {frame_count}/{total_frames} frames")
        
        # Show statistics for this frame
        foreground_pixels = np.sum(binary_mask == 255)
        background_pixels = np.sum(binary_mask == 0)
        total_pixels = binary_mask.size
        fg_percent = 100 * foreground_pixels / total_pixels
        
        print(f"  Frame {frame_count}: {fg_percent:.1f}% foreground, {100-fg_percent:.1f}% background")

# Clean up
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\n✅ Binary mask video saved: {output_path}")

# Convert to H.264 for better compatibility
output_h264 = output_path.replace('.mp4', '_h264.mp4')
print("\nConverting to H.264...")
import subprocess
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

# Create a side-by-side comparison video
print("\nCreating side-by-side comparison...")
comparison_output = "../../outputs/mask_comparison.mp4"

cmd = [
    "ffmpeg", "-y",
    "-i", green_screen_video,
    "-i", output_h264,
    "-filter_complex",
    "[0:v]pad=iw*2:ih[bg];"
    "[1:v]format=yuv420p[mask];"
    "[bg][mask]overlay=w:0[out]",
    "-map", "[out]",
    "-c:v", "libx264",
    "-preset", "fast",
    "-crf", "23",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
    comparison_output
]
subprocess.run(cmd, check=True, capture_output=True)
print(f"Side-by-side comparison: {comparison_output}")

print("\n📊 Summary:")
print(f"1. Binary mask: {output_h264}")
print(f"   - White pixels = Foreground (person)")
print(f"   - Black pixels = Background (green screen)")
print(f"2. Comparison: {comparison_output}")
print(f"   - Left: Original green screen from Replicate")
print(f"   - Right: Binary mask with tolerance={TOLERANCE}")

# Clean up temp file
Path(output_path).unlink()