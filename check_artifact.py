#!/usr/bin/env python3
"""Extract and inspect frames to check for gray line artifacts."""

import cv2
import numpy as np
import os

# Extract frames from the test video
video_path = "hello_world_fixed.mp4"
if not os.path.exists(video_path):
    print(f"Error: {video_path} not found")
    exit(1)

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {video_path}")
print(f"FPS: {fps}, Total frames: {total_frames}")

# Extract frames where W is dissolving (around frame 96-156)
# Space (index 5) would be at frames 81-141
# W (index 6) would be at frames 96-156

target_frames = [90, 96, 100, 110, 120, 130, 140, 150]

for frame_idx in target_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if ret:
        # Save frame
        output_name = f"frame_{frame_idx:03d}.png"
        cv2.imwrite(output_name, frame)
        
        # Check for gray artifacts in the area above W
        # W is approximately at x=570-700, y=190-300
        # Check area above it: y=150-190
        roi = frame[150:190, 570:700]  # [y1:y2, x1:x2]
        
        # Convert to grayscale and check for non-black pixels
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Count pixels that are gray (not black, not white)
        gray_pixels = np.sum((gray > 10) & (gray < 245))
        total_pixels = gray.shape[0] * gray.shape[1]
        
        # Check if there are horizontal lines
        # Sum along width to get vertical profile
        vertical_profile = np.mean(gray, axis=1)
        max_gray_value = np.max(vertical_profile)
        
        print(f"Frame {frame_idx}: Gray pixels in ROI: {gray_pixels}/{total_pixels} ({gray_pixels/total_pixels*100:.1f}%), Max gray: {max_gray_value:.1f}")
        
        if max_gray_value > 20:
            print(f"  ⚠️  Potential gray line detected! Max intensity: {max_gray_value:.1f}")
            # Save the ROI for inspection
            roi_enhanced = cv2.convertScaleAbs(roi, alpha=3.0, beta=50)  # Enhance contrast
            cv2.imwrite(f"roi_frame_{frame_idx:03d}_enhanced.png", roi_enhanced)

cap.release()
print("\nFrames extracted. Check the PNG files to visually inspect for artifacts.")
print("ROI files show the area above W where gray lines typically appear.")