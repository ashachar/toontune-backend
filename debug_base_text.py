#!/usr/bin/env python3
"""Debug why base text is still visible after dissolve."""

import os
os.environ['FRAME_DISSOLVE_DEBUG'] = '1'

import cv2
import numpy as np

# Check the video at different points
cap = cv2.VideoCapture("hello_world_fixed.mp4")
fps = int(cap.get(cv2.CAP_PROP_FPS))

print("Checking for base text visibility...")
print(f"FPS: {fps}")

# Key frames to check
# Based on timeline: WordDissolve starts at frame 60
# Stable phase: 60-66
# First letter starts: 66
# Last letter starts: 66 + 9*15 = 201
# Last letter ends: 201 + 60 = 261
# All complete: 261

test_frames = [60, 66, 100, 150, 200, 250, 261, 270, 280, 300]

for frame_idx in test_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    
    if ret:
        # Check for yellow text (255, 220, 0)
        # In BGR that's (0, 220, 255)
        yellow_mask = (frame[:,:,1] > 180) & (frame[:,:,2] > 180) & (frame[:,:,0] < 100)
        yellow_pixels = np.sum(yellow_mask)
        
        # Also check for semi-transparent yellow (50% alpha would be around 128, 110, 0 on background)
        semi_yellow_mask = (frame[:,:,1] > 90) & (frame[:,:,1] < 150) & \
                          (frame[:,:,2] > 90) & (frame[:,:,2] < 150) & \
                          (frame[:,:,0] < 50)
        semi_yellow_pixels = np.sum(semi_yellow_mask)
        
        wd_frame = frame_idx - 60 if frame_idx >= 60 else -1
        
        print(f"Frame {frame_idx:3d} (WD frame {wd_frame:3d}): "
              f"Yellow: {yellow_pixels:5d}, Semi-yellow: {semi_yellow_pixels:5d}")
        
        if frame_idx in [250, 270]:
            cv2.imwrite(f"debug_frame_{frame_idx}.png", frame)

cap.release()

print("\n" + "="*60)
print("Analyzing the issue...")

# Check what's happening at frame 250 specifically
cap = cv2.VideoCapture("hello_world_fixed.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 250)
ret, frame = cap.read()

if ret:
    # Extract text region
    text_region = frame[150:350, 50:1100]
    
    # Get unique colors in text region
    unique_colors = np.unique(text_region.reshape(-1, 3), axis=0)
    
    # Find colors that look like faded yellow
    yellow_like = []
    for color in unique_colors:
        b, g, r = color
        # Yellow (255, 220, 0) at 50% on greenish background would be around (25, 150, 127)
        if g > 100 and r > 100 and b < 80:
            yellow_like.append((b, g, r))
    
    print(f"Found {len(yellow_like)} yellow-like colors in text region")
    if len(yellow_like) > 0:
        print("Sample yellow-like colors (BGR):")
        for color in yellow_like[:5]:
            print(f"  {color}")

cap.release()

print("\nDiagnosis:")
print("The base text is being rendered even though letters should be dissolved.")
print("This suggests the 'not_started' check isn't working correctly,")