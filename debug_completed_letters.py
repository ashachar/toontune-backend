#!/usr/bin/env python3
"""Debug why completed letters are still visible."""

import os
os.environ['FRAME_DISSOLVE_DEBUG'] = '1'

import cv2
import numpy as np

# Create a simple test to understand the issue
print("Checking video frames to understand completed letter visibility...")

cap = cv2.VideoCapture("hello_world_fixed.mp4")
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Based on the logs:
# - Stable phase: frames 0-6
# - First letter (H) starts dissolving: frame 6
# - Last letter starts: frame 156
# - Last letter ends: frame 156 + 60 = 216

print(f"Video FPS: {fps}")
print("Expected timeline:")
print("  Stable phase: 0-6")
print("  Dissolve phase: 6-216")
print("  All completed: 216+")
print()

# Check specific frames
test_frames = [0, 6, 100, 150, 200, 216, 220, 250, 280, 300]

for frame_idx in test_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    
    if ret:
        # Check for yellow pixels (text color is yellow)
        yellow_mask = (frame[:,:,1] > 180) & (frame[:,:,2] > 180) & (frame[:,:,0] < 100)
        yellow_count = np.sum(yellow_mask)
        
        # Get average brightness in text region
        text_region = frame[150:350, 50:1100]
        avg_brightness = np.mean(text_region)
        
        phase = "stable" if frame_idx < 6 else ("dissolving" if frame_idx < 216 else "completed")
        
        print(f"Frame {frame_idx:3d} ({phase:10s}): Yellow pixels: {yellow_count:5d}, Avg brightness: {avg_brightness:.1f}")
        
        # Save frames where letters should be gone but aren't
        if frame_idx >= 216 and yellow_count > 100:
            cv2.imwrite(f"debug_frame_{frame_idx:03d}_still_visible.png", frame)

cap.release()

print("\n" + "="*60)
print("ANALYSIS:")

# Now let's check if it's a rendering issue or a masking issue
cap = cv2.VideoCapture("hello_world_fixed.mp4")

# Get frame 250 where everything should be dissolved
cap.set(cv2.CAP_PROP_POS_FRAMES, 250)
ret, frame_250 = cap.read()

# Get frame 5 before any dissolve
cap.set(cv2.CAP_PROP_POS_FRAMES, 5)
ret, frame_5 = cap.read()

if ret:
    # Compare the two frames
    diff = cv2.absdiff(frame_250, frame_5)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Find regions with text
    text_mask = diff_gray > 20
    
    # Extract just the text region
    y1, y2 = 150, 350
    x1, x2 = 50, 1100
    
    text_region_250 = frame_250[y1:y2, x1:x2]
    text_region_5 = frame_5[y1:y2, x1:x2]
    
    # Check if text is dimmed or still at full brightness
    yellow_250 = (text_region_250[:,:,1] > 180) & (text_region_250[:,:,2] > 180) & (text_region_250[:,:,0] < 100)
    yellow_5 = (text_region_5[:,:,1] > 180) & (text_region_5[:,:,2] > 180) & (text_region_5[:,:,0] < 100)
    
    if np.any(yellow_250):
        # Get average brightness of yellow pixels
        bright_250 = np.mean(text_region_250[yellow_250])
        bright_5 = np.mean(text_region_5[yellow_5]) if np.any(yellow_5) else 0
        
        print(f"Yellow pixel brightness at frame 5: {bright_5:.1f}")
        print(f"Yellow pixel brightness at frame 250: {bright_250:.1f}")
        print(f"Dimming ratio: {bright_250/bright_5*100:.1f}%" if bright_5 > 0 else "N/A")
        
        if bright_250 > bright_5 * 0.5:
            print("\n⚠️  Text is still quite bright - masking might not be working!")
        else:
            print("\n✓ Text is significantly dimmed but still visible")

cap.release()