#!/usr/bin/env python3
"""Debug W extraction to understand why we're only getting tiny components."""

import cv2
import numpy as np
from scipy import ndimage

# Load frames
cap = cv2.VideoCapture("hello_world_fixed.mp4")

# Try different frame pairs to extract W
frame_pairs = [
    (85, 95),   # Well before W and early W
    (90, 96),   # Just before W and W starting to dissolve
    (92, 94),   # Very close frames when W is stable
]

for before_frame, after_frame in frame_pairs:
    print(f"\n--- Extracting W using frames {before_frame} and {after_frame} ---")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, before_frame)
    ret, frame_before = cap.read()
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, after_frame)
    ret, frame_after = cap.read()
    
    # Full frame difference
    full_diff = cv2.absdiff(frame_after, frame_before)
    full_gray_diff = cv2.cvtColor(full_diff, cv2.COLOR_BGR2GRAY)
    
    # Check overall difference
    max_diff = np.max(full_gray_diff)
    mean_diff = np.mean(full_gray_diff)
    pixels_changed = np.sum(full_gray_diff > 10)
    
    print(f"  Full frame: max_diff={max_diff}, mean_diff={mean_diff:.2f}, pixels_changed={pixels_changed}")
    
    # Focus on W region
    x1, x2 = 520, 760
    y1, y2 = 140, 350
    
    roi_before = frame_before[y1:y2, x1:x2]
    roi_after = frame_after[y1:y2, x1:x2]
    
    # Calculate difference
    diff = cv2.absdiff(roi_after, roi_before)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Stats
    roi_max = np.max(gray_diff)
    roi_mean = np.mean(gray_diff)
    roi_pixels = np.sum(gray_diff > 10)
    
    print(f"  W region: max_diff={roi_max}, mean_diff={roi_mean:.2f}, pixels_changed={roi_pixels}")
    
    # Try different thresholds
    for threshold in [5, 10, 20, 30]:
        _, mask = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)
        labeled, num_features = ndimage.label(mask)
        
        if num_features > 0:
            sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
            largest = max(sizes)
            total_pixels = np.sum(mask > 0)
            print(f"    Threshold {threshold}: {num_features} components, largest={largest} px, total={total_pixels} px")
    
    # Save the difference image for inspection
    cv2.imwrite(f"w_diff_{before_frame}_{after_frame}.png", gray_diff)
    
    # Also save a color-coded version
    _, mask = cv2.threshold(gray_diff, 10, 255, cv2.THRESH_BINARY)
    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    colored[mask > 0] = [0, 255, 0]  # Green for detected W pixels
    cv2.imwrite(f"w_mask_{before_frame}_{after_frame}.png", colored)

cap.release()

print("\n--- Checking the actual frames ---")
# Let's also check what the frames actually look like
cap = cv2.VideoCapture("hello_world_fixed.mp4")
for frame_num in [85, 90, 92, 94, 95, 96]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if ret:
        # Save full frame
        cv2.imwrite(f"debug_frame_{frame_num:03d}.png", frame)
        
        # Check if W region has yellow pixels (the W is yellow)
        roi = frame[140:350, 520:760]
        
        # Check for yellow pixels (high R and G, low B)
        yellow_pixels = np.sum((roi[:,:,2] > 200) & (roi[:,:,1] > 200) & (roi[:,:,0] < 100))
        print(f"Frame {frame_num}: {yellow_pixels} yellow pixels in W region")

cap.release()