#!/usr/bin/env python3
"""Test if extract_foreground_mask is returning stale masks."""

import cv2
import numpy as np
import sys
sys.path.append('utils')
from video.segmentation.segment_extractor import extract_foreground_mask

# Create test frames with moving rectangle
frames = []
for i in range(5):
    frame = np.full((360, 640, 3), 200, dtype=np.uint8)
    # Rectangle moves from x=200 to x=400
    rect_x = 200 + i * 50
    cv2.rectangle(frame, (rect_x, 100), (rect_x + 100, 250), (50, 50, 100), -1)
    frames.append(frame)

print("Testing mask extraction on 5 frames with moving rectangle...")
print("Rectangle moves from x=200 to x=400")
print("="*60)

previous_mask = None
for i, frame in enumerate(frames):
    mask = extract_foreground_mask(frame)
    
    # Find mask bounds
    y_coords, x_coords = np.where(mask > 128)
    if len(x_coords) > 0:
        mask_left = x_coords.min()
        mask_right = x_coords.max()
        print(f"\nFrame {i}: Mask x=[{mask_left}-{mask_right}]")
        
        # Check if mask changed
        if previous_mask is not None:
            diff = np.sum(np.abs(mask.astype(float) - previous_mask.astype(float)))
            print(f"  Difference from previous: {diff:.0f} pixels")
            
            if diff < 100:
                print("  ⚠️ WARNING: Mask barely changed! Possible caching issue!")
        
        previous_mask = mask.copy()

print("\n" + "="*60)
print("If mask bounds don't change with the moving rectangle,")
print("then extract_foreground_mask has a caching/stale data issue.")