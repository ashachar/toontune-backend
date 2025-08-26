#!/usr/bin/env python3
"""Minimal reproduction of the stale mask bug."""

import cv2
import numpy as np
from PIL import Image
import sys
import os
sys.path.append('.')

# Create a video with moving person
frames = []
width, height = 640, 360
for i in range(30):
    frame = np.full((height, width, 3), 220, dtype=np.uint8)
    
    # Person (rectangle) moves continuously
    person_x = 200 + i * 10  # Moves from 200 to 490
    cv2.rectangle(frame, (person_x, 100), (person_x + 100, 250), (50, 50, 100), -1)
    
    frames.append(frame)

print("Created 30 frames with person moving from x=200 to x=490")

# Now test dissolve with these frames
from utils.animations.letter_3d_dissolve import Letter3DDissolve

dissolve = Letter3DDissolve(
    duration=1.0,  # 30 frames at 30 fps
    fps=30,
    resolution=(width, height),
    text="H",
    font_size=80,
    text_color=(255, 220, 0),
    initial_scale=1.0,
    initial_position=(300, 150),  # Fixed position where person passes through
    is_behind=True,  # Enable occlusion
    debug=True
)

print("\n" + "="*60)
print("Running dissolve animation...")
print("H is at x=300, person moves from x=200 to x=490")
print("="*60)

# Track mask edge position
mask_edges = []

for i in range(30):
    if i % 5 == 0:
        print(f"\n--- Frame {i} ---")
    
    output = dissolve.generate_frame(i, frames[i])
    
    # Analyze output to find where H is cut off
    # Look for the rightmost yellow pixel
    yellow_mask = (
        (output[:, :, 0] > 180) &
        (output[:, :, 1] > 180) &
        (output[:, :, 2] < 100)
    )
    
    if np.any(yellow_mask):
        x_coords = np.where(yellow_mask)[1]
        rightmost_x = x_coords.max()
        mask_edges.append(rightmost_x)
        
        if i % 5 == 0:
            print(f"  H visible up to x={rightmost_x}")

print("\n" + "="*60)
print("STALE MASK BUG CHECK:")
print("="*60)

if len(mask_edges) > 10:
    first_edges = mask_edges[:5]
    last_edges = mask_edges[-5:]
    
    print(f"First 5 frames: H cut off at x={first_edges}")
    print(f"Last 5 frames: H cut off at x={last_edges}")
    
    edge_variation = max(mask_edges) - min(mask_edges)
    print(f"\nEdge variation: {edge_variation} pixels")
    
    if edge_variation < 20:
        print("⚠️ BUG CONFIRMED: H is cut off at nearly the same position despite person moving!")
        print("   This proves the occlusion boundary is NOT updating with the mask.")
    else:
        print("✅ Occlusion boundary is updating correctly with the moving person.")