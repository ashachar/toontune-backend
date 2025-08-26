#!/usr/bin/env python3
"""Test just the dissolve animation mask extraction."""

import cv2
import numpy as np
from PIL import Image
import sys
sys.path.append('.')

# Create a simple test video with a moving rectangle
width, height = 640, 360
fps = 10
frames = []

# Create 20 frames with rectangle moving right
for i in range(20):
    frame = np.full((height, width, 3), 200, dtype=np.uint8)
    # Rectangle moves from x=200 to x=400
    rect_x = 200 + i * 10
    cv2.rectangle(frame, (rect_x, 100), (rect_x + 100, 250), (50, 50, 100), -1)
    frames.append(frame)

# Now test dissolve directly
from utils.animations.letter_3d_dissolve import Letter3DDissolve

dissolve = Letter3DDissolve(
    duration=2.0,
    fps=10,
    resolution=(width, height),
    text="H",
    font_size=60,
    text_color=(255, 220, 0),
    initial_scale=1.0,
    initial_position=(300, 150),
    is_behind=True,  # Enable occlusion
    debug=True
)

print("Testing dissolve animation mask extraction...")
print("Rectangle moves from x=200 to x=400 over 20 frames")
print("Letter 'H' is at x=300")
print("="*60)

# Generate frames
for i in range(20):
    print(f"\n--- Frame {i} ---")
    output = dissolve.generate_frame(i, frames[i])
    
print("\n" + "="*60)
print("Check the debug output above to see if mask position changes each frame.")
print("The mask bounds should move from x=[200-300] to x=[400-500]")