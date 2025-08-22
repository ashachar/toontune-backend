#!/usr/bin/env python3
"""Test WordDissolve directly at frame 220 to see what it returns."""

import os
os.environ['FRAME_DISSOLVE_DEBUG'] = '1'

import cv2
import numpy as np
from utils.animations.word_dissolve import WordDissolve

# Create a clean test frame
width, height = 1280, 492
clean_frame = np.ones((height, width, 3), dtype=np.uint8) * [50, 80, 50]  # Green

# Create WordDissolve with handoff data that has 50% alpha
# Simulate what happens in the test
dissolve = WordDissolve(
    element_path="test_element_3sec.mp4",
    background_path="test_element_3sec.mp4", 
    position=(width // 2, height // 2),
    word="HELLO WORLD",
    font_size=130,
    text_color=(255, 220, 0),
    stable_duration=0.1,     
    dissolve_duration=1.0,   
    dissolve_stagger=0.25,
    fps=60,
    debug=True
)

# Test frame 220 (after all letters complete)
result = dissolve.render_word_frame(clean_frame, 220, mask=None)

# Check for text
yellow = (result[:,:,0] > 180) & (result[:,:,1] > 180) & (result[:,:,2] < 100)
semi_yellow = (result[:,:,0] > 90) & (result[:,:,0] < 150) & (result[:,:,1] > 90) & (result[:,:,1] < 150) & (result[:,:,2] < 50)

print(f"Frame 220 direct render:")
print(f"  Yellow pixels: {np.sum(yellow)}")
print(f"  Semi-yellow pixels: {np.sum(semi_yellow)}")

# Save for inspection
cv2.imwrite("direct_render_220.png", result)
print("Saved direct_render_220.png")

# Also test frame 190
result = dissolve.render_word_frame(clean_frame, 190, mask=None)
semi_yellow = (result[:,:,0] > 90) & (result[:,:,0] < 150) & (result[:,:,1] > 90) & (result[:,:,1] < 150) & (result[:,:,2] < 50)
print(f"\nFrame 190 direct render:")
print(f"  Semi-yellow pixels: {np.sum(semi_yellow)}")