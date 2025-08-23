#!/usr/bin/env python3
"""Analyze why the mask isn't fully covering the girl's head"""

import cv2
import numpy as np
from utils.segmentation.segment_extractor import extract_foreground_mask

print("ANALYZING MASK COVERAGE ISSUE")
print("-" * 50)

# Load the problematic frame
video_path = "test_element_3sec.mp4"
cap = cv2.VideoCapture(video_path)

# Frame 25 seems to be the issue
cap.set(cv2.CAP_PROP_POS_FRAMES, 25)
ret, frame = cap.read()
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
H, W = frame_rgb.shape[:2]
cap.release()

print(f"Frame size: {W}x{H}")

# Extract mask
print("\n1. Extracting mask with rembg...")
mask = extract_foreground_mask(frame_rgb)

# Analyze the mask quality
print(f"   Mask unique values: {np.unique(mask)}")
print(f"   Total foreground pixels: {np.sum(mask > 128):,} / {mask.size:,}")
print(f"   Foreground percentage: {100 * np.sum(mask > 128) / mask.size:.1f}%")

# Check if the girl's head area is properly masked
# The girl's head appears to be around x=700, y=250 based on the image
head_x = 700
head_y = 250
head_size = 100

head_region = mask[head_y:head_y+head_size, head_x:head_x+head_size]
print(f"\n2. Girl's head region ({head_x}, {head_y}):")
print(f"   Foreground pixels in head: {np.sum(head_region > 128)} / {head_region.size}")
print(f"   Coverage: {100 * np.sum(head_region > 128) / head_region.size:.1f}%")

# Check where the W is positioned
w_x = 640  # Approximate W position
w_y = 250
w_size = 80

w_region = mask[w_y:w_y+w_size, w_x:w_x+w_size]
print(f"\n3. W letter region ({w_x}, {w_y}):")
print(f"   Foreground pixels at W: {np.sum(w_region > 128)} / {w_region.size}")
print(f"   Coverage: {100 * np.sum(w_region > 128) / w_region.size:.1f}%")

# The issue: The W region has only partial mask coverage
# Let's visualize the problem
print("\n4. Creating detailed visualization...")

# Create a figure showing the issue
fig = np.zeros((H*2, W*2, 3), dtype=np.uint8)

# Top-left: Original frame
fig[:H, :W] = frame_rgb

# Top-right: Mask
mask_rgb = np.stack([mask, mask, mask], axis=2)
fig[:H, W:] = mask_rgb

# Bottom-left: Mask overlay
overlay = frame_rgb.copy()
mask_colored = np.zeros_like(frame_rgb)
mask_colored[:, :, 0] = mask  # Red for mask
overlay = cv2.addWeighted(overlay, 0.6, mask_colored, 0.4, 0)
fig[H:, :W] = overlay

# Bottom-right: Problem areas highlighted
problem = frame_rgb.copy()
# Draw rectangles around problem areas
cv2.rectangle(problem, (w_x, w_y), (w_x+w_size, w_y+w_size), (255, 0, 0), 2)  # W area in red
cv2.rectangle(problem, (head_x, head_y), (head_x+head_size, head_y+head_size), (0, 255, 0), 2)  # Head area in green
cv2.putText(problem, "W", (w_x, w_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
cv2.putText(problem, "HEAD", (head_x, head_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
fig[H:, W:] = problem

# Add labels
cv2.putText(fig, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(fig, "Mask", (W+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(fig, "Overlay", (10, H+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(fig, "Problem Areas", (W+10, H+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cv2.imwrite("mask_analysis.png", cv2.cvtColor(fig, cv2.COLOR_RGB2BGR))

# Try alternative mask extraction
print("\n5. Testing mask improvements...")

# Method 1: More aggressive dilation
kernel = np.ones((7, 7), np.uint8)
mask_dilated = cv2.dilate(mask, kernel, iterations=2)
cv2.imwrite("mask_dilated.png", mask_dilated)

# Method 2: Morphological closing to fill gaps
kernel = np.ones((5, 5), np.uint8)
mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask_closed = cv2.dilate(mask_closed, kernel, iterations=1)
cv2.imwrite("mask_closed.png", mask_closed)

# Method 3: Combine both
mask_combined = cv2.dilate(mask_closed, kernel, iterations=1)
cv2.imwrite("mask_combined.png", mask_combined)

# Check coverage with improved masks
w_region_improved = mask_combined[w_y:w_y+w_size, w_x:w_x+w_size]
print(f"\n6. W coverage with improved mask:")
print(f"   Original: {100 * np.sum(w_region > 128) / w_region.size:.1f}%")
print(f"   Improved: {100 * np.sum(w_region_improved > 128) / w_region_improved.size:.1f}%")

print("\n" + "="*50)
print("ANALYSIS COMPLETE!")
print("="*50)
print("\nOutputs:")
print("  • mask_analysis.png - 4-panel analysis")
print("  • mask_dilated.png - Dilated mask")
print("  • mask_closed.png - Morphologically closed mask")
print("  • mask_combined.png - Combined improvements")
print("\nThe issue is that the mask doesn't fully cover")
print("the area where the W appears. We need stronger masking!")