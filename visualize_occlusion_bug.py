#!/usr/bin/env python3
"""Visualize exactly where letters are vs where mask is."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('utils')
from video.segmentation.segment_extractor import extract_foreground_mask

# Load frame 25
cap = cv2.VideoCapture('outputs/test_occlusion_proof_final_h264.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 25)
ret, frame = cap.read()
cap.release()

frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Extract mask
mask = extract_foreground_mask(frame_rgb)

# Find yellow pixels (correct BGR order)
yellow_mask = (
    (frame[:, :, 0] < 100) &  # Blue low
    (frame[:, :, 1] > 180) &  # Green high
    (frame[:, :, 2] > 180)    # Red high
)

print(f"Yellow pixels found: {np.sum(yellow_mask)}")
print(f"Mask pixels found: {np.sum(mask > 128)}")

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Original frame
axes[0, 0].imshow(frame_rgb)
axes[0, 0].set_title("Frame 25 - Original")
axes[0, 0].axis('off')

# Mask
axes[0, 1].imshow(mask, cmap='gray')
axes[0, 1].set_title(f"Person Mask ({np.sum(mask > 128):,} pixels)")
axes[0, 1].axis('off')

# Yellow letters
axes[0, 2].imshow(yellow_mask, cmap='hot')
axes[0, 2].set_title(f"Yellow Letters ({np.sum(yellow_mask):,} pixels)")
axes[0, 2].axis('off')

# Overlap visualization
overlap = yellow_mask & (mask > 128)
axes[1, 0].imshow(overlap, cmap='hot')
axes[1, 0].set_title(f"Overlap ({np.sum(overlap):,} pixels)")
axes[1, 0].axis('off')

# Create composite showing problem
composite = frame_rgb.copy()
# Mark yellow pixels in green
composite[yellow_mask] = [0, 255, 0]
# Mark mask pixels with transparency
mask_overlay = np.zeros_like(frame_rgb)
mask_overlay[mask > 128] = [255, 0, 0]
composite = cv2.addWeighted(composite, 0.7, mask_overlay, 0.3, 0)

axes[1, 1].imshow(composite)
axes[1, 1].set_title("Composite: Green=Letters, Red=Mask")
axes[1, 1].axis('off')

# Zoom in on problem area (around second 'l')
zoom_region = frame_rgb[290:350, 430:500]
axes[1, 2].imshow(zoom_region)
axes[1, 2].set_title("Zoom: Second 'l' region")
axes[1, 2].axis('off')

plt.suptitle("Occlusion Bug Visualization - Frame 25", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/occlusion_bug_visualization.png', dpi=150)
print("\n✅ Saved visualization to: outputs/occlusion_bug_visualization.png")

# Let's check specific letter positions
y_coords, x_coords = np.where(yellow_mask)
if len(x_coords) > 0:
    print(f"\nYellow letter pixels found at:")
    print(f"  X range: {x_coords.min()} - {x_coords.max()}")
    print(f"  Y range: {y_coords.min()} - {y_coords.max()}")
    
    # Check if any of these pixels overlap with mask
    for i in range(0, len(x_coords), 100):  # Sample every 100th pixel
        x, y = x_coords[i], y_coords[i]
        if mask[y, x] > 128:
            print(f"  ⚠️ Overlap at ({x}, {y}): letter pixel where mask is {mask[y, x]}")