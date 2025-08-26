#!/usr/bin/env python3
"""Debug exactly what mask region is being extracted."""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create test scenario
width, height = 640, 360

# Frame 1: Person at x=200-300, H at x=280-370
frame1 = np.full((height, width, 3), 220, dtype=np.uint8)
cv2.rectangle(frame1, (200, 100), (300, 250), (50, 50, 100), -1)

# Frame 2: Person at x=400-500, H still at x=280-370
frame2 = np.full((height, width, 3), 220, dtype=np.uint8)
cv2.rectangle(frame2, (400, 100), (500, 250), (50, 50, 100), -1)

# Extract masks
import sys
sys.path.append('utils')
from video.segmentation.segment_extractor import extract_foreground_mask

mask1 = extract_foreground_mask(frame1)
mask2 = extract_foreground_mask(frame2)

# H position (fixed)
h_x = 280
h_y = 150
h_w = 90
h_h = 90

# Extract mask regions as the code does
x1 = h_x
x2 = h_x + h_w
y1 = h_y
y2 = h_y + h_h

print("H letter bounds: x=[{}-{}], y=[{}-{}]".format(x1, x2, y1, y2))
print("="*60)

# Frame 1 mask region
mask_region1 = mask1[y1:y2, x1:x2]
print("\nFrame 1 (person at x=[200-300]):")
print(f"  Mask region shape: {mask_region1.shape}")
print(f"  Mask pixels in region: {np.sum(mask_region1 > 128)}")

# Find where mask is within the region
mask_y, mask_x = np.where(mask_region1 > 128)
if len(mask_x) > 0:
    local_left = mask_x.min()
    local_right = mask_x.max()
    global_left = x1 + local_left
    global_right = x1 + local_right
    print(f"  Mask in region: local x=[{local_left}-{local_right}]")
    print(f"  Mask in region: global x=[{global_left}-{global_right}]")
    print(f"  → H should be visible from x={global_right+1} to x={x2}")

# Frame 2 mask region
mask_region2 = mask2[y1:y2, x1:x2]
print("\nFrame 2 (person at x=[400-500]):")
print(f"  Mask region shape: {mask_region2.shape}")
print(f"  Mask pixels in region: {np.sum(mask_region2 > 128)}")

mask_y, mask_x = np.where(mask_region2 > 128)
if len(mask_x) > 0:
    local_left = mask_x.min()
    local_right = mask_x.max()
    global_left = x1 + local_left
    global_right = x1 + local_right
    print(f"  Mask in region: local x=[{local_left}-{local_right}]")
    print(f"  Mask in region: global x=[{global_left}-{global_right}]")
else:
    print("  No mask pixels in H region - H should be fully visible")

# Visualize
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Frame 1
axes[0,0].imshow(frame1)
axes[0,0].set_title("Frame 1: Person at x=[200-300]")
axes[0,0].add_patch(plt.Rectangle((h_x, h_y), h_w, h_h, fill=False, edgecolor='yellow', linewidth=2))
axes[0,0].axis('off')

axes[0,1].imshow(mask1, cmap='gray')
axes[0,1].set_title("Mask 1")
axes[0,1].add_patch(plt.Rectangle((h_x, h_y), h_w, h_h, fill=False, edgecolor='yellow', linewidth=2))
axes[0,1].axis('off')

axes[0,2].imshow(mask_region1, cmap='gray')
axes[0,2].set_title(f"Mask region at H: {np.sum(mask_region1>128)} pixels")
axes[0,2].axis('off')

# Frame 2
axes[1,0].imshow(frame2)
axes[1,0].set_title("Frame 2: Person at x=[400-500]")
axes[1,0].add_patch(plt.Rectangle((h_x, h_y), h_w, h_h, fill=False, edgecolor='yellow', linewidth=2))
axes[1,0].axis('off')

axes[1,1].imshow(mask2, cmap='gray')
axes[1,1].set_title("Mask 2")
axes[1,1].add_patch(plt.Rectangle((h_x, h_y), h_w, h_h, fill=False, edgecolor='yellow', linewidth=2))
axes[1,1].axis('off')

axes[1,2].imshow(mask_region2, cmap='gray')
axes[1,2].set_title(f"Mask region at H: {np.sum(mask_region2>128)} pixels")
axes[1,2].axis('off')

plt.suptitle("Mask Region Extraction Test", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/mask_region_test.png', dpi=150)
print("\n✅ Saved visualization to outputs/mask_region_test.png")

print("\n" + "="*60)
print("EXPECTED BEHAVIOR:")
print("="*60)
print("Frame 1: Person overlaps H → mask_region should have pixels")
print("Frame 2: Person doesn't overlap H → mask_region should be empty")
print("\nIf both frames show similar mask regions, there's a bug!")