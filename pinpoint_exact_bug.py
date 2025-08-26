#!/usr/bin/env python3
"""Pinpoint the exact stale mask bug."""

import cv2
import numpy as np
import sys
sys.path.append('utils')
from video.segmentation.segment_extractor import extract_foreground_mask

# Create two test frames
width, height = 640, 360

# Frame 1: Person at x=200
frame1 = np.full((height, width, 3), 220, dtype=np.uint8)
cv2.rectangle(frame1, (200, 100), (300, 250), (50, 50, 100), -1)

# Frame 2: Person at x=400  
frame2 = np.full((height, width, 3), 220, dtype=np.uint8)
cv2.rectangle(frame2, (400, 100), (500, 250), (50, 50, 100), -1)

print("Test setup:")
print("  Frame 1: Person at x=[200-300]")
print("  Frame 2: Person at x=[400-500]")
print("  H letter at x=[280-370] (overlaps person in frame 1, not in frame 2)")
print("="*60)

# Extract masks
mask1 = extract_foreground_mask(frame1)
mask2 = extract_foreground_mask(frame2)

print("\nMask extraction:")
y1, x1 = np.where(mask1 > 128)
if len(x1) > 0:
    print(f"  Mask 1: x=[{x1.min()}-{x1.max()}]")

y2, x2 = np.where(mask2 > 128)
if len(x2) > 0:
    print(f"  Mask 2: x=[{x2.min()}-{x2.max()}]")

# Now test the dissolve directly
from utils.animations.letter_3d_dissolve import Letter3DDissolve

dissolve = Letter3DDissolve(
    duration=0.2,  # Just 2 frames
    fps=10,
    resolution=(width, height),
    text="H",
    font_size=60,
    text_color=(255, 220, 0),
    initial_scale=1.0,
    initial_position=(280, 150),
    is_behind=True,
    debug=False  # Turn off debug for cleaner output
)

print("\nTesting dissolve:")
print("-"*40)

# Process frame 1
print("\n[Frame 0] Person at x=[200-300], H at x=[280-370]")
output1 = dissolve.generate_frame(0, frame1)

# Count visible yellow pixels
yellow1 = (output1[:, :, 0] > 180) & (output1[:, :, 1] > 180) & (output1[:, :, 2] < 100)
if np.any(yellow1):
    y_coords, x_coords = np.where(yellow1)
    print(f"  H visible: x=[{x_coords.min()}-{x_coords.max()}]")
    if x_coords.min() > 300:
        print(f"  ✅ H correctly starts after person (at x=300)")
    else:
        print(f"  ⚠️ H overlaps with person!")

# Process frame 2
print("\n[Frame 1] Person at x=[400-500], H at x=[280-370]")
output2 = dissolve.generate_frame(1, frame2)

# Count visible yellow pixels
yellow2 = (output2[:, :, 0] > 180) & (output2[:, :, 1] > 180) & (output2[:, :, 2] < 100)
if np.any(yellow2):
    y_coords, x_coords = np.where(yellow2)
    print(f"  H visible: x=[{x_coords.min()}-{x_coords.max()}]")
    if x_coords.max() < 400:
        print(f"  ✅ H correctly ends before person (at x=400)")
    else:
        print(f"  ⚠️ H overlaps with person!")

print("\n" + "="*60)
print("BUG CHECK:")
print("="*60)

# Compare the two outputs
if np.any(yellow1) and np.any(yellow2):
    y1, x1 = np.where(yellow1)
    y2, x2 = np.where(yellow2)
    
    range1 = (x1.min(), x1.max())
    range2 = (x2.min(), x2.max())
    
    print(f"Frame 0: H visible at x=[{range1[0]}-{range1[1]}]")
    print(f"Frame 1: H visible at x=[{range2[0]}-{range2[1]}]")
    
    if range1 == range2:
        print("\n❌ BUG CONFIRMED: H visibility doesn't change even though person moved!")
        print("   This proves the occlusion mask is stuck at the first frame.")
    elif abs(range1[0] - range2[0]) < 10 and abs(range1[1] - range2[1]) < 10:
        print("\n⚠️ PARTIAL BUG: H visibility barely changed")
        print(f"   Left edge moved {abs(range1[0] - range2[0])} pixels")
        print(f"   Right edge moved {abs(range1[1] - range2[1])} pixels")
    else:
        print("\n✅ H visibility changed correctly with person movement")

# Save visual proof
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(output1)
axes[0].set_title("Frame 0: Person at x=[200-300]")
axes[0].axvline(200, color='blue', linestyle='--', alpha=0.5)
axes[0].axvline(300, color='blue', linestyle='--', alpha=0.5)
axes[0].axis('off')

axes[1].imshow(output2)
axes[1].set_title("Frame 1: Person at x=[400-500]")
axes[1].axvline(400, color='blue', linestyle='--', alpha=0.5)
axes[1].axvline(500, color='blue', linestyle='--', alpha=0.5)
axes[1].axis('off')

plt.suptitle("Stale Mask Bug Test", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/stale_mask_pinpoint.png', dpi=150)
print("\n✅ Saved visual proof to outputs/stale_mask_pinpoint.png")