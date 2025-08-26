#!/usr/bin/env python3
"""Analyze why dissolve isn't applying masks correctly even though they're being extracted."""

import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Load the frame at 1 second where the issue is visible
frame = cv2.imread('outputs/frame_at_1s.png')
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Extract the current mask
import sys
sys.path.append('utils')
from video.segmentation.segment_extractor import extract_foreground_mask

current_mask = extract_foreground_mask(frame_rgb)

print("="*60)
print("ANALYZING STALE MASK BUG IN DISSOLVE")
print("="*60)

# Find yellow letter pixels
yellow_mask = (
    (frame_rgb[:, :, 0] > 180) & 
    (frame_rgb[:, :, 1] > 180) & 
    (frame_rgb[:, :, 2] < 100)
)

# Find where letters overlap with person
overlap = yellow_mask & (current_mask > 128)
overlap_pixels = np.sum(overlap)

print(f"\n📊 Frame Analysis (at 1 second):")
print(f"- Yellow pixels (letters): {np.sum(yellow_mask):,}")
print(f"- Mask pixels (person): {np.sum(current_mask > 128):,}")
print(f"- Overlap (letters that SHOULD be hidden): {overlap_pixels:,}")

if overlap_pixels > 0:
    print(f"\n❌ BUG CONFIRMED: {overlap_pixels} pixels showing through person!")
    y_coords, x_coords = np.where(overlap)
    print(f"   Problem areas: x=[{x_coords.min()}-{x_coords.max()}], y=[{y_coords.min()}-{y_coords.max()}]")

# Create detailed visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("STALE MASK BUG: Letters Show Through Moving Person", fontsize=14, fontweight='bold')

# Original frame
axes[0, 0].imshow(frame_rgb)
axes[0, 0].set_title("Frame at 1s - Letters visible through person")
axes[0, 0].axis('off')

# Current mask
axes[0, 1].imshow(current_mask, cmap='gray')
axes[0, 1].set_title(f"Current Mask: {np.sum(current_mask>128):,} pixels")
axes[0, 1].axis('off')

# Yellow pixels
axes[0, 2].imshow(yellow_mask, cmap='hot')
axes[0, 2].set_title(f"Letter Pixels: {np.sum(yellow_mask):,}")
axes[0, 2].axis('off')

# Overlap visualization
axes[1, 0].imshow(overlap, cmap='hot')
axes[1, 0].set_title(f"BUG: {overlap_pixels} pixels should be hidden!")
axes[1, 0].axis('off')

# Annotated frame
annotated = frame_rgb.copy()
# Highlight overlap areas in red
annotated[overlap] = [255, 0, 0]
axes[1, 1].imshow(annotated)
axes[1, 1].set_title("Red = Letters that should be hidden")
axes[1, 1].axis('off')

# Explanation
axes[1, 2].axis('off')
axes[1, 2].text(0.1, 0.8, "THE PROBLEM:", fontsize=12, fontweight='bold')
axes[1, 2].text(0.1, 0.6, "1. Masks ARE being extracted (logs confirm)", fontsize=10)
axes[1, 2].text(0.1, 0.5, "2. Pixel counts change (person moving)", fontsize=10)
axes[1, 2].text(0.1, 0.4, "3. BUT occlusion not applied correctly", fontsize=10, color='red')
axes[1, 2].text(0.1, 0.3, "4. Letters show where person IS now", fontsize=10, color='red')
axes[1, 2].text(0.1, 0.1, "Root cause: Mask applied wrong or cached", fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/dissolve_mask_bug_analysis.png', dpi=150)
print("\n✅ Saved analysis to: outputs/dissolve_mask_bug_analysis.png")

print("\n" + "="*60)
print("ROOT CAUSE ANALYSIS")
print("="*60)
print("\nThe dissolve animation code:")
print("1. ✅ Extracts fresh masks every frame (confirmed by logs)")
print("2. ✅ Has is_behind=True (confirmed by debug)")
print("3. ❌ BUT letters still show through the person")
print("\nPossible causes:")
print("- Mask is extracted but not applied to the right letters")
print("- Letter positions are wrong when checking against mask")
print("- Timing issue: mask from wrong frame being used")
print("- The occlusion calculation is incorrect")
print("\nNeed to check the exact occlusion application in letter_3d_dissolve.py")