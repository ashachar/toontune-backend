#!/usr/bin/env python3
"""
Debug the foreground mask extraction at a specific frame
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the original frame
frame = cv2.imread("outputs/debug_original_3.5s.png")
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Load the masking module
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pipelines.word_level_pipeline.masking import ForegroundMaskExtractor

# Extract the mask
mask_extractor = ForegroundMaskExtractor()
mask = mask_extractor.extract_foreground_mask(frame)

# Debug: Show what the mask looks like around the text area (y=216)
text_y = 216
text_region = mask[text_y-30:text_y+30, :]

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Original frame
axes[0, 0].imshow(frame_rgb)
axes[0, 0].set_title("Original Frame")
axes[0, 0].axis('off')

# Full mask
axes[0, 1].imshow(mask, cmap='gray')
axes[0, 1].set_title("Extracted Foreground Mask")
axes[0, 1].axis('off')

# Mask around text area
axes[0, 2].imshow(text_region, cmap='gray')
axes[0, 2].set_title(f"Mask at Text Region (y={text_y-30} to {text_y+30})")
axes[0, 2].axis('off')

# Overlay mask on original
overlay = frame_rgb.copy()
overlay[mask > 0] = [255, 0, 0]  # Red where mask is active
axes[1, 0].imshow(overlay)
axes[1, 0].set_title("Mask Overlay (Red = Foreground)")
axes[1, 0].axis('off')

# Horizontal profile at text line
profile = mask[text_y, :]
axes[1, 1].plot(profile)
axes[1, 1].set_title(f"Horizontal Mask Profile at y={text_y}")
axes[1, 1].set_xlabel("X Position")
axes[1, 1].set_ylabel("Mask Value")
axes[1, 1].grid(True)

# Find contours of the mask
contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_img = frame_rgb.copy()
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
axes[1, 2].imshow(contour_img)
axes[1, 2].set_title("Mask Contours")
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig("outputs/debug_mask_analysis.png", dpi=150)
print("Saved mask analysis to outputs/debug_mask_analysis.png")

# Print statistics
print(f"\nMask Statistics:")
print(f"  Total pixels: {mask.size}")
print(f"  Foreground pixels: {np.sum(mask > 0)} ({100*np.sum(mask > 0)/mask.size:.1f}%)")
print(f"  Mask values: min={mask.min()}, max={mask.max()}")

# Check specific area where text would be
text_x_start = 200  # Approximate
text_x_end = 1000
text_mask_region = mask[text_y-20:text_y+20, text_x_start:text_x_end]
print(f"\nText Region Analysis (y={text_y-20} to {text_y+20}, x={text_x_start} to {text_x_end}):")
print(f"  Foreground pixels in text area: {np.sum(text_mask_region > 0)} ({100*np.sum(text_mask_region > 0)/text_mask_region.size:.1f}%)")

# Check what method is being used
print(f"\nMask Extraction Method:")
print(f"  Using: {mask_extractor.__class__.__name__}")

# Save individual mask
cv2.imwrite("outputs/debug_mask_only.png", mask)
print("\nSaved mask only to outputs/debug_mask_only.png")