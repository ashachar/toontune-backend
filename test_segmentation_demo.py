#!/usr/bin/env python3
"""
Demo segmentation on do_re_mi video frame
Works without SAM2 installation
"""

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Extract and process frame
print("Extracting frame from video...")
cap = cv2.VideoCapture("do_re_mi_with_music_256x256_downsampled.mov")
cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Could not extract frame")
    exit(1)

# Convert BGR to RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
Image.fromarray(frame_rgb).save("test_frame_30.png")
print(f"Frame extracted: {frame_rgb.shape}")

# Create simulated segments using various methods
h, w = frame_rgb.shape[:2]
gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

segments = []
segment_names = []

# 1. Background/Foreground separation using Otsu
_, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
segments.append(otsu_mask > 0)
segment_names.append("Foreground (Otsu)")

# 2. Inverse of Otsu for background
segments.append(otsu_mask == 0)
segment_names.append("Background")

# 3. Edge regions
edges = cv2.Canny(gray, 50, 150)
kernel = np.ones((5, 5), np.uint8)
dilated_edges = cv2.dilate(edges, kernel, iterations=2)
segments.append(dilated_edges > 0)
segment_names.append("Edge Regions")

# 4. K-means clustering for color regions
pixels = frame_rgb.reshape(-1, 3).astype(np.float32)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 3
_, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
kmeans_seg = labels.reshape(h, w)

# Add k-means segments
for i in range(k):
    mask = kmeans_seg == i
    if np.sum(mask) > 100:  # Only add if segment is large enough
        segments.append(mask)
        segment_names.append(f"Color Region {i+1}")

# Limit to 6 segments
segments = segments[:6]
segment_names = segment_names[:6]

# Create visualization
print("Creating colored visualization...")
fig = plt.figure(figsize=(18, 12))

# Original image
ax1 = plt.subplot(3, 3, 1)
ax1.imshow(frame_rgb)
ax1.set_title("Original Frame", fontsize=12, fontweight='bold')
ax1.axis('off')

# Combined colored segments
combined_colored = np.zeros((h, w, 3), dtype=np.uint8)
colors = plt.cm.tab20(np.linspace(0, 1, len(segments)))

for idx, (segment, color) in enumerate(zip(segments, colors)):
    rgb_color = (color[:3] * 255).astype(np.uint8)
    # Add to combined with priority to later segments
    mask_area = segment.astype(bool)
    combined_colored[mask_area] = rgb_color

ax2 = plt.subplot(3, 3, 2)
ax2.imshow(combined_colored)
ax2.set_title("All Segments (Colored)", fontsize=12, fontweight='bold')
ax2.axis('off')

# Overlay on original
overlay = cv2.addWeighted(frame_rgb, 0.5, combined_colored, 0.5, 0)
ax3 = plt.subplot(3, 3, 3)
ax3.imshow(overlay)
ax3.set_title("Overlay on Original", fontsize=12, fontweight='bold')
ax3.axis('off')

# Individual segments
for idx, (segment, name, color) in enumerate(zip(segments, segment_names, colors)):
    ax = plt.subplot(3, 3, idx + 4)
    
    # Create individual colored overlay
    individual_colored = np.zeros((h, w, 3), dtype=np.uint8)
    individual_colored[segment] = (color[:3] * 255).astype(np.uint8)
    
    # Blend with original
    result = cv2.addWeighted(frame_rgb, 0.6, individual_colored, 0.4, 0)
    
    ax.imshow(result)
    ax.set_title(name, fontsize=10)
    ax.axis('off')

plt.suptitle("Segmentation Demo on do_re_mi.mov Frame", fontsize=14, fontweight='bold')
plt.tight_layout()

# Save outputs
output_path = "segmentation_demo_colored.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Visualization saved to: {output_path}")

# Save individual outputs
Image.fromarray(combined_colored).save("segments_colored_only.png")
Image.fromarray(overlay).save("segments_overlay.png")
print("Additional outputs saved: segments_colored_only.png, segments_overlay.png")

# Show statistics
print("\nSegmentation Statistics:")
for name, segment in zip(segment_names, segments):
    area = np.sum(segment)
    percentage = (area / (h * w)) * 100
    print(f"  {name}: {percentage:.1f}% of image")

print("\nDemo complete!")
print("\nNote: This is a simulation using traditional CV methods.")
print("For actual SAM2 segmentation, install SAM2 and run:")
print("  python utils/segmentation/sam2_local.py do_re_mi_with_music_256x256_downsampled.mov --automatic")