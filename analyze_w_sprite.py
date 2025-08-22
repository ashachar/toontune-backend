#!/usr/bin/env python3
"""Analyze the W sprite to understand disconnected components."""

import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

# Load frames to extract W sprite
cap = cv2.VideoCapture("hello_world_fixed.mp4")

# Get frame just before W appears (frame 90) and when W is visible (frame 92)
cap.set(cv2.CAP_PROP_POS_FRAMES, 90)
ret, frame_before = cap.read()

cap.set(cv2.CAP_PROP_POS_FRAMES, 92)
ret, frame_with_w = cap.read()

cap.release()

# Extract W region (approximate bounds)
x1, x2 = 520, 760
y1, y2 = 140, 350

before_roi = frame_before[y1:y2, x1:x2]
with_w_roi = frame_with_w[y1:y2, x1:x2]

# Get the difference to isolate W
diff = cv2.absdiff(with_w_roi, before_roi)
gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

# Threshold to get W mask
_, w_mask = cv2.threshold(gray_diff, 10, 255, cv2.THRESH_BINARY)

# Analyze connected components
labeled, num_features = ndimage.label(w_mask)
print(f"Found {num_features} connected components in W sprite")

# Get sizes of each component
component_sizes = []
for i in range(1, num_features + 1):
    size = np.sum(labeled == i)
    component_sizes.append((i, size))
    print(f"  Component {i}: {size} pixels")

# Sort by size
component_sizes.sort(key=lambda x: x[1], reverse=True)

# Visualize components
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Original mask
axes[0, 0].imshow(w_mask, cmap='gray')
axes[0, 0].set_title('Original W mask')
axes[0, 0].axis('off')

# All components colored
colored_labels = np.zeros((*labeled.shape, 3), dtype=np.uint8)
colors = plt.cm.rainbow(np.linspace(0, 1, num_features))
for i in range(1, min(num_features + 1, 11)):  # Show up to 10 components
    mask = labeled == i
    color = (colors[i-1][:3] * 255).astype(np.uint8)
    colored_labels[mask] = color

axes[0, 1].imshow(colored_labels)
axes[0, 1].set_title(f'All {num_features} components (colored)')
axes[0, 1].axis('off')

# Main component only
if num_features > 0:
    main_component_id = component_sizes[0][0]
    main_only = np.zeros_like(w_mask)
    main_only[labeled == main_component_id] = 255
    axes[0, 2].imshow(main_only, cmap='gray')
    axes[0, 2].set_title(f'Main component only ({component_sizes[0][1]} pixels)')
    axes[0, 2].axis('off')

# Small components only (the problematic ones)
if num_features > 1:
    small_only = np.zeros_like(w_mask)
    for comp_id, size in component_sizes[1:]:
        small_only[labeled == comp_id] = 255
    axes[1, 0].imshow(small_only, cmap='gray')
    axes[1, 0].set_title(f'Small components ({num_features-1} total)')
    axes[1, 0].axis('off')
    
    # Zoom in on problematic area (right side of W)
    zoom_x1, zoom_x2 = 180, 240  # Right side of ROI
    zoom_y1, zoom_y2 = 40, 60     # Top area where artifacts appear
    
    zoom_mask = w_mask[zoom_y1:zoom_y2, zoom_x1:zoom_x2]
    zoom_small = small_only[zoom_y1:zoom_y2, zoom_x1:zoom_x2]
    
    axes[1, 1].imshow(zoom_mask, cmap='gray', interpolation='nearest')
    axes[1, 1].set_title('Zoom: Right side of W (all)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(zoom_small, cmap='gray', interpolation='nearest')
    axes[1, 2].set_title('Zoom: Small components only')
    axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('w_sprite_analysis.png', dpi=150, bbox_inches='tight')
print(f"\nSaved analysis to w_sprite_analysis.png")

# Check distance of small components from main component
if num_features > 1:
    main_mask = labeled == main_component_id
    
    print(f"\nDistance analysis of small components:")
    for comp_id, size in component_sizes[1:]:
        if size < 100:  # Only analyze very small components
            comp_mask = labeled == comp_id
            
            # Find centroid of small component
            y_coords, x_coords = np.where(comp_mask)
            centroid_y = np.mean(y_coords)
            centroid_x = np.mean(x_coords)
            
            # Find nearest point in main component
            main_y, main_x = np.where(main_mask)
            if len(main_y) > 0:
                distances = np.sqrt((main_y - centroid_y)**2 + (main_x - centroid_x)**2)
                min_dist = np.min(distances)
                print(f"  Component {comp_id} ({size} px): {min_dist:.1f} pixels from main W")