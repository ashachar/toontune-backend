#!/usr/bin/env python3
"""Show all segments with their BLIP2 labels"""

import cv2
import numpy as np
from sklearn.cluster import KMeans

# Load images
frame = cv2.imread("output/fast_pipeline/frame0.png")
mask = cv2.imread("output/fast_pipeline/mask.png")

# The actual BLIP2 descriptions from the pipeline
descriptions = [
    "an image of a tropical beach with palm t",
    "a beach scene with palm trees and a yell",
    "a beach scene with palm trees and a yell"
]

# Apply same clustering as pipeline
pixels = mask.reshape(-1, 3)
non_black = pixels[np.any(pixels != [0,0,0], axis=1)]

kmeans = KMeans(n_clusters=5, random_state=42, n_init=1)
kmeans.fit(non_black[:5000])

labels = np.zeros(len(pixels), dtype=int) - 1
labels[np.any(pixels != [0,0,0], axis=1)] = kmeans.predict(non_black)
label_image = labels.reshape(mask.shape[:2])

# Find segments (same as pipeline)
segments = []
for label_id in range(5):
    mask_binary = (label_image == label_id)
    area = mask_binary.sum()
    
    if area < 1000:
        continue
    
    y_coords, x_coords = np.where(mask_binary)
    segments.append({
        'id': len(segments),
        'cluster': label_id,
        'centroid': (int(x_coords.mean()), int(y_coords.mean())),
        'area': area
    })
    
    if len(segments) >= 3:
        break

print(f"Found {len(segments)} segments")

# Create annotated image
annotated = frame.copy()

# Color overlay for each segment
colors = [(255,100,100), (100,255,100), (100,100,255)]

for i, seg in enumerate(segments):
    # Add colored overlay
    segment_mask = (label_image == seg['cluster'])
    color_overlay = np.zeros_like(frame)
    color_overlay[segment_mask] = colors[i % len(colors)]
    annotated = cv2.addWeighted(annotated, 1, color_overlay, 0.3, 0)
    
    # Add label with BLIP2 description
    cx, cy = seg['centroid']
    
    # Get description
    if i < len(descriptions):
        label = descriptions[i]
    else:
        label = f"Segment {i}"
    
    # Draw label background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
    
    padding = 5
    cv2.rectangle(annotated,
                (cx - text_size[0]//2 - padding, cy - text_size[1] - padding),
                (cx + text_size[0]//2 + padding, cy + padding),
                (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(annotated, label,
              (cx - text_size[0]//2, cy),
              font, font_scale,
              (255, 255, 255), thickness, cv2.LINE_AA)
    
    # Draw center dot
    cv2.circle(annotated, (cx, cy), 5, (255, 255, 0), -1)
    
    print(f"  Segment {i}: {label} at ({cx}, {cy})")

# Save result
output_path = "output/fast_pipeline/all_blip2_labels.png"
cv2.imwrite(output_path, annotated)
print(f"\n✅ Saved to: {output_path}")