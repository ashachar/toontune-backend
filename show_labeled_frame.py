#!/usr/bin/env python3
"""Show the first frame with labels"""

import cv2
import numpy as np

# Load the first frame and mask
frame = cv2.imread("output/fast_pipeline/frame0.png")
mask = cv2.imread("output/fast_pipeline/mask.png")

# The segments and their descriptions from the pipeline output
segments_info = [
    {"id": 0, "description": "an image of a tropical beach with palm t", "centroid": None},
    {"id": 1, "description": "a beach scene with palm trees and a yell", "centroid": None},
    {"id": 2, "description": "a beach scene with palm trees and a yell", "centroid": None}
]

# Find centroids from mask
mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(mask_gray, 10, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort by area and take top 3
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

# Create annotated image
annotated = frame.copy()

# Blend with mask for visualization
annotated = cv2.addWeighted(annotated, 0.6, mask, 0.4, 0)

# Add labels for each contour
for i, contour in enumerate(contours):
    if i >= len(segments_info):
        break
    
    # Get centroid
    M = cv2.moments(contour)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Draw contour
        cv2.drawContours(annotated, [contour], -1, (0, 255, 0), 2)
        
        # Add label with background
        label = segments_info[i]["description"]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Get text size
        text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
        text_width, text_height = text_size
        
        # Draw background rectangle
        padding = 5
        cv2.rectangle(annotated,
                    (cx - text_width//2 - padding, cy - text_height - padding),
                    (cx + text_width//2 + padding, cy + padding),
                    (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(annotated, label,
                  (cx - text_width//2, cy),
                  font, font_scale,
                  (255, 255, 255), thickness, cv2.LINE_AA)
        
        # Draw center point
        cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)

# Save result
output_path = "output/fast_pipeline/first_frame_labeled.png"
cv2.imwrite(output_path, annotated)
print(f"✅ Saved labeled first frame to: {output_path}")

# Also create a side-by-side comparison
comparison = np.hstack([frame, annotated])
comparison_path = "output/fast_pipeline/comparison.png"
cv2.imwrite(comparison_path, comparison)
print(f"✅ Saved comparison to: {comparison_path}")