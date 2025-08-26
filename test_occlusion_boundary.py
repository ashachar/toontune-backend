#!/usr/bin/env python3
"""Test to prove the occlusion boundary issue."""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Extract frames from the video with the issue
cap = cv2.VideoCapture('outputs/occlusion_fixed_final_hq.mp4')

frames_to_check = [22, 25, 28, 31]  # During dissolve
extracted_frames = {}

for frame_num in frames_to_check:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if ret:
        extracted_frames[frame_num] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

cap.release()

# Now analyze the H letter visibility
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for idx, frame_num in enumerate(frames_to_check):
    frame = extracted_frames[frame_num]
    
    # Zoom in on the H letter region
    h_region = frame[250:450, 300:500]  # Approximate H location
    
    axes[idx].imshow(h_region)
    axes[idx].set_title(f"Frame {frame_num}")
    axes[idx].axis('off')
    
    # Find yellow pixels in this region
    yellow = (
        (h_region[:, :, 0] > 180) &
        (h_region[:, :, 1] > 180) &
        (h_region[:, :, 2] < 100)
    )
    
    if np.any(yellow):
        y_coords, x_coords = np.where(yellow)
        leftmost_x = x_coords.min()
        rightmost_x = x_coords.max()
        
        # Mark the edges
        axes[idx].axvline(leftmost_x, color='green', linestyle='--', alpha=0.5)
        axes[idx].axvline(rightmost_x, color='red', linestyle='--', alpha=0.5)
        
        print(f"Frame {frame_num}: H visible from x={leftmost_x} to x={rightmost_x} (width={rightmost_x-leftmost_x})")
    else:
        print(f"Frame {frame_num}: H not visible")

plt.suptitle("H Letter Occlusion Boundary Analysis", fontsize=14)
plt.tight_layout()
plt.savefig('outputs/h_occlusion_boundary.png', dpi=150)
print("\n✅ Saved analysis to outputs/h_occlusion_boundary.png")

# Now let's check if the person mask is moving
import sys
sys.path.append('utils')
from video.segmentation.segment_extractor import extract_foreground_mask

print("\n" + "="*60)
print("Person Mask Movement Analysis")
print("="*60)

for frame_num in frames_to_check:
    frame = extracted_frames[frame_num]
    mask = extract_foreground_mask(frame)
    
    # Find mask bounds
    y_coords, x_coords = np.where(mask > 128)
    if len(x_coords) > 0:
        mask_left = x_coords.min()
        mask_right = x_coords.max()
        mask_center = (mask_left + mask_right) // 2
        print(f"Frame {frame_num}: Mask x=[{mask_left}-{mask_right}], center={mask_center}")

print("\nIf the mask center is moving but H's right edge stays at the same x coordinate,")
print("then the occlusion boundary is NOT updating with the mask position.")