#!/usr/bin/env python3
"""
Debug why 'd' still shows gap/missing pixels even after "fix".
Focus on frames 10-20 where the issue is visible.
"""

import cv2
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.video.segmentation.segment_extractor import extract_foreground_mask

VIDEO = "outputs/hello_world_2m20s_FINAL_FIX_compatible.mp4"

print("="*80)
print("🔍 DEBUGGING 'd' GAP ISSUE - Why pixels aren't restored")
print("="*80)

cap = cv2.VideoCapture(VIDEO)
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Focus on critical frames
test_frames = [10, 12, 14, 16, 18, 20]
masks = []
d_regions = []

for frame_num in test_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Extract mask
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mask = extract_foreground_mask(frame_rgb)
    
    # Focus on 'd' area (approximate position)
    d_x1, d_x2 = 820, 880  # Approximate 'd' location
    d_y1, d_y2 = 340, 380
    
    # Get mask in 'd' region
    d_mask_region = mask[d_y1:d_y2, d_x1:d_x2]
    
    # Count occluded pixels in 'd' region
    occluded_pixels = np.sum(d_mask_region > 128)
    
    # Find exact mask boundary in this region
    if occluded_pixels > 0:
        # Find the topmost occluding row in 'd' region
        rows_with_mask = np.any(d_mask_region > 128, axis=1)
        if np.any(rows_with_mask):
            top_mask_row = np.argmax(rows_with_mask)
            global_top_mask_y = d_y1 + top_mask_row
            print(f"\n🔴 Frame {frame_num}:")
            print(f"   'd' region ({d_x1},{d_y1})-({d_x2},{d_y2})")
            print(f"   Occluded pixels in 'd' region: {occluded_pixels}")
            print(f"   Top mask edge in 'd' region: y={global_top_mask_y}")
        else:
            print(f"\n✅ Frame {frame_num}: No occlusion in 'd' region")
    else:
        print(f"\n✅ Frame {frame_num}: 'd' region clear")
    
    # Store for comparison
    masks.append(mask)
    d_regions.append(d_mask_region)
    
    # Extract the actual 'd' pixels to see if they're missing
    d_area = frame[d_y1:d_y2, d_x1:d_x2]
    
    # Check for horizontal gaps (rows with very few colored pixels)
    for row_idx in range(d_area.shape[0]):
        row = d_area[row_idx]
        # Check if row has the yellow/orange text color (BGR format)
        yellow_pixels = 0
        if len(row.shape) == 3:  # Has color channels
            yellow_pixels = np.sum((row[:,1] > 150) & (row[:,2] > 100))  # Green and Red channels for yellow
            if yellow_pixels < 5:  # Very few yellow pixels = potential gap
                actual_y = d_y1 + row_idx
                print(f"   ⚠️ Potential gap at y={actual_y} (only {yellow_pixels} yellow pixels)")

cap.release()

# Compare masks between frames
print("\n" + "="*80)
print("📊 MASK MOVEMENT ANALYSIS")
print("="*80)

for i in range(1, len(masks)):
    prev_mask = masks[i-1]
    curr_mask = masks[i]
    
    # Calculate mask center movement
    prev_y, prev_x = np.where(prev_mask > 128)
    curr_y, curr_x = np.where(curr_mask > 128)
    
    if len(prev_y) > 0 and len(curr_y) > 0:
        prev_center_y = np.mean(prev_y)
        curr_center_y = np.mean(curr_y)
        movement = curr_center_y - prev_center_y
        
        frame1 = test_frames[i-1]
        frame2 = test_frames[i]
        
        print(f"\nFrames {frame1}→{frame2}: Mask moved {movement:+.1f} pixels vertically")
        
        # Check if 'd' region mask changed
        prev_d = d_regions[i-1]
        curr_d = d_regions[i]
        prev_d_pixels = np.sum(prev_d > 128)
        curr_d_pixels = np.sum(curr_d > 128)
        change = curr_d_pixels - prev_d_pixels
        
        print(f"   'd' region occlusion: {prev_d_pixels} → {curr_d_pixels} pixels ({change:+d})")
        
        if abs(movement) > 2 and abs(change) < 10:
            print(f"   🐛 BUG: Mask moved but 'd' occlusion didn't update properly!")

# Visualize the issue
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, frame_num in enumerate(test_frames[:6]):
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Zoom in on 'd' area
        d_area = frame[340:380, 820:880]
        d_area_rgb = cv2.cvtColor(d_area, cv2.COLOR_BGR2RGB)
        
        # Enhance to show gaps
        d_area_enhanced = cv2.convertScaleAbs(d_area_rgb, alpha=2.0, beta=30)
        
        # Mark any horizontal gaps (rows with very few pixels)
        for row_idx in range(d_area_enhanced.shape[0]):
            row = d_area_enhanced[row_idx]
            yellow_pixels = np.sum((row[:,:,0] > 150) & (row[:,:,1] > 150))
            if yellow_pixels < 5:
                # Draw red line to highlight gap
                cv2.line(d_area_enhanced, (0, row_idx), (59, row_idx), (255, 0, 0), 1)
        
        axes[idx].imshow(d_area_enhanced)
        axes[idx].set_title(f"Frame {frame_num} - 'd' area (gaps in red)", fontsize=10)
        axes[idx].axis('off')

plt.suptitle("'d' Gap Analysis - Red lines show missing pixel rows", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/d_gap_detailed_analysis.png", dpi=150)
print(f"\n✅ Saved detailed analysis to outputs/d_gap_detailed_analysis.png")

print("\n" + "="*80)
print("💡 HYPOTHESIS: The issue might be that:")
print("  1. Letter sprites are being modified in-place (occlusion baked in)")
print("  2. Once pixels are removed, they're not restored")
print("  3. Need to use FRESH sprite copies each frame")
print("="*80)