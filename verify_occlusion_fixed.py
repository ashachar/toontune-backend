#!/usr/bin/env python3
"""Verify that occlusion is now working correctly in the fixed video."""

import cv2
import numpy as np
import sys
sys.path.append('utils')
from video.segmentation.segment_extractor import extract_foreground_mask

# Load the fixed video
cap = cv2.VideoCapture('outputs/occlusion_fixed_final_hq.mp4')

# Check multiple frames during dissolve (after frame 22 when dissolve starts)
test_frames = [25, 30, 35, 40, 45]

print("="*60)
print("VERIFYING OCCLUSION FIX")
print("="*60)

for frame_num in test_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    
    if not ret:
        continue
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Extract current mask
    mask = extract_foreground_mask(frame_rgb)
    
    # Find yellow text pixels (BGR format)
    yellow_pixels = (
        (frame[:, :, 0] < 100) &  # Blue low
        (frame[:, :, 1] > 180) &  # Green high
        (frame[:, :, 2] > 180)    # Red high
    )
    
    # Check overlap
    overlap = yellow_pixels & (mask > 128)
    overlap_count = np.sum(overlap)
    
    yellow_count = np.sum(yellow_pixels)
    mask_count = np.sum(mask > 128)
    
    print(f"\nFrame {frame_num}:")
    print(f"  Yellow text pixels: {yellow_count:,}")
    print(f"  Person mask pixels: {mask_count:,}")
    print(f"  Overlapping pixels: {overlap_count:,}")
    
    if overlap_count > 100:
        print(f"  ⚠️ WARNING: {overlap_count} pixels visible through person!")
        print(f"     This means occlusion is NOT working properly.")
    else:
        print(f"  ✅ GOOD: Little to no overlap - occlusion is working!")

cap.release()

print("\n" + "="*60)
print("VERIFICATION COMPLETE")
print("="*60)
print("\nIf overlap counts are low (< 100 pixels), the fix is working!")
print("Letters should be hidden behind the person during dissolve.")