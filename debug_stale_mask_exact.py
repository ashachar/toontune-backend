#!/usr/bin/env python3
"""Debug the exact stale mask issue - prove that occlusion is using wrong positions."""

import cv2
import numpy as np
import sys
sys.path.append('utils')
from video.segmentation.segment_extractor import extract_foreground_mask

# Load the video
cap = cv2.VideoCapture('outputs/test_occlusion_proof_final_h264.mp4')

# Check frame 19 (end of motion) and frame 25 (during dissolve)
frames_to_check = {
    19: "End of motion (handoff)",
    25: "During dissolve (person moved)"
}

masks = {}
letter_regions = {}

for frame_num, desc in frames_to_check.items():
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    
    if not ret:
        continue
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Extract mask
    mask = extract_foreground_mask(frame_rgb)
    masks[frame_num] = mask
    
    # Find yellow letter pixels (BGR format from cv2)
    yellow_mask = (
        (frame[:, :, 0] < 100) &  # Blue channel low
        (frame[:, :, 1] > 180) &  # Green channel high  
        (frame[:, :, 2] > 180)    # Red channel high
    )
    letter_regions[frame_num] = yellow_mask
    
    print(f"\n{'='*60}")
    print(f"Frame {frame_num}: {desc}")
    print(f"{'='*60}")
    
    # Find where letters are
    y_coords, x_coords = np.where(yellow_mask)
    if len(x_coords) > 0:
        letter_bbox = (x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max())
        print(f"Letters visible at: x=[{letter_bbox[0]}-{letter_bbox[2]}], y=[{letter_bbox[1]}-{letter_bbox[3]}]")
        
        # Check if person overlaps with letters
        letter_region_mask = mask[letter_bbox[1]:letter_bbox[3]+1, letter_bbox[0]:letter_bbox[2]+1]
        overlap_pixels = np.sum(letter_region_mask > 128)
        
        if overlap_pixels > 0:
            print(f"⚠️ Person overlaps letters: {overlap_pixels} pixels")
            print(f"   → Letters SHOULD be hidden but are visible!")
    
    # Find where person is
    person_y, person_x = np.where(mask > 128)
    if len(person_x) > 0:
        person_bbox = (person_x.min(), person_y.min(), person_x.max(), person_y.max())
        print(f"Person mask at: x=[{person_bbox[0]}-{person_bbox[2]}], y=[{person_bbox[1]}-{person_bbox[3]}]")

cap.release()

# Now compare: Did person move between frames?
print("\n" + "="*60)
print("MASK MOVEMENT ANALYSIS")
print("="*60)

if 19 in masks and 25 in masks:
    mask_19 = masks[19]
    mask_25 = masks[25]
    
    # Calculate mask difference
    mask_diff = np.abs(mask_25.astype(float) - mask_19.astype(float))
    pixels_changed = np.sum(mask_diff > 0)
    
    print(f"Mask pixels changed: {pixels_changed:,}")
    
    # Find center of mass for each mask
    y19, x19 = np.where(mask_19 > 128)
    if len(x19) > 0:
        com_19 = (np.mean(x19), np.mean(y19))
        print(f"Frame 19 person center: ({com_19[0]:.0f}, {com_19[1]:.0f})")
    
    y25, x25 = np.where(mask_25 > 128)
    if len(x25) > 0:
        com_25 = (np.mean(x25), np.mean(y25))
        print(f"Frame 25 person center: ({com_25[0]:.0f}, {com_25[1]:.0f})")
        
        if 'com_19' in locals():
            dx = com_25[0] - com_19[0]
            dy = com_25[1] - com_19[1]
            print(f"Person moved: dx={dx:.0f}, dy={dy:.0f} pixels")
    
    # Check: Are letters visible where person WAS at frame 19?
    print("\n" + "="*60)
    print("STALE MASK BUG PROOF")
    print("="*60)
    
    letters_25 = letter_regions[25]
    
    # Check if letters at frame 25 overlap with where person WAS at frame 19
    overlap_with_old_position = letters_25 & (mask_19 > 128)
    old_overlap_pixels = np.sum(overlap_with_old_position)
    
    # Check if letters at frame 25 overlap with where person IS at frame 25
    overlap_with_current_position = letters_25 & (mask_25 > 128)
    current_overlap_pixels = np.sum(overlap_with_current_position)
    
    # Debug: Check why we're seeing different numbers
    print(f"\nDebug: Total yellow pixels at frame 25: {np.sum(letters_25)}")
    print(f"Debug: Total mask pixels at frame 25: {np.sum(mask_25 > 128)}")
    
    print(f"Letters visible at frame 25:")
    print(f"  - Overlap with person's OLD position (frame 19): {old_overlap_pixels} pixels")
    print(f"  - Overlap with person's CURRENT position (frame 25): {current_overlap_pixels} pixels")
    
    if current_overlap_pixels > 100:
        print(f"\n❌ BUG CONFIRMED: {current_overlap_pixels} letter pixels visible")
        print(f"   through person's CURRENT position!")
        print(f"   This proves occlusion is NOT working correctly.")
    elif old_overlap_pixels > 100 and current_overlap_pixels < 100:
        print(f"\n❌ DIFFERENT BUG: Letters showing where person WAS,")
        print(f"   not where person IS. This would mean stale mask is being used.")