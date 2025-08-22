#!/usr/bin/env python3
"""Check if letters remain visible after dissolve completion."""

import cv2
import numpy as np

video_path = "hello_world_fixed.mp4"
cap = cv2.VideoCapture(video_path)

# Get a frame before any dissolve starts
cap.set(cv2.CAP_PROP_POS_FRAMES, 5)
ret, initial_frame = cap.read()

# Get a frame after ALL letters should be dissolved (frame 220+)
# Based on logs: last letter completes around frame 216
cap.set(cv2.CAP_PROP_POS_FRAMES, 250)
ret, final_frame = cap.read()

if ret:
    # Calculate difference
    diff = cv2.absdiff(final_frame, initial_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Check for any remaining text pixels
    text_region = gray_diff[150:350, 50:1100]  # Approximate text area
    
    # Count pixels that differ from background
    changed_pixels = np.sum(text_region > 10)
    total_pixels = text_region.shape[0] * text_region.shape[1]
    
    print(f"After all letters should be dissolved (frame 250):")
    print(f"  Changed pixels in text region: {changed_pixels}/{total_pixels}")
    print(f"  Percentage: {changed_pixels/total_pixels*100:.2f}%")
    
    if changed_pixels > 100:
        print("\n⚠️  PROBLEM: Letters are still visible after dissolve!")
        
        # Save visualization
        highlight = final_frame.copy()
        # Highlight areas that differ from initial
        mask = gray_diff > 10
        highlight[mask] = [0, 0, 255]  # Red for remaining text
        cv2.imwrite("post_dissolve_remnants.png", highlight)
        print("Saved visualization to post_dissolve_remnants.png")
        
        # Also save the actual final frame
        cv2.imwrite("final_frame_250.png", final_frame)
        
        # Check for yellow pixels (text color)
        yellow_mask = (final_frame[:,:,1] > 180) & (final_frame[:,:,2] > 180) & (final_frame[:,:,0] < 100)
        yellow_count = np.sum(yellow_mask)
        print(f"\nYellow pixels remaining: {yellow_count}")
        
    else:
        print("\n✓ Letters appear to be fully dissolved")

cap.release()