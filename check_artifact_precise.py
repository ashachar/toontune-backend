#!/usr/bin/env python3
"""More precise check for gray line artifacts by comparing to background."""

import cv2
import numpy as np
import os

video_path = "hello_world_fixed.mp4"
if not os.path.exists(video_path):
    print(f"Error: {video_path} not found")
    exit(1)

cap = cv2.VideoCapture(video_path)

# Get frame before W starts dissolving to establish background
cap.set(cv2.CAP_PROP_POS_FRAMES, 85)  # Before W dissolves
ret, background_frame = cap.read()

# Define ROI for checking (area above and around W)
# W is at approximately x=570-700, y=190-300
# Check wider area: x=520-750, y=140-190
roi_x1, roi_x2 = 520, 750
roi_y1, roi_y2 = 140, 190

background_roi = background_frame[roi_y1:roi_y2, roi_x1:roi_x2]
bg_gray = cv2.cvtColor(background_roi, cv2.COLOR_BGR2GRAY)

print(f"Checking for artifacts different from background...")
print(f"Background mean intensity: {np.mean(bg_gray):.1f}")
print()

# Check frames during W dissolve
target_frames = [96, 100, 110, 120, 130, 140, 150]

artifacts_found = False
for frame_idx in target_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if ret:
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate difference from background
        diff = cv2.absdiff(gray, bg_gray)
        
        # Look for pixels that are significantly different from background
        # (more than 10 intensity units different)
        artifact_mask = diff > 10
        artifact_pixels = np.sum(artifact_mask)
        
        if artifact_pixels > 0:
            # Check if artifacts form horizontal lines
            # Sum along width to get vertical profile of artifacts
            vertical_profile = np.sum(artifact_mask, axis=1)
            
            # Find rows with significant artifacts (>10 pixels different)
            artifact_rows = np.where(vertical_profile > 10)[0]
            
            if len(artifact_rows) > 0:
                print(f"Frame {frame_idx}: Found {artifact_pixels} artifact pixels")
                print(f"  Artifact rows: {artifact_rows + roi_y1}")
                
                # Save difference image for inspection
                diff_enhanced = cv2.convertScaleAbs(diff, alpha=5.0, beta=0)
                cv2.imwrite(f"artifact_diff_{frame_idx:03d}.png", diff_enhanced)
                
                # Also save the actual frame roi with artifacts highlighted
                artifact_highlight = roi.copy()
                artifact_highlight[artifact_mask] = [0, 0, 255]  # Red for artifacts
                cv2.imwrite(f"artifact_highlight_{frame_idx:03d}.png", artifact_highlight)
                
                artifacts_found = True
            else:
                print(f"Frame {frame_idx}: No significant horizontal artifacts")
        else:
            print(f"Frame {frame_idx}: No artifacts detected")

cap.release()

if artifacts_found:
    print("\n⚠️  Artifacts detected! Check artifact_diff_*.png and artifact_highlight_*.png files")
else:
    print("\n✅ No significant artifacts detected!")