#!/usr/bin/env python3
"""Extract frames specifically around W dissolve to check right-side artifact."""

import cv2
import numpy as np
import os

video_path = "hello_world_fixed.mp4"
cap = cv2.VideoCapture(video_path)

# W starts dissolving around frame 96
# Extract frames 90-156 to see the full W dissolve
start_frame = 90
end_frame = 160
step = 5

os.makedirs("w_dissolve_frames", exist_ok=True)

for frame_idx in range(start_frame, end_frame, step):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    
    if ret:
        # Save full frame
        cv2.imwrite(f"w_dissolve_frames/frame_{frame_idx:03d}.png", frame)
        
        # Also save a zoomed version focused on W
        # W is approximately at x=570-750, y=150-340
        x1, y1, x2, y2 = 520, 140, 780, 360
        w_region = frame[y1:y2, x1:x2]
        
        # Scale up 2x for better visibility
        w_zoomed = cv2.resize(w_region, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(f"w_dissolve_frames/w_zoom_{frame_idx:03d}.png", w_zoomed)
        
        # Create edge detection to highlight artifacts
        gray = cv2.cvtColor(w_region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        
        # Dilate edges for visibility
        kernel = np.ones((2,2), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Color the edges
        edges_colored = cv2.cvtColor(edges_dilated, cv2.COLOR_GRAY2BGR)
        edges_colored[edges_dilated > 0] = [0, 0, 255]  # Red edges
        
        # Overlay on original
        overlay = w_region.copy()
        mask = edges_dilated > 0
        overlay[mask] = edges_colored[mask]
        
        cv2.imwrite(f"w_dissolve_frames/w_edges_{frame_idx:03d}.png", overlay)

cap.release()

print(f"Extracted {(end_frame - start_frame) // step} frames to w_dissolve_frames/")
print("Check w_zoom_* for magnified W region")
print("Check w_edges_* for edge-highlighted version")