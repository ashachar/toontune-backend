#!/usr/bin/env python3
"""
Analyze frames 10-30 where text is visible to find occlusion issues.
Focus on the 'd' in "World" to see if it has stale mask occlusion.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

VIDEO = "outputs/hello_world_2m20s_FIXED_compatible.mp4"

cap = cv2.VideoCapture(VIDEO)
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Focus on frames where text is visible
frames_to_analyze = [8, 10, 12, 14, 16, 18, 20, 22, 24]

fig, axes = plt.subplots(3, 3, figsize=(20, 20))
axes = axes.flatten()

print("Analyzing frames where text is visible...")

for idx, frame_num in enumerate(frames_to_analyze):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Zoom in on the 'd' area (right side of text)
        # The 'd' in "World" should be around x=750-850, y=340-380
        zoom_x1, zoom_x2 = 700, 900
        zoom_y1, zoom_y2 = 320, 400
        
        # Extract zoom region
        zoom_region = frame_rgb[zoom_y1:zoom_y2, zoom_x1:zoom_x2].copy()
        
        # Enhance contrast to see occlusion better
        zoom_region = cv2.convertScaleAbs(zoom_region, alpha=1.5, beta=20)
        
        # Mark where we expect 'd' to be
        d_rel_x = 120  # relative position in zoom
        d_rel_y = 40
        cv2.rectangle(zoom_region, (d_rel_x-20, d_rel_y-20), 
                     (d_rel_x+30, d_rel_y+30), (255, 0, 0), 1)
        cv2.putText(zoom_region, "d", (d_rel_x-10, d_rel_y-25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # Add frame info
        time_s = frame_num / fps
        phase = "MOTION" if frame_num < 20 else "HOLD"
        cv2.putText(zoom_region, f"F{frame_num} ({time_s:.2f}s)", 
                   (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Show zoomed region
        axes[idx].imshow(zoom_region)
        axes[idx].set_title(f"Frame {frame_num} - {phase} (Zoomed on 'd' area)", fontsize=10)
        axes[idx].axis('off')
        
        # Also save the full frame with zoom area marked
        cv2.rectangle(frame_rgb, (zoom_x1, zoom_y1), (zoom_x2, zoom_y2), (0, 255, 0), 2)
        cv2.putText(frame_rgb, "Zoom area", (zoom_x1, zoom_y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(f"outputs/frame_{frame_num}_with_zoom.png", 
                   cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

cap.release()

plt.suptitle("Zoomed View of 'd' Area - Look for Horizontal Cut Lines", fontsize=16)
plt.tight_layout()
plt.savefig("outputs/d_occlusion_analysis.png", dpi=150)
print(f"✅ Saved analysis to outputs/d_occlusion_analysis.png")

print("\n🔍 What to look for:")
print("  1. Horizontal cut lines on the 'd' (blue box)")
print("  2. Whether the cut line position changes between frames")
print("  3. If cut stays at same Y position while speaker moves = STALE MASK BUG")
print("  4. The 'd' should be visible in frames 10-20")