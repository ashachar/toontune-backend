#!/usr/bin/env python3
"""
Check when text is actually visible in the animation.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

VIDEO = "outputs/hello_world_2m20s_FIXED_compatible.mp4"

cap = cv2.VideoCapture(VIDEO)
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {total_frames} frames @ {fps} fps = {total_frames/fps:.2f}s")
print("\nExpected timeline:")
print("  0-20 frames (0.0-0.8s): Motion")
print("  20-32 frames (0.8-1.3s): Hold")
print("  32-82 frames (1.3-3.3s): Dissolve")

# Sample frames throughout the video
sample_frames = [0, 10, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]

fig, axes = plt.subplots(4, 4, figsize=(20, 20))
axes = axes.flatten()

for idx, frame_num in enumerate(sample_frames):
    if frame_num >= total_frames:
        axes[idx].axis('off')
        continue
        
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Add frame info
        time_s = frame_num / fps
        phase = "MOTION" if frame_num < 20 else "HOLD" if frame_num < 32 else "DISSOLVE"
        
        cv2.putText(frame_rgb, f"F{frame_num} ({time_s:.1f}s) - {phase}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Highlight center area where text should be
        center_x, center_y = 640, 360
        cv2.circle(frame_rgb, (center_x, center_y), 5, (255, 0, 0), -1)
        cv2.rectangle(frame_rgb, (400, 300), (880, 420), (0, 255, 0), 1)
        
        axes[idx].imshow(frame_rgb)
        axes[idx].set_title(f"Frame {frame_num}", fontsize=8)
        axes[idx].axis('off')

cap.release()

plt.suptitle("Text Visibility Throughout Animation (green box = text area)", fontsize=16)
plt.tight_layout()
plt.savefig("outputs/text_visibility_timeline.png", dpi=150)
print(f"\n✅ Saved timeline to outputs/text_visibility_timeline.png")

print("\n🔍 To find the issue:")
print("  1. Look for frames where text is partially visible")
print("  2. Check if text shows horizontal cut lines (occlusion)")
print("  3. Compare foreground position when text is occluded")
print("  4. The 'd' should be one of the last letters to dissolve")