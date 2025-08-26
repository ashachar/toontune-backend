#!/usr/bin/env python3
"""
Extract critical frames to identify where 'd' shows stale occlusion.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

VIDEO = "outputs/hello_world_2m20s_FIXED_compatible.mp4"

cap = cv2.VideoCapture(VIDEO)
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Extract frames 45-65 where 'd' is likely affected
frames_to_check = [45, 50, 52, 54, 56, 58, 60, 62, 65]

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()

for idx, frame_num in enumerate(frames_to_check):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Focus on the 'd' region (right side, around x=850-900)
        # Draw a box around where 'd' should be
        d_x1, d_x2 = 820, 920
        d_y1, d_y2 = 320, 400
        cv2.rectangle(frame_rgb, (d_x1, d_y1), (d_x2, d_y2), (255, 0, 0), 2)
        
        # Add frame info
        time_s = frame_num / fps
        cv2.putText(frame_rgb, f"Frame {frame_num} (t={time_s:.2f}s)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        axes[idx].imshow(frame_rgb)
        axes[idx].set_title(f"Frame {frame_num}")
        axes[idx].axis('off')
        
        # Also save individual frame
        cv2.imwrite(f"outputs/critical_frame_{frame_num}.png", frame)

cap.release()

plt.suptitle("Critical Frames: Check 'd' occlusion (blue box)", fontsize=16)
plt.tight_layout()
plt.savefig("outputs/critical_frames_analysis.png", dpi=150)
print(f"✅ Saved analysis to outputs/critical_frames_analysis.png")
print(f"✅ Individual frames saved as outputs/critical_frame_*.png")

# Now let's check the actual animation code to see if is_behind is set
print("\n🔍 Checking if is_behind=True is properly set...")

import subprocess
result = subprocess.run(
    ["grep", "-n", "is_behind", "utils/animations/apply_3d_text_animation.py"],
    capture_output=True, text=True
)
print("Lines with is_behind in apply_3d_text_animation.py:")
for line in result.stdout.strip().split('\n')[:10]:
    if line:
        print(f"  {line}")

print("\n📌 Key observation points:")
print("  1. Check if 'd' is visible in frames 50-60")
print("  2. Look for horizontal cut line on 'd' (stale occlusion)")
print("  3. Compare foreground position between frames")
print("  4. The cut should move with the foreground, not stay fixed")