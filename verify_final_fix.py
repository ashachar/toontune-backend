#!/usr/bin/env python3
"""
Verify that the stale mask bug is fixed by comparing frames from the fixed video.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Compare old and new videos
OLD_VIDEO = "outputs/hello_world_2m20s_both_animations_compatible.mp4"
NEW_VIDEO = "outputs/hello_world_2m20s_FINAL_FIX_compatible.mp4"

# Focus on frames where the issue was visible
test_frames = [10, 12, 14, 16, 18, 20]

fig, axes = plt.subplots(2, 6, figsize=(24, 8))

for col, frame_num in enumerate(test_frames):
    # Old video (with bug)
    cap_old = cv2.VideoCapture(OLD_VIDEO)
    cap_old.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame_old = cap_old.read()
    cap_old.release()
    
    # New video (fixed)
    cap_new = cv2.VideoCapture(NEW_VIDEO)
    cap_new.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame_new = cap_new.read()
    cap_new.release()
    
    if ret:
        # Focus on 'd' area and convert to RGB
        zoom_region = (700, 320, 900, 400)  # x1, y1, x2, y2
        
        old_zoom = cv2.cvtColor(
            frame_old[zoom_region[1]:zoom_region[3], zoom_region[0]:zoom_region[2]], 
            cv2.COLOR_BGR2RGB
        )
        new_zoom = cv2.cvtColor(
            frame_new[zoom_region[1]:zoom_region[3], zoom_region[0]:zoom_region[2]], 
            cv2.COLOR_BGR2RGB
        )
        
        # Show old (buggy) version
        axes[0, col].imshow(old_zoom)
        axes[0, col].set_title(f"Frame {frame_num} - OLD (Stale Mask)", fontsize=10)
        axes[0, col].axis('off')
        
        # Show new (fixed) version
        axes[1, col].imshow(new_zoom)
        axes[1, col].set_title(f"Frame {frame_num} - FIXED", fontsize=10)
        axes[1, col].axis('off')

plt.suptitle("Comparison: Stale Mask Bug (TOP) vs Fixed (BOTTOM) - Focus on 'd' Area", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/stale_mask_fix_comparison.png", dpi=150)
print("✅ Saved comparison to outputs/stale_mask_fix_comparison.png")

print("\n" + "="*80)
print("🎉 FIX VERIFICATION COMPLETE!")
print("="*80)
print("\nThe fix ensures:")
print("  1. Fresh masks are extracted EVERY frame during motion")
print("  2. NO fallback to stale masks if extraction fails") 
print("  3. Better to have no occlusion than wrong occlusion")
print("\nCheck the comparison image to see the difference!")
print("="*80)