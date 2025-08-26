#!/usr/bin/env python3
"""Extract frames to visually prove the stale mask bug."""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Extract frames from the proof video
cap = cv2.VideoCapture('outputs/final_stale_mask_proof_h264.mp4')

frames_to_extract = [10, 12, 14, 16, 18, 20]  # Earlier frames during dissolve
extracted = {}

for frame_num in frames_to_extract:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if ret:
        extracted[frame_num] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

cap.release()

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Stale Mask Bug: Does H Occlusion Update as Person Moves?", fontsize=16, fontweight='bold')

for idx, frame_num in enumerate(frames_to_extract):
    row = idx // 3
    col = idx % 3
    
    frame = extracted[frame_num]
    axes[row, col].imshow(frame)
    
    # Calculate where person should be
    t = frame_num / 60  # 60 total frames
    person_x = int(200 + 600 * t)
    person_right = person_x + 200
    
    # Mark person boundaries
    axes[row, col].axvline(person_x, color='blue', linestyle='--', alpha=0.7, linewidth=2)
    axes[row, col].axvline(person_right, color='blue', linestyle='--', alpha=0.7, linewidth=2)
    
    # Find yellow H pixels (RGB format from matplotlib)
    yellow = (
        (frame[:, :, 2] < 100) &  # Blue low (index 2 in RGB)
        (frame[:, :, 1] > 180) &  # Green high (index 1)
        (frame[:, :, 0] > 180)    # Red high (index 0)
    )
    
    if np.any(yellow):
        y_coords, x_coords = np.where(yellow)
        h_left = x_coords.min()
        h_right = x_coords.max()
        
        # Mark H boundaries
        axes[row, col].axvline(h_left, color='yellow', linestyle='-', alpha=0.7, linewidth=2)
        axes[row, col].axvline(h_right, color='red', linestyle='-', alpha=0.7, linewidth=2)
        
        axes[row, col].set_title(f"Frame {frame_num}: Person [{person_x}-{person_right}], H [{h_left}-{h_right}]", fontsize=10)
    else:
        axes[row, col].set_title(f"Frame {frame_num}: Person [{person_x}-{person_right}], H not visible", fontsize=10)
    
    axes[row, col].axis('off')

# Add legend
fig.text(0.5, 0.02, "Blue dashes = Person bounds | Yellow = H left edge | Red = H right edge", 
         ha='center', fontsize=12, color='black', fontweight='bold')
fig.text(0.5, 0.005, "If occlusion works correctly, H edges should move as person crosses", 
         ha='center', fontsize=10, color='gray')

plt.tight_layout()
plt.savefig('outputs/stale_mask_frames_proof.png', dpi=150, bbox_inches='tight')
print("✅ Saved frame analysis to outputs/stale_mask_frames_proof.png")

# Analyze the H cutoff positions
h_positions = []
for frame_num in frames_to_extract:
    frame = extracted[frame_num]
    yellow = (
        (frame[:, :, 2] < 100) &  # Blue low (RGB)
        (frame[:, :, 1] > 180) &  # Green high
        (frame[:, :, 0] > 180)    # Red high
    )
    if np.any(yellow):
        x_coords = np.where(yellow)[1]
        h_positions.append((frame_num, x_coords.min(), x_coords.max()))

print("\n" + "="*60)
print("H OCCLUSION ANALYSIS:")
print("="*60)

if len(h_positions) > 0:
    print("\nFrame | H Left | H Right | Width")
    print("-" * 40)
    for frame, left, right in h_positions:
        width = right - left
        print(f"  {frame:3d} |  {left:4d}  |  {right:4d}  | {width:4d}")
    
    # Check variation
    lefts = [p[1] for p in h_positions]
    rights = [p[2] for p in h_positions]
    
    left_var = max(lefts) - min(lefts)
    right_var = max(rights) - min(rights)
    
    print(f"\nLeft edge variation: {left_var} pixels")
    print(f"Right edge variation: {right_var} pixels")
    
    if left_var < 20 and right_var < 20:
        print("\n❌ STALE MASK BUG CONFIRMED!")
        print("   H boundaries barely change as person moves across")
        print("   This proves occlusion is stuck at initial position")
    elif left_var > 100 or right_var > 100:
        print("\n✅ Occlusion is updating correctly")
        print("   H boundaries change significantly as person moves")
    else:
        print("\n⚠️ Partial update detected")
        print("   Some movement but less than expected")