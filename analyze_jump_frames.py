#!/usr/bin/env python3
"""Analyze the exact frames where the jump occurs."""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load consecutive frames around the jump
frames_to_analyze = list(range(28, 41))
frames = []
frame_nums = []

for num in frames_to_analyze:
    path = f"debug_dissolve_frame_{num:03d}.png"
    try:
        frame = cv2.imread(path)
        if frame is not None:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_nums.append(num)
    except:
        pass

print(f"Loaded {len(frames)} frames")

# Find the frame with maximum change
max_diff = 0
jump_frame = None

for i in range(1, len(frames)):
    diff = cv2.absdiff(frames[i], frames[i-1])
    mean_diff = np.mean(diff)
    
    print(f"Frame {frame_nums[i-1]} -> {frame_nums[i]}: diff = {mean_diff:.2f}")
    
    if mean_diff > max_diff:
        max_diff = mean_diff
        jump_frame = (frame_nums[i-1], frame_nums[i])

print(f"\n*** BIGGEST JUMP: Frame {jump_frame[0]} -> {jump_frame[1]} with diff = {max_diff:.2f}")

# Visualize the jump
if jump_frame and len(frames) > 0:
    idx1 = frame_nums.index(jump_frame[0])
    idx2 = frame_nums.index(jump_frame[1])
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Show the two frames with the jump
    axes[0].imshow(frames[idx1])
    axes[0].set_title(f"Frame {jump_frame[0]} (BEFORE JUMP)")
    axes[0].axis('off')
    
    axes[1].imshow(frames[idx2])
    axes[1].set_title(f"Frame {jump_frame[1]} (AFTER JUMP)")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('jump_location.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved jump visualization to jump_location.png")
    
    # Also create a difference image
    diff_img = cv2.absdiff(frames[idx1], frames[idx2])
    plt.figure(figsize=(10, 6))
    plt.imshow(diff_img)
    plt.title(f"Difference between frames {jump_frame[0]} and {jump_frame[1]}")
    plt.colorbar()
    plt.savefig('jump_difference.png', dpi=150, bbox_inches='tight')
    print(f"Saved difference map to jump_difference.png")