#!/usr/bin/env python3
"""Verify that occlusion is working correctly with the position fix."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Read the output video
video_path = "outputs/final_stale_mask_proof_h264.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {width}x{height}, {fps} fps, {total_frames} frames")

# Analyze key frames
key_frames = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 59]
frames_data = []

for frame_idx in key_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_data.append((frame_idx, frame_rgb))

cap.release()

# Create visualization
fig, axes = plt.subplots(3, 5, figsize=(20, 12))
axes = axes.flatten()

# H position from the logs
h_x = 436  # From dissolve phase position
h_width = 129  # Approximate width
h_y = 276
h_height = 129

for idx, (frame_idx, frame) in enumerate(frames_data[:15]):
    ax = axes[idx]
    ax.imshow(frame)
    
    # Calculate person position (moves from x=200 to x=800)
    progress = frame_idx / 60.0
    person_x = 200 + progress * 600
    
    # Add rectangle for H position
    h_rect = Rectangle((h_x, h_y), h_width, h_height, 
                       fill=False, edgecolor='yellow', linewidth=2)
    ax.add_patch(h_rect)
    
    # Add rectangle for estimated person position
    person_rect = Rectangle((person_x, 250), 204, 200,
                           fill=False, edgecolor='red', linewidth=2, linestyle='--')
    ax.add_patch(person_rect)
    
    # Determine occlusion status
    person_right = person_x + 204
    if person_right > h_x:
        overlap = min(person_right, h_x + h_width) - h_x
        occlusion_pct = (overlap / h_width) * 100
        status = f"Occluded {occlusion_pct:.0f}%"
        color = 'red'
    else:
        status = "Visible"
        color = 'green'
    
    ax.set_title(f"Frame {frame_idx}\nPerson x={person_x:.0f}\n{status}", 
                 fontsize=10, color=color)
    ax.axis('off')

# Hide unused axes
for idx in range(len(frames_data), 15):
    axes[idx].axis('off')

plt.suptitle("Occlusion Verification - H Position Fixed", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/occlusion_verification.png', dpi=150)
print("\n✅ Saved visualization to outputs/occlusion_verification.png")

# Check specific frames for H visibility
print("\n" + "="*60)
print("OCCLUSION ANALYSIS:")
print("="*60)

for frame_idx in [10, 15, 20, 25, 30]:
    progress = frame_idx / 60.0
    person_x = 200 + progress * 600
    person_right = person_x + 204
    
    print(f"\nFrame {frame_idx}:")
    print(f"  Person: x=[{person_x:.0f}-{person_right:.0f}]")
    print(f"  H letter: x=[{h_x}-{h_x+h_width}]")
    
    if person_right < h_x:
        print(f"  ✅ H fully visible (person before H)")
    elif person_x > h_x + h_width:
        print(f"  ✅ H fully visible (person after H)")
    else:
        overlap_start = max(person_x, h_x)
        overlap_end = min(person_right, h_x + h_width)
        overlap = overlap_end - overlap_start
        print(f"  ⚠️ H occluded: overlap x=[{overlap_start:.0f}-{overlap_end:.0f}] ({overlap:.0f}px)")

print("\n" + "="*60)
print("EXPECTED BEHAVIOR:")
print("="*60)
print("- Frames 0-10: H should be partially/fully occluded")
print("- Frames 11-25: H should transition from occluded to visible")
print("- Frames 26-60: H should be fully visible (person past H)")