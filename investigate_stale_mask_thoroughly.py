#!/usr/bin/env python3
"""Thorough investigation of the stale mask bug."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# First, let's create a simple test case
width, height = 640, 360
frames = []

print("Creating test video...")
print("Person (blue rectangle) moves from x=200 to x=400")
print("H letter will be placed at x=300")
print("="*60)

# Create 30 frames
for i in range(30):
    frame = np.full((height, width, 3), 220, dtype=np.uint8)
    
    # Person moves from x=200 to x=400
    person_x = 200 + (i * 200 // 30)
    cv2.rectangle(frame, (person_x, 100), (person_x + 100, 250), (50, 50, 100), -1)
    
    # Add frame number
    cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    frames.append(frame)

# Save as H.264 video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('test_mask_movement.mp4', fourcc, 10, (width, height))
for frame in frames:
    out.write(frame)
out.release()

# Convert to H.264 with proper encoding
import subprocess
subprocess.run([
    'ffmpeg', '-i', 'test_mask_movement.mp4', 
    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
    '-y', 'test_mask_movement_h264.mp4'
], capture_output=True, check=True)

print("✅ Created test video with proper H.264 encoding")

# Now test the dissolve animation directly
print("\nTesting dissolve animation...")

from utils.animations.letter_3d_dissolve import Letter3DDissolve

# Create dissolve with debug enabled
dissolve = Letter3DDissolve(
    duration=3.0,  # 30 frames at 10 fps
    fps=10,
    resolution=(width, height),
    text="H",
    font_size=60,
    text_color=(255, 220, 0),
    initial_scale=1.0,
    initial_position=(280, 150),  # Place H at x=280 (will be occluded as person passes)
    is_behind=True,
    debug=True
)

# Track where the H is cut off each frame
cutoff_positions = []

print("\nProcessing frames...")
for i in range(30):
    # Get the output frame
    output = dissolve.generate_frame(i, frames[i])
    
    # Find yellow pixels (the H letter)
    yellow_mask = (
        (output[:, :, 0] > 180) &  # R
        (output[:, :, 1] > 180) &  # G
        (output[:, :, 2] < 100)    # B
    )
    
    if np.any(yellow_mask):
        # Find rightmost visible yellow pixel
        y_coords, x_coords = np.where(yellow_mask)
        rightmost_x = x_coords.max()
        leftmost_x = x_coords.min()
        cutoff_positions.append(rightmost_x)
        
        # Calculate where person is
        person_x = 200 + (i * 200 // 30)
        person_right = person_x + 100
        
        if i % 5 == 0:
            print(f"\nFrame {i}:")
            print(f"  Person: x=[{person_x}-{person_right}]")
            print(f"  H visible: x=[{leftmost_x}-{rightmost_x}]")
            
            # Check if cutoff matches person position
            if abs(rightmost_x - person_x) < 10:
                print(f"  ✅ H is correctly cut at person's left edge")
            elif rightmost_x < person_x - 10:
                print(f"  ⚠️ H cut too early (before person)")
            else:
                print(f"  ⚠️ H extends past person by {rightmost_x - person_x} pixels")

# Analyze the results
print("\n" + "="*60)
print("STALE MASK BUG ANALYSIS:")
print("="*60)

if len(cutoff_positions) > 10:
    # Check variation in cutoff position
    min_cutoff = min(cutoff_positions)
    max_cutoff = max(cutoff_positions)
    variation = max_cutoff - min_cutoff
    
    print(f"H cutoff range: x=[{min_cutoff}-{max_cutoff}]")
    print(f"Variation: {variation} pixels")
    
    # Person moves 200 pixels total
    expected_variation = 200
    
    if variation < 50:
        print("\n❌ BUG CONFIRMED: H cutoff barely moves!")
        print(f"   Expected ~{expected_variation} pixels movement")
        print(f"   Got only {variation} pixels")
        print("\n   This proves the occlusion boundary is FROZEN at the initial position.")
    else:
        print(f"\n✅ H cutoff varies by {variation} pixels (expected ~{expected_variation})")

# Create visualization
print("\nCreating visualization...")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
frames_to_show = [0, 5, 10, 15, 20, 25]

for idx, frame_num in enumerate(frames_to_show):
    row = idx // 3
    col = idx % 3
    
    # Generate the frame
    output = dissolve.generate_frame(frame_num, frames[frame_num])
    
    axes[row, col].imshow(output)
    axes[row, col].set_title(f"Frame {frame_num}")
    axes[row, col].axis('off')
    
    # Mark where person should be
    person_x = 200 + (frame_num * 200 // 30)
    axes[row, col].axvline(person_x, color='red', linestyle='--', alpha=0.5, label='Person left edge')
    axes[row, col].axvline(person_x + 100, color='blue', linestyle='--', alpha=0.5, label='Person right edge')

plt.suptitle("Stale Mask Bug Visualization", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/stale_mask_investigation.png', dpi=150)
print("✅ Saved visualization to outputs/stale_mask_investigation.png")

# Now let's trace the exact bug in the code
print("\n" + "="*60)
print("INVESTIGATING THE CODE:")
print("="*60)
print("\nThe bug is likely in letter_3d_dissolve.py where the mask is applied.")
print("The mask is extracted fresh (we see this in debug), but the occlusion")
print("boundary stays fixed. This suggests the mask region calculation is wrong.")