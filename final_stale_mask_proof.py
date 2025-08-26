#!/usr/bin/env python3
"""Final proof of the stale mask bug with visual output."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['DEBUG_3D_TEXT'] = '1'

from utils.animations.apply_3d_text_animation import apply_animation_to_video

# Create video with person moving right across the frame
width, height = 1280, 720
fps = 30
duration = 2
total_frames = fps * duration

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('test_stale_mask.mp4', fourcc, fps, (width, height))

for i in range(total_frames):
    frame = np.full((height, width, 3), 220, dtype=np.uint8)
    
    # Person moves from x=300 to x=700
    t = i / total_frames
    person_x = int(300 + 400 * t)
    
    # Draw person
    cv2.rectangle(frame, (person_x, 250), (person_x + 200, 500), (80, 40, 20), -1)
    cv2.circle(frame, (person_x + 100, 220), 60, (100, 60, 40), -1)
    
    # Add reference line at x=500 (where H will be)
    cv2.line(frame, (500, 0), (500, height), (255, 0, 0), 1)
    
    out.write(frame)

out.release()

# Convert to H.264
import subprocess
subprocess.run(['ffmpeg', '-i', 'test_stale_mask.mp4', '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p', '-y', 'test_stale_mask_h264.mp4'],
               capture_output=True, check=True)

print("✅ Created test video")
print("Person moves from x=300 to x=700")
print("H will be placed at x=500")
print("="*60)

# Apply animation
result = apply_animation_to_video(
    video_path="test_stale_mask_h264.mp4",
    text="H",
    font_size=100,
    position=(500, 380),  # Place H at fixed position
    motion_duration=0.5,  # 15 frames
    dissolve_duration=1.5,  # 45 frames
    output_path="outputs/stale_mask_proof.mp4",
    final_opacity=0.7,
    supersample=2,
    debug=False  # Turn off debug for cleaner output
)

print(f"\n✅ Created: {result}")

# Now analyze the output
cap = cv2.VideoCapture('outputs/stale_mask_proof.mp4')

# Extract frames during dissolve
frames_to_check = [15, 20, 25, 30, 35]  # During dissolve
extracted = {}

for frame_num in frames_to_check:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if ret:
        extracted[frame_num] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

cap.release()

# Analyze H visibility
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

h_right_edges = []

for idx, frame_num in enumerate(frames_to_check):
    frame = extracted[frame_num]
    
    # Find yellow pixels
    yellow = (
        (frame[:, :, 2] > 180) &  # Red high
        (frame[:, :, 1] > 180) &  # Green high
        (frame[:, :, 0] < 100)   # Blue low
    )
    
    axes[idx].imshow(frame[300:450, 400:600])  # Zoom on H region
    axes[idx].set_title(f"Frame {frame_num}")
    axes[idx].axis('off')
    
    # Find rightmost yellow pixel
    if np.any(yellow):
        y_coords, x_coords = np.where(yellow)
        rightmost = x_coords.max()
        h_right_edges.append(rightmost)
        
        # Mark the cutoff
        local_x = min(rightmost - 400, 199)  # Convert to local coords
        if local_x > 0:
            axes[idx].axvline(local_x, color='red', linestyle='--', alpha=0.7)
        
        # Calculate expected person position
        t = frame_num / 60  # 60 total frames
        expected_person_x = 300 + 400 * t
        expected_person_left = expected_person_x
        
        print(f"Frame {frame_num}:")
        print(f"  H visible up to x={rightmost}")
        print(f"  Person should be at x={expected_person_left:.0f}")
        print(f"  H should be cut at x={expected_person_left:.0f} (person's left edge)")

plt.suptitle("H Occlusion During Dissolve (Red line = cutoff)", fontsize=14)
plt.tight_layout()
plt.savefig('outputs/stale_mask_proof.png', dpi=150)

print("\n" + "="*60)
print("STALE MASK BUG ANALYSIS:")
print("="*60)

if len(h_right_edges) >= 3:
    edge_change = h_right_edges[-1] - h_right_edges[0]
    print(f"H cutoff moved {edge_change} pixels from frame {frames_to_check[0]} to {frames_to_check[-1]}")
    
    expected_movement = 400 * (frames_to_check[-1] - frames_to_check[0]) / 60
    print(f"Person should have moved {expected_movement:.0f} pixels in that time")
    
    if abs(edge_change - expected_movement) > 50:
        print("\n⚠️ BUG CONFIRMED: H cutoff is NOT tracking person movement!")
        print(f"   Discrepancy: {abs(edge_change - expected_movement):.0f} pixels")
    else:
        print("\n✅ H cutoff is tracking person movement correctly.")

print("\n✅ Saved visual proof to outputs/stale_mask_proof.png")