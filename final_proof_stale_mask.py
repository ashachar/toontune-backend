#!/usr/bin/env python3
"""Final proof of stale mask bug with proper video."""

import cv2
import numpy as np
import os
os.environ['DEBUG_3D_TEXT'] = '1'

from utils.animations.apply_3d_text_animation import apply_animation_to_video

# Create a test video where person moves across the H letter
width, height = 1280, 720
fps = 30
duration = 2
total_frames = fps * duration

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('test_crossing.mp4', fourcc, fps, (width, height))

for i in range(total_frames):
    frame = np.full((height, width, 3), 220, dtype=np.uint8)
    
    # Person moves from left to right, crossing the H
    t = i / total_frames
    person_x = int(200 + 600 * t)  # Move from 200 to 800
    
    # Draw person
    cv2.rectangle(frame, (person_x, 250), (person_x + 200, 550), (80, 40, 20), -1)
    cv2.circle(frame, (person_x + 100, 220), 70, (100, 60, 40), -1)
    
    # Add frame counter
    cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Add reference line at x=500 where H will be
    cv2.line(frame, (500, 0), (500, height), (0, 0, 255), 1)
    
    out.write(frame)

out.release()

# Convert to H.264 with proper encoding
import subprocess
subprocess.run([
    'ffmpeg', '-i', 'test_crossing.mp4',
    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
    '-y', 'test_crossing_h264.mp4'
], capture_output=True, check=True)

print("✅ Created test video")
print("Person moves from x=200 to x=800 over 60 frames (2 seconds)")
print("H will be at x=500")

# Apply animation with H at position where person crosses
result = apply_animation_to_video(
    video_path="test_crossing_h264.mp4",
    text="H",
    font_size=120,
    position=(500, 380),
    motion_duration=0.3,  # 9 frames
    dissolve_duration=1.7,  # 51 frames
    output_path="outputs/final_stale_mask_proof.mp4",
    final_opacity=0.8,
    supersample=2,
    debug=False
)

print(f"\n✅ Created: {result}")

# Convert to proper H.264 for viewing
subprocess.run([
    'ffmpeg', '-i', 'outputs/final_stale_mask_proof_hq.mp4',
    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
    '-y', 'outputs/final_stale_mask_proof_h264.mp4'
], capture_output=True, check=True)

print("✅ Converted to H.264: outputs/final_stale_mask_proof_h264.mp4")

# Analyze key frames
cap = cv2.VideoCapture('outputs/final_stale_mask_proof_h264.mp4')

frames_to_check = [10, 20, 30, 40, 50]  # During dissolve
h_visibility = []

for frame_num in frames_to_check:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    
    if not ret:
        continue
    
    # Find yellow pixels
    yellow = (
        (frame[:, :, 0] < 100) &  # Blue low
        (frame[:, :, 1] > 180) &  # Green high
        (frame[:, :, 2] > 180)    # Red high
    )
    
    if np.any(yellow):
        y_coords, x_coords = np.where(yellow)
        h_left = x_coords.min()
        h_right = x_coords.max()
        h_visibility.append((frame_num, h_left, h_right))
        
        # Calculate where person should be
        t = frame_num / 60
        person_x = int(200 + 600 * t)
        person_right = person_x + 200
        
        print(f"\nFrame {frame_num}:")
        print(f"  Person: x=[{person_x}-{person_right}]")
        print(f"  H visible: x=[{h_left}-{h_right}]")
        
        # Check occlusion
        if person_x < 500 < person_right:
            # Person overlaps H position
            if h_left >= person_right - 10:
                print(f"  ✅ H correctly occluded on left (starts at {h_left}, person ends at {person_right})")
            elif h_right <= person_x + 10:
                print(f"  ✅ H correctly occluded on right (ends at {h_right}, person starts at {person_x})")
            else:
                print(f"  ⚠️ H should be partially occluded but shows full width")

cap.release()

print("\n" + "="*60)
print("STALE MASK BUG VERIFICATION:")
print("="*60)

if len(h_visibility) >= 3:
    # Check if H cutoff position changes
    cutoffs = [v[2] for v in h_visibility]  # Right edges
    variation = max(cutoffs) - min(cutoffs)
    
    print(f"H right edge positions: {cutoffs}")
    print(f"Variation: {variation} pixels")
    
    if variation < 50:
        print("\n❌ STALE MASK BUG CONFIRMED!")
        print("   H occlusion boundary barely moves as person crosses")
    else:
        print(f"\n✅ H occlusion updates correctly ({variation} pixels variation)")

print("\n✅ Open outputs/final_stale_mask_proof_h264.mp4 to see the result")