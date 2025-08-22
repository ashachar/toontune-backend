#!/usr/bin/env python3
"""Simple test focusing only on WordDissolve to verify the fix."""

import os
os.environ['FRAME_DISSOLVE_DEBUG'] = '1'

import cv2
import numpy as np
from utils.animations.word_dissolve import WordDissolve

# Create a simple test video
width, height = 1280, 720
fps = 30
background_color = (50, 80, 50)  # Green background

# Create dissolve animator
dissolve = WordDissolve(
    element_path="test_element_3sec.mp4",  # Not used but required
    background_path="test_element_3sec.mp4",  # Not used but required
    position=(width // 2, height // 2),
    word="HELLO",
    font_size=100,
    text_color=(255, 220, 0),  # Yellow
    stable_duration=0.2,    # 6 frames at 30fps
    dissolve_duration=1.0,  # 30 frames at 30fps
    dissolve_stagger=0.5,   # 15 frames between letters
    fps=fps,
    debug=True
)

# Calculate total frames needed
stable_frames = int(0.2 * fps)  # 6
dissolve_frames = int(1.0 * fps)  # 30
stagger_frames = int(0.5 * fps)  # 15
num_letters = 5  # HELLO
total_dissolve_frames = stable_frames + (num_letters - 1) * stagger_frames + dissolve_frames
total_frames = total_dissolve_frames + 60  # Add 2 seconds after dissolve completes

print(f"Stable phase: frames 0-{stable_frames-1}")
print(f"Last letter starts dissolving: frame {stable_frames + (num_letters-1) * stagger_frames}")
print(f"All letters completed by: frame {total_dissolve_frames}")
print(f"Total frames: {total_frames}")
print()

# Create output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('dissolve_only_test.mp4', fourcc, fps, (width, height))

# Generate frames
for frame_idx in range(total_frames):
    # Create background frame
    frame = np.ones((height, width, 3), dtype=np.uint8) * background_color
    
    # Apply dissolve
    result = dissolve.render_word_frame(frame, frame_idx, mask=None)
    
    # Write frame
    out.write(result)
    
    # Log important frames
    if frame_idx in [0, stable_frames, total_dissolve_frames - 1, total_dissolve_frames, total_dissolve_frames + 10]:
        # Check for yellow pixels
        yellow_mask = (result[:,:,1] > 180) & (result[:,:,2] > 180) & (result[:,:,0] < 100)
        yellow_count = np.sum(yellow_mask)
        print(f"Frame {frame_idx:3d}: Yellow pixels: {yellow_count:5d}")

out.release()
print(f"\nCreated dissolve_only_test.mp4")

# Now verify the fix worked
cap = cv2.VideoCapture('dissolve_only_test.mp4')

# Check frame after all letters should be dissolved
cap.set(cv2.CAP_PROP_POS_FRAMES, total_dissolve_frames + 5)
ret, frame = cap.read()

if ret:
    # Check for any yellow pixels
    yellow_mask = (frame[:,:,1] > 180) & (frame[:,:,2] > 180) & (frame[:,:,0] < 100)
    yellow_count = np.sum(yellow_mask)
    
    if yellow_count == 0:
        print("\n✅ SUCCESS: No text visible after dissolve completion!")
    else:
        print(f"\n⚠️  PROBLEM: {yellow_count} yellow pixels still visible after dissolve!")
        cv2.imwrite("dissolve_problem_frame.png", frame)

cap.release()