#!/usr/bin/env python3
"""Track letter positions vs mask positions during dissolve."""

import cv2
import numpy as np
import sys
sys.path.append('.')
from utils.animations.apply_3d_text_animation import apply_animation_to_video

# First create a better test video with clear movement
width, height = 1280, 720
fps = 30
duration = 3
total_frames = fps * duration

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('test_moving_person.mp4', fourcc, fps, (width, height))

for i in range(total_frames):
    frame = np.full((height, width, 3), 220, dtype=np.uint8)
    
    # Person moves continuously from left to right
    t = i / total_frames
    person_x = int(200 + 600 * t)  # Move from 200 to 800
    
    # Draw person
    cv2.rectangle(frame, (person_x, 200), (person_x + 200, 550), (80, 40, 20), -1)
    cv2.circle(frame, (person_x + 100, 170), 70, (100, 60, 40), -1)
    
    # Add frame number for reference
    cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    out.write(frame)

out.release()

# Convert to H.264
import subprocess
subprocess.run(['ffmpeg', '-i', 'test_moving_person.mp4', '-c:v', 'libx264', 
                '-pix_fmt', 'yuv420p', '-y', 'test_moving_person_h264.mp4'],
               capture_output=True, check=True)

print("✅ Created test video with continuously moving person")

# Now run with detailed debug
import os
os.environ['DEBUG_3D_TEXT'] = '1'

print("\nRunning animation with debug...")
print("Motion: 0.5s (15 frames)")
print("Dissolve: 2.5s (75 frames)")
print("Person moves from x=200 to x=800 over 90 frames")
print("="*60)

result = apply_animation_to_video(
    video_path="test_moving_person_h264.mp4",
    text="H",  # Just one letter to track clearly
    font_size=100,
    position=(400, 350),  # Position where person will pass through
    motion_duration=0.5,
    dissolve_duration=2.5,
    output_path="outputs/letter_position_test.mp4",
    final_opacity=0.7,
    supersample=2,
    debug=True
)

print(f"\n✅ Created: {result}")
print("\nCheck debug output for letter position vs mask position.")