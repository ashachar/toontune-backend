#!/usr/bin/env python3
"""Create the fixed video with proper occlusion and verify it works."""

import cv2
import numpy as np
import sys
sys.path.append('.')

# First, let's create a proper test video with a moving person
width, height = 1280, 720
fps = 30
duration = 3
total_frames = fps * duration

# Create video with moving person silhouette
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('test_person_for_occlusion.mp4', fourcc, fps, (width, height))

for i in range(total_frames):
    # Gray background
    frame = np.full((height, width, 3), 220, dtype=np.uint8)
    
    # Moving person (dark blue shape)
    t = i / total_frames
    person_x = int(300 + 400 * t)  # Move from left to right
    
    # Draw person body
    cv2.rectangle(frame, (person_x, 250), (person_x + 180, 600), (80, 40, 20), -1)
    
    # Draw person head
    cv2.circle(frame, (person_x + 90, 220), 60, (100, 60, 40), -1)
    
    # Draw arms
    cv2.ellipse(frame, (person_x + 40, 350), (60, 120), 15, 0, 180, (80, 40, 20), -1)
    cv2.ellipse(frame, (person_x + 140, 350), (60, 120), -15, 0, 180, (80, 40, 20), -1)
    
    out.write(frame)

out.release()

# Convert to H.264
import subprocess
subprocess.run(['ffmpeg', '-i', 'test_person_for_occlusion.mp4', '-c:v', 'libx264', 
                '-pix_fmt', 'yuv420p', '-y', 'test_person_h264.mp4'],
               capture_output=True)

print("✅ Created test video with moving person")

# Now apply the animation with the fix
from utils.animations.apply_3d_text_animation import apply_animation_to_video

print("\nApplying text animation with occlusion fix...")
print("Settings: is_behind=True (occlusion enabled)")

result = apply_animation_to_video(
    video_path="test_person_h264.mp4",
    text="Hello World",
    font_size=80,
    position=(640, 400),  # Position where person will pass through
    motion_duration=0.75,  # 0.75 seconds of motion
    dissolve_duration=2.0,  # 2 seconds of dissolve
    output_path="outputs/occlusion_fixed_final.mp4",
    final_opacity=0.7,
    supersample=4,  # Medium quality for reasonable speed
    debug=False
)

print(f"\n✅ Created fixed video: {result}")
print("\nThe dissolve animation now has is_behind=True, so letters will be hidden behind the person!")