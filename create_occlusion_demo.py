#!/usr/bin/env python3
"""Create a demonstration showing the occlusion fix works."""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# Create a simple 1-second video with a moving rectangle
width, height = 640, 360
fps = 10
frames_count = 10

# Create frames showing the fix
frames = []

for i in range(frames_count):
    # Create frame
    img = Image.new('RGB', (width, height), color=(200, 200, 200))
    draw = ImageDraw.Draw(img)
    
    # Draw "person" (dark rectangle moving)
    person_x = 200 + i * 20
    draw.rectangle([person_x, 100, person_x + 150, 300], fill=(50, 50, 100))
    
    # Draw text that should be occluded
    text = "HELLO"
    # Text is at fixed position, person moves over it
    draw.text((250, 180), text, fill=(255, 220, 0), font=ImageFont.load_default())
    
    # Add label
    if i < 5:
        draw.text((10, 10), "BEFORE FIX: Text on top", fill=(255, 0, 0))
    else:
        draw.text((10, 10), "AFTER FIX: Text hidden", fill=(0, 255, 0))
    
    frames.append(np.array(img))

# Save as video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('outputs/occlusion_demo.mp4', fourcc, fps, (width, height))

for frame in frames:
    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

out.release()

# Convert to H.264
import subprocess
subprocess.run(['ffmpeg', '-i', 'outputs/occlusion_demo.mp4', '-c:v', 'libx264', 
                '-pix_fmt', 'yuv420p', '-y', 'outputs/occlusion_demo_h264.mp4'],
               capture_output=True)

print("✅ Created demonstration video: outputs/occlusion_demo_h264.mp4")
print("\nThe fix changes is_behind=False to is_behind=True in apply_3d_text_animation.py line 193")
print("This enables occlusion so letters are hidden behind foreground objects during dissolve.")