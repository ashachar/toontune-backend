#!/usr/bin/env python3
"""Test with a much simpler eraser wipe"""

import subprocess

# Test with existing files
character_video = "outputs/temp_eraser_source.mp4"
original_video = "uploads/assets/runway_experiment/runway_demo_input.mp4"
eraser_image = "uploads/assets/images/eraser.png"
output_video = "outputs/test_simple_eraser.mp4"

# Build a simple filter that just overlays the eraser
filter_complex = (
    "[0:v]format=rgba[char];"
    "[1:v]format=rgba[orig];"
    "[2:v]format=rgba,scale=iw*0.7:ih*0.7[eraser];"
    # Simple blend from character to original
    "[orig][char]blend=all_expr='A*(1-T/0.6)+B*(T/0.6)':enable='between(t,0,0.6)'[blend];"
    # Overlay eraser on top
    "[blend][eraser]overlay=x='W/2-overlay_w/2':y='H/2-overlay_h/2':enable='between(t,0,0.6)'[outv]"
)

cmd = [
    'ffmpeg', '-y',
    '-i', character_video,
    '-i', original_video, 
    '-stream_loop', '-1',
    '-i', eraser_image,
    '-filter_complex', filter_complex,
    '-map', '[outv]',
    '-map', '1:a?',
    '-c:a', 'copy',
    '-c:v', 'libx264',
    '-preset', 'fast',
    '-crf', '18',
    '-pix_fmt', 'yuv420p',
    '-t', '2',  # Just 2 seconds for testing
    output_video
]

print("Testing simple eraser overlay...")
print(f"Command: {' '.join(cmd[:6])}...")

result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode == 0:
    print(f"Success! Output: {output_video}")
else:
    print(f"Error: {result.stderr[:500]}")