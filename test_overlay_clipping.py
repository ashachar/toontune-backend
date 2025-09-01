#!/usr/bin/env python3
"""
Test to verify FFmpeg overlay clipping behavior
"""

import subprocess
import os

# Create a small base video (360p)
base_cmd = [
    'ffmpeg', '-y',
    '-f', 'lavfi', '-i', 'color=blue:s=640x360:d=2',
    '-c:v', 'libx264', '-preset', 'fast',
    'test_base.mp4'
]
print("Creating base video (640x360)...")
subprocess.run(base_cmd)

# Create a tall red rectangle (2000px tall)
tall_cmd = [
    'ffmpeg', '-y',
    '-f', 'lavfi', '-i', 'color=red:s=100x2000',
    '-frames:v', '1',
    'tall_rect.png'
]
print("Creating tall rectangle (100x2000)...")
subprocess.run(tall_cmd)

# Test 1: Overlay at y=50 (should extend to y=2050, way past frame bottom of 360)
test1_cmd = [
    'ffmpeg', '-y',
    '-i', 'test_base.mp4',
    '-i', 'tall_rect.png',
    '-filter_complex', '[1:v][0:v]overlay=x=100:y=50',
    '-c:v', 'libx264', '-preset', 'fast',
    'test_overlay_y50.mp4'
]
print("\nTest 1: Overlaying at y=50...")
subprocess.run(test1_cmd)

# Test 2: Overlay at y=-1500 (bottom should be at y=500, still past frame)
test2_cmd = [
    'ffmpeg', '-y',
    '-i', 'test_base.mp4',
    '-i', 'tall_rect.png',
    '-filter_complex', '[1:v][0:v]overlay=x=100:y=-1500',
    '-c:v', 'libx264', '-preset', 'fast',
    'test_overlay_y-1500.mp4'
]
print("\nTest 2: Overlaying at y=-1500...")
subprocess.run(test2_cmd)

# Extract frames to check
for video in ['test_overlay_y50.mp4', 'test_overlay_y-1500.mp4']:
    frame_cmd = [
        'ffmpeg', '-y',
        '-i', video,
        '-vframes', '1',
        f'{video.replace(".mp4", "_frame.png")}'
    ]
    subprocess.run(frame_cmd)

print("\nTest complete! Check the output frames:")
print("- test_overlay_y50_frame.png - Red should extend to bottom if not clipped")
print("- test_overlay_y-1500_frame.png - Red should be visible at bottom if not clipped")
print("\nIf red stops at y=360, overlay is being clipped to frame bounds.")

# Cleanup
for f in ['test_base.mp4', 'tall_rect.png']:
    if os.path.exists(f):
        os.remove(f)