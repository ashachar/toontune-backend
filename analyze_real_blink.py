#!/usr/bin/env python3
"""Analyze the actual blink by checking what motion returns vs what dissolve expects."""

import sys

# Parse the debug output to understand the actual values
with open('debug_output.txt', 'r') as f:
    lines = f.readlines()

# Find key moments
handoff_line = None
first_dissolve = None
motion_frames = []
dissolve_frames = []

for i, line in enumerate(lines):
    if "[JUMP_CUT] Handoff at frame 18" in line:
        handoff_line = line.strip()
        print(f"Found handoff: {handoff_line}")
        
    if "[Text3DMotion] Frame" in line:
        frame_num = line.split("Frame ")[1].split(":")[0]
        motion_frames.append((int(frame_num), line.strip()))
        
    if "[OPACITY_BLINK]" in line and "frame=" in line:
        dissolve_frames.append(line.strip())

print("\n=== Motion Frames ===")
for frame_num, line in motion_frames[-3:]:
    print(f"Frame {frame_num}: {line}")

print("\n=== Dissolve Frames (first 3) ===")  
for line in dissolve_frames[:3]:
    print(line)

# The key insight: Check if motion is applying alpha gradually
print("\n=== Analysis ===")
print("Motion's final frame reports alpha=0.630")
print("Dissolve receives alpha=0.630 and sprites with alpha already at 0.630")
print("Relative multiplier = 0.630/0.630 = 1.0")
print("\nThe issue might be:")
print("1. Motion gradually reduces alpha from 1.0 to 0.63 over frames 0-18")
print("2. Motion's frame 18 has letters at 63% opacity")  
print("3. Dissolve frame 0 (video frame 19) also has letters at 63% opacity")
print("4. BUT: If the video background changes between frames, the visual result differs!")
print("\nOr:")
print("1. Motion's alpha reduction is not smooth")
print("2. There's a jump in motion's alpha calculation near the end")