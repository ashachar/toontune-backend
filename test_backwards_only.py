#!/usr/bin/env python3
"""Test backwards-only animation (no grow phase)"""

import cv2
import numpy as np
from utils.animations.text_3d_behind_segment_backwards_only import Text3DBehindSegment

print("="*60)
print("TESTING BACKWARDS-ONLY ANIMATION")
print("="*60)
print("\n✨ Key changes:")
print("  • NO grow phase - starts fully visible")
print("  • Only moves backwards behind subject")
print("  • Starts big, shrinks as it goes behind")
print("  • Ends with small wiggle behind subject\n")

video_path = "test_element_3sec.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Load 90 frames (1.5 seconds)
frames = []
for i in range(90):
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

print(f"Loaded {len(frames)} frames at {W}x{H} @ {fps}fps")

# Create backwards-only animation
anim = Text3DBehindSegment(
    duration=1.5,
    fps=fps,
    resolution=(W, H),
    text="HELLO WORLD",
    segment_mask=None,  # Dynamic masking
    font_size=140,
    text_color=(255, 220, 0),
    depth_color=(200, 170, 0),
    depth_layers=10,
    depth_offset=3,
    start_scale=2.2,  # Start big and visible
    end_scale=0.9,    # End small behind
    shrink_duration=1.2,  # Most time spent going backwards
    wiggle_duration=0.3,  # Brief wiggle at end
    center_position=(W//2, H//2),
    shadow_offset=6,
    outline_width=2,
    perspective_angle=0,  # No perspective
    supersample_factor=3,  # High quality
    debug=True,  # Show debug info
)

print("\nGenerating BACKWARDS-ONLY animation...")
print("Text starts fully visible and only moves backwards\n")

output_frames = []
for i in range(len(frames)):
    if i % 15 == 0:
        print(f"  Frame {i}/{len(frames)}...")
    
    # Note key moments
    if i == 0:
        print("    → Starting fully visible (no grow)")
    elif i == 36:  # 50% of shrink phase
        print("    → Starting to fade (passing behind)")
    elif i == 72:  # End of shrink
        print("    → Fully behind, starting wiggle")
    
    frame = anim.generate_frame(i, frames[i])
    
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    
    output_frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

# Save video
print("\nSaving backwards-only video...")
temp = "temp_backwards.mp4"
out = cv2.VideoWriter(temp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
for f in output_frames:
    out.write(f)
out.release()

# Convert to H.264
print("Converting to H.264...")
import subprocess
cmd = [
    'ffmpeg', '-y', '-i', temp,
    '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
    'BACKWARDS_ONLY_h264.mp4'
]
subprocess.run(cmd, capture_output=True)

import os
if os.path.exists(temp):
    os.remove(temp)

# Save key frames
cv2.imwrite("backwards_frame_0_start.png", output_frames[0])
cv2.imwrite("backwards_frame_45_midway.png", output_frames[45])
cv2.imwrite("backwards_frame_72_behind.png", output_frames[72])

print("\n" + "="*60)
print("✅ BACKWARDS-ONLY ANIMATION COMPLETE!")
print("="*60)
print("\n📹 Video: BACKWARDS_ONLY_h264.mp4")
print("\n📸 Key frames:")
print("  • backwards_frame_0_start.png - Starts fully visible")
print("  • backwards_frame_45_midway.png - Moving backwards")
print("  • backwards_frame_72_behind.png - Fully behind subject")
print("\n🎯 Animation flow:")
print("  1. Starts big and fully visible (no grow)")
print("  2. Shrinks while moving backwards")
print("  3. Fades as it passes behind subject")
print("  4. Ends small behind with subtle wiggle")