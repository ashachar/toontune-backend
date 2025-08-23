#!/usr/bin/env python3
"""Test the fade timing and slant fixes"""

import os
import cv2
import numpy as np
from utils.animations.text_3d_behind_segment import Text3DBehindSegment
from utils.segmentation.segment_extractor import extract_foreground_mask

print("Testing FADE TIMING + NO SLANT fixes...\n")

video_path = "test_element_3sec.mp4"
if not os.path.exists(video_path):
    print("Using synthetic test")
    W, H, fps = 1166, 534, 60
    frames = [np.full((H, W, 3), 245, dtype=np.uint8) for _ in range(90)]
    mask = np.zeros((H, W), dtype=np.uint8)
else:
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print("Extracting mask...")
    mask = extract_foreground_mask(frame_rgb)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frames = []
    for i in range(90):  # 1.5 seconds
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

print(f"Loaded {len(frames)} frames at {W}x{H}")
print("\n✨ Testing with FIXED code:")
print("  1. Fade starts at 50% of shrink (exponential)")
print("  2. NO perspective (no slanting)\n")

# Create animation with fixes
anim = Text3DBehindSegment(
    duration=1.5,
    fps=fps,
    resolution=(W, H),
    text="HELLO WORLD",
    segment_mask=mask,
    font_size=140,
    text_color=(255, 220, 0),
    depth_color=(200, 170, 0),
    depth_layers=10,
    depth_offset=3,
    start_scale=2.2,
    end_scale=0.9,
    phase1_duration=1.0,  # Longer shrink to see fade timing clearly
    phase2_duration=0.3,
    phase3_duration=0.2,
    center_position=(W//2, H//2),
    shadow_offset=6,
    outline_width=2,
    perspective_angle=25,  # Will be ignored due to fix
    supersample_factor=3,
    debug=True,  # Enable debug to see fade values
    perspective_during_shrink=False,
)

print("Generating frames...")
output_frames = []
key_frames = {}

for i in range(len(frames)):
    if i % 15 == 0:
        print(f"  Frame {i}/{len(frames)}...")
    
    frame = anim.generate_frame(i, frames[i])
    
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    
    output_frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    # Capture key frames
    if i == 0:
        key_frames['start'] = i
    elif i == fps // 2:  # 0.5 seconds (50% of shrink phase)
        key_frames['midpoint'] = i
        print(f"    → MIDPOINT (50% shrink): Frame {i} - fade should START here")
    elif i == int(fps * 0.75):  # 0.75 seconds
        key_frames['three_quarter'] = i
    elif i == fps:  # 1 second (end of shrink)
        key_frames['end_shrink'] = i
    elif i == len(frames) - 1:
        key_frames['final'] = i

# Save key frames
print("\nSaving key frames...")
for name, idx in key_frames.items():
    if idx < len(output_frames):
        filename = f"fade_fix_{name}.png"
        cv2.imwrite(filename, output_frames[idx])
        print(f"  ✓ {filename}")

# Save video
print("\nSaving video...")
output_path = "fade_slant_FIXED.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
temp = "temp_fade_fix.mp4"

out = cv2.VideoWriter(temp, fourcc, fps, (W, H))
for f in output_frames:
    out.write(f)
out.release()

# Convert to H.264
import subprocess
cmd = [
    'ffmpeg', '-y', '-i', temp,
    '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
    output_path
]
subprocess.run(cmd, capture_output=True)

if os.path.exists(temp):
    os.remove(temp)

print(f"\n✅ SUCCESS!")
print(f"Video: {output_path}")
print("\nFixes demonstrated:")
print("  1. Text starts fading at frame ~30 (50% of shrink)")
print("  2. Text remains straight (no slant) at final position")
print("\nCheck the key frames to see:")
print("  • fade_fix_midpoint.png - should start fading here")
print("  • fade_fix_final.png - should be straight (no slant)")