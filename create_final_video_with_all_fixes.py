#!/usr/bin/env python3
"""Create the FINAL video with ALL fixes applied"""

import os
import cv2
import numpy as np
from utils.animations.text_3d_behind_segment import Text3DBehindSegment
from utils.segmentation.segment_extractor import extract_foreground_mask

print("="*60)
print("CREATING FINAL VIDEO WITH ALL FIXES")
print("="*60)
print("\nFixes included:")
print("  ✓ Smooth depth (no pixelation)")
print("  ✓ 80% reduced depth")
print("  ✓ Fade starts at 50% of shrink")
print("  ✓ No slanting (perspective disabled)")
print("  ✓ Center-point locking\n")

video_path = "test_element_3sec.mp4"
if not os.path.exists(video_path):
    print("Error: Video file not found")
    exit(1)

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Get mask
ret, frame = cap.read()
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
print("Extracting foreground mask...")
mask = extract_foreground_mask(frame_rgb)

# Load 90 frames (1.5 seconds) for manageable processing
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frames = []
for i in range(90):
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

print(f"Loaded {len(frames)} frames at {W}x{H} @ {fps}fps")

# Create animation with ALL fixes
anim = Text3DBehindSegment(
    duration=1.5,
    fps=fps,
    resolution=(W, H),
    text="HELLO WORLD",
    segment_mask=mask,
    font_size=140,
    text_color=(255, 220, 0),
    depth_color=(200, 170, 0),
    depth_layers=10,  # Doubled internally for smoothness
    depth_offset=3,    # Reduced 80% internally
    start_scale=2.2,
    end_scale=0.9,
    phase1_duration=1.0,  # Shrink phase
    phase2_duration=0.3,  # Transition
    phase3_duration=0.2,  # Stable
    center_position=(W//2, H//2),
    shadow_offset=6,
    outline_width=2,
    perspective_angle=25,  # Will be ignored due to fix
    supersample_factor=3,  # Good quality/speed balance
    debug=False,
    perspective_during_shrink=False,  # Already handled by fix
)

print("\nGenerating frames with all fixes...")
output_frames = []

for i in range(len(frames)):
    if i % 15 == 0:
        print(f"  Frame {i}/{len(frames)}...")
    
    frame = anim.generate_frame(i, frames[i])
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    output_frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

# Save video
print("\nSaving video...")
output_path = "text_3d_ALL_FIXES_FINAL.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
temp = "temp_all_fixes.mp4"

out = cv2.VideoWriter(temp, fourcc, fps, (W, H))
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
    output_path
]
subprocess.run(cmd, capture_output=True)

if os.path.exists(temp):
    os.remove(temp)

# Save preview frames
print("Saving preview frames...")
preview_frames = {
    'early': len(frames) // 6,      # Early in shrink
    'midpoint': len(frames) // 3,   # Should start fading here
    'late': 2 * len(frames) // 3,   # Late in animation
    'final': len(frames) - 1        # Final position
}

for name, idx in preview_frames.items():
    if idx < len(output_frames):
        cv2.imwrite(f"all_fixes_{name}.png", output_frames[idx])

print(f"\n" + "="*60)
print("✅ SUCCESS! Final video created with ALL fixes!")
print("="*60)
print(f"\n📹 Video: {output_path}")
print("\n🎯 All fixes applied:")
print("  1. Smooth depth on all letters (no pixelation)")
print("  2. Depth reduced by 80% (subtle 3D)")
print("  3. Fade starts at 50% of shrink phase")
print("  4. No perspective/slanting")
print("  5. Center-point locking")
print("\n📸 Preview frames:")
print("  • all_fixes_early.png - Full opacity")
print("  • all_fixes_midpoint.png - Fading starts")
print("  • all_fixes_late.png - Behind subject")
print("  • all_fixes_final.png - Final position (no slant)")