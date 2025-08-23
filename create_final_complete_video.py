#!/usr/bin/env python3
"""Create FINAL video with ALL fixes including dynamic masking"""

import os
import cv2
import numpy as np
from utils.animations.text_3d_behind_segment import Text3DBehindSegment

print("="*60)
print("FINAL VIDEO WITH ALL FIXES INCLUDING DYNAMIC MASKING")
print("="*60)
print("\n✅ All fixes included:")
print("  1. Smooth depth (no pixelation)")
print("  2. 80% reduced depth")
print("  3. Fade starts at 50% of shrink")
print("  4. No slanting (perspective disabled)")
print("  5. Center-point locking")
print("  6. ⭐ DYNAMIC MASK - recalculated EVERY frame\n")

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

# Create animation - NO pre-calculated mask!
anim = Text3DBehindSegment(
    duration=1.5,
    fps=fps,
    resolution=(W, H),
    text="HELLO WORLD",
    segment_mask=None,  # ⭐ No static mask - will calculate per frame
    font_size=140,
    text_color=(255, 220, 0),
    depth_color=(200, 170, 0),
    depth_layers=10,
    depth_offset=3,
    start_scale=2.2,
    end_scale=0.9,
    phase1_duration=1.0,
    phase2_duration=0.3,
    phase3_duration=0.2,
    center_position=(W//2, H//2),
    shadow_offset=6,
    outline_width=2,
    perspective_angle=25,  # Will be ignored
    supersample_factor=3,
    debug=False,  # Disable debug for cleaner output
    perspective_during_shrink=False,
)

print("\nGenerating frames with DYNAMIC masking...")
print("(Mask recalculated for each frame when text is behind)")
output_frames = []

for i in range(len(frames)):
    if i % 15 == 0:
        print(f"  Frame {i}/{len(frames)}...")
    
    # Pass current frame - mask will be calculated from it
    frame = anim.generate_frame(i, frames[i])
    
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    
    output_frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

# Save video
print("\nSaving final video...")
output_path = "text_3d_COMPLETE_FINAL.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
temp = "temp_complete.mp4"

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

# Save preview
preview_idx = len(frames) // 2
cv2.imwrite("complete_final_preview.png", output_frames[preview_idx])

print(f"\n" + "="*60)
print("✅ COMPLETE! Final video with ALL fixes!")
print("="*60)
print(f"\n📹 Video: {output_path}")
print(f"📸 Preview: complete_final_preview.png")
print("\n🎯 All issues fixed:")
print("  1. ✓ Smooth depth (no pixelation)")
print("  2. ✓ 80% reduced depth")
print("  3. ✓ Fade timing (starts at 50% of shrink)")
print("  4. ✓ No slanting")
print("  5. ✓ Center-point locking")
print("  6. ✓ Dynamic masking (tracks moving subjects)")
print("\nThe mask is recalculated for EVERY frame,")
print("ensuring proper occlusion even with moving subjects!")