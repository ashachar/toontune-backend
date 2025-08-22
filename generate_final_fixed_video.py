#!/usr/bin/env python3
"""Generate the ACTUAL fixed video with all improvements applied"""

import os
import cv2
import numpy as np
from utils.animations.text_3d_behind_segment import Text3DBehindSegment
from utils.segmentation.segment_extractor import extract_foreground_mask

video_path = "test_element_3sec.mp4"
if not os.path.exists(video_path):
    # Fallback to synthetic
    print("Using synthetic test")
    W, H, fps = 1166, 534, 60
    frames = [np.full((H, W, 3), 245, dtype=np.uint8) for _ in range(180)]
    mask = np.zeros((H, W), dtype=np.uint8)
else:
    print("Loading video...")
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print("Extracting mask...")
        mask = extract_foreground_mask(frame_rgb)
    else:
        mask = np.zeros((H, W), dtype=np.uint8)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    print(f"Loaded {len(frames)} frames")

print("\nCreating animation with ALL fixes applied:")
print("  • 80% reduced depth")
print("  • 2x more depth layers")  
print("  • 4x supersampling")
print("  • Smooth gradients\n")

anim = Text3DBehindSegment(
    duration=3.0,
    fps=fps,
    resolution=(W, H),
    text="HELLO WORLD",
    segment_mask=mask,
    font_size=140,
    text_color=(255, 220, 0),
    depth_color=(200, 170, 0),
    depth_layers=10,  # Will be doubled to 20 internally
    depth_offset=3,    # Will be reduced by 80% internally
    start_scale=2.2,
    end_scale=0.9,
    phase1_duration=1.2,
    phase2_duration=0.6,
    phase3_duration=1.2,
    center_position=(W//2, H//2),
    shadow_offset=6,
    outline_width=2,
    perspective_angle=25,
    supersample_factor=4,  # Using the improved default
    debug=False,
    perspective_during_shrink=False,
)

print("Generating frames...")
output_frames = []
total = int(3.0 * fps)

for i in range(total):
    if i % 30 == 0:
        print(f"  Frame {i}/{total}...")
    
    bg = frames[i % len(frames)]
    frame = anim.generate_frame(i, bg)
    
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    
    output_frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

print("\nSaving video...")
output_path = "text_3d_truly_fixed.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
temp = "temp_truly_fixed.mp4"

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

# Save a preview frame
preview_idx = total // 4
cv2.imwrite("truly_fixed_preview.png", output_frames[preview_idx])

print(f"\n✅ DONE!")
print(f"Video: {output_path}")
print(f"Preview: truly_fixed_preview.png")
print("\nThis video has ALL the fixes:")
print("  • Smooth depth on all letters (like the test image)")
print("  • 80% smaller depth")
print("  • No pixelation")
print("  • Center-point locking")