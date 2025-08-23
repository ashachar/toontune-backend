#!/usr/bin/env python3
"""Quick demo with dynamic masking"""

import cv2
import numpy as np
from utils.animations.text_3d_behind_segment import Text3DBehindSegment

video_path = "test_element_3sec.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Load 45 frames
frames = []
for i in range(45):
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

print(f"Creating 0.75s demo with DYNAMIC masking...")

anim = Text3DBehindSegment(
    duration=0.75,
    fps=fps,
    resolution=(W, H),
    text="HELLO WORLD",
    segment_mask=None,  # Dynamic!
    font_size=140,
    text_color=(255, 220, 0),
    depth_color=(200, 170, 0),
    depth_layers=10,
    depth_offset=3,
    start_scale=2.0,
    end_scale=1.0,
    phase1_duration=0.5,
    phase2_duration=0.15,
    phase3_duration=0.1,
    center_position=(W//2, H//2),
    shadow_offset=6,
    outline_width=2,
    perspective_angle=0,
    supersample_factor=2,
    debug=False,
)

output_frames = []
for i in range(45):
    print(f"Frame {i+1}/45...", end='\r')
    frame = anim.generate_frame(i, frames[i])
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    output_frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

print("\nSaving...")
out = cv2.VideoWriter("FINAL_dynamic_mask.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
for f in output_frames:
    out.write(f)
out.release()

cv2.imwrite("final_dynamic_preview.png", output_frames[30])

print("✅ Done!")
print("Video: FINAL_dynamic_mask.mp4")
print("Preview: final_dynamic_preview.png")
print("\nDynamic masking ensures proper occlusion")
print("even when subjects move!")
