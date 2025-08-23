#!/usr/bin/env python3
"""Compare original vs my broken 'fix' to understand the issue"""

import cv2
import numpy as np

# Import both versions
from utils.animations.text_3d_behind_segment import Text3DBehindSegment as OriginalVersion
from utils.animations.text_3d_behind_segment_fixed import Text3DBehindSegment as BrokenVersion

print("COMPARING ORIGINAL vs BROKEN 'FIX'")
print("=" * 50)

# Load test frame
video_path = "test_element_3sec.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Get multiple frames around the critical moment
frames = []
for i in range(30, 60):  # Frames 30-59
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if ret:
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

print(f"Loaded {len(frames)} frames")

# Common parameters
params = {
    "duration": 1.0,
    "fps": fps,
    "resolution": (W, H),
    "text": "HELLO WORLD",
    "segment_mask": None,  # Dynamic
    "font_size": 140,
    "text_color": (255, 220, 0),
    "depth_color": (200, 170, 0),
    "depth_layers": 10,
    "depth_offset": 3,
    "start_scale": 1.5,
    "end_scale": 0.9,
    "phase1_duration": 0.5,
    "phase2_duration": 0.4,
    "phase3_duration": 0.1,
    "center_position": (W//2, H//2),
    "shadow_offset": 6,
    "outline_width": 2,
    "perspective_angle": 0,
    "supersample_factor": 2,
    "debug": False,
}

print("\nGenerating with ORIGINAL version...")
original = OriginalVersion(**params)

print("Generating with BROKEN 'fix' version...")
broken = BrokenVersion(**params)

# Generate frame 48 (critical W occlusion)
frame_idx = 18  # This is frame 48 in the video (30 + 18)
frame_rgb = frames[frame_idx]

original_result = original.generate_frame(48, frame_rgb)
broken_result = broken.generate_frame(48, frame_rgb)

# Convert to BGR for saving
if original_result.shape[2] == 4:
    original_result = original_result[:, :, :3]
if broken_result.shape[2] == 4:
    broken_result = broken_result[:, :, :3]

cv2.imwrite("compare_1_original.png", cv2.cvtColor(original_result, cv2.COLOR_RGB2BGR))
cv2.imwrite("compare_2_broken.png", cv2.cvtColor(broken_result, cv2.COLOR_RGB2BGR))

# Create side-by-side
comparison = np.zeros((H, W*2, 3), dtype=np.uint8)
comparison[:, :W] = original_result
comparison[:, W:] = broken_result

# Add labels
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(comparison, "ORIGINAL (Working)", (20, 40), font, 1.0, (255, 255, 255), 2)
cv2.putText(comparison, "MY 'FIX' (Broken)", (W+20, 40), font, 1.0, (255, 255, 255), 2)
cv2.line(comparison, (W, 0), (W, H), (255, 255, 255), 2)

cv2.imwrite("compare_3_side_by_side.png", cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

# Generate a short video showing the difference
print("\nGenerating comparison video...")
output_frames = []
for i in range(len(frames)):
    print(f"Frame {i+1}/{len(frames)}", end='\r')
    
    orig_frame = original.generate_frame(30 + i, frames[i])
    broken_frame = broken.generate_frame(30 + i, frames[i])
    
    if orig_frame.shape[2] == 4:
        orig_frame = orig_frame[:, :, :3]
    if broken_frame.shape[2] == 4:
        broken_frame = broken_frame[:, :, :3]
    
    # Create side-by-side
    comp = np.zeros((H, W*2, 3), dtype=np.uint8)
    comp[:, :W] = orig_frame
    comp[:, W:] = broken_frame
    
    # Add labels on first frame only
    if i == 0:
        cv2.putText(comp, "ORIGINAL", (20, 40), font, 1.0, (255, 255, 255), 2)
        cv2.putText(comp, "BROKEN", (W+20, 40), font, 1.0, (255, 255, 255), 2)
    
    cv2.line(comp, (W, 0), (W, H), (255, 255, 255), 1)
    output_frames.append(cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))

# Save video
print("\nSaving comparison video...")
out = cv2.VideoWriter("comparison_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (W*2, H))
for f in output_frames:
    out.write(f)
out.release()

print("\n" + "="*50)
print("COMPARISON COMPLETE!")
print("="*50)
print("\nGenerated:")
print("1. compare_1_original.png - Original version (working)")
print("2. compare_2_broken.png - My 'fix' (broken)")
print("3. compare_3_side_by_side.png - Side by side at frame 48")
print("4. comparison_video.mp4 - Full comparison video")
print("\nThe broken version likely has incorrect mask mapping!")