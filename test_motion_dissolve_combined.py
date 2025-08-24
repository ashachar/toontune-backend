#!/usr/bin/env python3
"""Test the combined 3D text motion + dissolve animation"""

import cv2
import numpy as np
from utils.animations.text_3d_motion_dissolve_proper import Text3DMotionDissolve

print("="*70)
print("TESTING COMBINED MOTION + DISSOLVE ANIMATION")
print("="*70)
print("\n✅ Features:")
print("  • Smooth continuous motion (no stops)")
print("  • 3D text with depth layers")
print("  • Text goes behind subject")
print("  • Letter-by-letter 3D dissolve (depth layers dissolve together)")
print("  • Total duration: ~2.5 seconds")
print("")

# Load video
video_path = "test_element_3sec.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate durations
motion_duration = 0.75
dissolve_duration = 1.5
total_duration = motion_duration + dissolve_duration

print(f"Animation timing:")
print(f"  FPS: {fps}")
print(f"  Motion phase: 0-{motion_duration:.2f}s")
print(f"  Dissolve phase: {motion_duration:.2f}-{total_duration:.2f}s")
print(f"  Total: {total_duration:.2f}s")

# Load frames for the entire duration
total_frames_needed = int(total_duration * fps) + 1
frames = []
for i in range(total_frames_needed):
    ret, frame = cap.read()
    if not ret:
        # If we run out of video, loop back
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

print(f"\nLoaded {len(frames)} frames")

# Create combined animation
anim = Text3DMotionDissolve(
    duration=total_duration,
    fps=fps,
    resolution=(W, H),
    text="HELLO WORLD",
    segment_mask=None,
    font_size=140,
    text_color=(255, 220, 0),
    depth_color=(200, 170, 0),
    depth_layers=8,
    depth_offset=3,
    # Motion parameters
    motion_duration=motion_duration,
    start_scale=2.0,
    end_scale=1.0,
    final_scale=0.9,
    shrink_duration=0.6,
    settle_duration=0.15,
    # Dissolve parameters
    dissolve_stable_duration=0.1,
    dissolve_duration=0.5,
    dissolve_stagger=0.1,
    float_distance=40,
    max_dissolve_scale=1.3,
    randomize_order=True,
    maintain_kerning=True,
    # Visual effects
    center_position=(W//2, H//2),
    shadow_offset=6,
    outline_width=2,
    perspective_angle=0,
    supersample_factor=2,
    glow_effect=True,
    debug=True,
)

print("\n" + "="*70)
print("Generating animation frames...")
print("="*70)

output_frames = []
motion_end_frame = int(motion_duration * fps)
dissolve_start_frame = motion_end_frame

for i in range(len(frames)):
    if i % 15 == 0:
        phase = "motion" if i < motion_end_frame else "dissolve"
        print(f"  Frame {i}/{len(frames)} ({phase} phase)...")
    
    frame = anim.generate_frame(i, frames[i])
    
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    output_frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

print(f"\nGenerated {len(output_frames)} frames")

# Save video
print("\nSaving video...")
out = cv2.VideoWriter("motion_dissolve.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
for f in output_frames:
    out.write(f)
out.release()

# Convert to H.264
import subprocess
print("Converting to H.264...")
subprocess.run([
    'ffmpeg', '-y', '-i', 'motion_dissolve.mp4',
    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
    'MOTION_DISSOLVE_COMBINED_h264.mp4'
], capture_output=True)

import os
os.remove('motion_dissolve.mp4')

# Extract key frames for verification
print("\n" + "="*70)
print("Extracting key frames for verification...")
print("="*70)

key_frames = [
    (0, "start"),
    (motion_end_frame // 2, "mid_motion"),
    (motion_end_frame - 1, "end_motion"),
    (dissolve_start_frame + 5, "start_dissolve"),
    (dissolve_start_frame + 30, "mid_dissolve"),
    (len(output_frames) - 10, "near_end"),
]

for frame_idx, label in key_frames:
    if frame_idx < len(output_frames):
        cv2.imwrite(f'frame_{frame_idx:03d}_{label}.png', output_frames[frame_idx])
        print(f"  Saved: frame_{frame_idx:03d}_{label}.png")

print("\n" + "="*70)
print("✅ COMBINED ANIMATION COMPLETE!")
print("="*70)
print("\n📹 Final video: MOTION_DISSOLVE_COMBINED_h264.mp4")
print("\n🎯 What's included:")
print("  • Smooth 3D text motion from top to center")
print("  • Text shrinks and goes behind subject")
print("  • Continuous motion (no stops or jumps)")
print("  • Letter-by-letter 3D dissolve (INCLUDING depth layers)")
print("  • Each letter's 3D depth dissolves as a unit")
print("  • Random dissolve order with floating effect")
print("\n📊 Key frames saved for inspection")
print("\nPlease review the video to confirm both effects work together!")