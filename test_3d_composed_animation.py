#!/usr/bin/env python3
"""
Test the composed 3D text animation using separate Text3DMotion and Letter3DDissolve classes.
This demonstrates proper animation composition without code duplication.
"""

import cv2
import numpy as np
from utils.animations.text_3d_motion import Text3DMotion
from utils.animations.letter_3d_dissolve import Letter3DDissolve

print("="*80)
print("TESTING COMPOSED 3D TEXT ANIMATION")
print("="*80)

# Load test video
video_path = "test_element_3sec.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Read background frames
frames = []
for i in range(90):  # 3 seconds at 30fps
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

print(f"\nLoaded {len(frames)} frames")
print(f"Resolution: {W}x{H}")
print(f"FPS: {fps}")

# Create motion animation
print("\nCreating motion animation...")
motion_duration = 0.75
motion = Text3DMotion(
    duration=motion_duration,
    fps=fps,
    resolution=(W, H),
    text="HELLO WORLD",
    segment_mask=None,  # Will use dynamic masking
    font_size=140,
    text_color=(255, 220, 0),
    depth_color=(200, 170, 0),
    depth_layers=8,
    depth_offset=3,
    start_scale=2.0,
    end_scale=1.0,
    final_scale=0.9,
    start_position=(W//2, H//2 - H//6),  # Start above center
    end_position=(W//2, H//2),           # End at center
    shrink_duration=0.6,
    settle_duration=0.15,
    shadow_offset=6,
    outline_width=2,
    perspective_angle=0,
    supersample_factor=2,
    glow_effect=True,
    debug=True,
)

# Create dissolve animation
print("\nCreating dissolve animation...")
dissolve_duration = 1.5
dissolve = Letter3DDissolve(
    duration=dissolve_duration,
    fps=fps,
    resolution=(W, H),
    text="HELLO WORLD",
    font_size=140,
    text_color=(255, 220, 0),
    depth_color=(200, 170, 0),
    depth_layers=8,
    depth_offset=3,
    initial_scale=0.9,  # Will be set from motion's final state
    initial_position=(W//2, H//2),  # Will be set from motion's final state
    stable_duration=0.1,
    dissolve_duration=0.5,
    dissolve_stagger=0.1,
    float_distance=40,
    max_dissolve_scale=1.3,
    randomize_order=False,
    shadow_offset=6,
    outline_width=2,
    supersample_factor=2,
    debug=True,
)

# Generate composed video
print("\nGenerating composed animation...")
output_path = "text_3d_composed_animation.mp4"

height, width = frames[0].shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

motion_frames = int(motion_duration * fps)
dissolve_frames = int(dissolve_duration * fps)
total_frames = motion_frames + dissolve_frames

print(f"Motion frames: {motion_frames}")
print(f"Dissolve frames: {dissolve_frames}")
print(f"Total frames: {total_frames}")

# Generate motion phase
print("\nGenerating motion phase...")
for i in range(motion_frames):
    bg_idx = i % len(frames)
    background = frames[bg_idx]
    
    frame = motion.generate_frame(i, background)
    
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write(frame_bgr)
    
    if i % 10 == 0:
        print(f"  Frame {i}/{motion_frames}")

# Get final state from motion and pass to dissolve
final_state = motion.get_final_state()
if final_state:
    print(f"\nHandoff state from motion to dissolve:")
    print(f"  Scale: {final_state.scale:.3f}")
    print(f"  Position: {final_state.position}")
    print(f"  Center: {final_state.center_position}")
    
    # Set initial state for dissolve
    dissolve.set_initial_state(
        scale=final_state.scale,
        position=final_state.center_position
    )

# Generate dissolve phase
print("\nGenerating dissolve phase...")
for i in range(dissolve_frames):
    bg_idx = (motion_frames + i) % len(frames)
    background = frames[bg_idx]
    
    frame = dissolve.generate_frame(i, background)
    
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write(frame_bgr)
    
    if i % 10 == 0:
        print(f"  Frame {i}/{dissolve_frames}")

out.release()
print(f"\nVideo saved to {output_path}")

# Convert to H.264
print("\nConverting to H.264...")
h264_path = "text_3d_composed_animation_h264.mp4"
import subprocess
subprocess.run([
    'ffmpeg', '-i', output_path,
    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
    h264_path, '-y'
], check=True)

print(f"\n✅ H.264 video saved to: {h264_path}")
print("\n" + "="*80)
print("COMPOSED ANIMATION TEST COMPLETE")
print("="*80)
print("\nThe animation demonstrates:")
print("1. Text3DMotion: Shrinking and moving text behind subject")
print("2. Letter3DDissolve: Individual letter dissolve effect")
print("3. Proper handoff between animations with state preservation")
print("4. No code duplication - each animation is a separate, reusable class")