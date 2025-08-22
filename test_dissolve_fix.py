#!/usr/bin/env python3
"""Test dissolve animation with proper timing and soft glow fix."""

import os
os.environ.setdefault('FRAME_DISSOLVE_DEBUG','1')

import cv2
import numpy as np
from rembg import remove, new_session
from utils.animations.text_behind_segment import TextBehindSegment
from utils.animations.word_dissolve import WordDissolve

# Configuration
input_video = "test_element_3sec.mp4"
output_video = "test_dissolve_fix.mp4"
text = "HELLO WORLD"
RECOMPUTE_MASK_EVERY_N = 30  # Less frequent for speed

# Initialize rembg
print("[TEST] Initializing rembg...")
session = new_session('u2net')

# Load video
cap = cv2.VideoCapture(input_video)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"[TEST] Video: {width}x{height} @ {fps}fps, {total_frames} frames")

# Get first frame for initial mask
ret, first_frame = cap.read()
initial_mask = remove(first_frame, session=session, only_mask=True)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Animation phases (in frames)
phase1_frames = 30  # Shrink
phase2_frames = 20  # Move behind  
phase3_frames = 40  # Stable behind

# Create animations
center_position = (width // 2, int(height * 0.45))
font_size = int(height * 0.26)

# Create TextBehindSegment for handoff
text_animator = TextBehindSegment(
    element_path=input_video,
    background_path=input_video,
    position=center_position,
    text=text,
    font_size=font_size,
    text_color=(255, 220, 0),
    center_position=center_position,
    phase1_duration=0.5,  # Shrink duration
    phase2_duration=0.33,  # Move behind duration
    phase3_duration=0.67,  # Stable duration
    fps=fps
)

# Get handoff data
handoff_frame_idx = phase1_frames + phase2_frames + phase3_frames - 1
print(f"[TEST] Getting handoff at frame {handoff_frame_idx}")
cap.set(cv2.CAP_PROP_POS_FRAMES, handoff_frame_idx)
ret, frame = cap.read()
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
_ = text_animator.render_text_frame(frame_rgb, handoff_frame_idx)
handoff_data = text_animator.get_handoff_data()

# Create WordDissolve
word_dissolver = WordDissolve(
    element_path=input_video,
    background_path=input_video,
    position=center_position,
    word=text,
    font_size=font_size,
    text_color=(255, 220, 0),
    stable_duration=0.17,
    dissolve_duration=2.0,
    dissolve_stagger=0.5,
    float_distance=30,
    randomize_order=False,
    maintain_kerning=True,
    center_position=center_position,
    handoff_data=handoff_data,
    fps=fps,
    sprite_pad_ratio=0.35  # increased padding to prevent all artifacts
)

# Calculate actual WD length
active_letters = sum(1 for s in word_dissolver.letter_sprites if s is not None)
wd_total_frames = word_dissolver.stable_frames + max(0, active_letters - 1) * word_dissolver.stagger_frames + word_dissolver.dissolve_frames
total_animation_frames = phase1_frames + phase2_frames + phase3_frames + wd_total_frames

print(f"[TEST] Total animation: {total_animation_frames} frames ({total_animation_frames/fps:.1f}s)")
print(f"[TEST] WD will run for {wd_total_frames} frames")

# Setup video writer
temp_output = output_video.replace('.mp4', '_temp.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

# Process frames
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_idx = 0
current_mask = initial_mask
last_mask_update = 0

print("\n[TEST] Processing frames...")
while frame_idx < total_animation_frames:
    ret, frame = cap.read()
    if not ret:
        # Loop video if needed
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if not ret:
            break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Update mask periodically
    if frame_idx - last_mask_update >= RECOMPUTE_MASK_EVERY_N:
        current_mask = remove(frame, session=session, only_mask=True)
        last_mask_update = frame_idx
        if frame_idx % 60 == 0:
            print(f"[TEST] Progress: frame {frame_idx}/{total_animation_frames}")
    
    # Render appropriate phase
    if frame_idx < phase1_frames + phase2_frames + phase3_frames:
        # TextBehindSegment phase
        frame_rgb = text_animator.render_text_frame(frame_rgb, frame_idx, mask=current_mask)
    else:
        # WordDissolve phase
        dissolve_frame = frame_idx - (phase1_frames + phase2_frames + phase3_frames)
        frame_rgb = word_dissolver.render_word_frame(frame_rgb, dissolve_frame, mask=current_mask)
    
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    out.write(frame_bgr)
    frame_idx += 1

cap.release()
out.release()

# Convert to H.264
print("\n[TEST] Converting to H.264...")
import subprocess
cmd = [
    'ffmpeg', '-y', '-i', temp_output,
    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
    output_video
]
subprocess.run(cmd, capture_output=True)

import os
os.remove(temp_output)

print(f"[TEST] Output saved to: {output_video}")
print(f"[TEST] Duration: {total_animation_frames/fps:.1f} seconds")