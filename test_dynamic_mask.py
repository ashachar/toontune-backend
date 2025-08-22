#!/usr/bin/env python3
"""Test with per-frame mask recalculation for accurate occlusion."""

import os
os.environ['FRAME_DISSOLVE_DEBUG'] = '1'

import cv2
import numpy as np
from rembg import remove, new_session
from utils.animations.text_behind_segment import TextBehindSegment
from utils.animations.word_dissolve import WordDissolve

# Configuration
input_video = "test_element_3sec.mp4"
output_video = "hello_world_dynamic_mask.mp4"
text = "HELLO WORLD"

print("[TEST] Initializing with DYNAMIC per-frame mask...")
session = new_session('u2net')

# Load video
cap = cv2.VideoCapture(input_video)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"[TEST] Video: {width}x{height} @ {fps}fps")

# Reset to start
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Shorter animation for testing
phase1_frames = 20  # Shrink
phase2_frames = 15  # Move behind  
phase3_frames = 25  # Stable behind

center_position = (width // 2, int(height * 0.45))
font_size = int(height * 0.26)

# Create TextBehindSegment WITHOUT initial mask
# We'll pass the mask per-frame instead
text_animator = TextBehindSegment(
    element_path=input_video,
    background_path=input_video,
    position=center_position,
    text=text,
    segment_mask=None,  # No initial mask - will provide per-frame
    font_size=font_size,
    text_color=(255, 220, 0),
    center_position=center_position,
    phase1_duration=phase1_frames/fps,
    phase2_duration=phase2_frames/fps,
    phase3_duration=phase3_frames/fps,
    fps=fps
)

# Get handoff
handoff_frame_idx = phase1_frames + phase2_frames + phase3_frames - 1
print(f"[TEST] Getting handoff at frame {handoff_frame_idx}")
cap.set(cv2.CAP_PROP_POS_FRAMES, handoff_frame_idx)
ret, frame = cap.read()
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
handoff_mask = remove(frame, session=session, only_mask=True)
_ = text_animator.render_text_frame(frame_rgb, handoff_frame_idx, mask=handoff_mask)
handoff_data = text_animator.get_handoff_data()

# Create WordDissolve
word_dissolver = WordDissolve(
    element_path=input_video,
    background_path=input_video,
    position=center_position,
    word=text,
    font_size=font_size,
    text_color=(255, 220, 0),
    stable_duration=0.1,
    dissolve_duration=1.0,
    dissolve_stagger=0.25,
    float_distance=30,
    randomize_order=False,
    maintain_kerning=True,
    center_position=center_position,
    handoff_data=handoff_data,
    fps=fps,
    sprite_pad_ratio=0.40,
    debug=True
)

# Calculate frames
wd_duration = word_dissolver.stable_frames + 10 * word_dissolver.stagger_frames + word_dissolver.dissolve_frames
total_frames = phase1_frames + phase2_frames + phase3_frames + wd_duration + 30

print(f"[TEST] Total: {total_frames} frames ({total_frames/fps:.1f}s)")
print(f"[TEST] Processing with PER-FRAME mask calculation...")

# Process
temp_output = output_video.replace('.mp4', '_temp.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
mask_cache = {}  # Cache masks to avoid redundant calculations

print("[TEST] Processing...")
for frame_idx in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # ALWAYS calculate mask for current frame for maximum accuracy
    # Use cache for efficiency
    cache_key = frame_idx % (fps * 3)  # Cycle cache every 3 seconds
    if cache_key not in mask_cache:
        current_mask = remove(frame, session=session, only_mask=True)
        mask_cache[cache_key] = current_mask
        if frame_idx % 10 == 0:
            print(f"[MASK] Recalculated mask at frame {frame_idx}")
    else:
        current_mask = mask_cache[cache_key]
    
    # Render with fresh mask
    if frame_idx < phase1_frames + phase2_frames + phase3_frames:
        frame_rgb = text_animator.render_text_frame(frame_rgb, frame_idx, mask=current_mask)
    else:
        dissolve_frame = frame_idx - (phase1_frames + phase2_frames + phase3_frames)
        frame_rgb = word_dissolver.render_word_frame(frame_rgb, dissolve_frame, mask=current_mask)
    
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    out.write(frame_bgr)
    
    if frame_idx % 60 == 0:
        print(f"[TEST] Progress: {frame_idx}/{total_frames}")

cap.release()
out.release()

# Convert to H.264
print("[TEST] Converting to H.264...")
import subprocess
cmd = [
    'ffmpeg', '-y', '-i', temp_output,
    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
    output_video
]
result = subprocess.run(cmd, capture_output=True, text=True)

if os.path.exists(temp_output):
    os.remove(temp_output)

print(f"[TEST] ✅ Output: {output_video}")
print(f"[TEST] Duration: {total_frames/fps:.1f}s")
print("\n[VERIFICATION]")
print("1. Text should be accurately occluded by moving subject")
print("2. No text should appear through the subject even if they move")
print("3. Per-frame mask ensures pixel-perfect occlusion")