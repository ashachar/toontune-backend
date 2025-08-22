#!/usr/bin/env python3
"""Test with random letter dissolve order."""

import os
os.environ['FRAME_DISSOLVE_DEBUG'] = '1'

import cv2
import numpy as np
from rembg import remove, new_session
from utils.animations.text_behind_segment import TextBehindSegment
from utils.animations.word_dissolve import WordDissolve

# Configuration
input_video = "test_element_3sec.mp4"
output_video = "hello_world_random_dissolve.mp4"
text = "HELLO WORLD"

print("[TEST] Initializing with RANDOM dissolve order...")
session = new_session('u2net')

# Load video
cap = cv2.VideoCapture(input_video)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"[TEST] Video: {width}x{height} @ {fps}fps")

# Reset to start
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Animation phases
phase1_frames = 20  # Shrink
phase2_frames = 15  # Move behind  
phase3_frames = 25  # Stable behind

center_position = (width // 2, int(height * 0.45))
font_size = int(height * 0.26)

# Create TextBehindSegment
text_animator = TextBehindSegment(
    element_path=input_video,
    background_path=input_video,
    position=center_position,
    text=text,
    segment_mask=None,
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

# Create WordDissolve with RANDOM order
print("\n[DISSOLVE] Creating WordDissolve with randomize_order=True")
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
    randomize_order=True,  # RANDOM ORDER!
    maintain_kerning=True,
    center_position=center_position,
    handoff_data=handoff_data,
    fps=fps,
    sprite_pad_ratio=0.40,
    debug=True
)

# Show the random order
print(f"[DISSOLVE] Random order generated:")
if hasattr(word_dissolver, 'letter_indices'):
    order_chars = [text[i] if i < len(text) else '?' for i in word_dissolver.letter_indices]
    print(f"  Original: {text}")
    print(f"  Dissolve: {''.join(order_chars)}")
    print(f"  Indices:  {word_dissolver.letter_indices}")

# Calculate frames
wd_duration = word_dissolver.stable_frames + 10 * word_dissolver.stagger_frames + word_dissolver.dissolve_frames
total_frames = phase1_frames + phase2_frames + phase3_frames + wd_duration + 30

print(f"\n[TEST] Total: {total_frames} frames ({total_frames/fps:.1f}s)")

# Process
temp_output = output_video.replace('.mp4', '_temp.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
current_mask = None

print("[TEST] Processing...")
for frame_idx in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Update mask periodically
    if frame_idx % 30 == 0:
        current_mask = remove(frame, session=session, only_mask=True)
    
    # Render
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

print(f"\n[TEST] ✅ Output: {output_video}")
print(f"[TEST] Duration: {total_frames/fps:.1f}s")
print("\n[VERIFICATION]")
print("1. Letters dissolve in RANDOM order, not left-to-right")
print("2. Each run will have a different random sequence")
print("3. Visual interest increased with unpredictable dissolve pattern")