#!/usr/bin/env python3
"""Test that text only starts fading AFTER it passes behind the subject."""

import os
os.environ['FRAME_DISSOLVE_DEBUG'] = '1'

import cv2
import numpy as np
from rembg import remove, new_session
from utils.animations.text_behind_segment import TextBehindSegment
from utils.animations.word_dissolve import WordDissolve
import math

# Configuration
input_video = "test_element_3sec.mp4"
output_video = "hello_world_fade_after_behind.mp4"
text = "HELLO WORLD"

print("[TEST] Text fading ONLY after passing behind subject...")
print("[TEST] Phase 1 (shrink): Text stays FULLY OPAQUE")
print("[TEST] Phase 2 (transition): Text fades to 50% transparency")
print("[TEST] Phase 3 (stable behind): Text maintains 50% transparency\n")

session = new_session('u2net')

# Load video
cap = cv2.VideoCapture(input_video)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"[TEST] Video: {width}x{height} @ {fps}fps")

# Phases
phase1_frames = 30  # Shrink (text stays FULLY OPAQUE)
phase2_frames = 20  # Transition behind (text FADES here)
phase3_frames = 25  # Stable behind (text at 50% alpha)

# Show expected alpha values
print("\n[ALPHA TIMELINE]")
print(f"Frames 0-{phase1_frames}: Alpha = 1.0 (fully opaque during shrink)")
print(f"Frames {phase1_frames+1}-{phase1_frames+phase2_frames}: Alpha fades 1.0 → 0.5 (during transition)")
print(f"Frames {phase1_frames+phase2_frames+1}+: Alpha = 0.5 (stable behind)\n")

# Preview transition alpha curve
print("[TRANSITION FADE] Alpha values during phase 2:")
k = 3.0  # Same as in the fix
for i in range(0, phase2_frames + 1, 4):
    progress = i / phase2_frames if phase2_frames > 0 else 1
    exp_progress = (math.exp(k * progress) - 1) / (math.exp(k) - 1)
    target_alpha = 1.0 - (0.5 * exp_progress)
    abs_frame = phase1_frames + i
    print(f"  Frame {abs_frame:2d}: progress={progress:.2f} → alpha={target_alpha:.3f}")

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
print(f"\n[TEST] Getting handoff at frame {handoff_frame_idx}")
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
    randomize_order=True,
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
    
    # Update mask at phase transitions
    if frame_idx == 0 or frame_idx == phase1_frames or frame_idx == phase1_frames + phase2_frames:
        current_mask = remove(frame, session=session, only_mask=True)
        if frame_idx == phase1_frames:
            print(f"[PHASE] Transition to phase 2 at frame {frame_idx} - fading starts NOW")
        elif frame_idx == phase1_frames + phase2_frames:
            print(f"[PHASE] Transition to phase 3 at frame {frame_idx} - alpha stabilized at 0.5")
    
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
print("1. During shrink (phase 1): Text remains FULLY OPAQUE")
print("2. During transition (phase 2): Text fades from 100% to 50% opacity")  
print("3. Text ONLY starts fading when it actually passes behind subject")
print("4. No premature transparency during the shrink phase")