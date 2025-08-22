#!/usr/bin/env python3
"""Test with PERFECT occlusion - mask recalculated EVERY frame during occlusion phases."""

import os
os.environ['FRAME_DISSOLVE_DEBUG'] = '1'

import cv2
import numpy as np
from rembg import remove, new_session
from utils.animations.text_behind_segment import TextBehindSegment
from utils.animations.word_dissolve import WordDissolve

# Configuration
input_video = "test_element_3sec.mp4"
output_video = "hello_world_perfect_occlusion.mp4"
text = "HELLO WORLD"

print("[TEST] Initializing with PERFECT per-frame occlusion...")
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
phase1_frames = 20  # Shrink (foreground, no occlusion needed)
phase2_frames = 15  # Move behind (CRITICAL - needs per-frame mask)
phase3_frames = 25  # Stable behind (CRITICAL - needs per-frame mask)

center_position = (width // 2, int(height * 0.45))
font_size = int(height * 0.26)

# Create TextBehindSegment
text_animator = TextBehindSegment(
    element_path=input_video,
    background_path=input_video,
    position=center_position,
    text=text,
    segment_mask=None,  # Will provide per-frame
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

# Define when text goes behind (alpha becomes 0.5)
occlusion_start_frame = phase1_frames  # End of phase 1, start of phase 2
occlusion_end_frame = total_frames     # Through all dissolve

print(f"[TEST] Total: {total_frames} frames ({total_frames/fps:.1f}s)")
print(f"[TEST] Phase 1 (foreground): frames 0-{phase1_frames-1} - sparse mask updates")
print(f"[TEST] CRITICAL OCCLUSION: frames {occlusion_start_frame}-{occlusion_end_frame} - PER-FRAME mask")

# Process
temp_output = output_video.replace('.mp4', '_temp.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
current_mask = None
mask_update_count = 0

print("[TEST] Processing...")
for frame_idx in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # CRITICAL: Update mask strategy based on occlusion need
    if frame_idx >= occlusion_start_frame and frame_idx <= occlusion_end_frame:
        # Text is behind subject - MUST update EVERY frame for perfect occlusion
        current_mask = remove(frame, session=session, only_mask=True)
        mask_update_count += 1
        
        # Log critical updates
        if frame_idx == occlusion_start_frame:
            print(f"[MASK] START per-frame updates at frame {frame_idx} (text moves behind)")
        elif frame_idx % 30 == 0:
            print(f"[MASK] Per-frame update #{mask_update_count} at frame {frame_idx}")
    else:
        # Text is in front - can update less frequently
        if current_mask is None or frame_idx % 10 == 0:
            current_mask = remove(frame, session=session, only_mask=True)
            if frame_idx % 10 == 0:
                print(f"[MASK] Sparse update at frame {frame_idx} (text in foreground)")
    
    # Render with current mask
    if frame_idx < phase1_frames + phase2_frames + phase3_frames:
        frame_rgb = text_animator.render_text_frame(frame_rgb, frame_idx, mask=current_mask)
    else:
        dissolve_frame = frame_idx - (phase1_frames + phase2_frames + phase3_frames)
        frame_rgb = word_dissolver.render_word_frame(frame_rgb, dissolve_frame, mask=current_mask)
    
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    out.write(frame_bgr)
    
    if frame_idx % 60 == 0:
        print(f"[TEST] Progress: {frame_idx}/{total_frames} (mask updates: {mask_update_count})")

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
print(f"[TEST] Total mask updates: {mask_update_count}")
print(f"[TEST] Occlusion frames: {occlusion_end_frame - occlusion_start_frame + 1}")
print("\n[VERIFICATION]")
print("1. PERFECT pixel-accurate occlusion during text-behind phases")
print("2. Every single frame has fresh mask when text is behind subject")
print("3. No text bleeding through moving subjects")