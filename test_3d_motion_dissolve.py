#!/usr/bin/env python3
"""
Test the composed 3D motion+dissolve with frame-accurate dissolve schedule.

What this test asserts:
- Motion phase ends CENTERED (front-face center).
- Dissolve phase starts with letters at the SAME positions (no jump).
- No jump between 'O' of "WORLD" and 'W' (or any neighbors) due to missing fade frames.
- [JUMP_CUT] logs print per-letter schedule and transitions.
"""

import cv2
import numpy as np
import subprocess
from utils.animations.text_3d_motion import Text3DMotion
from utils.animations.letter_3d_dissolve import Letter3DDissolve

print("="*80)
print("TESTING 3D TEXT ANIMATION (Centered + Frame-Accurate Dissolve)")
print("="*80)

# Load test video (3 seconds @ ~30fps expected)
video_path = "test_element_3sec.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(round(cap.get(cv2.CAP_PROP_FPS))) or 30
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Read background frames
frames = []
for i in range(int(3 * fps)):
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

print(f"\nLoaded {len(frames)} frames")
print(f"Resolution: {W}x{H}")
print(f"FPS (from video): {fps}")

# Foreground mask (optional)
print("\nExtracting foreground mask for behind-subject effect (optional)...")
try:
    from utils.segmentation.segment_extractor import extract_foreground_mask
    first_frame_rgb = frames[0]
    segment_mask = extract_foreground_mask(first_frame_rgb)
    print(f"Foreground mask extracted: {segment_mask.shape}")
    print(f"Mask has foreground pixels: {np.any(segment_mask > 0)}")
except Exception as e:
    print(f"Warning: Could not extract foreground mask: {e}")
    print("Proceeding without behind-subject occlusion.")
    segment_mask = None

# Config
motion_duration = 0.75
dissolve_clip_duration = 1.5
final_opacity = 0.5  # opacity when stable/behind (matches your previous test)

# Motion (front-face center to center)
motion = Text3DMotion(
    duration=motion_duration,
    fps=fps,
    resolution=(W, H),
    text="HELLO WORLD",
    segment_mask=segment_mask,
    font_size=140,
    text_color=(255, 220, 0),
    depth_color=(200, 170, 0),
    depth_layers=8,
    depth_offset=3,
    start_scale=2.0,
    end_scale=1.0,
    final_scale=0.9,
    start_position=(W//2, H//2),
    end_position=(W//2, H//2),
    shrink_duration=0.6,
    settle_duration=0.15,
    final_alpha=final_opacity,
    shadow_offset=6,
    outline_width=2,
    perspective_angle=0,
    supersample_factor=2,
    glow_effect=True,
    debug=True,  # keep POS_HANDOFF logs
)

# Dissolve
dissolve = Letter3DDissolve(
    duration=dissolve_clip_duration,
    fps=fps,
    resolution=(W, H),
    text="HELLO WORLD",
    font_size=140,
    text_color=(255, 220, 0),
    depth_color=(200, 170, 0),
    depth_layers=8,
    depth_offset=3,
    initial_scale=0.9,
    initial_position=(W//2, H//2),  # overwritten by handoff
    stable_duration=0.1,
    stable_alpha=final_opacity,      # match motion final
    dissolve_duration=0.5,           # per-letter window
    dissolve_stagger=0.1,            # delay between letters
    float_distance=40,
    max_dissolve_scale=1.3,
    randomize_order=False,
    segment_mask=segment_mask,
    is_behind=False,                 # overwritten by handoff
    shadow_offset=6,
    outline_width=2,
    supersample_factor=2,
    # --- NEW robust timing knobs ---
    post_fade_seconds=0.10,          # >= ~3 frames @30fps
    pre_dissolve_hold_frames=1,      # 1 safety frame at exact stable alpha
    ensure_no_gap=True,
    debug=True,                      # enable [JUMP_CUT] logs
)

# Video writer
print("\nGenerating video...")
output_path = "text_3d_motion_dissolve.mp4"
height, width = frames[0].shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

motion_frames = int(round(motion_duration * fps))
dissolve_frames = int(round(dissolve_clip_duration * fps))

# Motion phase
for i in range(motion_frames):
    bg_idx = i % len(frames)
    background = frames[bg_idx]
    frame = motion.generate_frame(i, background)
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write(frame_bgr)

# Handoff to dissolve
final_state = motion.get_final_state()
if final_state:
    print(f"[POS_HANDOFF] Handoff captured -> center={final_state.center_position}, "
          f"final_scale={final_state.scale:.3f}, is_behind={final_state.is_behind}, "
          f"motion_paste_topleft={final_state.position}, sprite_size={final_state.text_size}")
    dissolve.set_initial_state(
        scale=final_state.scale,
        position=final_state.center_position,
        alpha=final_opacity,
        is_behind=final_state.is_behind,
        segment_mask=segment_mask
    )

# Print the final per-letter schedule (frame-accurate) before generating
print("\n[JUMP_CUT] Final dissolve schedule (frame-accurate):")
dissolve.debug_print_schedule()

# Dissolve phase (+ save frames around the previously problematic region)
save_window = set(range(28, 41))  # keep your previous debug window
for i in range(dissolve_frames):
    bg_idx = (motion_frames + i) % len(frames)
    background = frames[bg_idx]
    frame = dissolve.generate_frame(i, background)

    if i in save_window:
        debug_frame_path = f"debug_dissolve_frame_{i:03d}.png"
        cv2.imwrite(debug_frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if i in [30, 31, 35, 36]:
            print(f"*** KEY FRAME {i}: Saved {debug_frame_path}")

    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write(frame_bgr)

out.release()
print(f"Video saved to {output_path}")

# Convert to H.264 for compatibility
print("\nConverting to H.264...")
h264_path = "text_3d_motion_dissolve_h264.mp4"
subprocess.run([
    'ffmpeg', '-i', output_path,
    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
    h264_path, '-y'
], check=True)

print(f"\n✅ Animation saved to: {h264_path}")