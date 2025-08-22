#!/usr/bin/env python3
"""
Test FULL animation pipeline: TextBehindSegment + WordDissolve
Tests the complete two-step animation framework with mask updates
"""
import sys
import os
import numpy as np
import cv2
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import animation modules
from utils.animations.text_behind_segment import TextBehindSegment
from utils.animations.word_dissolve import WordDissolve

# Import rembg for mask generation
from rembg import remove, new_session
from PIL import Image

print("[MASK_DISSOLVE] Initializing rembg session...")
REMBG_SESSION = new_session('u2net')
print("[MASK_DISSOLVE] rembg session initialized.")

# Mask refresh interval (frames) - MUST BE 1 for scenes with multiple people!
RECOMPUTE_MASK_EVERY_N = 1


def generate_mask_for_frame(frame: np.ndarray, frame_idx: int) -> np.ndarray:
    """Generate segmentation mask for a frame using rembg."""
    print(f"[MASK_DISSOLVE] mask refresh @frame={frame_idx} size={frame.shape[1]}x{frame.shape[0]}")

    # Convert frame (BGR) to PIL Image (RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # Apply rembg to get foreground with transparent background
    output = remove(pil_image, session=REMBG_SESSION)

    # Convert to numpy array and extract alpha channel as mask
    output_np = np.array(output)

    if output_np.ndim == 3 and output_np.shape[2] == 4:  # Has alpha channel
        mask = output_np[:, :, 3]  # Extract alpha
    else:
        # Fallback: create mask from non-black pixels
        gray = cv2.cvtColor(output_np, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Denoise mask
    mask = cv2.medianBlur(mask, 5)
    
    # Ensure binary values
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    print(f"[MASK_DISSOLVE] Mask coverage: {(mask > 0).mean():.2%}")
    return mask  # uint8 in [0,255], 255=subject, 0=background


def test_full_pipeline():
    """Test the complete animation pipeline."""
    input_video = "test_element_3sec.mp4"
    output_video = "test_full_animation_pipeline.mp4"
    text = "HELLO WORLD"

    if not os.path.exists(input_video):
        print(f"[MASK_DISSOLVE] Error: {input_video} not found!")
        return

    print(f"[MASK_DISSOLVE] Testing FULL animation pipeline")
    print(f"[MASK_DISSOLVE] Input video: {input_video}")
    print(f"[MASK_DISSOLVE] Text: {text}")

    # Open video to get properties
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"[MASK_DISSOLVE] Cannot open video: {input_video}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[MASK_DISSOLVE] Video: {width}x{height} @ {fps}fps, {total_frames} frames")

    # Get first frame for initial mask
    ret, first_frame = cap.read()
    if not ret:
        print("[MASK_DISSOLVE] Cannot read first frame")
        cap.release()
        return

    # Generate initial mask
    print("[MASK_DISSOLVE] Generating initial mask...")
    initial_mask = generate_mask_for_frame(first_frame, 0)

    # Ensure mask matches video size
    if initial_mask.shape[0] != height or initial_mask.shape[1] != width:
        initial_mask = cv2.resize(initial_mask, (width, height), interpolation=cv2.INTER_LINEAR)

    # Animation parameters
    center_position = (width // 2, int(height * 0.45))
    font_size = int(height * 0.26)

    # Phase durations (in frames)
    phase1_frames = 30  # Shrinking
    phase2_frames = 20  # Moving behind
    phase3_frames = 40  # Stable behind
    dissolve_frames = 60  # Dissolve

    # --- IMPORTANT: use wd_total_frames instead of a fixed 'dissolve_frames' ---
    # Note: wd_total_frames will be computed after creating word_dissolver
    # For now, use dissolve_frames as placeholder
    total_animation_frames_initial = phase1_frames + phase2_frames + phase3_frames + dissolve_frames

    print(f"\n[MASK_DISSOLVE] Animation setup:")
    print(f"[MASK_DISSOLVE]   Center: {center_position}")
    print(f"[MASK_DISSOLVE]   Font size: {font_size}")
    print(f"[MASK_DISSOLVE]   Phases: shrink={phase1_frames}, behind={phase2_frames}, stable={phase3_frames}, dissolve={dissolve_frames}")
    print(f"[MASK_DISSOLVE]   Total animation (initial): {total_animation_frames_initial} frames")
    print(f"[MASK_DISSOLVE]   Mask refresh: every {RECOMPUTE_MASK_EVERY_N} frames")

    # Create TextBehindSegment animation
    text_animator = TextBehindSegment(
        element_path=input_video,
        background_path=input_video,
        position=center_position,
        text=text,
        segment_mask=initial_mask,
        font_size=font_size,
        font_path=None,
        text_color=(255, 220, 0),  # Yellow
        start_scale=2.0,
        end_scale=1.0,
        phase1_duration=phase1_frames / fps,
        phase2_duration=phase2_frames / fps,
        phase3_duration=phase3_frames / fps,
        center_position=center_position,
        fps=fps
    )
    print("[MASK_DISSOLVE] TextBehindSegment animator created")

    # Get handoff data for WordDissolve
    handoff_frame_idx = phase1_frames + phase2_frames + phase3_frames - 1
    
    # Render the handoff frame to establish text state
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    if ret:
        print(f"[MASK_DISSOLVE] Rendering handoff frame at index {handoff_frame_idx}")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _ = text_animator.render_text_frame(frame_rgb, handoff_frame_idx)
    
    handoff_data = text_animator.get_handoff_data()
    print(f"[MASK_DISSOLVE] Handoff data acquired: {len(handoff_data.get('final_letter_positions', []))} letters")

    # Create WordDissolve animation
    word_dissolver = WordDissolve(
        element_path=input_video,
        background_path=input_video,
        position=center_position,
        word=text,
        font_size=font_size,
        text_color=(255, 220, 0),
        stable_duration=0.17,  # 5 frames at 30fps
        dissolve_duration=2.0,  # 60 frames at 30fps
        dissolve_stagger=0.5,
        float_distance=30,
        randomize_order=False,
        maintain_kerning=True,
        center_position=center_position,
        handoff_data=handoff_data,
        fps=fps
    )
    print("[MASK_DISSOLVE] WordDissolve animator created")
    
    # --- NEW: compute true WD length based on active letters ---
    active_letters = sum(1 for s in word_dissolver.letter_sprites if s is not None)
    wd_stable   = word_dissolver.stable_frames
    wd_stagger  = word_dissolver.stagger_frames
    wd_dissolve = word_dissolver.dissolve_frames
    wd_total_frames = wd_stable + max(0, active_letters - 1) * wd_stagger + wd_dissolve
    
    print(f"[DISSOLVE_TIMING] WD frames: stable={wd_stable} stagger={wd_stagger} "
          f"dissolve={wd_dissolve} active_letters={active_letters} "
          f"-> wd_total_frames={wd_total_frames}")
    
    # First + last schedule (skip empty sprites like spaces)
    ordered = [i for i in word_dissolver.letter_indices if word_dissolver.letter_sprites[i] is not None]
    if ordered:
        first_i = ordered[0]; last_i = ordered[-1]
        first_start = wd_stable + ordered.index(first_i) * wd_stagger
        last_start  = wd_stable + (len(ordered) - 1) * wd_stagger
        print(f"[DISSOLVE_TIMING] first='{text[first_i]}' start={first_start} "
              f"last='{text[last_i]}' start={last_start} end={last_start + wd_dissolve}")
    
    # Update total animation frames with actual WD length
    total_animation_frames = phase1_frames + phase2_frames + phase3_frames + wd_total_frames
    print(f"[DISSOLVE_TIMING] Updated total animation frames: {total_animation_frames}")

    # Rewind for processing
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Setup video writer
    temp_output = output_video.replace('.mp4', '_temp.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    frame_idx = 0
    current_mask = initial_mask
    last_mask_update = 0
    in_occluded = False

    print("\n[MASK_DISSOLVE] Processing frames...")

    while frame_idx < total_animation_frames + fps:  # Process animation + 1 second buffer
        ret, frame = cap.read()
        if not ret:
            # Loop the video if we run out of frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        # Track when we enter occluded phases
        if frame_idx >= phase1_frames and not in_occluded:
            in_occluded = True
            print(f"[MASK_DISSOLVE] Entering occluded phase at frame {frame_idx}")

        # Refresh mask periodically during occluded phases
        if in_occluded and (frame_idx - last_mask_update) >= RECOMPUTE_MASK_EVERY_N:
            current_mask = generate_mask_for_frame(frame, frame_idx)
            if current_mask.shape[0] != height or current_mask.shape[1] != width:
                current_mask = cv2.resize(current_mask, (width, height), interpolation=cv2.INTER_LINEAR)
            # Update animator's mask
            text_animator.segment_mask = current_mask
            last_mask_update = frame_idx

        # Convert BGR to RGB for processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply animations based on frame index
        if frame_idx < phase1_frames:
            # SHRINKING phase
            phase_name = "SHRINKING"
            frame_rgb = text_animator.render_text_frame(frame_rgb, frame_idx, mask=current_mask)
        elif frame_idx < phase1_frames + phase2_frames:
            # MOVING_BEHIND phase
            phase_name = "MOVING_BEHIND"
            frame_rgb = text_animator.render_text_frame(frame_rgb, frame_idx, mask=current_mask)
        elif frame_idx < phase1_frames + phase2_frames + phase3_frames:
            # STABLE_BEHIND phase
            phase_name = "STABLE_BEHIND"
            frame_rgb = text_animator.render_text_frame(frame_rgb, frame_idx, mask=current_mask)
        elif frame_idx < total_animation_frames:
            # DISSOLVING phase
            phase_name = "DISSOLVING"
            dissolve_frame = frame_idx - (phase1_frames + phase2_frames + phase3_frames)
            # Convert mask to boolean for WordDissolve
            # Pass raw mask (WD normalizes it internally)
            frame_rgb = word_dissolver.render_word_frame(frame_rgb, dissolve_frame, mask=current_mask)
            
            # Refresh mask periodically during dissolve
            if dissolve_frame > 0 and dissolve_frame % RECOMPUTE_MASK_EVERY_N == 0:
                current_mask = generate_mask_for_frame(frame, frame_idx)
                print(f"[MASK_DISSOLVE] Mask updated during dissolve at frame {frame_idx}")
        else:
            phase_name = "COMPLETE"

        # Convert back to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

        # Progress with phase info
        if frame_idx % 30 == 0:
            mask_status = "with mask" if current_mask is not None else "no mask"
            print(f"[MASK_DISSOLVE] Progress: frame {frame_idx}... phase={phase_name} ({mask_status})")

        frame_idx += 1

    # Release resources
    cap.release()
    out.release()

    print(f"\n[MASK_DISSOLVE] ✓ Processing complete! {frame_idx} frames written")
    print(f"[MASK_DISSOLVE] Animation phases completed: shrink→behind→stable→dissolve")

    # Convert to H.264 for compatibility
    print("\n[MASK_DISSOLVE] Converting to H.264...")
    import subprocess
    try:
        ffmpeg_cmd = [
            'ffmpeg', '-i', temp_output,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-y', output_video
        ]
        subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
        os.remove(temp_output)
        print("[MASK_DISSOLVE] ✅ Converted to H.264 format")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[MASK_DISSOLVE] Warning: FFmpeg conversion failed, using original format")
        os.replace(temp_output, output_video)

    print(f"\n[MASK_DISSOLVE] ✅ Full animation pipeline complete!")
    print(f"[MASK_DISSOLVE] Output: {output_video}")
    print("\n[MASK_DISSOLVE] Expected behavior:")
    print("[MASK_DISSOLVE]   Frames 0-29: Text SHRINKING from 2x to 1.3x (in FRONT)")
    print("[MASK_DISSOLVE]   Frames 30-49: Text MOVING BEHIND subject")
    print("[MASK_DISSOLVE]   Frames 50-89: Text STABLE BEHIND subject")
    print("[MASK_DISSOLVE]   Frames 90-149: Text DISSOLVING letter by letter")
    print(f"\n[MASK_DISSOLVE] Mask was refreshed every {RECOMPUTE_MASK_EVERY_N} frames during occluded phases")


if __name__ == "__main__":
    test_full_pipeline()