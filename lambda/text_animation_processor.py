#!/usr/bin/env python3
"""
Text Animation Processor for Lambda
Applies text animation effects to videos with dynamic mask generation via rembg
"""

import sys
import os
import numpy as np
import cv2
from pathlib import Path
from typing import Optional

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import animation modules
from utils.animations.text_behind_segment import TextBehindSegment
from utils.animations.word_dissolve import WordDissolve

# Import rembg for mask generation - REQUIRED
try:
    from rembg import remove, new_session
    from PIL import Image
    print("[LAMBDA_MANIFEST] rembg imported successfully")
    print("[LAMBDA_ANIM] rembg imported successfully")
except ImportError as e:
    print(f"[LAMBDA_MANIFEST] FATAL: rembg import failed: {e}")
    print(f"[LAMBDA_ANIM] FATAL: rembg import failed: {e}")
    print("[LAMBDA_ANIM] Please ensure Lambda Layer with rembg is attached")
    sys.exit(1)

# Initialize rembg session globally
try:
    print("[LAMBDA_MANIFEST] Preparing rembg session (u2net)")
    REMBG_SESSION = new_session('u2net')
    print("[LAMBDA_MANIFEST] rembg session initialized (u2net)")
    print("[LAMBDA_ANIM] rembg session initialized (u2net).")
except Exception as e:
    print(f"[LAMBDA_MANIFEST] FATAL: rembg session init failed: {e}")
    print(f"[LAMBDA_ANIM] FATAL: rembg session init failed: {e}")
    sys.exit(1)


def generate_mask_for_frame(frame: np.ndarray, frame_idx: int) -> np.ndarray:
    """
    Generate segmentation mask for a frame using rembg.
    Returns binary mask where foreground (subject) is white.
    """
    print(f"[MASK_DISSOLVE] mask refresh @frame={frame_idx} size={frame.shape[1]}x{frame.shape[0]}")
    
    # Convert frame to PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    # Apply rembg to get foreground with transparent background
    output = remove(pil_image, session=REMBG_SESSION)
    
    # Convert to numpy array and extract alpha channel as mask
    output_np = np.array(output)
    
    if output_np.shape[2] == 4:  # Has alpha channel
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
    return mask


def process_video(input_path, text, output_path, mask_refresh_interval: int = 6):
    """
    Process video with text animation combo.
    """
    print(f"[LAMBDA_ANIM] Processing: {input_path}")
    print(f"[LAMBDA_ANIM] Text: {text}")
    print(f"[LAMBDA_ANIM] Output: {output_path}")
    print(f"[LAMBDA_ANIM] Mask refresh interval: every {mask_refresh_interval} frames")
    
    # Open video to get properties
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    print(f"[LAMBDA_ANIM] Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Animation parameters
    center_position = (width // 2, int(height * 0.45))
    # Use environment variable for font size scaling
    font_rel = float(os.environ.get('TEXT_FONT_REL', '0.26'))
    font_size = int(height * font_rel)
    print(f"[LAMBDA_ANIM] font_px selected: {font_size} (rel={font_rel})")
    
    # Phase durations (in frames)
    phase1_frames = 30  # Shrinking
    phase2_frames = 20  # Moving behind
    phase3_frames = 40  # Stable behind
    dissolve_frames = 60  # Dissolve
    
    total_animation_frames = phase1_frames + phase2_frames + phase3_frames + dissolve_frames
    
    print(f"[LAMBDA_ANIM] Animation setup: center={center_position}, font_size={font_size}")
    print(f"[LAMBDA_ANIM] Animation phases: shrink={phase1_frames}, behind={phase2_frames}, stable={phase3_frames}, dissolve={dissolve_frames}")
    
    # Generate initial mask for the first frame
    cap_temp = cv2.VideoCapture(input_path)
    ret, first_frame = cap_temp.read()
    cap_temp.release()
    
    if not ret:
        raise ValueError("Cannot read first frame for mask generation")
    
    # Generate mask for occlusion effect
    initial_mask = generate_mask_for_frame(first_frame, 0)
    print(f"[LAMBDA_ANIM] Initial mask generated, shape: {initial_mask.shape}")
    
    # Create TextBehindSegment animation WITH MASK
    text_animator = TextBehindSegment(
        element_path=input_path,
        background_path=input_path,
        position=center_position,
        text=text,
        segment_mask=initial_mask,  # CRITICAL: Pass the mask!
        font_size=font_size,
        font_path=os.environ.get('TEXT_FONT_PATH'),  # Optional custom font
        text_color=(255, 220, 0),  # Yellow
        start_scale=2.0,
        end_scale=1.0,
        phase1_duration=phase1_frames / fps,
        phase2_duration=phase2_frames / fps,
        phase3_duration=phase3_frames / fps,
        center_position=center_position,
        fps=fps
    )
    print(f"[LAMBDA_ANIM] TextBehindSegment animator created with mask")
    
    # Get handoff data after stable phase
    handoff_frame_idx = phase1_frames + phase2_frames + phase3_frames - 1
    
    # Read a frame to establish handoff
    cap = cv2.VideoCapture(input_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Render the handoff frame to freeze the text state
        print(f"[LAMBDA_ANIM] Handoff pre-freeze rendered at frame={handoff_frame_idx}")
        _ = text_animator.render_text_frame(frame, handoff_frame_idx)
    
    handoff_data = text_animator.get_handoff_data()
    print(f"[LAMBDA_ANIM] Handoff data acquired: {len(handoff_data.get('final_letter_positions', []))} letters")
    print(f"[LAMBDA_ANIM] Letter positions: {handoff_data.get('final_letter_positions', [])}")
    
    # Create WordDissolve animation with handoff
    word_dissolver = WordDissolve(
        element_path=input_path,
        background_path=input_path,
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
    
    # Open video for processing
    cap = cv2.VideoCapture(input_path)
    
    # Setup video writer with H.264 codec
    # Use a temporary file first, then convert to H.264
    temp_output = output_path.replace('.mp4', '_temp.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise ValueError(f"Cannot create output video: {output_path}")
    
    frame_idx = 0
    frames_processed = 0
    current_mask = initial_mask  # Track current mask
    last_mask_update = 0
    
    print("[LAMBDA_ANIM] Processing frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB for processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Refresh mask periodically during occlusion phase
        if (frame_idx > phase1_frames and 
            frame_idx < phase1_frames + phase2_frames + phase3_frames and
            frame_idx - last_mask_update >= mask_refresh_interval):
            current_mask = generate_mask_for_frame(frame, frame_idx)
            text_animator.segment_mask = current_mask
            last_mask_update = frame_idx
        
        # Apply animations based on frame index
        if frame_idx < phase1_frames:
            # SHRINKING phase
            phase_name = "SHRINKING"
            frame_rgb = text_animator.render_text_frame(frame_rgb, frame_idx)
        elif frame_idx < phase1_frames + phase2_frames:
            # MOVING_BEHIND phase
            phase_name = "MOVING_BEHIND"
            frame_rgb = text_animator.render_text_frame(frame_rgb, frame_idx)
        elif frame_idx < phase1_frames + phase2_frames + phase3_frames:
            # STABLE_BEHIND phase
            phase_name = "STABLE_BEHIND"
            frame_rgb = text_animator.render_text_frame(frame_rgb, frame_idx)
        elif frame_idx < total_animation_frames:
            # DISSOLVING phase
            phase_name = "DISSOLVING"
            dissolve_frame = frame_idx - (phase1_frames + phase2_frames + phase3_frames)
            # Convert mask to boolean for WordDissolve
            current_mask_bool = current_mask > 127 if current_mask is not None else None
            frame_rgb = word_dissolver.render_word_frame(frame_rgb, dissolve_frame, mask=current_mask_bool)
            
            # Refresh mask periodically during dissolve
            if dissolve_frame > 0 and dissolve_frame % mask_refresh_interval == 0:
                current_mask = generate_mask_for_frame(frame, frame_idx)
                print(f"[MASK_DISSOLVE] Mask updated during dissolve at frame {frame_idx}")
        else:
            phase_name = "COMPLETE"
        
        # Convert back to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
        
        frames_processed += 1
        frame_idx += 1
        
        # Stop after animation completes + 1 second buffer
        if frame_idx >= total_animation_frames + fps:
            break
        
        # Progress indicator with phase info
        if frames_processed % 30 == 0:
            mask_status = "with mask" if current_mask is not None else "no mask"
            print(f"[MASK_DISSOLVE] Progress: {frames_processed} frames... phase={phase_name} ({mask_status})")
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"[MASK_DISSOLVE] ✓ Processing complete! {frames_processed} frames written")
    print(f"[MASK_DISSOLVE] Animation phases completed: shrink→behind→stable→dissolve")
    
    # Convert to H.264 for web compatibility using imageio-ffmpeg
    print("[LAMBDA_ANIM] Converting to H.264 for web compatibility...")
    import imageio_ffmpeg as ffmpeg
    import subprocess
    
    # Get the ffmpeg executable from imageio-ffmpeg
    ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
    
    # Use ffmpeg to convert to H.264 with web-compatible settings
    ffmpeg_cmd = [
        ffmpeg_exe,
        '-i', temp_output,
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',  # Enable fast start for web streaming
        '-y',  # Overwrite output
        output_path
    ]
    
    try:
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
        print("[LAMBDA_ANIM] ✓ Successfully converted to H.264")
    except subprocess.CalledProcessError as e:
        print(f"[LAMBDA_ANIM] FFmpeg error: {e.stderr}")
        # If ffmpeg fails, just rename the temp file
        os.rename(temp_output, output_path)
        print("[LAMBDA_ANIM] Warning: Using fallback format (may not be web-compatible)")
    except Exception as e:
        print(f"[LAMBDA_ANIM] Conversion error: {str(e)}")
        # If anything fails, just rename the temp file
        if os.path.exists(temp_output):
            os.rename(temp_output, output_path)
        print("[LAMBDA_ANIM] Warning: Using fallback format (may not be web-compatible)")
    else:
        # Remove temporary file
        if os.path.exists(temp_output):
            os.remove(temp_output)
    
    # Verify output file
    if not os.path.exists(output_path):
        raise ValueError(f"Output file was not created: {output_path}")
    
    output_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[LAMBDA_ANIM] ✓ Output file size: {output_size:.2f} MB")


def main():
    """Main entry point for command-line usage."""
    if len(sys.argv) != 4:
        print("Usage: python text_animation_processor.py <input_video> <text> <output_video>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    text = sys.argv[2].upper()
    output_path = sys.argv[3]
    
    if not os.path.exists(input_path):
        print(f"Error: Input video not found: {input_path}")
        sys.exit(1)
    
    try:
        process_video(input_path, text, output_path)
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()