#!/usr/bin/env python3
"""
Test TextBehindSegment animation locally (without Lambda)
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

# Import rembg for mask generation
from rembg import remove, new_session
from PIL import Image

print("[OCCLUSION] Initializing rembg session...")
REMBG_SESSION = new_session('u2net')
print("[OCCLUSION] rembg session initialized.")

# Optional: recompute mask every N frames if subject moves (0 disables)
RECOMPUTE_MASK_EVERY_N = 0  # set to e.g. 10 if your subject moves significantly


def generate_mask_for_frame(frame: np.ndarray) -> np.ndarray:
    """Generate segmentation mask for a frame using rembg."""
    print(f"[OCCLUSION] Generating mask for frame with shape={frame.shape}")

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

    return mask  # uint8 in [0,255], 255=subject, 0=background


def test_animation():
    """Test the text animation locally."""
    input_video = "test_element_3sec.mp4"
    output_video = "test_local_animation.mp4"
    text = "HELLO WORLD"

    if not os.path.exists(input_video):
        print(f"[OCCLUSION] Error: {input_video} not found!")
        return

    print(f"[OCCLUSION] Input video: {input_video}")
    print(f"[OCCLUSION] Text: {text}")

    # Open video to get properties
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"[OCCLUSION] Cannot open video: {input_video}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[OCCLUSION] Video: {width}x{height} @ {fps}fps, {total_frames} frames")

    # Get first frame for mask
    ret, first_frame = cap.read()
    if not ret:
        print("[OCCLUSION] Cannot read first frame")
        cap.release()
        return

    # Generate mask
    print("[OCCLUSION] Generating mask from first frame...")
    mask = generate_mask_for_frame(first_frame)

    # Ensure mask matches video size
    if mask.shape[0] != height or mask.shape[1] != width:
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)

    # Coverage diagnostics
    coverage = float((mask > 0).mean())
    print(f"[OCCLUSION] Mask generated: shape={mask.shape} coverage(subject ratio)={coverage:.3f}")

    # Animation parameters
    center_position = (width // 2, int(height * 0.45))
    font_size = int(height * 0.26)

    # Phase durations (in frames)
    phase1_frames = 30  # Shrinking
    phase2_frames = 20  # Moving behind
    phase3_frames = 40  # Stable behind

    print(f"\n[OCCLUSION] Animation setup:")
    print(f"[OCCLUSION]   Center: {center_position}")
    print(f"[OCCLUSION]   Font size: {font_size}")
    print(f"[OCCLUSION]   Phases: shrink={phase1_frames}, behind={phase2_frames}, stable={phase3_frames}")
    print(f"[OCCLUSION]   Scale: 2.0 → 1.0")

    # Create TextBehindSegment animation
    text_animator = TextBehindSegment(
        element_path=input_video,
        background_path=input_video,
        position=center_position,
        text=text,
        segment_mask=mask,             # Store on the instance (also passed per-frame below)
        font_size=font_size,
        font_path=None,
        text_color=(255, 220, 0),      # Yellow
        start_scale=2.0,               # Start at 2x size
        end_scale=1.0,                 # End at normal size
        phase1_duration=phase1_frames / fps,
        phase2_duration=phase2_frames / fps,
        phase3_duration=phase3_frames / fps,
        center_position=center_position,
        fps=fps
    )

    print("[OCCLUSION] TextBehindSegment animator created")

    # Rewind to frame 0 for processing
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Setup video writer - use temporary file first
    temp_output = output_video.replace('.mp4', '_temp.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    frame_idx = 0
    print("\n[OCCLUSION] Processing frames...")

    while frame_idx < 90:  # Process first 90 frames to see shrink + behind phases
        ret, frame = cap.read()
        if not ret:
            break

        # Optionally keep mask in sync if there is motion
        if RECOMPUTE_MASK_EVERY_N > 0 and (frame_idx % RECOMPUTE_MASK_EVERY_N == 0):
            mask = generate_mask_for_frame(frame)
            if mask.shape[0] != height or mask.shape[1] != width:
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
            cov = float((mask > 0).mean())
            print(f"[OCCLUSION] REGEN mask at frame={frame_idx} coverage={cov:.3f}")

        # Convert BGR to RGB for processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply animation (IMPORTANT: pass mask here)
        frame_rgb = text_animator.render_text_frame(frame_rgb, frame_idx, mask=mask)

        # Convert back to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

        # Progress
        if frame_idx % 10 == 0:
            print(f"[OCCLUSION] Frame {frame_idx} processed")

        frame_idx += 1

    # Release resources
    cap.release()
    out.release()

    # Convert to H.264 for compatibility
    print("\n[OCCLUSION] Converting to H.264...")
    import subprocess
    try:
        # Use ffmpeg to convert to H.264
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
        os.remove(temp_output)  # Remove temp file
        print("[OCCLUSION] ✅ Converted to H.264 format")
    except subprocess.CalledProcessError:
        print("[OCCLUSION] Warning: FFmpeg conversion failed, using original format")
        os.replace(temp_output, output_video)
    except FileNotFoundError:
        print("[OCCLUSION] Warning: ffmpeg not found, using original format")
        os.replace(temp_output, output_video)

    print(f"\n[OCCLUSION] ✅ Animation complete! Output: {output_video}")
    print("[OCCLUSION] Expected behavior:")
    print("[OCCLUSION]   Frames 0-29: Text starts LARGE (2x) and shrinks to 1.3x (in FRONT)")
    print("[OCCLUSION]   Frames 30-49: Text continues shrinking to 1x and begins hiding BEHIND subject")
    print("[OCCLUSION]   Frames 50-89: Text stays at 1x size BEHIND the subject")

if __name__ == "__main__":
    test_animation()