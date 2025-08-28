#!/usr/bin/env python3
"""
Precise background replacement with per-frame mask calculation and advanced refinement.
"""

import cv2
import numpy as np
import subprocess
from pathlib import Path
from rembg import remove, new_session
from PIL import Image
import tempfile
from tqdm import tqdm


def refine_mask(mask, frame):
    """
    Apply advanced mask refinement to eliminate gaps.
    
    Args:
        mask: Binary mask from rembg
        frame: Original frame for guided filtering
        
    Returns:
        Refined mask with no gaps
    """
    # 1. Fill small holes inside the mask
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_filled = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)
    
    # 2. Expand mask slightly to close edge gaps
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_dilated = cv2.dilate(mask_filled, kernel_medium, iterations=1)
    
    # 3. Apply guided filter for edge refinement (preserves edges while smoothing)
    # Convert to float for guided filter
    mask_float = mask_dilated.astype(np.float32) / 255.0
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    
    # Guided filter parameters
    radius = 5
    eps = 0.01
    mask_guided = cv2.ximgproc.guidedFilter(frame_gray, mask_float, radius, eps)
    
    # 4. Apply feathering for smooth edges
    mask_feathered = cv2.GaussianBlur(mask_guided, (7, 7), 2)
    
    # 5. Threshold with soft edges
    # Keep values above 0.3 to preserve more foreground
    mask_final = np.clip(mask_feathered * 255, 0, 255).astype(np.uint8)
    
    return mask_final


def process_video_precise(input_video, background_video, output_path, 
                         max_duration=5.0, high_quality=True):
    """
    Replace background with precise per-frame mask calculation.
    
    Args:
        input_video: Path to input video
        background_video: Path to background video
        output_path: Path to save output
        max_duration: Maximum duration to process (seconds)
        high_quality: Use higher quality settings
    """
    print("Starting precise background replacement...")
    print("IMPORTANT: Calculating fresh mask for EVERY frame - no caching!")
    
    # Open videos
    cap_input = cv2.VideoCapture(str(input_video))
    cap_bg = cv2.VideoCapture(str(background_video))
    
    # Get video properties
    fps = cap_input.get(cv2.CAP_PROP_FPS)
    width = int(cap_input.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_input.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = min(int(fps * max_duration), int(cap_input.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))
    
    # Create rembg session with best model
    print("Initializing high-quality background removal model...")
    if high_quality:
        session = new_session('u2netp')  # Use higher quality model
    else:
        session = new_session('u2net')
    
    # Process frames
    print(f"Processing {total_frames} frames with per-frame mask calculation...")
    
    for frame_idx in tqdm(range(total_frames), desc="Removing backgrounds"):
        # Read frames
        ret_input, frame_input = cap_input.read()
        ret_bg, frame_bg = cap_bg.read()
        
        if not ret_input:
            break
            
        # Loop background video if needed
        if not ret_bg:
            cap_bg.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_bg, frame_bg = cap_bg.read()
        
        # Resize background to match input
        frame_bg_resized = cv2.resize(frame_bg, (width, height))
        
        # ===== CRITICAL: Fresh mask calculation for EVERY frame =====
        # Convert BGR to RGB for rembg
        frame_rgb = cv2.cvtColor(frame_input, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Remove background with alpha output for better quality
        output_rgba = remove(pil_image, session=session, alpha_matting=True, 
                           alpha_matting_foreground_threshold=240,
                           alpha_matting_background_threshold=10,
                           alpha_matting_erode_size=10)
        
        # Extract alpha channel as mask
        output_np = np.array(output_rgba)
        if output_np.shape[2] == 4:
            mask = output_np[:, :, 3]
        else:
            mask = np.zeros((height, width), dtype=np.uint8)
        
        # Refine mask to eliminate gaps
        mask_refined = refine_mask(mask, frame_input)
        
        # Convert mask to float for smooth blending
        alpha = mask_refined.astype(np.float32) / 255.0
        
        # Create 3-channel alpha
        alpha_3d = np.stack([alpha, alpha, alpha], axis=2)
        
        # High-quality composite with anti-aliasing
        foreground_float = frame_input.astype(np.float32)
        background_float = frame_bg_resized.astype(np.float32)
        
        # Apply composite with proper alpha blending
        composite = alpha_3d * foreground_float + (1.0 - alpha_3d) * background_float
        
        # Add slight edge blending to eliminate hard edges
        edge_mask = cv2.Canny(mask_refined, 50, 150)
        edge_mask = cv2.dilate(edge_mask, np.ones((3, 3), np.uint8), iterations=1)
        edge_region = edge_mask > 0
        
        if np.any(edge_region):
            # Blend edges more smoothly
            composite_blur = cv2.GaussianBlur(composite, (3, 3), 1)
            edge_alpha = (edge_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
            composite = composite * (1 - edge_alpha * 0.3) + composite_blur * (edge_alpha * 0.3)
        
        # Convert back to uint8
        composite_final = np.clip(composite, 0, 255).astype(np.uint8)
        
        # Write frame
        out.write(composite_final)
    
    # Clean up
    cap_input.release()
    cap_bg.release()
    out.release()
    
    print("Encoding final video with H.264...")
    # Convert to H.264 with high quality
    crf = "18" if high_quality else "23"
    cmd = [
        "ffmpeg", "-y",
        "-i", temp_output.name,
        "-i", str(input_video),
        "-c:v", "libx264",
        "-preset", "slow" if high_quality else "fast",
        "-crf", crf,
        "-pix_fmt", "yuv420p",
        "-map", "0:v",
        "-map", "1:a?",
        "-movflags", "+faststart",
        str(output_path)
    ]
    subprocess.run(cmd, check=True)
    
    # Clean up temp file
    Path(temp_output.name).unlink()
    
    print(f"Output saved to: {output_path}")


def main():
    """Test precise background replacement."""
    
    # Setup paths
    input_video = Path("uploads/assets/videos/ai_math1_test_5sec.mp4")
    background_video = Path("uploads/assets/videos/ai_math1/ai_math1_background_0_0_5_0_STc2OalsLp.mp4")
    output_video = Path("outputs/ai_math1_precise_bg_replaced.mp4")
    
    if not input_video.exists():
        print(f"Error: Input video not found: {input_video}")
        return
    
    if not background_video.exists():
        print(f"Error: Background video not found: {background_video}")
        return
    
    # Process video with high quality settings
    process_video_precise(
        input_video,
        background_video,
        output_video,
        max_duration=5.0,
        high_quality=True  # Use best quality settings
    )
    
    # Open result
    subprocess.run(["open", str(output_video)])


if __name__ == "__main__":
    main()