#!/usr/bin/env python3
"""
Fast background replacement using batch processing and optimizations.
"""

import cv2
import numpy as np
import subprocess
from pathlib import Path
from rembg import remove, new_session
from PIL import Image
import tempfile
from tqdm import tqdm


def process_video_with_bg_replacement(input_video, background_video, output_path, 
                                     max_duration=5.0, batch_size=10):
    """
    Replace background in video using batch processing for efficiency.
    
    Args:
        input_video: Path to input video
        background_video: Path to background video
        output_path: Path to save output
        max_duration: Maximum duration to process (seconds)
        batch_size: Number of frames to process in batch
    """
    print("Starting fast background replacement...")
    
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
    
    # Create rembg session once
    print("Initializing background removal model...")
    session = new_session('u2net')
    
    # Process frames
    print(f"Processing {total_frames} frames...")
    
    for frame_idx in tqdm(range(total_frames)):
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
        
        # CRITICAL: Process EVERY frame with rembg for accurate masks
        # Convert BGR to RGB for rembg
        frame_rgb = cv2.cvtColor(frame_input, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Remove background - get RGBA output with alpha for better edge handling
        output_rgba = remove(pil_image, session=session, alpha_matting=True,
                           alpha_matting_foreground_threshold=270,  # More aggressive
                           alpha_matting_background_threshold=0,     # Exclude all background
                           alpha_matting_erode_size=2)              # Slight erosion
        
        # Get both mask and color-corrected foreground
        output_np = np.array(output_rgba)
        if output_np.shape[2] == 4:
            mask = output_np[:, :, 3]
            # Extract RGB without background contamination
            foreground_rgb = output_np[:, :, :3]
        else:
            mask = np.zeros((height, width), dtype=np.uint8)
            foreground_rgb = frame_rgb
        
        # CRITICAL: Remove white/light edge contamination
        # Find edge pixels
        edge_detector = cv2.Canny(mask, 50, 150)
        edge_mask = cv2.dilate(edge_detector, np.ones((7, 7), np.uint8), iterations=1)
        
        # In edge regions, detect light-colored pixels (likely from white background)
        edge_pixels = edge_mask > 0
        if np.any(edge_pixels):
            # Convert frame to HSV for better color detection
            frame_hsv = cv2.cvtColor(frame_input, cv2.COLOR_BGR2HSV)
            
            # Detect near-white/light pixels (low saturation, high value)
            # These are likely background bleed
            light_pixels = (frame_hsv[:, :, 1] < 30) & (frame_hsv[:, :, 2] > 200)  # Low sat, high value
            
            # Remove light edge pixels from mask
            contaminated_edges = edge_pixels & light_pixels
            mask[contaminated_edges] = 0
        
        # Apply morphological operations to clean up
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # Close small gaps
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)
        
        # Slight erosion to ensure no background pixels
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        mask = cv2.erode(mask, kernel_erode, iterations=1)
        
        # Smooth mask edges with smaller blur for tighter edges
        mask = cv2.GaussianBlur(mask, (3, 3), 1)
        
        # Apply mask with proper alpha blending
        mask_float = mask.astype(np.float32) / 255.0
        mask_3d = np.stack([mask_float, mask_float, mask_float], axis=2)
        
        # Use decontaminated foreground if available, otherwise original
        if 'foreground_rgb' in locals() and foreground_rgb is not None:
            # Convert RGB back to BGR for OpenCV
            foreground_bgr = cv2.cvtColor(foreground_rgb, cv2.COLOR_RGB2BGR)
            foreground_to_use = foreground_bgr
        else:
            foreground_to_use = frame_input
        
        # Composite with proper alpha blending
        composite = (mask_3d * foreground_to_use.astype(np.float32) + 
                    (1.0 - mask_3d) * frame_bg_resized.astype(np.float32)).astype(np.uint8)
        
        # Write frame
        out.write(composite)
    
    # Clean up
    cap_input.release()
    cap_bg.release()
    out.release()
    
    print("Encoding final video with H.264...")
    # Convert to H.264 with audio from original
    cmd = [
        "ffmpeg", "-y",
        "-i", temp_output.name,
        "-i", str(input_video),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
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
    """Test fast background replacement."""
    
    # Setup paths
    input_video = Path("uploads/assets/videos/ai_math1_test_5sec.mp4")
    background_video = Path("uploads/assets/videos/ai_math1/ai_math1_background_0_0_5_0_STc2OalsLp.mp4")
    output_video = Path("outputs/ai_math1_fast_bg_replaced.mp4")
    
    if not input_video.exists():
        print(f"Error: Input video not found: {input_video}")
        return
    
    if not background_video.exists():
        print(f"Error: Background video not found: {background_video}")
        return
    
    # Process video
    process_video_with_bg_replacement(
        input_video,
        background_video,
        output_video,
        max_duration=5.0
    )
    
    # Open result
    subprocess.run(["open", str(output_video)])


if __name__ == "__main__":
    main()