#!/usr/bin/env python3
"""
Background replacement using rembg with NO post-processing.
Pure rembg output with feature flag for optional post-processing.
"""

import cv2
import numpy as np
import subprocess
from pathlib import Path
from rembg import remove, new_session
from PIL import Image
import tempfile
from tqdm import tqdm


def process_rembg_no_postprocess(input_video, background_video, output_path, 
                                 max_duration=None, apply_postprocessing=False):
    """
    Replace background using rembg with optional post-processing.
    
    Args:
        input_video: Path to input video
        background_video: Path to background video
        output_path: Path to save output
        max_duration: Maximum duration to process (None for full video)
        apply_postprocessing: Whether to apply edge cleanup (default: False)
    """
    print("=" * 60)
    print("Background Replacement with rembg")
    print(f"Post-processing: {'ENABLED' if apply_postprocessing else 'DISABLED (Pure rembg output)'}")
    print("=" * 60)
    
    # Open videos
    cap_input = cv2.VideoCapture(str(input_video))
    cap_bg = cv2.VideoCapture(str(background_video))
    
    # Get video properties
    fps = cap_input.get(cv2.CAP_PROP_FPS)
    width = int(cap_input.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_input.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if max_duration:
        total_frames = min(int(fps * max_duration), int(cap_input.get(cv2.CAP_PROP_FRAME_COUNT)))
    else:
        total_frames = int(cap_input.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing at {width}x{height} resolution, {total_frames} frames")
    
    # Setup temp output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))
    
    # Create rembg session with highest quality
    print("Initializing rembg with u2netp model (highest quality)...")
    session = new_session('u2netp')
    
    print(f"Processing frames...")
    
    for frame_idx in tqdm(range(total_frames), desc="Processing"):
        # Read frames
        ret_input, frame_input = cap_input.read()
        ret_bg, frame_bg = cap_bg.read()
        
        if not ret_input:
            break
        
        # Loop background if needed
        if not ret_bg:
            cap_bg.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_bg, frame_bg = cap_bg.read()
        
        # Resize background
        frame_bg = cv2.resize(frame_bg, (width, height), interpolation=cv2.INTER_CUBIC)
        
        # Convert to RGB for rembg
        frame_rgb = cv2.cvtColor(frame_input, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Remove background - get RGBA with alpha channel
        output_rgba = remove(pil_image, session=session,
                           alpha_matting=True,
                           alpha_matting_foreground_threshold=270,
                           alpha_matting_background_threshold=10,
                           alpha_matting_erode_size=0)  # No erosion by default
        
        # Get RGBA output
        output_np = np.array(output_rgba)
        
        if output_np.shape[2] == 4:
            # Extract alpha channel and foreground
            alpha = output_np[:, :, 3].astype(np.float32) / 255.0
            foreground_rgb = output_np[:, :, :3]
            foreground_bgr = cv2.cvtColor(foreground_rgb, cv2.COLOR_RGB2BGR)
        else:
            # Fallback if no alpha
            alpha = np.ones((height, width), dtype=np.float32)
            foreground_bgr = frame_input
        
        # OPTIONAL POST-PROCESSING
        if apply_postprocessing:
            # Apply minimal edge cleanup
            alpha_uint8 = (alpha * 255).astype(np.uint8)
            
            # Light erosion to remove edge artifacts
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            alpha_uint8 = cv2.erode(alpha_uint8, kernel, iterations=1)
            
            # Light blur for smooth edges
            alpha_uint8 = cv2.GaussianBlur(alpha_uint8, (3, 3), 1)
            
            # Convert back to float
            alpha = alpha_uint8.astype(np.float32) / 255.0
        
        # Create 3-channel alpha
        alpha_3d = np.stack([alpha, alpha, alpha], axis=2)
        
        # Composite using pure rembg foreground (no contamination from original)
        composite = (alpha_3d * foreground_bgr.astype(np.float32) + 
                    (1.0 - alpha_3d) * frame_bg.astype(np.float32))
        composite = np.clip(composite, 0, 255).astype(np.uint8)
        
        # Write frame
        out.write(composite)
    
    # Cleanup
    cap_input.release()
    cap_bg.release()
    out.release()
    
    print("Encoding final video with H.264...")
    
    # Convert to H.264 with high quality
    cmd = [
        "ffmpeg", "-y",
        "-i", temp_output.name,
        "-i", str(input_video),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "17",  # High quality
        "-pix_fmt", "yuv420p",
        "-map", "0:v",
        "-map", "1:a?",
        "-movflags", "+faststart",
        str(output_path)
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    
    # Remove temp file
    Path(temp_output.name).unlink()
    
    print(f"✓ Output saved to: {output_path}")


def main():
    """Test rembg without post-processing."""
    
    # Setup paths
    input_video = Path("uploads/assets/videos/ai_math1.mp4")
    background_video = Path("uploads/assets/videos/ai_math1/ai_math1_background_0_0_5_0_STc2OalsLp.mp4")
    output_video = Path("outputs/ai_math1_rembg_no_postprocess.mp4")
    
    if not input_video.exists():
        print(f"Error: Input video not found: {input_video}")
        return
    
    if not background_video.exists():
        print(f"Error: Background video not found: {background_video}")
        return
    
    # Process WITHOUT post-processing (pure rembg output)
    process_rembg_no_postprocess(
        input_video,
        background_video,
        output_video,
        max_duration=5.0,  # 5-second test
        apply_postprocessing=False  # DISABLED - pure rembg output
    )
    
    # Open result
    if output_video.exists():
        print("\nOpening result...")
        subprocess.run(["open", str(output_video)])
    
    print("\nTo enable post-processing, set apply_postprocessing=True")


if __name__ == "__main__":
    main()