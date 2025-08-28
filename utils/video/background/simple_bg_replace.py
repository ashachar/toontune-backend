#!/usr/bin/env python3
"""
Simple but effective background replacement with aggressive edge cleanup.
Optimized for speed while maintaining quality.
"""

import cv2
import numpy as np
import subprocess
from pathlib import Path
from rembg import remove, new_session
from PIL import Image
import tempfile
from tqdm import tqdm


def process_simple_bg_replacement(input_video, background_video, output_path, max_duration=5.0):
    """
    Simple background replacement with aggressive edge cleanup.
    
    Args:
        input_video: Path to input video
        background_video: Path to background video
        output_path: Path to save output
        max_duration: Maximum duration to process
    """
    print("Starting optimized background replacement...")
    print("Focus: Remove white edges effectively and quickly")
    
    # Open videos
    cap_input = cv2.VideoCapture(str(input_video))
    cap_bg = cv2.VideoCapture(str(background_video))
    
    # Get video properties
    fps = cap_input.get(cv2.CAP_PROP_FPS)
    width = int(cap_input.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_input.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = min(int(fps * max_duration), int(cap_input.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    # Setup temp output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))
    
    # Create rembg session - use u2net for speed
    print("Initializing background removal...")
    session = new_session('u2net')
    
    print(f"Processing {total_frames} frames...")
    
    for frame_idx in tqdm(range(total_frames), desc="Removing backgrounds"):
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
        frame_bg = cv2.resize(frame_bg, (width, height))
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame_input, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Remove background with alpha matting
        output_rgba = remove(pil_image, session=session,
                           alpha_matting=True,
                           alpha_matting_foreground_threshold=240,
                           alpha_matting_background_threshold=50,
                           alpha_matting_erode_size=10)
        
        # Get mask
        output_np = np.array(output_rgba)
        if output_np.shape[2] == 4:
            mask = output_np[:, :, 3]
        else:
            mask = np.zeros((height, width), dtype=np.uint8)
        
        # AGGRESSIVE EDGE CLEANUP
        # 1. Strong erosion to remove ALL edge contamination
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.erode(mask, kernel_large, iterations=1)
        
        # 2. Moderate dilation to restore size
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel_medium, iterations=1)
        
        # 3. Gaussian blur for smooth edges
        mask = cv2.GaussianBlur(mask, (7, 7), 3)
        
        # 4. Strong threshold - anything below 200 is background
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        
        # 5. Final slight erosion for safety
        kernel_tiny = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.erode(mask, kernel_tiny, iterations=1)
        
        # Create alpha channel
        alpha = mask.astype(np.float32) / 255.0
        alpha_3d = np.stack([alpha, alpha, alpha], axis=2)
        
        # Composite
        composite = (alpha_3d * frame_input.astype(np.float32) + 
                    (1.0 - alpha_3d) * frame_bg.astype(np.float32))
        composite = np.clip(composite, 0, 255).astype(np.uint8)
        
        # Write frame
        out.write(composite)
    
    # Cleanup
    cap_input.release()
    cap_bg.release()
    out.release()
    
    print("Encoding with H.264...")
    
    # Convert to H.264 with high quality
    cmd = [
        "ffmpeg", "-y",
        "-i", temp_output.name,
        "-i", str(input_video),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
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
    """Test simple background replacement."""
    
    print("=" * 60)
    print("Simple Background Replacement with Edge Cleanup")
    print("=" * 60)
    
    # Setup paths
    input_video = Path("uploads/assets/videos/ai_math1_test_5sec.mp4")
    background_video = Path("uploads/assets/videos/ai_math1/ai_math1_background_0_0_5_0_STc2OalsLp.mp4")
    output_video = Path("outputs/ai_math1_simple_bg_replaced.mp4")
    
    # Create test video if needed
    if not input_video.exists():
        print("Creating 5-second test video...")
        original = Path("uploads/assets/videos/ai_math1.mp4")
        if original.exists():
            cmd = [
                "ffmpeg", "-y",
                "-i", str(original),
                "-t", "5",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "18",
                str(input_video)
            ]
            subprocess.run(cmd, check=True)
            print(f"✓ Created: {input_video}")
        else:
            print(f"Error: Original not found: {original}")
            return
    
    if not background_video.exists():
        print(f"Error: Background not found: {background_video}")
        return
    
    # Process
    process_simple_bg_replacement(
        input_video,
        background_video,
        output_video,
        max_duration=5.0
    )
    
    # Open result
    if output_video.exists():
        print("\nOpening result...")
        subprocess.run(["open", str(output_video)])


if __name__ == "__main__":
    main()