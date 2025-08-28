#!/usr/bin/env python3
"""
High-quality background replacement preserving fine details.
Removes white edges without excessive aliasing.
"""

import cv2
import numpy as np
import subprocess
from pathlib import Path
from rembg import remove, new_session
from PIL import Image
import tempfile
from tqdm import tqdm


def process_high_quality_bg_replacement(input_video, background_video, output_path, max_duration=None):
    """
    High-quality background replacement preserving details.
    
    Args:
        input_video: Path to input video
        background_video: Path to background video
        output_path: Path to save output
        max_duration: Maximum duration to process (None for full video)
    """
    print("Starting high-quality background replacement...")
    print("Focus: Preserve quality while removing white edges")
    
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
    
    print(f"Processing at {width}x{height} resolution")
    
    # Setup temp output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))
    
    # Create rembg session - use high quality model
    print("Initializing high-quality background removal (u2netp)...")
    session = new_session('u2netp')
    
    print(f"Processing {total_frames} frames...")
    
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
        
        # Resize background to match input resolution exactly
        frame_bg = cv2.resize(frame_bg, (width, height), interpolation=cv2.INTER_CUBIC)
        
        # Convert to RGB for rembg
        frame_rgb = cv2.cvtColor(frame_input, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Remove background with high-quality alpha matting
        output_rgba = remove(pil_image, session=session,
                           alpha_matting=True,
                           alpha_matting_foreground_threshold=270,  # Very selective
                           alpha_matting_background_threshold=10,   # Exclude background
                           alpha_matting_erode_size=0)              # No erosion initially
        
        # Get mask and RGBA output
        output_np = np.array(output_rgba)
        if output_np.shape[2] == 4:
            mask = output_np[:, :, 3].astype(np.float32) / 255.0
            # Use the clean foreground from rembg (already has background removed)
            foreground_rgb = output_np[:, :, :3]
            foreground_bgr = cv2.cvtColor(foreground_rgb, cv2.COLOR_RGB2BGR)
        else:
            mask = np.zeros((height, width), dtype=np.float32)
            foreground_bgr = frame_input
        
        # REFINED EDGE CLEANUP (less aggressive)
        # 1. Detect edge region
        mask_uint8 = (mask * 255).astype(np.uint8)
        edges = cv2.Canny(mask_uint8, 50, 150)
        
        # 2. Create edge mask (where we'll apply cleanup)
        kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edge_region = cv2.dilate(edges, kernel_edge, iterations=1) > 0
        
        # 3. In edge regions only, apply threshold to remove semi-transparent pixels
        # This removes the white halo without affecting the main subject
        edge_mask = mask.copy()
        edge_mask[edge_region] = np.where(mask[edge_region] > 0.9, 1.0, 0.0)
        
        # 4. Slight erosion ONLY on edges to ensure no white pixels
        kernel_tiny = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edge_mask_uint8 = (edge_mask * 255).astype(np.uint8)
        edge_mask_uint8 = cv2.erode(edge_mask_uint8, kernel_tiny, iterations=1)
        
        # 5. Gaussian blur for smooth transitions (smaller kernel for less blur)
        edge_mask_uint8 = cv2.GaussianBlur(edge_mask_uint8, (3, 3), 1)
        
        # Convert back to float
        final_mask = edge_mask_uint8.astype(np.float32) / 255.0
        
        # Create 3-channel mask
        mask_3d = np.stack([final_mask, final_mask, final_mask], axis=2)
        
        # High-quality composite using clean foreground from rembg
        # This avoids color contamination from the original background
        composite = (mask_3d * foreground_bgr.astype(np.float32) + 
                    (1.0 - mask_3d) * frame_bg.astype(np.float32))
        composite = np.clip(composite, 0, 255).astype(np.uint8)
        
        # Write frame
        out.write(composite)
    
    # Cleanup
    cap_input.release()
    cap_bg.release()
    out.release()
    
    print("Encoding with H.264 at high quality...")
    
    # Convert to H.264 with very high quality to preserve details
    cmd = [
        "ffmpeg", "-y",
        "-i", temp_output.name,
        "-i", str(input_video),
        "-c:v", "libx264",
        "-preset", "slow",      # Slow preset for best quality
        "-crf", "15",           # Very high quality (lower = better)
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
    """Test high-quality background replacement."""
    
    print("=" * 60)
    print("High-Quality Background Replacement")
    print("=" * 60)
    
    # Setup paths - process FULL video this time
    input_video = Path("uploads/assets/videos/ai_math1.mp4")  # Full video
    background_video = Path("uploads/assets/videos/ai_math1/ai_math1_background_0_0_5_0_STc2OalsLp.mp4")
    output_video = Path("outputs/ai_math1_high_quality_bg.mp4")
    
    if not input_video.exists():
        print(f"Error: Input video not found: {input_video}")
        return
    
    if not background_video.exists():
        print(f"Error: Background video not found: {background_video}")
        return
    
    # Get user choice
    print("\nOptions:")
    print("1. Process 5-second test")
    print("2. Process full video")
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        max_duration = 5.0
        print("\nProcessing 5-second test...")
    else:
        max_duration = None
        print("\nProcessing full video...")
    
    # Process
    process_high_quality_bg_replacement(
        input_video,
        background_video,
        output_video,
        max_duration=max_duration
    )
    
    # Open result
    if output_video.exists():
        print("\nOpening result...")
        subprocess.run(["open", str(output_video)])


if __name__ == "__main__":
    main()