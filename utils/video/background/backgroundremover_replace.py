#!/usr/bin/env python3
"""
Background replacement using backgroundremover library for more robust video processing.
This processes the entire video at once rather than frame-by-frame.
"""

import cv2
import numpy as np
import subprocess
import tempfile
from pathlib import Path
from tqdm import tqdm
import shutil


def process_video_with_backgroundremover(input_video, background_video, output_path, 
                                        max_duration=5.0):
    """
    Replace background using backgroundremover for the entire video.
    
    Args:
        input_video: Path to input video
        background_video: Path to background video
        output_path: Path to save output
        max_duration: Maximum duration to process (seconds)
    """
    print("Starting background replacement with backgroundremover...")
    print("This processes the entire video for temporal consistency")
    
    # Create temp directory for intermediate files
    temp_dir = tempfile.mkdtemp()
    temp_dir_path = Path(temp_dir)
    
    try:
        # First, extract the segment we want to process
        segment_path = temp_dir_path / "segment.mp4"
        print(f"Extracting {max_duration} second segment...")
        extract_cmd = [
            "ffmpeg", "-y",
            "-i", str(input_video),
            "-t", str(max_duration),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            str(segment_path)
        ]
        subprocess.run(extract_cmd, check=True, capture_output=True)
        
        # Process with backgroundremover
        print("Running backgroundremover on entire video...")
        print("This ensures temporal consistency across all frames")
        
        # Output will be a video with transparent background (webm with alpha)
        transparent_output = temp_dir_path / "foreground_alpha.webm"
        
        # Run backgroundremover command
        # Using the highest quality model (u2net) for best results
        br_cmd = [
            "backgroundremover",
            "-i", str(segment_path),
            "-tv",  # Video mode with transparency
            "-o", str(transparent_output),
            "-m", "u2netp"  # Use best model
        ]
        
        print("Command:", " ".join(br_cmd))
        result = subprocess.run(br_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error running backgroundremover: {result.stderr}")
            # Fallback to basic model if u2netp fails
            print("Trying with u2net model instead...")
            br_cmd[-1] = "u2net"
            subprocess.run(br_cmd, check=True)
        
        # Check if output exists
        if not transparent_output.exists():
            raise FileNotFoundError(f"Backgroundremover did not create output: {transparent_output}")
        
        print("Background removed successfully. Now compositing...")
        
        # Composite with new background using FFmpeg
        # The WebM with alpha channel will be overlaid on the background
        composite_cmd = [
            "ffmpeg", "-y",
            "-i", str(background_video),  # Background
            "-i", str(transparent_output),  # Foreground with alpha
            "-filter_complex",
            f"[0:v]scale=1280:720,loop=loop=-1:size={int(max_duration*25)}[bg];"  # Loop background if needed
            "[1:v]scale=1280:720[fg];"  # Scale foreground
            "[bg][fg]overlay=0:0:shortest=1[out]",  # Overlay with alpha
            "-map", "[out]",
            "-map", "1:a?",  # Use audio from original if present
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path)
        ]
        
        print("Compositing foreground with new background...")
        subprocess.run(composite_cmd, check=True)
        
        print(f"Output saved to: {output_path}")
        
    finally:
        # Clean up temp directory
        if temp_dir_path.exists():
            shutil.rmtree(temp_dir)


def install_backgroundremover():
    """Check and install backgroundremover if needed."""
    try:
        import backgroundremover
        print("backgroundremover is already installed")
    except ImportError:
        print("Installing backgroundremover...")
        subprocess.run(["pip", "install", "backgroundremover"], check=True)
        print("backgroundremover installed successfully")


def process_with_edge_cleanup(input_video, background_video, output_path, max_duration=5.0):
    """
    Alternative approach: Use backgroundremover and apply edge cleanup in post.
    """
    print("Processing with backgroundremover + edge cleanup...")
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    temp_dir_path = Path(temp_dir)
    
    try:
        # Step 1: Extract segment
        segment_path = temp_dir_path / "segment.mp4" 
        print(f"Extracting {max_duration} second segment...")
        extract_cmd = [
            "ffmpeg", "-y",
            "-i", str(input_video),
            "-t", str(max_duration),
            "-c", "copy",
            str(segment_path)
        ]
        subprocess.run(extract_cmd, check=True, capture_output=True)
        
        # Step 2: Process with backgroundremover to get mask video
        print("Extracting masks with backgroundremover...")
        mask_video = temp_dir_path / "mask.mp4"
        
        # Generate mask video (black and white)
        br_mask_cmd = [
            "backgroundremover",
            "-i", str(segment_path),
            "-mk",  # Mask output mode
            "-o", str(mask_video),
            "-m", "u2net"
        ]
        
        result = subprocess.run(br_mask_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Warning: {result.stderr}")
            # Try alternative approach
            print("Using video mode with transparency...")
            transparent_video = temp_dir_path / "transparent.webm"
            br_cmd = [
                "backgroundremover", 
                "-i", str(segment_path),
                "-tv",
                "-o", str(transparent_video),
                "-m", "u2net"
            ]
            subprocess.run(br_cmd, check=True)
            
            # Extract alpha channel from transparent video
            print("Extracting alpha channel...")
            extract_alpha_cmd = [
                "ffmpeg", "-y",
                "-i", str(transparent_video),
                "-vf", "alphaextract",
                "-pix_fmt", "gray",
                str(mask_video)
            ]
            subprocess.run(extract_alpha_cmd, check=True)
        
        # Step 3: Apply edge cleanup to mask
        print("Applying edge cleanup to remove white borders...")
        cleaned_mask = temp_dir_path / "mask_cleaned.mp4"
        
        # Use FFmpeg filters to clean edges
        # - Erosion to remove thin edges
        # - Gaussian blur for smooth edges
        # - Levels adjustment to ensure clean separation
        cleanup_cmd = [
            "ffmpeg", "-y",
            "-i", str(mask_video),
            "-vf", (
                "erosion=coordinates=3:3,"  # Erode edges
                "dilation=coordinates=2:2,"  # Slight dilation back
                "gblur=sigma=1.5,"  # Smooth edges
                "curves=preset=increase_contrast"  # Sharpen mask boundaries
            ),
            "-c:v", "libx264",
            "-crf", "0",  # Lossless for mask
            str(cleaned_mask)
        ]
        subprocess.run(cleanup_cmd, check=True)
        
        # Step 4: Composite using cleaned mask
        print("Compositing with cleaned mask...")
        composite_cmd = [
            "ffmpeg", "-y",
            "-i", str(background_video),  # Background
            "-i", str(segment_path),      # Original foreground
            "-i", str(cleaned_mask),      # Cleaned mask
            "-filter_complex",
            "[0:v]scale=1280:720,loop=loop=-1:size={int(max_duration*25)}[bg];"
            "[1:v]scale=1280:720[fg];"
            "[2:v]scale=1280:720,format=gray[mask];"
            "[bg][fg][mask]maskedmerge[out]",
            "-map", "[out]",
            "-map", "1:a?",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path)
        ]
        subprocess.run(composite_cmd, check=True)
        
        print(f"Output saved to: {output_path}")
        
    finally:
        # Clean up
        if temp_dir_path.exists():
            shutil.rmtree(temp_dir)


def main():
    """Test backgroundremover-based background replacement."""
    
    # Check/install backgroundremover
    install_backgroundremover()
    
    # Setup paths
    input_video = Path("uploads/assets/videos/ai_math1_test_5sec.mp4")
    background_video = Path("uploads/assets/videos/ai_math1/ai_math1_background_0_0_5_0_STc2OalsLp.mp4")
    output_video = Path("outputs/ai_math1_backgroundremover_replaced.mp4")
    
    if not input_video.exists():
        print(f"Error: Input video not found: {input_video}")
        return
    
    if not background_video.exists():
        print(f"Error: Background video not found: {background_video}")
        return
    
    # Process video with backgroundremover
    try:
        # Try direct approach first
        process_video_with_backgroundremover(
            input_video,
            background_video,
            output_video,
            max_duration=5.0
        )
    except Exception as e:
        print(f"Direct approach failed: {e}")
        print("Trying alternative approach with edge cleanup...")
        process_with_edge_cleanup(
            input_video,
            background_video,
            output_video,
            max_duration=5.0
        )
    
    # Open result
    subprocess.run(["open", str(output_video)])


if __name__ == "__main__":
    main()