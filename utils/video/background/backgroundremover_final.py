#!/usr/bin/env python3
"""
Background replacement using backgroundremover library.
No post-processing by default - the library should handle it.
"""

import cv2
import numpy as np
import subprocess
import tempfile
from pathlib import Path
from tqdm import tqdm
import shutil


def process_with_backgroundremover(input_video, background_video, output_path, 
                                   max_duration=None, apply_postprocessing=False):
    """
    Replace background using backgroundremover CLI.
    
    Args:
        input_video: Path to input video
        background_video: Path to background video
        output_path: Path to save output
        max_duration: Maximum duration to process (None for full video)
        apply_postprocessing: Whether to apply edge cleanup post-processing
    """
    print("Starting background replacement with backgroundremover...")
    print(f"Post-processing: {'ENABLED' if apply_postprocessing else 'DISABLED'}")
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    temp_dir_path = Path(temp_dir)
    
    try:
        # Step 1: Prepare input video segment
        if max_duration:
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
            processing_input = segment_path
        else:
            processing_input = input_video
            print("Processing full video...")
        
        # Step 2: Get video info
        cap = cv2.VideoCapture(str(processing_input))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        print(f"Video info: {width}x{height} @ {fps} fps")
        
        # Step 3: Create mask video using backgroundremover
        print("Creating mask video with backgroundremover...")
        mask_video = temp_dir_path / "mask.mp4"
        
        # Use backgroundremover CLI to create mask video
        br_cmd = [
            "backgroundremover",
            "-i", str(processing_input),
            "-mk",  # Mask output mode (black and white mask)
            "-o", str(mask_video),
            "-m", "u2netp"  # Use best model
        ]
        
        print(f"Running: {' '.join(br_cmd)}")
        print("This may take a moment...")
        
        # Run backgroundremover
        process = subprocess.Popen(br_cmd, 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT,
                                 text=True, bufsize=1)
        
        # Show progress
        for line in iter(process.stdout.readline, ''):
            if line and "%" in line:
                print(f"  Progress: {line.strip()}")
        
        process.wait()
        
        # If mask video doesn't exist, try alternative approach
        if not mask_video.exists() or process.returncode != 0:
            print("Trying alternative: creating transparent video first...")
            
            # Create transparent video
            transparent_video = temp_dir_path / "transparent.webm"
            br_cmd_transparent = [
                "backgroundremover",
                "-i", str(processing_input),
                "-tv",  # Transparent video mode
                "-o", str(transparent_video),
                "-m", "u2netp"
            ]
            
            print(f"Running: {' '.join(br_cmd_transparent)}")
            subprocess.run(br_cmd_transparent, check=True)
            
            # Extract alpha channel as mask
            print("Extracting alpha channel from transparent video...")
            extract_alpha_cmd = [
                "ffmpeg", "-y",
                "-i", str(transparent_video),
                "-vf", "alphaextract,format=gray",
                "-c:v", "libx264",
                "-crf", "0",  # Lossless
                str(mask_video)
            ]
            subprocess.run(extract_alpha_cmd, check=True)
        
        # Step 4: Apply post-processing if enabled
        if apply_postprocessing:
            print("Applying edge cleanup post-processing...")
            cleaned_mask = temp_dir_path / "mask_cleaned.mp4"
            
            cleanup_cmd = [
                "ffmpeg", "-y",
                "-i", str(mask_video),
                "-vf", (
                    "erosion,"      # Light erosion
                    "dilation,"     # Light dilation  
                    "gblur=sigma=1" # Minimal blur
                ),
                "-c:v", "libx264",
                "-crf", "0",
                str(cleaned_mask)
            ]
            subprocess.run(cleanup_cmd, check=True)
            mask_video = cleaned_mask
        else:
            print("Skipping post-processing - using raw backgroundremover output")
        
        # Step 5: Composite with new background
        print("Compositing with new background...")
        
        composite_cmd = [
            "ffmpeg", "-y",
            "-i", str(background_video),     # Background
            "-i", str(processing_input),     # Original foreground
            "-i", str(mask_video),           # Mask from backgroundremover
            "-filter_complex",
            f"[0:v]scale={width}:{height},loop=loop=-1:size=10000[bg];"
            f"[1:v]scale={width}:{height}[fg];"
            f"[2:v]scale={width}:{height},format=gray[mask];"
            "[bg][fg][mask]maskedmerge[out]",
            "-map", "[out]",
            "-map", "1:a?",  # Keep audio from original
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path)
        ]
        
        subprocess.run(composite_cmd, check=True)
        print(f"✓ Output saved to: {output_path}")
        
    finally:
        # Clean up temp directory
        if temp_dir_path.exists():
            shutil.rmtree(temp_dir)


def check_backgroundremover():
    """Check if backgroundremover is properly installed."""
    try:
        # Try to run backgroundremover help
        result = subprocess.run(
            ["python", "-m", "backgroundremover", "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0 or "usage:" in result.stdout.lower() or "usage:" in result.stderr.lower():
            print("✓ backgroundremover is available")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    print("✗ backgroundremover not working properly")
    print("Attempting to reinstall...")
    
    # Try reinstalling
    subprocess.run(["pip", "uninstall", "backgroundremover", "-y"], capture_output=True)
    subprocess.run(["pip", "install", "backgroundremover"], check=True)
    return True


def main():
    """Test backgroundremover-based background replacement."""
    
    print("=" * 60)
    print("Background Replacement using backgroundremover")
    print("NO POST-PROCESSING - Pure backgroundremover output")
    print("=" * 60)
    
    # Check installation
    if not check_backgroundremover():
        print("Failed to setup backgroundremover")
        return
    
    # Setup paths
    input_video = Path("uploads/assets/videos/ai_math1.mp4")
    background_video = Path("uploads/assets/videos/ai_math1/ai_math1_background_0_0_5_0_STc2OalsLp.mp4")
    output_video = Path("outputs/ai_math1_backgroundremover_nopost.mp4")
    
    if not input_video.exists():
        print(f"Error: Input video not found: {input_video}")
        return
    
    if not background_video.exists():
        print(f"Error: Background video not found: {background_video}")
        return
    
    # Process with backgroundremover - NO post-processing
    print("\nProcessing 5-second test with backgroundremover...")
    print("Post-processing is DISABLED\n")
    
    process_with_backgroundremover(
        input_video,
        background_video,
        output_video,
        max_duration=5.0,
        apply_postprocessing=False  # DISABLED - pure backgroundremover output
    )
    
    # Open result
    if output_video.exists():
        print("\nOpening result...")
        subprocess.run(["open", str(output_video)])


if __name__ == "__main__":
    main()