#!/usr/bin/env python3
"""
Background replacement using backgroundremover CLI for robust video processing.
Uses the command-line interface to avoid import issues.
"""

import cv2
import numpy as np
import subprocess
import tempfile
from pathlib import Path
from tqdm import tqdm
import shutil
import time


def process_with_backgroundremover_cli(input_video, background_video, output_path, 
                                       max_duration=5.0):
    """
    Replace background using backgroundremover CLI for the entire video.
    This ensures temporal consistency across frames.
    
    Args:
        input_video: Path to input video
        background_video: Path to background video
        output_path: Path to save output
        max_duration: Maximum duration to process (seconds)
    """
    print("Starting background replacement with backgroundremover CLI...")
    print("This processes the entire video for temporal consistency")
    
    # Create temp directory for intermediate files
    temp_dir = tempfile.mkdtemp()
    temp_dir_path = Path(temp_dir)
    
    try:
        # Step 1: Extract the segment we want to process
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
        
        # Step 2: Use backgroundremover to create transparent video
        print("Running backgroundremover on entire video...")
        transparent_output = temp_dir_path / "foreground_alpha.mp4"
        
        # Use backgroundremover CLI with video transparency mode
        br_cmd = [
            "backgroundremover",
            "-i", str(segment_path),
            "-tv",  # Transparent video mode
            "-o", str(transparent_output),
            "-m", "u2netp"  # Use best model
        ]
        
        print(f"Command: {' '.join(br_cmd)}")
        print("This may take a moment for the first run as models are downloaded...")
        
        # Run with real-time output
        process = subprocess.Popen(br_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 text=True, bufsize=1)
        
        for line in iter(process.stdout.readline, ''):
            if line:
                print(f"  {line.strip()}")
        
        process.wait()
        
        if process.returncode != 0:
            print("Trying with u2net model instead...")
            br_cmd[-1] = "u2net"
            subprocess.run(br_cmd, check=True)
        
        # Check output exists
        if not transparent_output.exists():
            # Try WebM format which has better alpha support
            transparent_output = temp_dir_path / "foreground_alpha.webm"
            br_cmd[-2] = str(transparent_output)
            print("Trying WebM output format for better alpha channel support...")
            subprocess.run(br_cmd, check=True)
        
        if not transparent_output.exists():
            raise FileNotFoundError(f"Backgroundremover did not create output")
        
        print("Background removed successfully. Now compositing...")
        
        # Step 3: Create clean mask video from the alpha channel
        mask_video = temp_dir_path / "mask.mp4"
        
        # Extract alpha channel as mask
        print("Extracting clean alpha channel...")
        alpha_cmd = [
            "ffmpeg", "-y",
            "-i", str(transparent_output),
            "-vf", "alphaextract,format=gray",
            "-c:v", "libx264",
            "-crf", "0",  # Lossless for mask
            str(mask_video)
        ]
        
        result = subprocess.run(alpha_cmd, capture_output=True, text=True)
        
        if result.returncode != 0 or not mask_video.exists():
            # Alternative: Create mask using color difference
            print("Creating mask from transparent video...")
            create_mask_from_transparent(segment_path, transparent_output, mask_video)
        
        # Step 4: Apply edge cleanup to remove white borders
        print("Applying edge cleanup to remove white borders...")
        cleaned_mask = temp_dir_path / "mask_cleaned.mp4"
        
        cleanup_cmd = [
            "ffmpeg", "-y",
            "-i", str(mask_video),
            "-vf", (
                "erosion=coordinates=5:5,"     # Aggressive erosion to remove edges
                "dilation=coordinates=3:3,"     # Slight dilation back
                "gblur=sigma=2,"                # Smooth edges  
                "curves=preset=increase_contrast,"  # Sharp boundaries
                "eq=contrast=1.5"               # Enhance contrast
            ),
            "-c:v", "libx264",
            "-crf", "0",
            str(cleaned_mask)
        ]
        subprocess.run(cleanup_cmd, check=True)
        
        # Step 5: Final composite with cleaned mask
        print("Creating final composite with new background...")
        composite_cmd = [
            "ffmpeg", "-y",
            "-i", str(background_video),   # Background
            "-i", str(segment_path),       # Original foreground
            "-i", str(cleaned_mask),       # Cleaned mask
            "-filter_complex",
            f"[0:v]scale=1280:720,loop=loop=-1:size={int(max_duration*30)}[bg];"
            "[1:v]scale=1280:720[fg];"
            "[2:v]scale=1280:720,format=gray[mask];"
            "[bg][fg][mask]maskedmerge[out]",
            "-map", "[out]",
            "-map", "1:a?",  # Keep audio from original
            "-c:v", "libx264",
            "-preset", "slow",  # Better quality
            "-crf", "18",       # High quality
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


def create_mask_from_transparent(original_video, transparent_video, mask_output):
    """
    Create mask by comparing original and transparent videos.
    
    Args:
        original_video: Path to original video
        transparent_video: Path to transparent video
        mask_output: Path to save mask video
    """
    print("Creating mask from video comparison...")
    
    cap_orig = cv2.VideoCapture(str(original_video))
    cap_trans = cv2.VideoCapture(str(transparent_video))
    
    fps = cap_orig.get(cv2.CAP_PROP_FPS)
    width = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(mask_output), fourcc, fps, (width, height), isColor=False)
    
    while True:
        ret_orig, frame_orig = cap_orig.read()
        ret_trans, frame_trans = cap_trans.read()
        
        if not ret_orig or not ret_trans:
            break
        
        # Convert to grayscale
        gray_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
        gray_trans = cv2.cvtColor(frame_trans, cv2.COLOR_BGR2GRAY)
        
        # Create mask from difference
        diff = cv2.absdiff(gray_orig, gray_trans)
        _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY_INV)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        out.write(mask)
    
    cap_orig.release()
    cap_trans.release()
    out.release()


def verify_backgroundremover_installation():
    """Check if backgroundremover is properly installed."""
    try:
        result = subprocess.run(["backgroundremover", "--help"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ backgroundremover is installed and ready")
            return True
        else:
            print("✗ backgroundremover is not properly installed")
            return False
    except FileNotFoundError:
        print("✗ backgroundremover command not found")
        print("Installing backgroundremover...")
        subprocess.run(["pip", "install", "backgroundremover"], check=True)
        return verify_backgroundremover_installation()


def main():
    """Test backgroundremover CLI-based background replacement."""
    
    print("=" * 60)
    print("Background Replacement using backgroundremover CLI")
    print("=" * 60)
    
    # Verify installation
    if not verify_backgroundremover_installation():
        return
    
    # Setup paths  
    input_video = Path("uploads/assets/videos/ai_math1_test_5sec.mp4")
    background_video = Path("uploads/assets/videos/ai_math1/ai_math1_background_0_0_5_0_STc2OalsLp.mp4")
    output_video = Path("outputs/ai_math1_bgremover_cli.mp4")
    
    # Create 5-second test if needed
    if not input_video.exists():
        print(f"Creating 5-second test video from ai_math1.mp4...")
        original = Path("uploads/assets/videos/ai_math1.mp4")
        if original.exists():
            cmd = [
                "ffmpeg", "-y",
                "-i", str(original),
                "-t", "5",
                "-c", "copy",
                str(input_video)
            ]
            subprocess.run(cmd, check=True)
            print(f"✓ Created test video: {input_video}")
        else:
            print(f"Error: Original video not found: {original}")
            return
    
    if not background_video.exists():
        print(f"Error: Background video not found: {background_video}")
        return
    
    # Process video
    print("\nProcessing video...")
    process_with_backgroundremover_cli(
        input_video,
        background_video,
        output_video,
        max_duration=5.0
    )
    
    # Open result
    if output_video.exists():
        print("\nOpening result...")
        subprocess.run(["open", str(output_video)])
    else:
        print(f"Error: Output video was not created: {output_video}")


if __name__ == "__main__":
    main()