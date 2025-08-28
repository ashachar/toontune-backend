#!/usr/bin/env python3
"""
Background replacement using Replicate's Robust Video Matting model.
State-of-the-art video matting with temporal consistency.
NO post-processing - pure model output.
"""

import os
import subprocess
import tempfile
from pathlib import Path
import shutil
import replicate
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()


def upload_video_to_replicate(video_path):
    """
    Upload video directly to Replicate.
    
    Args:
        video_path: Path to local video file
        
    Returns:
        File object for Replicate
    """
    print(f"Uploading video to Replicate...")
    
    # Open the file for Replicate
    with open(video_path, 'rb') as f:
        # Replicate accepts file objects directly
        return f


def process_with_robust_video_matting(input_video, background_video, output_path, 
                                      max_duration=None):
    """
    Replace background using Replicate's Robust Video Matting.
    
    Args:
        input_video: Path to input video
        background_video: Path to background video
        output_path: Path to save output
        max_duration: Maximum duration to process
    """
    print("=" * 60)
    print("Robust Video Matting (Replicate)")
    print("State-of-the-art temporal consistency")
    print("NO POST-PROCESSING - PURE MODEL OUTPUT")
    print("=" * 60)
    
    # Check for Replicate API token
    api_token = os.getenv('REPLICATE_API_TOKEN')
    if not api_token:
        raise ValueError("REPLICATE_API_TOKEN not found in .env file")
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    temp_dir_path = Path(temp_dir)
    
    try:
        # Step 1: Prepare input video
        if max_duration:
            segment_path = temp_dir_path / "segment.mp4"
            print(f"\nExtracting {max_duration} second segment...")
            cmd = [
                "ffmpeg", "-y",
                "-i", str(input_video),
                "-t", str(max_duration),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "18",
                str(segment_path)
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            processing_input = segment_path
        else:
            processing_input = input_video
        
        # Step 2: Run Robust Video Matting on Replicate
        print("\n🚀 Running Robust Video Matting on Replicate...")
        print("This uses advanced temporal consistency algorithms")
        
        # Open the file and pass it directly
        with open(processing_input, 'rb') as video_file:
            input_params = {
                "input_video": video_file
            }
            
            # Run the model
            output = replicate.run(
                "arielreplicate/robust_video_matting:73d2128a371922d5d1abf0712a1d974be0e4e2358cc1218e4e34714767232bac",
                input=input_params
            )
        
        # Download the matted video
        print("\nDownloading matted video...")
        matted_video_path = temp_dir_path / "matted.mp4"
        
        # Get the output URL - it's directly a string URL
        output_url = str(output)
        print(f"Output URL: {output_url}")
        
        # Download the file
        response = requests.get(output_url, stream=True)
        response.raise_for_status()
        
        with open(matted_video_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("✓ Matted video downloaded")
        
        # Step 4: Check what type of output we got
        # Robust Video Matting typically outputs video with alpha channel
        print("\nAnalyzing output format...")
        
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=codec_name,pix_fmt",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(matted_video_path)
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        output_info = result.stdout.strip()
        print(f"Output format: {output_info}")
        
        # Step 5: Extract alpha channel as mask
        print("\nExtracting mask from matted video...")
        mask_video = temp_dir_path / "mask.mp4"
        
        # Try to extract alpha channel
        extract_alpha_cmd = [
            "ffmpeg", "-y",
            "-i", str(matted_video_path),
            "-vf", "alphaextract,format=gray",
            "-c:v", "libx264",
            "-crf", "0",
            str(mask_video)
        ]
        
        result = subprocess.run(extract_alpha_cmd, capture_output=True)
        
        if result.returncode != 0 or not mask_video.exists():
            # If no alpha channel, the video might be green screen or mask directly
            print("No alpha channel found, using as mask directly...")
            # Convert to grayscale mask
            convert_cmd = [
                "ffmpeg", "-y",
                "-i", str(matted_video_path),
                "-vf", "format=gray",
                "-c:v", "libx264",
                "-crf", "0",
                str(mask_video)
            ]
            subprocess.run(convert_cmd, check=True)
        
        # Step 6: Get dimensions
        probe_dim_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=s=x:p=0",
            str(processing_input)
        ]
        result = subprocess.run(probe_dim_cmd, capture_output=True, text=True)
        width, height = result.stdout.strip().split('x')
        
        # Step 7: Composite with new background (NO post-processing)
        print("\nCompositing with new background...")
        print("NO post-processing applied - pure Robust Video Matting output")
        
        composite_cmd = [
            "ffmpeg", "-y",
            "-i", str(background_video),      # Background
            "-i", str(processing_input),      # Original video
            "-i", str(mask_video),            # Mask from RVM
            "-filter_complex",
            f"[0:v]scale={width}:{height},loop=loop=-1:size=10000[bg];"
            f"[1:v]scale={width}:{height}[fg];"
            f"[2:v]scale={width}:{height},format=gray[mask];"
            "[bg][fg][mask]maskedmerge[out]",  # Direct merge, no processing
            "-map", "[out]",
            "-map", "1:a?",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path)
        ]
        
        subprocess.run(composite_cmd, check=True)
        print(f"\n✅ Output saved to: {output_path}")
        print("This is PURE Robust Video Matting output - NO post-processing")
        
    finally:
        # Cleanup
        if temp_dir_path.exists():
            shutil.rmtree(temp_dir)


def main():
    """Test Robust Video Matting from Replicate."""
    
    input_video = Path("uploads/assets/videos/ai_math1.mp4")
    background_video = Path("uploads/assets/videos/ai_math1/ai_math1_background_0_0_5_0_STc2OalsLp.mp4")
    output_video = Path("outputs/ai_math1_robust_video_matting.mp4")
    
    if not input_video.exists():
        print(f"Error: Input not found: {input_video}")
        return
    
    if not background_video.exists():
        print(f"Error: Background not found: {background_video}")
        return
    
    # Process with Robust Video Matting - NO post-processing
    process_with_robust_video_matting(
        input_video,
        background_video,
        output_video,
        max_duration=5.0  # 5-second test
    )
    
    # Open result
    if output_video.exists():
        print("\nOpening result...")
        subprocess.run(["open", str(output_video)])


if __name__ == "__main__":
    main()