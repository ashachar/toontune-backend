#!/usr/bin/env python3
"""
Background replacement using backgroundremover library directly.
NO post-processing - pure backgroundremover output.
"""

import subprocess
import tempfile
from pathlib import Path
import shutil


def process_with_backgroundremover_direct(input_video, background_video, output_path, 
                                          max_duration=None):
    """
    Replace background using backgroundremover library DIRECTLY.
    NO post-processing - pure backgroundremover output.
    
    Args:
        input_video: Path to input video
        background_video: Path to background video
        output_path: Path to save output
        max_duration: Maximum duration to process
    """
    print("=" * 60)
    print("PURE backgroundremover Library")
    print("https://github.com/nadermx/backgroundremover")
    print("NO POST-PROCESSING - RAW OUTPUT")
    print("=" * 60)
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    temp_dir_path = Path(temp_dir)
    
    try:
        # Step 1: Prepare input
        if max_duration:
            segment_path = temp_dir_path / "segment.mp4"
            print(f"\nExtracting {max_duration} second segment...")
            cmd = [
                "ffmpeg", "-y",
                "-i", str(input_video),
                "-t", str(max_duration),
                "-c", "copy",
                str(segment_path)
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            processing_input = segment_path
        else:
            processing_input = input_video
        
        # Step 2: Use backgroundremover to create video with removed background
        print("\n🚀 Running backgroundremover library...")
        print("Creating video with removed background (no alpha needed)...")
        
        removed_bg_video = temp_dir_path / "removed_bg.mp4"
        
        # Use backgroundremover CLI - output video with background removed (green screen)
        br_cmd = [
            "backgroundremover",
            "-i", str(processing_input),
            "-o", str(removed_bg_video),
            "-m", "u2netp",  # Best model
            "-a",  # Alpha matting ON
            "-af", "270",  
            "-ab", "10",
            "-ae", "0"  # NO erosion - pure output
        ]
        
        print(f"Command: {' '.join(br_cmd)}")
        print("Processing with backgroundremover...")
        
        # Run backgroundremover
        process = subprocess.Popen(br_cmd, 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT,
                                 text=True, bufsize=1)
        
        for line in iter(process.stdout.readline, ''):
            if line and any(x in line for x in ["FRAME", "%", "frame=", "WORKER"]):
                print(f"  {line.strip()}")
        
        process.wait()
        
        if not removed_bg_video.exists():
            # Try with mask output instead
            print("\nTrying mask output mode...")
            mask_video = temp_dir_path / "mask.mp4"
            br_cmd_mask = [
                "backgroundremover",
                "-i", str(processing_input),
                "-mk",  # Mask mode
                "-o", str(mask_video),
                "-m", "u2netp"
            ]
            subprocess.run(br_cmd_mask, check=True)
            
            # Use mask to composite
            print("\nCompositing with mask from backgroundremover...")
            composite_cmd = [
                "ffmpeg", "-y",
                "-i", str(background_video),
                "-i", str(processing_input),
                "-i", str(mask_video),
                "-filter_complex",
                "[0:v]scale=1280:720,loop=loop=-1:size=1000[bg];"
                "[1:v]scale=1280:720[fg];"
                "[2:v]scale=1280:720,format=gray[mask];"
                "[bg][fg][mask]maskedmerge[out]",
                "-map", "[out]",
                "-map", "1:a?",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                str(output_path)
            ]
            subprocess.run(composite_cmd, check=True)
        else:
            # The output has removed background (likely green/black)
            # Composite it over new background
            print("\n✓ backgroundremover processing complete")
            print("Compositing with new background...")
            
            composite_cmd = [
                "ffmpeg", "-y",
                "-i", str(background_video),
                "-i", str(removed_bg_video),
                "-filter_complex",
                "[0:v]scale=1280:720,loop=loop=-1:size=1000[bg];"
                "[1:v]scale=1280:720,chromakey=green:0.1:0.2[fg];"  # Remove green if present
                "[bg][fg]overlay=0:0[out]",
                "-map", "[out]",
                "-map", "1:a?",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                str(output_path)
            ]
            
            # First check if it has green screen
            probe_cmd = ["ffmpeg", "-i", str(removed_bg_video), "-vframes", "1", "-f", "image2pipe", "-"]
            result = subprocess.run(probe_cmd, capture_output=True)
            
            # If it's already composited (not green), use it directly with overlay
            composite_simple = [
                "ffmpeg", "-y",
                "-i", str(background_video),
                "-i", str(removed_bg_video),
                "-filter_complex",
                "[0:v]scale=1280:720,loop=loop=-1:size=1000[bg];"
                "[1:v]scale=1280:720[fg];"
                "[bg][fg]overlay=0:0[out]",
                "-map", "[out]",
                "-map", "1:a?",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                str(output_path)
            ]
            
            subprocess.run(composite_simple, check=True, capture_output=True)
        
        print(f"\n✅ Output saved to: {output_path}")
        print("This is PURE backgroundremover output - NO post-processing applied")
        
    finally:
        # Cleanup
        if temp_dir_path.exists():
            shutil.rmtree(temp_dir)


def main():
    """Run backgroundremover directly."""
    
    input_video = Path("uploads/assets/videos/ai_math1.mp4")
    background_video = Path("uploads/assets/videos/ai_math1/ai_math1_background_0_0_5_0_STc2OalsLp.mp4")
    output_video = Path("outputs/ai_math1_backgroundremover_direct.mp4")
    
    if not input_video.exists():
        print(f"Error: Input not found: {input_video}")
        return
    
    if not background_video.exists():
        print(f"Error: Background not found: {background_video}")
        return
    
    # Process with backgroundremover - NO post-processing
    process_with_backgroundremover_direct(
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