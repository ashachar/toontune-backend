#!/usr/bin/env python3
"""
Background replacement using the backgroundremover library.
No post-processing - pure backgroundremover output.
"""

import subprocess
import tempfile
from pathlib import Path
import shutil


def process_with_backgroundremover(input_video, background_video, output_path, 
                                   max_duration=None, apply_postprocessing=False):
    """
    Replace background using backgroundremover library.
    
    Args:
        input_video: Path to input video
        background_video: Path to background video
        output_path: Path to save output
        max_duration: Maximum duration to process (None for full)
        apply_postprocessing: Whether to apply edge cleanup (default: False)
    """
    print("=" * 60)
    print("Background Replacement using backgroundremover")
    print("https://github.com/nadermx/backgroundremover")
    print(f"Post-processing: {'ENABLED' if apply_postprocessing else 'DISABLED (Pure output)'}")
    print("=" * 60)
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    temp_dir_path = Path(temp_dir)
    
    try:
        # Step 1: Prepare input segment
        if max_duration:
            segment_path = temp_dir_path / "segment.mp4"
            print(f"\nExtracting {max_duration} second segment...")
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
            print("\nProcessing full video...")
        
        # Step 2: Create transparent video using backgroundremover
        print("\nRunning backgroundremover to create transparent video...")
        print("Using u2netp model (highest quality)")
        
        transparent_output = temp_dir_path / "transparent.mp4"
        
        # Use backgroundremover CLI
        br_cmd = [
            "backgroundremover",
            "-i", str(processing_input),
            "-tv",  # Transparent video output
            "-o", str(transparent_output),
            "-m", "u2netp",  # Best quality model
            "-a",  # Enable alpha matting for better edges
            "-af", "270",  # Foreground threshold
            "-ab", "10",   # Background threshold
            "-ae", "0"     # No erosion (we want pure output)
        ]
        
        print(f"Command: {' '.join(br_cmd)}")
        print("This may take a moment as backgroundremover processes the video...")
        
        # Run with progress output
        process = subprocess.Popen(br_cmd, 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT,
                                 text=True, bufsize=1)
        
        for line in iter(process.stdout.readline, ''):
            if line:
                # Show progress
                if "%" in line or "Processing" in line or "frame" in line.lower():
                    print(f"  {line.strip()}")
        
        process.wait()
        
        if process.returncode != 0:
            raise RuntimeError("backgroundremover failed to process video")
        
        if not transparent_output.exists():
            raise FileNotFoundError(f"Output not created: {transparent_output}")
        
        print("✓ Transparent video created successfully")
        
        # Step 3: Extract mask from transparent video
        print("\nExtracting mask from transparent video...")
        mask_video = temp_dir_path / "mask.mp4"
        
        extract_mask_cmd = [
            "ffmpeg", "-y",
            "-i", str(transparent_output),
            "-vf", "alphaextract,format=gray",
            "-c:v", "libx264",
            "-crf", "0",  # Lossless for mask
            str(mask_video)
        ]
        subprocess.run(extract_mask_cmd, check=True, capture_output=True)
        
        # Step 4: Optional post-processing
        if apply_postprocessing:
            print("\nApplying minimal post-processing...")
            processed_mask = temp_dir_path / "mask_processed.mp4"
            
            # Very light processing only
            process_cmd = [
                "ffmpeg", "-y",
                "-i", str(mask_video),
                "-vf", "gblur=sigma=0.5",  # Very light blur only
                "-c:v", "libx264",
                "-crf", "0",
                str(processed_mask)
            ]
            subprocess.run(process_cmd, check=True, capture_output=True)
            mask_video = processed_mask
        
        # Step 5: Get video dimensions
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=s=x:p=0",
            str(processing_input)
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        width, height = result.stdout.strip().split('x')
        
        # Step 6: Composite with background
        print("\nCompositing with new background...")
        
        composite_cmd = [
            "ffmpeg", "-y",
            "-i", str(background_video),      # Background
            "-i", str(processing_input),      # Original video
            "-i", str(mask_video),            # Mask from backgroundremover
            "-filter_complex",
            f"[0:v]scale={width}:{height},loop=loop=-1:size=10000[bg];"
            f"[1:v]scale={width}:{height}[fg];"
            f"[2:v]scale={width}:{height},format=gray[mask];"
            "[bg][fg][mask]maskedmerge[out]",
            "-map", "[out]",
            "-map", "1:a?",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path)
        ]
        
        subprocess.run(composite_cmd, check=True, capture_output=True)
        print(f"✓ Output saved to: {output_path}")
        
    finally:
        # Cleanup
        if temp_dir_path.exists():
            shutil.rmtree(temp_dir)


def main():
    """Test backgroundremover without post-processing."""
    
    # Setup paths
    input_video = Path("uploads/assets/videos/ai_math1.mp4")
    background_video = Path("uploads/assets/videos/ai_math1/ai_math1_background_0_0_5_0_STc2OalsLp.mp4")
    output_video = Path("outputs/ai_math1_backgroundremover_pure.mp4")
    
    if not input_video.exists():
        print(f"Error: Input video not found: {input_video}")
        return
    
    if not background_video.exists():
        print(f"Error: Background video not found: {background_video}")
        return
    
    # Process WITHOUT post-processing
    process_with_backgroundremover(
        input_video,
        background_video,
        output_video,
        max_duration=5.0,  # 5-second test
        apply_postprocessing=False  # DISABLED - pure backgroundremover output
    )
    
    # Open result
    if output_video.exists():
        print("\nOpening result...")
        subprocess.run(["open", str(output_video)])
    
    print("\nThis used the backgroundremover library (https://github.com/nadermx/backgroundremover)")
    print("Post-processing was DISABLED - this is pure backgroundremover output")


if __name__ == "__main__":
    main()