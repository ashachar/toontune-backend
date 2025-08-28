#!/usr/bin/env python3
"""
Apply simple animated backgrounds using RVM masks.
Uses FFmpeg's built-in filters for reliable animated backgrounds.
"""

import subprocess
from pathlib import Path


def create_animated_background(pattern_type, duration, output_path):
    """Create simple animated backgrounds using reliable FFmpeg filters."""
    
    if pattern_type == "blue_pulse":
        # Blue pulsing background
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c=0x0040FF:s=1280x720:d={duration}",
            "-vf", f"geq=r='32+32*sin(2*PI*T)':g='64+64*sin(2*PI*T)':b='255-32*sin(2*PI*T)'",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(output_path)
        ]
    elif pattern_type == "purple_gradient":
        # Moving purple gradient
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c=0x800080:s=1280x720:d={duration}",
            "-vf", f"geq=r='128+127*sin(X/100+T)':g='0':b='128+127*cos(Y/100+T)'",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(output_path)
        ]
    elif pattern_type == "green_data":
        # Green matrix/data effect
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c=black:s=1280x720:d={duration}",
            "-vf", f"geq=r='32*mod(X+T*100,40)/40':g='255*mod(Y+T*50,30)/30':b='0'",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(output_path)
        ]
    else:
        # Orange abstract pattern
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c=0xFF4500:s=1280x720:d={duration}",
            "-vf", f"geq=r='255*abs(sin(hypot(X-640,Y-360)/50+T))':g='69*abs(cos(hypot(X-640,Y-360)/50+T))':b='0'",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(output_path)
        ]
    
    subprocess.run(cmd, check=True, capture_output=True)


def composite_with_mask(foreground_video, background_video, mask_video, output_path, start_time, duration):
    """Apply mask-based compositing with simplified pipeline."""
    
    # Single FFmpeg command that does everything
    composite_cmd = [
        "ffmpeg", "-y",
        "-i", str(background_video),         # Background (full duration)
        "-ss", str(start_time),             # Seek in foreground
        "-t", str(duration),
        "-i", str(foreground_video),        # Foreground segment
        "-ss", str(start_time),             # Seek in mask
        "-t", str(duration),
        "-i", str(mask_video),              # Mask segment
        "-filter_complex",
        "[0:v]scale=1280:720[bg];"
        "[1:v]scale=1280:720[fg];"
        "[2:v]scale=1280:720,format=gray[alpha];"
        "[bg][fg][alpha]maskedmerge[out]",
        "-map", "[out]",
        "-map", "1:a?",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]
    
    result = subprocess.run(composite_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error in compositing: {result.stderr[-500:]}")  # Show last 500 chars of error
        raise RuntimeError("Compositing failed")


def main():
    """Create demo with simple animated backgrounds."""
    
    # Paths
    project_folder = Path("uploads/assets/videos/ai_math1")
    original_video = project_folder / "ai_math1_final.mp4"
    mask_video = project_folder / "ai_math1_rvm_mask_5s_024078685789.mp4"
    
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Define 4 segments within the first 5 seconds (where we have mask)
    segments = [
        {
            "start": 0.0,
            "duration": 1.25,
            "pattern": "blue_pulse",
            "description": "Math formulas - blue pulsing"
        },
        {
            "start": 1.25,
            "duration": 1.25,
            "pattern": "purple_gradient",
            "description": "AI technology - purple gradient"
        },
        {
            "start": 2.5,
            "duration": 1.25,
            "pattern": "green_data",
            "description": "Data visualization - green matrix"
        },
        {
            "start": 3.75,
            "duration": 1.25,
            "pattern": "orange_abstract",
            "description": "Discovery - orange abstract"
        }
    ]
    
    print("Creating video segments with animated backgrounds...\n")
    
    processed = []
    
    for i, seg in enumerate(segments):
        print(f"[{i+1}/{len(segments)}] {seg['description']}")
        
        # Create animated background
        bg_path = output_dir / f"bg_{i}_{seg['pattern']}.mp4"
        print(f"  Creating {seg['pattern']} background...")
        create_animated_background(seg['pattern'], seg['duration'], bg_path)
        
        # Composite with mask
        output_path = output_dir / f"segment_{i}_masked.mp4"
        print(f"  Compositing with mask...")
        composite_with_mask(
            original_video, bg_path, mask_video, output_path,
            seg['start'], seg['duration']
        )
        
        processed.append(output_path)
        print(f"  ✓ Complete\n")
    
    # Create final montage
    print("Creating final montage...")
    concat_list = output_dir / "concat_animated.txt"
    with open(concat_list, 'w') as f:
        for path in processed:
            f.write(f"file '{path.absolute()}'\n")
    
    final = output_dir / "ai_math1_animated_backgrounds_final.mp4"
    concat_cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(final)
    ]
    subprocess.run(concat_cmd, check=True, capture_output=True)
    
    print(f"✅ Final video created: {final}\n")
    
    # Also create a side-by-side comparison
    print("Creating side-by-side comparison...")
    comparison = output_dir / "background_comparison.mp4"
    
    # Extract original 5-second segment
    original_5s = output_dir / "original_5s.mp4"
    extract_cmd = [
        "ffmpeg", "-y",
        "-i", str(original_video),
        "-t", "5",
        "-c", "copy",
        str(original_5s)
    ]
    subprocess.run(extract_cmd, check=True, capture_output=True)
    
    # Create side-by-side
    sidebyside_cmd = [
        "ffmpeg", "-y",
        "-i", str(original_5s),
        "-i", str(final),
        "-filter_complex",
        "[0:v]scale=640:360[left];"
        "[1:v]scale=640:360[right];"
        "[left][right]hstack[out]",
        "-map", "[out]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(comparison)
    ]
    subprocess.run(sidebyside_cmd, check=True, capture_output=True)
    
    print(f"✅ Comparison video: {comparison}")
    
    # Open both videos
    subprocess.run(["open", str(final)])
    subprocess.run(["open", str(comparison)])


if __name__ == "__main__":
    main()