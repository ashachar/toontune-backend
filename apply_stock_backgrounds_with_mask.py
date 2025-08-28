#!/usr/bin/env python3
"""
Apply stock video backgrounds using pre-calculated RVM masks.
This version properly composites the backgrounds using the alpha mask.
"""

import subprocess
from pathlib import Path
import json


def create_background_video(color, duration, output_path):
    """Create a solid color or gradient background video."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c={color}:s=1280x720:d={duration}",
        "-vf", "format=yuv420p",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        str(output_path)
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def apply_background_with_mask(original_video, background_video, mask_video, output_path, start_time, duration):
    """
    Apply background using RVM mask with proper alpha compositing.
    
    The mask video contains the alpha channel that defines which parts 
    are foreground (white/255) and which are background (black/0).
    """
    
    # Extract segment from original video
    segment_path = output_path.parent / f"temp_segment_{start_time}.mp4"
    extract_cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-i", str(original_video),
        "-t", str(duration),
        "-c", "copy",
        str(segment_path)
    ]
    subprocess.run(extract_cmd, check=True, capture_output=True)
    
    # Apply the mask to composite foreground over background
    composite_cmd = [
        "ffmpeg", "-y",
        "-i", str(background_video),     # Background (input 0)
        "-i", str(segment_path),         # Original video segment (input 1)
        "-i", str(mask_video),           # Alpha mask (input 2)
        "-filter_complex",
        # Scale all inputs to same size
        "[0:v]scale=1280:720,format=yuv420p[bg];"
        "[1:v]scale=1280:720,format=yuv420p[fg];"
        "[2:v]scale=1280:720,format=gray,geq=lum='lum(X,Y)':a='lum(X,Y)'[mask];"
        # Use the mask to blend foreground over background
        "[bg][fg][mask]maskedmerge[out]",
        "-map", "[out]",
        "-map", "1:a?",  # Copy audio from original if present
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]
    
    result = subprocess.run(composite_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise RuntimeError("FFmpeg compositing failed")
    
    # Clean up temp file
    segment_path.unlink()
    
    print(f"✓ Created: {output_path}")


def main():
    """Apply different backgrounds at specific timestamps."""
    
    # Setup paths
    project_folder = Path("uploads/assets/videos/ai_math1")
    original_video = project_folder / "ai_math1_final.mp4"
    mask_video = project_folder / "ai_math1_rvm_mask_5s_024078685789.mp4"
    
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Define segments with different colored backgrounds
    segments = [
        {
            "start": 0.5,
            "duration": 2.0,
            "color": "blue",
            "description": "Math formulas - blue background"
        },
        {
            "start": 2.5,
            "duration": 2.0,
            "color": "purple", 
            "description": "AI technology - purple background"
        },
        {
            "start": 4.5,
            "duration": 0.5,
            "color": "green",
            "description": "Data visualization - green background"
        }
    ]
    
    print("Applying stock backgrounds with RVM mask...\n")
    
    processed_segments = []
    
    for i, seg in enumerate(segments):
        print(f"[{i+1}/{len(segments)}] {seg['description']}")
        
        # Create background video
        bg_path = output_dir / f"background_{i}_{seg['color']}.mp4"
        print(f"  Creating {seg['color']} background...")
        create_background_video(seg['color'], seg['duration'], bg_path)
        
        # Apply background with mask
        output_path = output_dir / f"masked_segment_{i}_{seg['start']}s.mp4"
        print(f"  Applying mask and compositing...")
        apply_background_with_mask(
            original_video, bg_path, mask_video, output_path,
            seg['start'], seg['duration']
        )
        
        processed_segments.append(output_path)
    
    # Concatenate all segments
    print("\nCreating final montage...")
    concat_list = output_dir / "concat_list.txt"
    with open(concat_list, 'w') as f:
        for path in processed_segments:
            f.write(f"file '{path.absolute()}'\n")
    
    final_output = output_dir / "ai_math1_with_colored_backgrounds.mp4"
    concat_cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(final_output)
    ]
    subprocess.run(concat_cmd, check=True)
    
    print(f"\n✅ Final video created: {final_output}")
    
    # Open the result
    subprocess.run(["open", str(final_output)])


if __name__ == "__main__":
    main()