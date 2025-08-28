#!/usr/bin/env python3
"""
Apply animated/gradient backgrounds using RVM masks.
Creates visually appealing animated backgrounds instead of solid colors.
"""

import subprocess
from pathlib import Path


def create_animated_background(pattern_type, duration, output_path):
    """Create different types of animated backgrounds."""
    
    if pattern_type == "particles":
        # Animated particle effect (blue particles)
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"life=size=1280x720:rate=25:ratio=0.1:death_color=blue:life_color=white:duration={duration}",
            "-vf", "scale=1280:720,colorchannelmixer=rr=0.2:bb=1.5:gg=0.3,gblur=sigma=2",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(output_path)
        ]
    elif pattern_type == "gradient_wave":
        # Animated gradient wave (purple/pink)
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"gradients=size=1280x720:duration={duration}:speed=0.01:c0=0x8B008B:c1=0xFF1493",
            "-vf", "scale=1280:720",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(output_path)
        ]
    elif pattern_type == "tech_grid":
        # Tech/data visualization grid (green)
        cmd = [
            "ffmpeg", "-y", 
            "-f", "lavfi",
            "-i", f"testsrc2=size=1280x720:rate=25:duration={duration}",
            "-vf", "scale=1280:720,colorchannelmixer=rr=0:gg=1.5:bb=0.5,eq=brightness=-0.3:contrast=1.5",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(output_path)
        ]
    elif pattern_type == "abstract_flow":
        # Abstract flowing pattern
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", 
            "-i", f"mandelbrot=size=1280x720:rate=25:maxiter=100:end_pts={duration}",
            "-vf", "scale=1280:720,hue=H=2*PI*t/5,colorbalance=rs=0.3:gs=-0.1:bs=0.2",
            "-t", str(duration),
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(output_path)
        ]
    else:
        # Default: animated noise pattern
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"nullsrc=size=1280x720:rate=25:duration={duration}",
            "-vf", "geq=random(1)/hypot(X-640,Y-360),scale=1280:720,colorize=hue=200:saturation=0.3",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(output_path)
        ]
    
    subprocess.run(cmd, check=True, capture_output=True)


def composite_with_mask(foreground_video, background_video, mask_video, output_path, start_time, duration):
    """Apply mask-based compositing."""
    
    # Extract the foreground segment
    fg_segment = output_path.parent / f"fg_temp_{start_time}.mp4"
    extract_cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-i", str(foreground_video),
        "-t", str(duration),
        "-c", "copy",
        str(fg_segment)
    ]
    subprocess.run(extract_cmd, check=True, capture_output=True)
    
    # Extract corresponding mask segment
    mask_segment = output_path.parent / f"mask_temp_{start_time}.mp4"
    mask_extract_cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-i", str(mask_video),
        "-t", str(duration),
        "-c", "copy", 
        str(mask_segment)
    ]
    subprocess.run(mask_extract_cmd, check=True, capture_output=True)
    
    # Composite using the mask
    composite_cmd = [
        "ffmpeg", "-y",
        "-i", str(background_video),   # New background
        "-i", str(fg_segment),         # Original foreground
        "-i", str(mask_segment),       # Alpha mask
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
    
    subprocess.run(composite_cmd, check=True, capture_output=True)
    
    # Cleanup
    fg_segment.unlink()
    mask_segment.unlink()


def main():
    """Create demo with animated backgrounds."""
    
    # Paths
    project_folder = Path("uploads/assets/videos/ai_math1")
    original_video = project_folder / "ai_math1_final.mp4"
    mask_video = project_folder / "ai_math1_rvm_mask_5s_024078685789.mp4"
    
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Define segments with animated backgrounds
    segments = [
        {
            "start": 0.0,
            "duration": 1.25,
            "pattern": "particles",
            "description": "Math/calculus - particle effect"
        },
        {
            "start": 1.25,
            "duration": 1.25,
            "pattern": "gradient_wave",
            "description": "AI/ChatGPT - gradient wave"
        },
        {
            "start": 2.5,
            "duration": 1.25,
            "pattern": "tech_grid",
            "description": "Data science - tech grid"
        },
        {
            "start": 3.75,
            "duration": 1.25,
            "pattern": "abstract_flow",
            "description": "Theory/discovery - abstract flow"
        }
    ]
    
    print("Creating video with animated backgrounds...\n")
    
    processed = []
    
    for i, seg in enumerate(segments):
        print(f"[{i+1}/{len(segments)}] {seg['description']}")
        
        # Create animated background
        bg_path = output_dir / f"animated_bg_{i}_{seg['pattern']}.mp4"
        print(f"  Creating {seg['pattern']} background...")
        create_animated_background(seg['pattern'], seg['duration'], bg_path)
        
        # Composite with mask
        output_path = output_dir / f"animated_segment_{i}.mp4"
        print(f"  Applying mask and compositing...")
        composite_with_mask(
            original_video, bg_path, mask_video, output_path,
            seg['start'], seg['duration']
        )
        
        processed.append(output_path)
        print(f"  ✓ Done\n")
    
    # Create montage
    print("Creating final montage...")
    concat_list = output_dir / "animated_concat.txt"
    with open(concat_list, 'w') as f:
        for path in processed:
            f.write(f"file '{path.absolute()}'\n")
    
    final = output_dir / "ai_math1_animated_backgrounds_demo.mp4"
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
    subprocess.run(concat_cmd, check=True)
    
    print(f"✅ Created: {final}")
    subprocess.run(["open", str(final)])


if __name__ == "__main__":
    main()