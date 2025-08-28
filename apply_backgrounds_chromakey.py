#!/usr/bin/env python3
"""
Apply stock backgrounds using the green screen version with chromakey.
This properly replaces the background instead of overlaying.
"""

import subprocess
from pathlib import Path


def create_animated_background(pattern_type, duration, output_path):
    """Create animated backgrounds."""
    
    if pattern_type == "blue_tech":
        # Blue technology pattern
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"testsrc2=size=1280x720:rate=25:duration={duration}",
            "-vf", "colorchannelmixer=rr=0:bb=2:gg=0.5,hue=h=200",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(output_path)
        ]
    elif pattern_type == "purple_wave":
        # Purple animated wave
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c=purple:size=1280x720:d={duration}",
            "-vf", f"geq=r='128+127*sin(X/50+T*2)':g='0+128*sin(Y/50+T*2)':b='128+127*cos(X/50+T*2)'",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(output_path)
        ]
    elif pattern_type == "green_grid":
        # Green data grid
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=black:size=1280x720:d={duration}",
            "-vf", f"drawgrid=width=40:height=40:thickness=1:color=green@0.3,hue=h=120:s=1.5",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(output_path)
        ]
    else:
        # Orange abstract
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", 
            "-i", f"color=c=orange:size=1280x720:d={duration}",
            "-vf", f"geq=r='255*abs(sin(hypot(X-640,Y-360)/100+T))':g='165*abs(cos(hypot(X-640,Y-360)/100+T))':b='0'",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(output_path)
        ]
    
    subprocess.run(cmd, check=True, capture_output=True)


def apply_chromakey_background(green_screen_video, background_video, output_path, start_time, duration):
    """
    Replace green screen with new background using chromakey.
    """
    
    # Use chromakey to replace green with new background
    cmd = [
        "ffmpeg", "-y",
        "-i", str(background_video),          # New background
        "-ss", str(start_time),              # Seek to start time
        "-t", str(duration),                 # Duration
        "-i", str(green_screen_video),       # Green screen foreground
        "-filter_complex",
        "[0:v]scale=1280:720[bg];"
        "[1:v]scale=1280:720[fg];"
        # Apply chromakey with despill for clean edges
        "[fg]chromakey=green:0.10:0.08[keyed];"
        "[keyed]despill=type=green:mix=0.5:expand=1.0[clean];"
        # Overlay the cleaned foreground on new background
        "[bg][clean]overlay[out]",
        "-map", "[out]",
        "-map", "1:a?",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr[-500:]}")
        raise RuntimeError("Chromakey failed")


def main():
    """Apply backgrounds using green screen chromakey."""
    
    # Paths
    project_folder = Path("uploads/assets/videos/ai_math1")
    green_screen_video = project_folder / "ai_math1_rvm_green_5s_024078685789.mp4"
    original_video = project_folder / "ai_math1_final.mp4"
    
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Check if green screen exists
    if not green_screen_video.exists():
        print(f"Error: Green screen video not found at {green_screen_video}")
        return
    
    # Define segments with different backgrounds
    segments = [
        {
            "start": 0.0,
            "duration": 1.25,
            "pattern": "blue_tech",
            "description": "Math/Tech - blue pattern"
        },
        {
            "start": 1.25,
            "duration": 1.25,
            "pattern": "purple_wave",
            "description": "AI - purple waves"
        },
        {
            "start": 2.5,
            "duration": 1.25,
            "pattern": "green_grid",
            "description": "Data - green grid"
        },
        {
            "start": 3.75,
            "duration": 1.25,
            "pattern": "orange_abstract",
            "description": "Discovery - orange abstract"
        }
    ]
    
    print("Replacing backgrounds using chromakey...\n")
    
    processed = []
    
    for i, seg in enumerate(segments):
        print(f"[{i+1}/{len(segments)}] {seg['description']}")
        
        # Create animated background
        bg_path = output_dir / f"chromakey_bg_{i}.mp4"
        print(f"  Creating {seg['pattern']} background...")
        create_animated_background(seg['pattern'], seg['duration'], bg_path)
        
        # Apply chromakey
        output_path = output_dir / f"chromakey_segment_{i}.mp4"
        print(f"  Applying chromakey...")
        apply_chromakey_background(
            green_screen_video, bg_path, output_path,
            seg['start'], seg['duration']
        )
        
        processed.append(output_path)
        print(f"  ✓ Complete\n")
    
    # Create final montage
    print("Creating final video...")
    concat_list = output_dir / "chromakey_concat.txt"
    with open(concat_list, 'w') as f:
        for path in processed:
            f.write(f"file '{path.absolute()}'\n")
    
    final = output_dir / "ai_math1_chromakey_backgrounds.mp4"
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
    
    print(f"✅ Final video: {final}\n")
    
    # Create comparison video
    print("Creating comparison video...")
    
    # Extract 5 seconds of original
    original_5s = output_dir / "original_5s.mp4"
    extract_cmd = [
        "ffmpeg", "-y",
        "-i", str(original_video),
        "-t", "5",
        "-c", "copy",
        str(original_5s)
    ]
    subprocess.run(extract_cmd, check=True, capture_output=True)
    
    # Stack original, green screen, and new backgrounds
    comparison = output_dir / "chromakey_comparison.mp4"
    stack_cmd = [
        "ffmpeg", "-y",
        "-i", str(original_5s),
        "-i", str(green_screen_video),
        "-i", str(final),
        "-filter_complex",
        "[0:v]scale=426:240,drawtext=text='Original':x=10:y=10:fontsize=20:fontcolor=white[v0];"
        "[1:v]scale=426:240,drawtext=text='Green Screen':x=10:y=10:fontsize=20:fontcolor=white[v1];"
        "[2:v]scale=426:240,drawtext=text='New Backgrounds':x=10:y=10:fontsize=20:fontcolor=white[v2];"
        "[v0][v1][v2]hstack=inputs=3[out]",
        "-map", "[out]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(comparison)
    ]
    subprocess.run(stack_cmd, check=True, capture_output=True)
    
    print(f"✅ Comparison: {comparison}")
    
    # Open both videos
    subprocess.run(["open", str(final)])
    subprocess.run(["open", str(comparison)])


if __name__ == "__main__":
    main()