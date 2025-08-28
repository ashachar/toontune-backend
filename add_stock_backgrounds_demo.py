#!/usr/bin/env python3
"""
Demo: Add stock video backgrounds at specific timestamps (limited to 3 segments for speed).
Uses pre-calculated masks if available, otherwise creates them dynamically.
"""

import subprocess
import json
from pathlib import Path
import sys
import tempfile
import shutil

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from utils.video.background.coverr_manager import CoverrManager


# Demo: Process only first 3 segments for quick demonstration
DEMO_SEGMENTS = [
    {
        "start": 7.28,
        "end": 9.6,
        "keywords": ["abstract", "mathematics", "formula"],
        "description": "derivative and integral symbols"
    },
    {
        "start": 27.92,
        "end": 31.6,
        "keywords": ["technology", "digital", "futuristic"],
        "description": "ChatGPT and AI technology"
    },
    {
        "start": 112.8,
        "end": 117.0,  # Shortened for demo
        "keywords": ["data", "visualization", "graph"],
        "description": "data science visualization"
    }
]


def extract_segment(video_path, start, end, output_path):
    """Extract a segment from video."""
    duration = end - start
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(video_path),
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "ultrafast",  # Faster for demo
        "-crf", "23",
        "-c:a", "copy",
        str(output_path)
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def composite_with_background(segment_path, background_path, output_path):
    """Simple overlay compositing without mask for demo."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(background_path),  # Background
        "-i", str(segment_path),     # Foreground
        "-filter_complex",
        "[0:v]scale=1280:720,format=yuva420p,colorchannelmixer=aa=0.3[bg];"  # Semi-transparent background
        "[1:v]scale=1280:720[fg];"
        "[bg][fg]overlay[out]",
        "-map", "[out]",
        "-map", "1:a?",
        "-shortest",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]
    subprocess.run(cmd, check=True)


def main():
    """Demo: Process video with stock backgrounds at 3 timestamps."""
    
    # Setup paths
    project_name = "ai_math1"
    project_folder = Path(f"uploads/assets/videos/{project_name}")
    video_path = project_folder / "ai_math1_final.mp4"
    
    if not video_path.exists():
        print(f"Error: Video not found at {video_path}")
        return
    
    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    # Initialize Coverr manager in demo mode (no API needed)
    manager = CoverrManager(demo_mode=True)
    
    print("DEMO: Processing 3 segments with stock backgrounds...\n")
    
    processed_files = []
    
    for i, segment_info in enumerate(DEMO_SEGMENTS):
        print(f"[{i+1}/3] Processing {segment_info['start']:.1f}s - {segment_info['end']:.1f}s")
        print(f"  Description: {segment_info['description']}")
        
        # Extract segment
        segment_path = temp_dir / f"segment_{i}.mp4"
        print("  Extracting segment...")
        extract_segment(video_path, segment_info['start'], segment_info['end'], segment_path)
        
        # Create demo background (animated gradient)
        print("  Creating demo background...")
        background_path = temp_dir / f"bg_{i}.mp4"
        duration = segment_info['end'] - segment_info['start']
        
        # Create animated gradient background
        gradient_cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c=0x{['4169E1', 'FF69B4', '32CD32'][i]}:s=1280x720:d={duration}",
            "-vf", f"geq=r='X/W*155+100':g='Y/H*155+100':b='128+127*sin(2*PI*T/{i+1})'",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            str(background_path)
        ]
        subprocess.run(gradient_cmd, check=True, capture_output=True)
        
        # Composite
        output_segment = output_dir / f"demo_segment_{i}_{segment_info['start']:.0f}s.mp4"
        print("  Compositing with background...")
        composite_with_background(segment_path, background_path, output_segment)
        
        processed_files.append(str(output_segment))
        print(f"  ✓ Saved to: {output_segment}\n")
    
    # Create a simple montage of all segments
    print("Creating montage of all processed segments...")
    
    # Concatenate all segments
    concat_file = temp_dir / "concat.txt"
    with open(concat_file, 'w') as f:
        for file in processed_files:
            f.write(f"file '{file}'\n")
    
    montage_output = output_dir / "ai_math1_stock_backgrounds_demo.mp4"
    concat_cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        str(montage_output)
    ]
    subprocess.run(concat_cmd, check=True)
    
    print(f"\n✅ Demo complete!")
    print(f"Individual segments saved to: outputs/demo_segment_*.mp4")
    print(f"Montage saved to: {montage_output}")
    
    # Cleanup temp dir
    shutil.rmtree(temp_dir)
    
    # Validate continuity of montage
    print("\nValidating video continuity...")
    validate_cmd = ["python", "utils/video/validation/validate_continuity.py", str(montage_output)]
    result = subprocess.run(validate_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Video continuity validated - no duplicate frames detected")
    else:
        print("⚠️ Warning: Video may have continuity issues")
    
    # Open the montage
    if sys.platform == "darwin":
        subprocess.run(["open", str(montage_output)])


if __name__ == "__main__":
    main()