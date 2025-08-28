#!/usr/bin/env python3
"""
Add stock video backgrounds at specific timestamps in the AI math video.
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
from utils.video.background.replace_background import BackgroundReplacer


# Define timestamps and search keywords for each segment
STOCK_VIDEO_SEGMENTS = [
    {
        "start": 7.28,
        "end": 9.6,
        "keywords": ["mathematics", "formula", "calculus", "equation", "blackboard"],
        "description": "derivative and integral symbols"
    },
    {
        "start": 27.92,
        "end": 31.6,
        "keywords": ["AI", "technology", "ChatGPT", "artificial intelligence", "digital"],
        "description": "ChatGPT and large language models"
    },
    {
        "start": 42.88,
        "end": 45.68,
        "keywords": ["abstract", "building", "blocks", "structure", "geometric"],
        "description": "building math in four parts"
    },
    {
        "start": 65.6,
        "end": 70.88,
        "keywords": ["cooking", "baking", "kitchen", "ingredients", "cake"],
        "description": "baking cake and ingredients metaphor"
    },
    {
        "start": 74.56,
        "end": 80.16,
        "keywords": ["scientist", "researcher", "technology", "innovation", "leader"],
        "description": "AI research leaders"
    },
    {
        "start": 89.84,
        "end": 95.04,
        "keywords": ["robot", "science", "laboratory", "future", "technology"],
        "description": "AI doing science vs robots"
    },
    {
        "start": 112.8,
        "end": 127.28,
        "keywords": ["data", "visualization", "graph", "analytics", "trends"],
        "description": "data science and calculus applications"
    },
    {
        "start": 127.28,
        "end": 134.72,
        "keywords": ["ancient", "history", "mathematics", "discovery", "invention"],
        "description": "math discovered vs invented debate"
    },
    {
        "start": 145.28,
        "end": 152.0,
        "keywords": ["trends", "graph", "visualization", "data", "analysis"],
        "description": "local trends in scientific literature"
    },
    {
        "start": 171.52,
        "end": 177.6,
        "keywords": ["creative", "art", "music", "generation", "AI"],
        "description": "AI generating images and music"
    }
]


def extract_segment(video_path, start, end, output_path):
    """Extract a segment from video using precise timestamps."""
    duration = end - start
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(video_path),
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-c:a", "copy",
        str(output_path)
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def process_with_background(segment_path, background_path, mask_path, output_path):
    """Composite segment with background using mask."""
    if mask_path and mask_path.exists():
        # Use existing mask
        cmd = [
            "ffmpeg", "-y",
            "-i", str(background_path),  # Background video
            "-i", str(segment_path),     # Foreground video  
            "-i", str(mask_path),        # Mask video
            "-filter_complex",
            "[0:v]scale=1920:1080,loop=loop=-1:size=1000[bg];"
            "[1:v]scale=1920:1080[fg];"
            "[2:v]scale=1920:1080,format=gray,loop=loop=-1:size=1000[mask];"
            "[bg][fg][mask]maskedmerge[out]",
            "-map", "[out]",
            "-map", "1:a?",
            "-shortest",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            str(output_path)
        ]
    else:
        # Create mask dynamically using BackgroundReplacer
        print("Creating mask dynamically...")
        replacer = BackgroundReplacer()
        temp_mask = Path(tempfile.mkdtemp()) / "mask.mp4"
        replacer.create_mask_video(segment_path, temp_mask)
        
        # Now composite with mask
        cmd = [
            "ffmpeg", "-y",
            "-i", str(background_path),
            "-i", str(segment_path),
            "-i", str(temp_mask),
            "-filter_complex",
            "[0:v]scale=1920:1080,loop=loop=-1:size=1000[bg];"
            "[1:v]scale=1920:1080[fg];"
            "[2:v]scale=1920:1080,format=gray[mask];"
            "[bg][fg][mask]maskedmerge[out]",
            "-map", "[out]",
            "-map", "1:a?",
            "-shortest",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            str(output_path)
        ]
    
    subprocess.run(cmd, check=True)


def main():
    """Process video with stock backgrounds at specified timestamps."""
    
    # Setup paths
    project_name = "ai_math1"
    project_folder = Path(f"uploads/assets/videos/{project_name}")
    video_path = project_folder / "ai_math1_final.mp4"
    mask_path = project_folder / "ai_math1_rvm_mask_5s_024078685789.mp4"
    
    if not video_path.exists():
        print(f"Error: Video not found at {video_path}")
        return
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize Coverr manager
    manager = CoverrManager(demo_mode=False)  # Set to True if no API key
    
    # Load transcript for context
    transcript_path = project_folder / "transcript.json"
    with open(transcript_path, 'r') as f:
        transcript_data = json.load(f)
    transcript_text = transcript_data['text']
    
    processed_segments = []
    
    print(f"Processing {len(STOCK_VIDEO_SEGMENTS)} segments with stock backgrounds...\n")
    
    for i, segment_info in enumerate(STOCK_VIDEO_SEGMENTS):
        print(f"[{i+1}/{len(STOCK_VIDEO_SEGMENTS)}] Processing {segment_info['start']:.1f}s - {segment_info['end']:.1f}s")
        print(f"  Description: {segment_info['description']}")
        
        # Extract segment
        segment_path = temp_dir / f"segment_{i:03d}.mp4"
        extract_segment(video_path, segment_info['start'], segment_info['end'], segment_path)
        
        # Search for and download background
        print(f"  Searching for background: {segment_info['keywords'][:3]}...")
        videos = manager.search_videos(segment_info['keywords'])
        
        if videos:
            # Select best video
            best_video = manager.select_best_video(videos, segment_info['description'])
            print(f"  Selected: {best_video['title']}")
            
            # Download background
            background_path = manager.download_video(
                best_video, project_name, project_folder,
                segment_info['start'], segment_info['end']
            )
        else:
            print("  No videos found, creating demo background...")
            background_path = temp_dir / f"demo_bg_{i}.mp4"
            manager.create_demo_background(background_path, 
                                         segment_info['end'] - segment_info['start'])
        
        # Process with background
        output_segment = temp_dir / f"output_{i:03d}.mp4"
        
        # Check if we have a mask for this timestamp range
        segment_mask = None
        if mask_path.exists() and segment_info['start'] < 5.0:
            # Use the 5-second mask for early segments
            segment_mask = mask_path
        
        print("  Compositing with background...")
        process_with_background(segment_path, background_path, segment_mask, output_segment)
        
        processed_segments.append({
            'path': output_segment,
            'start': segment_info['start'],
            'end': segment_info['end']
        })
        print(f"  ✓ Complete\n")
    
    # Now we need to merge all segments with the original video
    print("Creating final video with stock backgrounds...")
    
    # Create concat list
    concat_file = temp_dir / "concat.txt"
    with open(concat_file, 'w') as f:
        last_end = 0
        for seg in processed_segments:
            # Add original video before this segment
            if seg['start'] > last_end:
                f.write(f"file '{video_path}'\n")
                f.write(f"inpoint {last_end}\n")
                f.write(f"outpoint {seg['start']}\n")
            
            # Add processed segment
            f.write(f"file '{seg['path']}'\n")
            last_end = seg['end']
        
        # Add remaining original video
        video_duration = float(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
        ]).decode().strip())
        
        if last_end < video_duration:
            f.write(f"file '{video_path}'\n")
            f.write(f"inpoint {last_end}\n")
    
    # Concatenate
    final_output = output_dir / f"{project_name}_with_stock_backgrounds.mp4"
    concat_cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        str(final_output)
    ]
    
    subprocess.run(concat_cmd, check=True)
    
    print(f"\n✅ Complete! Output saved to: {final_output}")
    
    # Cleanup
    shutil.rmtree(temp_dir)
    
    # Open result
    if sys.platform == "darwin":
        subprocess.run(["open", str(final_output)])


if __name__ == "__main__":
    main()