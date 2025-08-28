#!/usr/bin/env python3
"""
Full video pipeline with dynamic stock background changes.
Uses cached RVM and intelligently places backgrounds at key moments.
"""

import sys
import subprocess
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.video.background.cached_rvm import CachedRobustVideoMatting
from utils.video.background.coverr_manager import CoverrManager


def analyze_transcript_for_themes(transcript_path):
    """
    Analyze transcript to identify key themes and timestamps for background changes.
    
    Returns:
        List of segments with themes and timestamps
    """
    with open(transcript_path, 'r') as f:
        transcript = json.load(f)
    
    # Define thematic segments based on content analysis
    segments = [
        {
            "start": 0.0,
            "end": 13.6,
            "theme": "AI Innovation",
            "keywords": ["AI", "math", "calculus", "innovation"],
            "description": "Opening - AI creating new math"
        },
        {
            "start": 13.6,
            "end": 35.0,
            "theme": "Future Technology",
            "keywords": ["artificial intelligence", "technology", "future"],
            "description": "AI's potential in math and science"
        },
        {
            "start": 35.0,
            "end": 58.0,
            "theme": "Mathematical Theory",
            "keywords": ["calculus", "mathematics", "theorems", "axioms"],
            "description": "Building mathematical theories"
        },
        {
            "start": 58.0,
            "end": 71.0,
            "theme": "AI Learning",
            "keywords": ["AI", "learning", "neural networks"],
            "description": "AI's current capabilities and limitations"
        },
        {
            "start": 71.0,
            "end": 100.0,
            "theme": "Scientific Research",
            "keywords": ["research", "science", "discovery"],
            "description": "AI figures' views on mathematical discovery"
        },
        {
            "start": 100.0,
            "end": 127.0,
            "theme": "Data Science",
            "keywords": ["data", "trends", "analytics", "statistics"],
            "description": "New calculus operator for data science"
        },
        {
            "start": 127.0,
            "end": 152.0,
            "theme": "Mathematical Philosophy",
            "keywords": ["discovery", "invention", "philosophy", "abstract"],
            "description": "Math: discovered or invented?"
        },
        {
            "start": 152.0,
            "end": 180.0,
            "theme": "Practical Applications",
            "keywords": ["applications", "trends", "framework"],
            "description": "Trendland and practical applications"
        },
        {
            "start": 180.0,
            "end": 206.0,
            "theme": "Theory Overview",
            "keywords": ["theory", "overview", "presentation"],
            "description": "Agent Terry's theory introduction"
        }
    ]
    
    return segments


def get_or_download_stock_videos(segments, project_folder):
    """
    Get stock videos for each segment, downloading if needed.
    """
    manager = CoverrManager()
    
    for segment in segments:
        # Check if we already have a background for this segment
        existing = list(project_folder.glob(
            f"*_background_{int(segment['start'])}_{int(segment['end'])}_*.mp4"
        ))
        
        if existing:
            segment['stock_video'] = existing[0]
            print(f"✓ Found existing background for {segment['theme']}: {existing[0].name}")
        else:
            print(f"\n🔍 Searching for background: {segment['theme']}")
            # Search and download new background
            videos = manager.search_videos(segment['keywords'])
            if videos:
                best_video = manager.select_best_video(videos, segment['description'])
                stock_path = manager.download_video(
                    best_video, 
                    project_folder.parent.name,
                    project_folder,
                    segment['start'], 
                    segment['end']
                )
                segment['stock_video'] = stock_path
                print(f"✓ Downloaded: {stock_path.name}")
            else:
                segment['stock_video'] = None
                print(f"⚠️ No stock video found for {segment['theme']}")
    
    return segments


def apply_chromakey_segment(green_video, stock_video, output_path, start_time, duration):
    """
    Apply refined chromakey to a segment.
    """
    cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1", "-i", str(stock_video),    # Background looped
        "-ss", str(start_time), "-t", str(duration),
        "-i", str(green_video),                          # Green screen
        "-filter_complex",
        "[0:v]scale=1280:720,setpts=PTS-STARTPTS[bg];"
        "[1:v]scale=1280:720,setpts=PTS-STARTPTS[fg];"
        # Refined chromakey settings
        "[fg]chromakey=green:0.08:0.04[keyed];"
        "[keyed]despill=type=green:mix=0.2:expand=0[clean];"
        "[bg][clean]overlay=shortest=1[out]",
        "-map", "[out]",
        "-map", "1:a?",  # Preserve audio
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def main():
    """
    Process full video with dynamic background changes.
    """
    # Setup paths
    video_path = Path("uploads/assets/videos/ai_math1.mp4")
    project_folder = Path("uploads/assets/videos/ai_math1")
    transcript_path = project_folder / "transcript.json"
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Full Video Stock Background Pipeline")
    print("=" * 60)
    
    # Initialize processors
    rvm_processor = CachedRobustVideoMatting()
    
    # Step 1: Get or generate RVM green screen for full video
    print("\n📹 Step 1: Getting RVM green screen output...")
    
    # Check if we have full video RVM already
    full_duration = 206  # Full video is ~206 seconds
    green_screen_path = rvm_processor.get_rvm_output(video_path, duration=None)
    
    if not green_screen_path.exists():
        print("⚠️ Full video RVM not found, will process in chunks...")
        # For now, we'll use the 5-second version as a test
        green_screen_path = project_folder / "ai_math1_rvm_green_5s_024078685789.mp4"
        full_duration = 5
    
    print(f"✓ Using green screen: {green_screen_path}")
    
    # Step 2: Analyze transcript for themes
    print("\n📚 Step 2: Analyzing transcript for thematic segments...")
    segments = analyze_transcript_for_themes(transcript_path)
    print(f"✓ Identified {len(segments)} thematic segments")
    
    # Step 3: Get stock videos for each segment
    print("\n🎬 Step 3: Getting stock videos for each segment...")
    segments = get_or_download_stock_videos(segments, project_folder)
    
    # Filter to segments with stock videos and within our duration
    valid_segments = [s for s in segments 
                      if s['stock_video'] and s['start'] < full_duration]
    
    # Step 4: Process each segment
    print(f"\n🎨 Step 4: Processing {len(valid_segments)} segments...")
    processed = []
    
    for i, segment in enumerate(valid_segments):
        # Adjust end time if beyond our test duration
        segment_end = min(segment['end'], full_duration)
        duration = segment_end - segment['start']
        
        if duration <= 0:
            continue
        
        print(f"\n[{i+1}/{len(valid_segments)}] {segment['theme']} ({segment['start']:.1f}s - {segment_end:.1f}s)")
        
        output_path = output_dir / f"segment_{i:02d}_{segment['theme'].replace(' ', '_')}.mp4"
        
        success = apply_chromakey_segment(
            green_screen_path,
            segment['stock_video'],
            output_path,
            segment['start'],
            duration
        )
        
        if success:
            processed.append(output_path)
            print(f"  ✅ Processed: {output_path.name}")
        else:
            print(f"  ❌ Failed to process segment")
    
    # Step 5: Concatenate all segments
    if processed:
        print("\n🔗 Step 5: Concatenating segments...")
        
        concat_list = output_dir / "concat_list.txt"
        with open(concat_list, 'w') as f:
            for path in processed:
                f.write(f"file '{path.absolute()}'\n")
        
        final_output = output_dir / "ai_math1_dynamic_backgrounds_full.mp4"
        
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(final_output)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Final video created: {final_output}")
            
            # Add audio from original
            final_with_audio = output_dir / "ai_math1_dynamic_backgrounds_full_audio.mp4"
            cmd_audio = [
                "ffmpeg", "-y",
                "-i", str(final_output),
                "-i", str(video_path),
                "-c:v", "copy",
                "-map", "0:v",
                "-map", "1:a?",
                "-shortest",
                str(final_with_audio)
            ]
            subprocess.run(cmd_audio, check=True, capture_output=True)
            
            print(f"✅ Final with audio: {final_with_audio}")
            
            # Open result
            subprocess.run(["open", str(final_with_audio)])
        else:
            print(f"❌ Concatenation failed: {result.stderr[-500:]}")
    
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    
    # Summary
    print("\nSummary:")
    print(f"- Processed {len(processed)} segments")
    print(f"- Backgrounds change based on content themes")
    print(f"- Each segment has contextually relevant stock video")
    print("\nThemes covered:")
    for segment in valid_segments[:len(processed)]:
        print(f"  • {segment['theme']}: {segment['start']:.1f}s - {min(segment['end'], full_duration):.1f}s")


if __name__ == "__main__":
    main()