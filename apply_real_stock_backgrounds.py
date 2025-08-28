#!/usr/bin/env python3
"""
Apply real stock videos from Coverr as backgrounds using chromakey.
Uses the videos downloaded by the Coverr API subagent.
"""

import subprocess
from pathlib import Path
import json


def apply_stock_background_chromakey(green_screen_video, stock_video, output_path, start_time, duration):
    """
    Replace green screen with real stock video background.
    """
    
    # Apply chromakey to replace green with stock video
    cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1",  # Loop background if shorter than segment
        "-i", str(stock_video),              # Stock video background
        "-ss", str(start_time),              # Seek in green screen
        "-t", str(duration),                 # Duration
        "-i", str(green_screen_video),       # Green screen foreground
        "-filter_complex",
        # Scale both to same size and apply chromakey
        "[0:v]scale=1280:720,setpts=PTS-STARTPTS[bg];"
        "[1:v]scale=1280:720,setpts=PTS-STARTPTS[fg];"
        # Apply chromakey with optimized settings
        "[fg]chromakey=green:0.15:0.10[keyed];"
        "[keyed]despill=type=green:mix=0.5:expand=0.8[clean];"
        # Overlay foreground on stock background
        "[bg][clean]overlay=shortest=1[out]",
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
        return False
    return True


def main():
    """Apply real Coverr stock videos as backgrounds."""
    
    # Paths
    project_folder = Path("uploads/assets/videos/ai_math1")
    green_screen_video = project_folder / "ai_math1_rvm_green_5s_024078685789.mp4"
    original_video = project_folder / "ai_math1_final.mp4"
    
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Stock videos downloaded by the subagent
    stock_segments = [
        {
            "start": 0.5,  # Using segments within our 5s green screen
            "duration": 1.2,
            "stock_video": "ai_math1_background_7_2_9_5_zwvgcNhxei.mp4",
            "description": "Math - Drawing a cube"
        },
        {
            "start": 1.7,
            "duration": 1.3,
            "stock_video": "ai_math1_background_27_9_31_6_6l4lJ9gGfk.mp4",
            "description": "AI - Man watching AI art"
        },
        {
            "start": 3.0,
            "duration": 1.0,
            "stock_video": "ai_math1_background_112_7_127_2_isS82K91sI.mp4",
            "description": "Data - Notebook flipping"
        },
        {
            "start": 4.0,
            "duration": 1.0,
            "stock_video": "ai_math1_background_145_2_152_0_pZm2kl1dD3.mp4",
            "description": "Trends - Crypto analysis"
        }
    ]
    
    print("Applying REAL Coverr stock videos as backgrounds...\n")
    
    processed = []
    
    for i, seg in enumerate(stock_segments):
        print(f"[{i+1}/{len(stock_segments)}] {seg['description']}")
        
        # Check if stock video exists
        stock_path = project_folder / seg['stock_video']
        if not stock_path.exists():
            print(f"  ⚠️  Stock video not found: {stock_path}")
            # Try without the 'ai_math1_' prefix if it was added twice
            alt_path = project_folder / seg['stock_video'].replace('ai_math1_ai_math1_', 'ai_math1_')
            if alt_path.exists():
                stock_path = alt_path
                print(f"  ✓ Found at alternate path: {alt_path.name}")
            else:
                print(f"  ❌ Skipping segment")
                continue
        
        # Apply chromakey with stock background
        output_path = output_dir / f"real_stock_segment_{i}.mp4"
        print(f"  Applying chromakey with {stock_path.name}...")
        
        success = apply_stock_background_chromakey(
            green_screen_video, stock_path, output_path,
            seg['start'], seg['duration']
        )
        
        if success:
            processed.append(output_path)
            print(f"  ✓ Complete\n")
        else:
            print(f"  ❌ Failed\n")
    
    if not processed:
        print("❌ No segments processed successfully")
        return
    
    # Create final montage
    print("Creating final video with REAL stock backgrounds...")
    concat_list = output_dir / "real_stock_concat.txt"
    with open(concat_list, 'w') as f:
        for path in processed:
            f.write(f"file '{path.absolute()}'\n")
    
    final = output_dir / "ai_math1_REAL_stock_backgrounds.mp4"
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
    
    print(f"✅ Final video with REAL stock backgrounds: {final}\n")
    
    # Create comparison showing original vs real stock backgrounds
    print("Creating comparison video...")
    
    # Extract matching segment from original
    original_segment = output_dir / "original_segment.mp4"
    extract_cmd = [
        "ffmpeg", "-y",
        "-ss", "0.5",
        "-i", str(original_video),
        "-t", "4.5",
        "-c", "copy",
        str(original_segment)
    ]
    subprocess.run(extract_cmd, check=True, capture_output=True)
    
    # Side-by-side comparison
    comparison = output_dir / "real_stock_comparison.mp4"
    compare_cmd = [
        "ffmpeg", "-y",
        "-i", str(original_segment),
        "-i", str(final),
        "-filter_complex",
        "[0:v]scale=640:360,drawtext=text='ORIGINAL':x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5[left];"
        "[1:v]scale=640:360,drawtext=text='REAL STOCK VIDEOS':x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5[right];"
        "[left][right]hstack[out]",
        "-map", "[out]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(comparison)
    ]
    subprocess.run(compare_cmd, check=True, capture_output=True)
    
    print(f"✅ Comparison video: {comparison}")
    
    # List the stock videos used
    print("\n📹 Stock Videos Used:")
    for seg in stock_segments:
        stock_path = project_folder / seg['stock_video']
        if stock_path.exists():
            size_mb = stock_path.stat().st_size / (1024 * 1024)
            print(f"  • {seg['description']}: {seg['stock_video']} ({size_mb:.1f} MB)")
    
    # Open results
    subprocess.run(["open", str(final)])
    subprocess.run(["open", str(comparison)])


if __name__ == "__main__":
    main()