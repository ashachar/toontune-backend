#!/usr/bin/env python3
"""
The RVM mask is not an alpha mask - it's just grayscale video!
Use the GREEN SCREEN version instead with proper chromakey.
"""

import subprocess
from pathlib import Path


def apply_stock_via_green_screen(green_video, stock_video, output_path, start_time, duration):
    """
    Use the green screen version with careful chromakey settings.
    """
    
    cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1", "-i", str(stock_video),    # Background
        "-ss", str(start_time), "-t", str(duration),
        "-i", str(green_video),                          # Green screen
        "-filter_complex",
        "[0:v]scale=1280:720,setpts=PTS-STARTPTS[bg];"
        "[1:v]scale=1280:720,setpts=PTS-STARTPTS[fg];"
        # Use chromakey with very specific green detection
        # Pure green in RVM output should be around RGB(0,255,0)
        "[fg]chromakey=0x00FF00:0.3:0.1[keyed];"
        # Remove green spill
        "[keyed]despill=type=green:mix=0.5:expand=0[clean];"
        # Overlay
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
    return result.returncode == 0


def main():
    """Use green screen since the mask is not actually an alpha mask."""
    
    project_folder = Path("uploads/assets/videos/ai_math1")
    green_video = project_folder / "ai_math1_rvm_green_5s_024078685789.mp4"
    
    output_dir = Path("outputs")
    
    # Stock videos
    stock_segments = [
        {
            "start": 0.0,
            "duration": 1.25,
            "stock_video": "ai_math1_background_7_2_9_5_zwvgcNhxei.mp4",
            "description": "Math - Drawing cube"
        },
        {
            "start": 1.25,
            "duration": 1.25,
            "stock_video": "ai_math1_background_27_9_31_6_6l4lJ9gGfk.mp4",
            "description": "AI - Man watching AI art"
        },
        {
            "start": 2.5,
            "duration": 1.25,
            "stock_video": "ai_math1_background_112_7_127_2_isS82K91sI.mp4",
            "description": "Data - Notebook"
        },
        {
            "start": 3.75,
            "duration": 1.25,
            "stock_video": "ai_math1_background_145_2_152_0_pZm2kl1dD3.mp4",
            "description": "Trends - Crypto"
        }
    ]
    
    print("🎬 Using GREEN SCREEN with chromakey (mask file is not an alpha mask)...\n")
    
    processed = []
    
    for i, seg in enumerate(stock_segments):
        print(f"[{i+1}/{len(stock_segments)}] {seg['description']}")
        
        stock_path = project_folder / seg['stock_video']
        if not stock_path.exists():
            continue
        
        output_path = output_dir / f"green_screen_segment_{i}.mp4"
        print(f"  Applying chromakey to green screen...")
        
        success = apply_stock_via_green_screen(
            green_video, stock_path, output_path,
            seg['start'], seg['duration']
        )
        
        if success:
            processed.append(output_path)
            print(f"  ✅ Complete\n")
    
    if processed:
        # Create final
        concat_list = output_dir / "green_concat.txt"
        with open(concat_list, 'w') as f:
            for path in processed:
                f.write(f"file '{path.absolute()}'\n")
        
        final = output_dir / "ai_math1_GREEN_SCREEN_backgrounds.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(final)
        ], check=True, capture_output=True)
        
        print(f"✅ FINAL: {final}")
        print("\n📌 The RVM 'mask' file was not an alpha mask - just grayscale video!")
        print("   Using the green screen version with chromakey instead.")
        
        subprocess.run(["open", str(final)])


if __name__ == "__main__":
    main()