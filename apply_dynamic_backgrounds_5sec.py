#!/usr/bin/env python3
"""
Apply dynamic backgrounds to 5-second test video using existing assets.
Demonstrates changing backgrounds at different timestamps.
"""

import subprocess
from pathlib import Path


def apply_dynamic_backgrounds():
    """Apply existing stock videos at different timestamps."""
    
    project_folder = Path("uploads/assets/videos/ai_math1")
    green_video = project_folder / "ai_math1_rvm_green_5s_024078685789.mp4"
    output_dir = Path("outputs")
    
    # Define segments with existing stock videos
    segments = [
        {
            "start": 0.0,
            "duration": 1.5,
            "stock_video": "ai_math1_background_7_2_9_5_zwvgcNhxei.mp4",
            "theme": "Math Innovation"
        },
        {
            "start": 1.5,
            "duration": 1.5,
            "stock_video": "ai_math1_background_27_9_31_6_6l4lJ9gGfk.mp4",
            "theme": "AI Technology"
        },
        {
            "start": 3.0,
            "duration": 1.0,
            "stock_video": "ai_math1_background_112_7_127_2_isS82K91sI.mp4",
            "theme": "Data Science"
        },
        {
            "start": 4.0,
            "duration": 1.0,
            "stock_video": "ai_math1_background_145_2_152_0_pZm2kl1dD3.mp4",
            "theme": "Future Trends"
        }
    ]
    
    print("🎬 Creating 5-second video with dynamic background changes\n")
    print("Background schedule:")
    for seg in segments:
        print(f"  {seg['start']:.1f}s - {seg['start']+seg['duration']:.1f}s: {seg['theme']}")
    print()
    
    processed = []
    
    for i, seg in enumerate(segments):
        stock_path = project_folder / seg['stock_video']
        if not stock_path.exists():
            print(f"⚠️ Stock video not found: {seg['stock_video']}")
            continue
        
        output_path = output_dir / f"dynamic_segment_{i}.mp4"
        print(f"Processing {seg['theme']} segment...")
        
        # Apply chromakey with refined settings
        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1", "-i", str(stock_path),
            "-ss", str(seg['start']), "-t", str(seg['duration']),
            "-i", str(green_video),
            "-filter_complex",
            "[0:v]scale=1280:720,setpts=PTS-STARTPTS[bg];"
            "[1:v]scale=1280:720,setpts=PTS-STARTPTS[fg];"
            "[fg]chromakey=green:0.08:0.04[keyed];"
            "[keyed]despill=type=green:mix=0.2:expand=0[clean];"
            "[bg][clean]overlay=shortest=1[out]",
            "-map", "[out]",
            "-map", "1:a?",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            processed.append(output_path)
            print(f"  ✅ Processed: {output_path.name}")
    
    # Concatenate segments
    if processed:
        concat_list = output_dir / "dynamic_concat.txt"
        with open(concat_list, 'w') as f:
            for path in processed:
                f.write(f"file '{path.absolute()}'\n")
        
        final = output_dir / "ai_math1_dynamic_backgrounds_5sec.mp4"
        
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(final)
        ], check=True, capture_output=True)
        
        print(f"\n✅ Final video with dynamic backgrounds: {final}")
        print("\n📊 Background changes:")
        print("  0.0s: Math cube drawing background")
        print("  1.5s: AI art generation background")
        print("  3.0s: Data notebook background")
        print("  4.0s: Crypto trends background")
        
        subprocess.run(["open", str(final)])


if __name__ == "__main__":
    apply_dynamic_backgrounds()