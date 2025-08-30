#!/usr/bin/env python3
"""
Test the pipeline with existing 5-second RVM to verify it works.
"""

from pathlib import Path
import subprocess
import json
from utils.video.background.background_cache_manager import BackgroundCacheManager

# Setup
video_path = Path("uploads/assets/videos/ai_math1.mp4")
project_folder = video_path.parent / video_path.stem
green_screen = project_folder / "ai_math1_rvm_green_5s_024078685789.mp4"
output_dir = Path("outputs")

print("Testing pipeline with 5-second RVM")
print("=" * 60)

# Mini allocation for 5 seconds
segments = [
    {"start_time": 0.0, "end_time": 2.0, "theme": "abstract_tech", "keywords": ["AI", "technology"]},
    {"start_time": 2.0, "end_time": 3.5, "theme": "mathematics", "keywords": ["math", "calculus"]},
    {"start_time": 3.5, "end_time": 5.0, "theme": "innovation", "keywords": ["create", "new"]}
]

cache_manager = BackgroundCacheManager()
processed = []

for i, seg in enumerate(segments):
    print(f"\n[{i+1}/3] {seg['theme']} ({seg['start_time']}-{seg['end_time']}s)")
    
    # Get background
    bg = cache_manager.get_best_match(theme=seg['theme'], keywords=seg['keywords'])
    
    if not bg:
        print(f"  No background for {seg['theme']}")
        continue
    
    print(f"  Using: {bg.name}")
    
    output = output_dir / f"test_seg_{i}.mp4"
    duration = seg['end_time'] - seg['start_time']
    
    cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1", "-i", str(bg),
        "-ss", str(seg['start_time']), "-t", str(duration),
        "-i", str(green_screen),
        "-filter_complex",
        "[0:v]scale=1280:720[bg];[1:v]scale=1280:720[fg];"
        "[fg]chromakey=green:0.08:0.04[keyed];"
        "[keyed]despill=type=green:mix=0.2[clean];"
        "[bg][clean]overlay[out]",
        "-map", "[out]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        str(output)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        processed.append(output)
        print(f"  ✅ Processed")

if processed:
    # Concatenate
    concat_list = output_dir / "test_concat.txt"
    with open(concat_list, 'w') as f:
        for p in processed:
            f.write(f"file '{p.absolute()}'\n")
    
    final = output_dir / "test_5sec_pipeline.mp4"
    
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        str(final)
    ], check=True, capture_output=True)
    
    print(f"\n✅ Test complete: {final}")
    subprocess.run(["open", str(final)])
else:
    print("\n❌ No segments processed")